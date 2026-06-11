"""Phase J: held-out secondary blind set for the northern-Sierra placer model.

Combines two secondary positive-set sources that none of the placer
classifiers were trained on:

  1. USMIN placer/gravel points (USGS historic mine features from topo
     sheets; ~hundreds of CA placer/gravel records inside the AOI).
  2. Lindgren PP73 secondary diggings centroids (~15-25 hand-geocoded
     districts named in PP73 plates 1/2/6 but NOT in the 7 anchor
     districts that already define the Phase 1 gate).

For each surviving secondary point we look up:

  - Phase 1 index decile (from `phase1_index_250m.parquet`)
  - Phase 2 fused decile (from `prospectivity_placer_northern_sierra_250m_fused.parquet`)
  - Per-population calibrated decile (`pop_calibrated_<pop>_250m.parquet`)

A secondary positive is dropped from the test set if it lies within 500 m
of any training positive (Hydraulic Pit polygon edge, MRDS placer point,
or anchor-district centroid). The remaining points are the true held-out
test set.

Decision rule (consumed by the Phase L decision matrix in the model card):

  secondary_median_decile within 1 of anchor median       -> clean transfer
  1-2 deciles worse                                       -> marginal; flag
  > 2 deciles worse                                       -> anchor-overfit; flag prominently

Outputs (under data/derived/northern_sierra_placer/):

  lindgren_secondary_blind_set_results.csv
      point_name, point_source (USMIN / Lindgren), x, y, cell_idx,
      <score>_decile for each score column present.
  lindgren_secondary_summary.csv
      model, n_points, median_decile, frac_top_decile, frac_top_quintile,
      anchor_median_decile, delta_median.

Usage:
    .venv/bin/python scripts/northern_sierra_placer/lindgren_blind_set.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from ai_minerals.data.adapters import get_adapter
from ai_minerals.data.adapters.occurrences.mrds import _PLACER_DEP_TYPE_RE
from ai_minerals.data.usmin import fetch as fetch_usmin
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS
from ai_minerals.regions._northern_sierra_lindgren_secondaries import (
    LINDGREN_QUATERNARY_NAMES,
    LINDGREN_SECONDARY_DIGGINGS,
    LINDGREN_TERTIARY_NAMES,
)
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
DEDUP_BUFFER_M = 500.0

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
OUT_DIR = DATA_DERIVED / REGION.data_prefix

IN_PHASE1 = OUT_DIR / "phase1_index_250m.parquet"
IN_FUSED = OUT_DIR / "prospectivity_placer_northern_sierra_250m_fused.parquet"
IN_CAL_T = OUT_DIR / "pop_calibrated_placer_tertiary_250m.parquet"
IN_CAL_Q = OUT_DIR / "pop_calibrated_placer_quaternary_250m.parquet"

IN_ANCHOR_TABLE = OUT_DIR / "anchor_districts_decile_table.csv"

OUT_RESULTS = OUT_DIR / "lindgren_secondary_blind_set_results.csv"
OUT_SUMMARY = OUT_DIR / "lindgren_secondary_summary.csv"


def _load_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_grid_xy(phase1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return phase1["x"].to_numpy(), phase1["y"].to_numpy()


def _nearest_cell_idx(
    pts_xy: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, frame_index: pd.Index
) -> np.ndarray:
    """For each point in `pts_xy` (N x 2, working CRS), return the frame index of
    the nearest grid cell. Brute force; OK for ~thousands of points x ~500k cells.
    """
    out = np.empty(len(pts_xy), dtype=np.int64)
    for i, (px, py) in enumerate(pts_xy):
        d2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
        out[i] = int(frame_index[np.argmin(d2)])
    return out


def _within_buffer_mask(
    pts_xy: np.ndarray, ref_xy: np.ndarray, buffer_m: float
) -> np.ndarray:
    """True for each point that lies within `buffer_m` of any reference point.

    Brute O(N*M); fine for our sizes (M up to a few thousand on each side).
    """
    if len(ref_xy) == 0:
        return np.zeros(len(pts_xy), dtype=bool)
    out = np.zeros(len(pts_xy), dtype=bool)
    b2 = buffer_m ** 2
    for i, (px, py) in enumerate(pts_xy):
        d2 = (ref_xy[:, 0] - px) ** 2 + (ref_xy[:, 1] - py) ** 2
        if d2.min() <= b2:
            out[i] = True
    return out


def _within_polygon_buffer_mask(
    pts_geom_wgs84: gpd.GeoSeries, polys_gdf: gpd.GeoDataFrame, buffer_m: float
) -> np.ndarray:
    """True for each point within `buffer_m` of a polygon edge.

    Polygons buffered in the region working CRS to keep meters honest.
    """
    if len(polys_gdf) == 0:
        return np.zeros(len(pts_geom_wgs84), dtype=bool)
    polys_proj = polys_gdf.to_crs(REGION.working_crs)
    buf = polys_proj.geometry.buffer(buffer_m).union_all()
    if isinstance(pts_geom_wgs84, gpd.GeoSeries) and pts_geom_wgs84.crs is not None:
        pts_proj = pts_geom_wgs84.to_crs(REGION.working_crs)
    else:
        pts_proj = gpd.GeoSeries(pts_geom_wgs84, crs="EPSG:4326").to_crs(REGION.working_crs)
    return np.array([buf.intersects(p) for p in pts_proj], dtype=bool)


def _load_usmin_points() -> gpd.GeoDataFrame:
    """Fetch + load USMIN placer/gravel points (canonical occurrences)."""
    usmin_path = fetch_usmin(REGION.aoi)
    loader = get_adapter("occurrences", "usmin")
    gdf = loader(usmin_path, REGION.aoi)
    # Final AOI clip (the adapter trusts the fetcher's clip, but belt-and-suspenders).
    aoi_box = REGION.aoi.polygon
    return gdf[gdf.intersects(aoi_box)].copy()


def _load_lindgren_points() -> gpd.GeoDataFrame:
    """Build a canonical-occurrence GeoDataFrame from the Lindgren fixture."""
    names = list(LINDGREN_SECONDARY_DIGGINGS)
    geom = [Point(lon, lat) for (lon, lat) in (LINDGREN_SECONDARY_DIGGINGS[n] for n in names)]
    return gpd.GeoDataFrame(
        {
            "geometry": geom,
            "source": "Lindgren",
            "ftr_name": names,
            "raw_record_id": names,
        },
        crs="EPSG:4326",
    )


def _load_mrds_placer_points() -> gpd.GeoDataFrame:
    """Load MRDS records whose dep_type matches the placer regex."""
    mrds_path = REGION.raw_paths.get("lode_mrds")
    if mrds_path is None or not Path(mrds_path).exists():
        print(f"WARN: lode_mrds path missing ({mrds_path}); skipping MRDS dedup.",
              file=sys.stderr)
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    gdf = gpd.read_file(mrds_path)
    if "dep_type" not in gdf.columns:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    dep = gdf["dep_type"].astype("string").fillna("")
    mask = dep.apply(lambda s: bool(_PLACER_DEP_TYPE_RE.search(s))) if len(gdf) else pd.Series([], dtype=bool)
    return gdf.loc[mask, ["geometry"]].copy()


def _decile_rank(score: pd.Series) -> pd.Series:
    """Decile of each cell in score's distribution; 0 = top decile, 9 = bottom."""
    ranked = score.rank(method="first", ascending=False, na_option="keep")
    return pd.qcut(ranked, q=10, labels=range(10))


def _project_to_working_crs(
    geom: gpd.GeoSeries,
) -> np.ndarray:
    """Project a point series to the region working CRS as an (N, 2) ndarray."""
    if isinstance(geom, gpd.GeoSeries) and geom.crs is not None:
        series = geom
    else:
        series = gpd.GeoSeries(geom, crs="EPSG:4326")
    projected = series.to_crs(REGION.working_crs)
    return np.column_stack([projected.x.to_numpy(), projected.y.to_numpy()])


def _anchor_xy_working() -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    return np.array(
        [transformer.transform(lon, lat) for (lon, lat) in ANCHOR_DISTRICTS.values()],
        dtype=float,
    )


def _classify_point_source(row: gpd.GeoSeries) -> str:
    """USMIN or Lindgren."""
    src = str(row.get("source", "") or "")
    if src == "Lindgren":
        return "Lindgren"
    return "USMIN"


def _lindgren_pop(name: str) -> str | None:
    if name in LINDGREN_TERTIARY_NAMES:
        return "placer_tertiary"
    if name in LINDGREN_QUATERNARY_NAMES:
        return "placer_quaternary"
    return None


def _summarize_per_model(
    results: pd.DataFrame, decile_col: str, anchor_median: float | None
) -> dict[str, float | int]:
    """Aggregate stats for one score column."""
    vals = results[decile_col].dropna().astype(int)
    n = len(vals)
    if n == 0:
        return {
            "n_points": 0,
            "median_decile": float("nan"),
            "frac_top_decile": float("nan"),
            "frac_top_quintile": float("nan"),
            "anchor_median_decile": anchor_median if anchor_median is not None else float("nan"),
            "delta_median": float("nan"),
        }
    median = float(np.median(vals))
    return {
        "n_points": int(n),
        "median_decile": median,
        "frac_top_decile": float((vals == 0).mean()),
        "frac_top_quintile": float((vals <= 1).mean()),
        "anchor_median_decile": anchor_median if anchor_median is not None else float("nan"),
        "delta_median": (
            float(median - anchor_median) if anchor_median is not None else float("nan")
        ),
    }


def _anchor_medians_from_table(
    anchor_table: pd.DataFrame | None,
) -> dict[str, float]:
    """Per-model anchor median decile from anchor_districts_decile_table.csv."""
    if anchor_table is None:
        return {}
    out: dict[str, float] = {}
    for col in anchor_table.columns:
        m = re.match(r"^(.+)_decile$", col)
        if not m:
            continue
        score_col = m.group(1)
        vals = pd.to_numeric(anchor_table[col], errors="coerce").dropna()
        if len(vals):
            out[score_col] = float(np.median(vals))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=DEDUP_BUFFER_M,
        help=f"Dedup buffer in meters against training positives (default {DEDUP_BUFFER_M}).",
    )
    args = parser.parse_args(argv)

    # ---- 1. Phase 1 / fused / per-pop frames ----
    if not IN_PHASE1.exists():
        print(f"ERROR: {IN_PHASE1} missing. Run Phase 1 first.", file=sys.stderr)
        return 2
    phase1 = pd.read_parquet(IN_PHASE1)
    print(f"==> Loaded Phase 1 ({len(phase1):,} cells)")

    fused = _load_optional_parquet(IN_FUSED)
    if fused is None:
        print(f"ERROR: {IN_FUSED} missing. Phase F (calibrate-and-fuse) has not run; "
              f"can't score secondary points against the fused raster.", file=sys.stderr)
        return 2

    cal_t = _load_optional_parquet(IN_CAL_T)
    cal_q = _load_optional_parquet(IN_CAL_Q)
    if cal_t is None:
        print(f"WARN: {IN_CAL_T} missing; will skip placer_tertiary score column.")
    if cal_q is None:
        print(f"WARN: {IN_CAL_Q} missing; will skip placer_quaternary score column.")

    # ---- 2. Build a single merged score table on (row, col) ----
    df = phase1[["row", "col", "x", "y", "phase1_score"]].copy()
    df = df.merge(
        fused[["row", "col", "p_fused"]], on=["row", "col"], how="left"
    )
    if cal_t is not None:
        df = df.merge(
            cal_t[["row", "col", "p_cal"]].rename(columns={"p_cal": "p_cal_placer_tertiary"}),
            on=["row", "col"], how="left",
        )
    if cal_q is not None:
        df = df.merge(
            cal_q[["row", "col", "p_cal"]].rename(columns={"p_cal": "p_cal_placer_quaternary"}),
            on=["row", "col"], how="left",
        )
    df = df.reset_index(drop=True)
    grid_x, grid_y = _load_grid_xy(df)

    # ---- 3. Pull secondary candidate points ----
    print("==> Fetching USMIN placer/gravel points")
    try:
        usmin_gdf = _load_usmin_points()
    except Exception as exc:  # narrow: network or unzip failure
        print(f"WARN: USMIN fetch/load failed ({exc}); proceeding with Lindgren only.",
              file=sys.stderr)
        usmin_gdf = gpd.GeoDataFrame(
            {"geometry": [], "source": [], "ftr_name": [], "raw_record_id": []},
            crs="EPSG:4326",
        )
    print(f"    USMIN points in AOI (placer/gravel filter applied): {len(usmin_gdf):,}")
    usmin_gap_note = (len(usmin_gdf) == 0)

    print("==> Loading Lindgren PP73 secondary diggings fixture")
    lindgren_gdf = _load_lindgren_points()
    print(f"    Lindgren centroids: {len(lindgren_gdf):,}")

    cand = pd.concat(
        [
            usmin_gdf[["geometry", "source", "ftr_name", "raw_record_id"]],
            lindgren_gdf[["geometry", "source", "ftr_name", "raw_record_id"]],
        ],
        ignore_index=True,
    )
    cand = gpd.GeoDataFrame(cand, geometry="geometry", crs="EPSG:4326")
    n_pre_dedup = len(cand)

    # ---- 4. Dedup against training positives (hydraulic pits, anchors, MRDS placer) ----
    print(f"==> Dedup secondaries against training positives (buffer={args.buffer_m:.0f} m)")
    # Hydraulic-pit polygons
    pit_path = REGION.raw_paths.get("hydraulic_pits")
    if pit_path is not None and Path(pit_path).exists():
        pit_loader = get_adapter("geology", "hydraulic_pits")
        pit_polys = pit_loader(Path(pit_path), REGION.aoi)
    else:
        print(f"WARN: hydraulic_pits path missing ({pit_path}); skipping pit dedup.",
              file=sys.stderr)
        pit_polys = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    near_pit = _within_polygon_buffer_mask(cand.geometry, pit_polys, args.buffer_m)
    # Anchor centroids
    anchor_xy = _anchor_xy_working()
    cand_xy_working = _project_to_working_crs(cand.geometry)
    near_anchor = _within_buffer_mask(cand_xy_working, anchor_xy, args.buffer_m)
    # MRDS placer points
    mrds_placer = _load_mrds_placer_points()
    if len(mrds_placer):
        mrds_xy = _project_to_working_crs(mrds_placer.geometry)
    else:
        mrds_xy = np.empty((0, 2), dtype=float)
    near_mrds = _within_buffer_mask(cand_xy_working, mrds_xy, args.buffer_m)

    drop = near_pit | near_anchor | near_mrds
    print(
        f"    pre-dedup={n_pre_dedup}, "
        f"near_pit={int(near_pit.sum())}, "
        f"near_anchor={int(near_anchor.sum())}, "
        f"near_mrds={int(near_mrds.sum())}, "
        f"dropped={int(drop.sum())}"
    )
    cand = cand.loc[~drop].copy().reset_index(drop=True)
    cand_xy_working = cand_xy_working[~drop]
    if len(cand) == 0:
        print("ERROR: zero secondary points remain after dedup; nothing to evaluate.",
              file=sys.stderr)
        return 2

    # ---- 5. Snap survivors to nearest grid cell ----
    cell_idx = _nearest_cell_idx(cand_xy_working, grid_x, grid_y, df.index)

    # ---- 6. Score columns + per-point deciles ----
    score_cols = [
        c for c in ("phase1_score", "p_fused", "p_cal_placer_tertiary", "p_cal_placer_quaternary")
        if c in df.columns
    ]
    deciles_by_col = {c: _decile_rank(df[c]) for c in score_cols}

    rows: list[dict[str, object]] = []
    for i, (_, row) in enumerate(cand.iterrows()):
        idx = int(cell_idx[i])
        rec: dict[str, object] = {
            "point_name": row.get("ftr_name") or row.get("raw_record_id") or f"pt_{i}",
            "point_source": _classify_point_source(row),
            "x": float(df.at[idx, "x"]),
            "y": float(df.at[idx, "y"]),
            "cell_idx": idx,
        }
        for c in score_cols:
            d = deciles_by_col[c].loc[idx]
            rec[f"{c}_decile"] = int(d) if pd.notna(d) else np.nan
            v = df.at[idx, c]
            rec[f"{c}_value"] = float(v) if pd.notna(v) else np.nan
        rows.append(rec)
    results = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_RESULTS, index=False)
    print(f"==> Wrote per-point blind-set results: {OUT_RESULTS} ({len(results)} rows)")

    # ---- 7. Per-model summary + anchor delta ----
    anchor_table = pd.read_csv(IN_ANCHOR_TABLE) if IN_ANCHOR_TABLE.exists() else None
    if anchor_table is None:
        print(f"WARN: {IN_ANCHOR_TABLE} missing; anchor delta will be NaN.")
    anchor_medians = _anchor_medians_from_table(anchor_table)

    summary_rows: list[dict[str, object]] = []
    for c in score_cols:
        decile_col = f"{c}_decile"
        s = _summarize_per_model(results, decile_col, anchor_medians.get(c))
        s["model"] = c
        summary_rows.append(s)

    # Lindgren-population-aware row: for each Lindgren-only subset, also
    # report against the matching p_cal_<pop> column (so we can see if a
    # Tertiary-named secondary lands strongly in the Tertiary model).
    lindgren_only = results[results["point_source"] == "Lindgren"].copy()
    if len(lindgren_only):
        lindgren_only["lindgren_pop"] = lindgren_only["point_name"].map(_lindgren_pop)
        for pop in ("placer_tertiary", "placer_quaternary"):
            col = f"p_cal_{pop}"
            decile_col = f"{col}_decile"
            if decile_col not in lindgren_only.columns:
                continue
            sub = lindgren_only[lindgren_only["lindgren_pop"] == pop]
            if len(sub) == 0:
                continue
            s = _summarize_per_model(sub, decile_col, anchor_medians.get(col))
            s["model"] = f"{col}__lindgren_{pop}_only"
            summary_rows.append(s)

    summary = pd.DataFrame(summary_rows)
    # Stable column order.
    summary = summary[
        [
            "model", "n_points", "median_decile",
            "frac_top_decile", "frac_top_quintile",
            "anchor_median_decile", "delta_median",
        ]
    ]
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"==> Wrote per-model summary: {OUT_SUMMARY}")

    # ---- 8. Stdout summary ----
    print()
    print("Per-model decile performance on the secondary blind set:")
    for _, r in summary.iterrows():
        delta = r["delta_median"]
        delta_str = f"{delta:+.1f}" if pd.notna(delta) else "n/a"
        frac_top = r["frac_top_decile"]
        frac_top_str = f"{100 * frac_top:.0f}%" if pd.notna(frac_top) else "n/a"
        print(
            f"  {r['model']:48s}  n={int(r['n_points']):3d}  "
            f"median decile={r['median_decile']:.1f}  "
            f"top-decile frac={frac_top_str}  "
            f"anchor delta={delta_str}"
        )

    if usmin_gap_note:
        print()
        print("NOTE: zero USMIN placer/gravel points landed in the AOI after the filter "
              "(check FTR_TYPE values in data/raw/usmin/usmin_<aoi>.gpkg). Summary reflects "
              "Lindgren centroids only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
