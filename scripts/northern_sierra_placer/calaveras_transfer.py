"""Phase I (coarse): Sierra -> Calaveras placer-Au transfer dry-run.

Informational only. Builds a stripped-down Calaveras 250 m feature frame
covering only the placer features the deterministic Phase 1 USGS-Alaska
scorer needs (`distance_downstream_from_lode_m`, `hydraulic_pit_proximity_m`,
`spi_band`, `catchment_au_hawkes`, `is_quaternary_alluvium`, `twi`,
`geomorphon_terrace_mask`, `slope`), runs the same scorer with the same
default weights as the northern-Sierra Phase 1 driver, snaps each
Calaveras anchor district to its nearest grid cell, and reports the
decile rank under the (deterministic) Sierra Phase 1 model.

Why the deterministic Phase 1 scorer and not the trained Phase 2 stack:
  - the Phase 2 train_predict driver writes per-cell predictions but does
    not serialize the fitted estimator, so transferring the calibrated
    stack to a new AOI would require an in-script retraining pass on the
    Sierra features. That's expensive and orthogonal to the dry-run's
    purpose, which is to confirm that the placer feature stack carries
    real signal outside the data the Sierra model was tuned to.
  - the Phase 1 scorer is closed-form: every weight is in
    `scorers/usgs_alaska_placer.DEFAULT_WEIGHTS`, no fitted state, so
    transferring it to Calaveras is exactly "score Calaveras with the
    same weights and inspect the result."

Wiring up a Phase 2 transfer is left as a TODO: serialize the stacking
+ isotonic estimator from `northern_sierra_placer_train_predict_250m.py`
via joblib, load it here, run on Calaveras features.

Outputs (under data/derived/calaveras_placer/):
  transfer_metrics.csv     per-district decile rank + capture summary
  phase1_calaveras.parquet (row, col, x, y, phase1_score) for re-plot

Usage:
    .venv/bin/python scripts/northern_sierra_placer/calaveras_transfer.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import from_origin

from ai_minerals.data.adapters import get_adapter
from ai_minerals.features.hydrology import (
    distance_downstream_from_lode,
    flow_accumulation,
    geomorphon_terrace_mask,
    stream_power_index_band,
    topographic_wetness_index,
)
from ai_minerals.features.placer_geology import (
    hawkes_dual_decay_catchment,
    hydraulic_pit_proximity_m,
    is_quaternary_alluvium,
)
from ai_minerals.features.rasters import sample_raster, slope_and_tri
from ai_minerals.grid import build_grid
from ai_minerals.metrics import bootstrap_capture_ci
from ai_minerals.regions._calaveras_anchors import ANCHOR_DISTRICTS as CALAVERAS_ANCHORS
from ai_minerals.regions.calaveras_placer import CALAVERAS_PLACER_REGION
from ai_minerals.scorers.usgs_alaska_placer import (
    PALEOCHANNEL_PHASE1_PROXY,
    usgs_alaska_placer_index,
)


REGION = CALAVERAS_PLACER_REGION
RES_M = 250

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DERIVED = REPO_ROOT / "data" / "derived"
SIERRA_DIR = DATA_DERIVED / "northern_sierra_placer"

OUT_DIR = DATA_DERIVED / REGION.data_prefix
OUT_METRICS = OUT_DIR / "transfer_metrics.csv"
OUT_PHASE1 = OUT_DIR / "phase1_calaveras.parquet"

SIERRA_ANCHOR_TABLE = SIERRA_DIR / "anchor_districts_decile_table.csv"


def _make_grid():
    return build_grid(REGION.aoi, resolution_m=RES_M, working_crs=REGION.working_crs)


def _dem_at_grid(dem_path: Path, grid):
    dem_asc_y = sample_raster(dem_path, grid)
    r = grid.resolution_m
    transform = from_origin(grid.xs[0] - r / 2, grid.ys[-1] + r / 2, r, r)
    return dem_asc_y[::-1, :], transform


def _ravel_north_up(arr_north_up: np.ndarray) -> np.ndarray:
    return arr_north_up[::-1, :].ravel()


def _placer_dep_type_mask(dep_type: pd.Series) -> pd.Series:
    pattern = r"placer|alluvial|stream.?placer|paleo.?placer|black.?sand|residual|eluvial"
    dep = dep_type.astype("string").fillna("").str.lower()
    return dep.str.contains(pattern, regex=True, na=False)


def _filter_lode_seeds(mrds: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "dev_stat" not in mrds.columns:
        mrds["dev_stat"] = pd.Series([None] * len(mrds), dtype="string")
    if "dep_type" not in mrds.columns:
        mrds["dep_type"] = pd.Series([None] * len(mrds), dtype="string")
    dev = mrds["dev_stat"].astype("string").fillna("")
    keep_dev = dev.isin(["Past Producer", "Producer"])
    is_placer = _placer_dep_type_mask(mrds["dep_type"])
    return mrds.loc[keep_dev & ~is_placer].copy()


def build_calaveras_feature_frame() -> pd.DataFrame:
    """Stripped-down per-cell feature frame for the Phase 1 scorer on Calaveras.

    Builds the eight Phase 1 features plus identity columns; everything
    else is filled with NaN. Documents which inputs were available and
    which were synthesized.
    """
    grid = _make_grid()
    centroid_gdf = grid.centroid_gdf()
    df = pd.DataFrame({
        "row": np.repeat(np.arange(grid.shape[0]), grid.shape[1]),
        "col": np.tile(np.arange(grid.shape[1]), grid.shape[0]),
        "x": centroid_gdf.geometry.x.to_numpy(),
        "y": centroid_gdf.geometry.y.to_numpy(),
    })

    # --- DEM derivatives.
    dem_path = REGION.raw_paths["dem"]
    if not dem_path.exists():
        warnings.warn(f"DEM missing at {dem_path}; slope/twi/spi/geomorphon NaN.")
        df["slope"] = np.nan
        df["twi"] = np.nan
        df["spi_band"] = np.nan
        df["geomorphon_terrace_mask"] = np.nan
    else:
        dem_north_up, transform = _dem_at_grid(dem_path, grid)
        slope_north_up, _ = slope_and_tri(dem_north_up, grid.resolution_m)
        df["slope"] = _ravel_north_up(slope_north_up)
        try:
            flow_acc_north_up = flow_accumulation(dem_north_up, transform=transform)
        except RuntimeError as exc:
            warnings.warn(f"flow_accumulation unavailable ({exc}); twi/spi NaN.")
            flow_acc_north_up = None
        if flow_acc_north_up is None:
            df["twi"] = np.nan
            df["spi_band"] = np.nan
        else:
            df["spi_band"] = _ravel_north_up(
                stream_power_index_band(flow_acc_north_up, slope_north_up)
            )
            df["twi"] = _ravel_north_up(
                topographic_wetness_index(flow_acc_north_up, slope_north_up)
            )
        try:
            mask = geomorphon_terrace_mask(dem_path)
        except RuntimeError as exc:
            warnings.warn(f"geomorphon unavailable ({exc}); terrace mask NaN.")
            mask = None
        if mask is None:
            df["geomorphon_terrace_mask"] = np.nan
        else:
            gmpath = dem_path.with_suffix(dem_path.suffix + ".geomorphon.tif")
            mask_grid = sample_raster(gmpath, grid)
            df["geomorphon_terrace_mask"] = np.isin(
                np.rint(mask_grid).astype(np.int32), (1, 7, 8)
            ).astype(np.float32).ravel()

    # --- Hydraulic-pit proximity. The CA-wide pit polygons cover Calaveras.
    pit_path = REGION.raw_paths.get("hydraulic_pits")
    if pit_path is None or not pit_path.exists():
        warnings.warn(f"hydraulic_pits missing at {pit_path}; proximity NaN.")
        df["hydraulic_pit_proximity_m"] = np.nan
    else:
        pit_polys = get_adapter("geology", "hydraulic_pits")(pit_path, REGION.aoi)
        if len(pit_polys) == 0:
            warnings.warn(
                f"hydraulic_pits has zero polygons inside Calaveras AOI. "
                f"hydraulic_pit_proximity_m falls back to NaN; the Phase 1 "
                f"paleochannel proxy will also be NaN."
            )
            df["hydraulic_pit_proximity_m"] = np.nan
        else:
            prox = hydraulic_pit_proximity_m(pit_polys, grid)
            df["hydraulic_pit_proximity_m"] = prox.to_numpy(dtype=np.float32)

    # --- Quaternary-alluvium mask via CGS 2010.
    geo_path = REGION.raw_paths.get("geology")
    if geo_path is None or not geo_path.exists():
        warnings.warn(
            f"geology missing at {geo_path}; is_quaternary_alluvium NaN."
        )
        df["is_quaternary_alluvium"] = np.nan
    else:
        geo_poly = get_adapter("geology", "cgs_2010")(geo_path, REGION.aoi)
        qal = is_quaternary_alluvium(geo_poly, grid)
        df["is_quaternary_alluvium"] = qal.to_numpy(dtype=bool)

    # --- NHD flowlines + lode-Au seeds for distance-downstream-from-lode.
    nhd_path = REGION.raw_paths.get("nhd_flowlines")
    lode_path = REGION.raw_paths.get("lode_mrds")
    nhd_loaded = None
    if nhd_path is not None and nhd_path.exists():
        nhd_loaded = get_adapter("hydrology", "nhdplus_hr")(nhd_path, REGION.aoi)
    if (
        lode_path is None or not lode_path.exists()
        or nhd_loaded is None
    ):
        warnings.warn(
            "lode_mrds and/or nhd_flowlines missing; "
            "distance_downstream_from_lode_m NaN."
        )
        df["distance_downstream_from_lode_m"] = np.nan
    else:
        lode_raw = gpd.read_file(lode_path)
        lode_seeds = _filter_lode_seeds(lode_raw)
        dist_km = distance_downstream_from_lode(lode_seeds, nhd_loaded, grid)
        df["distance_downstream_from_lode_m"] = (
            dist_km.to_numpy(dtype=np.float32) * 1000.0
        )

    # --- Hawkes catchment Au.
    geochem_path = REGION.raw_paths.get("geochem")
    samples: gpd.GeoDataFrame | None = None
    if geochem_path is not None and geochem_path.exists():
        samples = get_adapter("geochem", "ngdb")(
            geochem_path, REGION.aoi, elements=REGION.pathfinder_elements,
        )
    if nhd_loaded is None or samples is None or len(samples) == 0:
        warnings.warn("Hawkes catchment_au skipped (missing NHD or samples); NaN.")
        df["catchment_au_hawkes"] = np.nan
    elif "Au_ppm" not in samples.columns:
        warnings.warn("Hawkes: Au_ppm absent from samples; catchment_au_hawkes NaN.")
        df["catchment_au_hawkes"] = np.nan
    else:
        series = hawkes_dual_decay_catchment(
            samples, nhd_loaded, grid, element="Au_ppm",
        )
        df["catchment_au_hawkes"] = series.to_numpy(dtype=np.float32)

    return df


def _anchor_cell_indices(df: pd.DataFrame) -> pd.Series:
    """Snap each Calaveras anchor (lon, lat) to its nearest grid-cell row index."""
    transformer = Transformer.from_crs(
        "EPSG:4326", REGION.working_crs, always_xy=True,
    )
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    names, idxs, ax_list, ay_list = [], [], [], []
    for name, (lon, lat) in CALAVERAS_ANCHORS.items():
        ax, ay = transformer.transform(lon, lat)
        d2 = (xs - ax) ** 2 + (ys - ay) ** 2
        cell = int(np.argmin(d2))
        names.append(name)
        idxs.append(df.index[cell])
        ax_list.append(ax)
        ay_list.append(ay)
    return pd.DataFrame({
        "district": names,
        "cell_idx": idxs,
        "x": ax_list,
        "y": ay_list,
    })


def _decile_rank(score: pd.Series) -> pd.Series:
    return pd.qcut(
        score.rank(method="first", ascending=False, pct=False, na_option="keep"),
        q=10,
        labels=range(10),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    # Early bail-out: at least the motherlode MRDS + the hydraulic-pit polygon
    # set must be on disk to compute anything; everything else can degrade to
    # NaN. If both are missing the dry-run can't produce a number.
    must_have = [REGION.raw_paths["lode_mrds"], REGION.raw_paths["hydraulic_pits"]]
    missing = [p for p in must_have if not Path(p).exists()]
    if missing:
        print("ERROR: required Calaveras inputs missing:")
        for p in missing:
            print(f"  - {p}")
        print("\nFetch motherlode MRDS + hydraulic-mine-pits CA polygons first "
              "(both already pulled for the Sierra build).")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"==> Building Calaveras feature frame (RES={RES_M} m, "
          f"CRS={REGION.working_crs})")
    df = build_calaveras_feature_frame()
    print(f"    cells: {len(df):,}  columns: {len(df.columns)}")
    for c in (
        "distance_downstream_from_lode_m", "hydraulic_pit_proximity_m",
        "spi_band", "catchment_au_hawkes", "is_quaternary_alluvium",
        "twi", "geomorphon_terrace_mask", "slope",
    ):
        present = c in df.columns and df[c].notna().any()
        print(f"    {c:38s} {'OK' if present else 'NaN-fill'}")

    print("==> Computing Phase 1 USGS-Alaska placer index (Sierra weights)")
    score = usgs_alaska_placer_index(
        df, paleochannel_proxy=PALEOCHANNEL_PHASE1_PROXY,
    ).rename("phase1_score")
    n_finite = int(score.notna().sum())
    if n_finite == 0:
        print("ERROR: Phase 1 index is all NaN. Every input feature is missing.")
        return 2
    print(f"    score: min={np.nanmin(score):.3f}  "
          f"mean={np.nanmean(score):.3f}  max={np.nanmax(score):.3f}  "
          f"NaN={int(score.isna().sum()):,}")

    # Persist the Calaveras Phase 1 raster for any downstream re-plot.
    out_df = df[["row", "col", "x", "y"]].copy()
    out_df["phase1_score"] = score.values
    out_df.to_parquet(OUT_PHASE1, index=False)
    print(f"    wrote {OUT_PHASE1}")

    print("==> Snapping Calaveras anchors to nearest grid cells")
    anchors = _anchor_cell_indices(df)
    print(anchors.to_string(index=False))

    print("==> Decile rank per anchor under Sierra-trained Phase 1")
    deciles = _decile_rank(score)
    anchors["sierra_trained_phase1_score"] = [
        float(score.iloc[int(i)]) if pd.notna(score.iloc[int(i)]) else np.nan
        for i in anchors["cell_idx"]
    ]
    anchors["sierra_trained_phase1_decile"] = [
        (int(deciles.iloc[int(i)]) if pd.notna(deciles.iloc[int(i)]) else np.nan)
        for i in anchors["cell_idx"]
    ]

    # Capture rates at top-1/5/10% on the predicted side's anchor set.
    finite_mask = score.notna()
    s_finite = score[finite_mask].to_numpy()
    anchor_mask_full = np.zeros(len(df), dtype=bool)
    anchor_mask_full[anchors["cell_idx"].to_numpy(dtype=np.int64)] = True
    anchor_mask_finite = anchor_mask_full[finite_mask.to_numpy()]
    n_anchors_finite = int(anchor_mask_finite.sum())
    if n_anchors_finite == 0:
        print("    WARN: no anchors land on finite-score cells.")
        capture_summary = {
            "n_anchors_finite": 0,
            "capture_1pct": float("nan"),
            "capture_5pct": float("nan"),
            "capture_10pct": float("nan"),
        }
    else:
        ci = bootstrap_capture_ci(
            s_finite, anchor_mask_finite,
            ks_percent=(1.0, 5.0, 10.0),
            n_resamples=500, seed=42,
        )
        capture_summary = {
            "n_anchors_finite": n_anchors_finite,
            "capture_1pct": ci[1.0][0],
            "capture_1pct_ci_lo": ci[1.0][1],
            "capture_1pct_ci_hi": ci[1.0][2],
            "capture_5pct": ci[5.0][0],
            "capture_5pct_ci_lo": ci[5.0][1],
            "capture_5pct_ci_hi": ci[5.0][2],
            "capture_10pct": ci[10.0][0],
            "capture_10pct_ci_lo": ci[10.0][1],
            "capture_10pct_ci_hi": ci[10.0][2],
        }

    # AUC-PA proxy: report median anchor decile (lower = better).
    median_anchor_decile = float(np.nanmedian(
        anchors["sierra_trained_phase1_decile"].astype("float64")
    ))

    # Compare to Sierra in-sample (median anchor decile under Phase 1).
    sierra_in_sample_median = float("nan")
    if SIERRA_ANCHOR_TABLE.exists():
        sierra_tab = pd.read_csv(SIERRA_ANCHOR_TABLE)
        if "phase1_score_decile" in sierra_tab.columns:
            sierra_in_sample_median = float(
                sierra_tab["phase1_score_decile"].median(skipna=True)
            )
        else:
            print(f"    WARN: {SIERRA_ANCHOR_TABLE} lacks phase1_score_decile; "
                  f"comparison metric falls back to NaN.")

    # Comparison metric column on the per-anchor frame (so the CSV carries
    # the in-sample baseline alongside each district row).
    anchors["sierra_in_sample_median_phase1_decile"] = sierra_in_sample_median
    anchors["comparison_metric"] = (
        anchors["sierra_trained_phase1_decile"].astype("float64")
        - sierra_in_sample_median
    )
    # Phase 2 transfer is out of scope for this dry-run; column present for
    # downstream wiring once the Phase 2 estimator is serialized.
    anchors["sierra_trained_p2_decile"] = np.nan

    print("\n==> Per-anchor results")
    print(anchors[[
        "district", "x", "y", "cell_idx",
        "sierra_trained_phase1_score",
        "sierra_trained_phase1_decile",
        "sierra_trained_p2_decile",
        "comparison_metric",
    ]].to_string(index=False))

    print("\n==> Capture summary (Calaveras anchors under Sierra-trained Phase 1)")
    for k, v in capture_summary.items():
        print(f"    {k:25s}  {v}")

    anchors.to_csv(OUT_METRICS, index=False)
    print(f"\n==> wrote {OUT_METRICS}")

    if np.isnan(sierra_in_sample_median):
        print(f"\nSierra -> Calaveras transfer: median anchor decile "
              f"{median_anchor_decile:.1f}; sierra in-sample anchor decile "
              f"unavailable (run scripts/northern_sierra_placer/validation.py "
              f"first).")
    else:
        drop = median_anchor_decile - sierra_in_sample_median
        print(f"\nSierra -> Calaveras transfer: median anchor decile "
              f"{median_anchor_decile:.1f}; sierra in-sample anchor decile "
              f"{sierra_in_sample_median:.1f}; drop = {drop:+.1f} deciles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
