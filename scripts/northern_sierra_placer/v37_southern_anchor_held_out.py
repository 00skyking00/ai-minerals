"""H2.5: southern Mother Lode anchor held-out + MRDS per-county gate.

Two validations of the v3.7.0 calibrated raster:

1. **Southern anchor capture** — for each documented southern placer
   anchor district, extract the max probability in a 2 km radius from
   the district centroid and report rank against the AOI distribution.

2. **MRDS per-county held-out gate** — for each of the 4 weak counties
   (Butte, Yuba, Amador, Mariposa), score every MRDS placer-Au record
   against the v3.7 raster. If the median MRDS-cell probability lands
   in the bottom half of the AOI distribution, fires the v3.7.0.1
   augmentation patch decision per plan H2.5.

Inputs: v3.7 fused calibrated raster + v3.7 per-population rasters +
MRDS-placer subset from data/raw/mrds_motherlode/.

Outputs:
- data/derived/northern_sierra_placer/v37_southern_anchor_held_out.parquet
- data/derived/northern_sierra_placer/v37_mrds_per_county_gate.parquet
- stdout summary with PASS/FAIL gates and recommendation

Usage:
    .venv/bin/python scripts/northern_sierra_placer/v37_southern_anchor_held_out.py

References:
    Plan H2.5 in ~/.claude/plans/hazy-humming-lynx.md
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

REPO = Path(__file__).resolve().parent.parent
DERIVED = REPO / "data/derived/northern_sierra_placer"
FUSED_TIF = DERIVED / "prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
QUAT_TIF = DERIVED / "prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"
TERT_TIF = DERIVED / "prospectivity_placer_placer_tertiary_250m_calibrated_4326.tif"
MRDS_GPKG_CANDIDATES = [
    REPO / "data/raw/mrds/mrds_northern_sierra_placer.gpkg",
    REPO / "data/raw/mrds_motherlode/mrds_motherlode_au.gpkg",
    REPO / "data/raw/mrds/mrds_placer_au.gpkg",
    REPO / "data/derived/mrds_motherlode_placer_au.gpkg",
]

OUT_ANCHOR = DERIVED / "v37_southern_anchor_held_out.parquet"
OUT_COUNTY = DERIVED / "v37_mrds_per_county_gate.parquet"
OUT_REPORT = DERIVED / "v37_held_out_report.md"

# Southern Mother Lode placer-Au anchor districts in the FOUR weak counties
# (Butte/Yuba are deep-gravel; Amador/Mariposa are canonical Mother Lode).
# Lon/lat in EPSG:4326. Sources: USGS GNIS + Lindgren PP73 references.
SOUTHERN_ANCHORS = [
    ("Mokelumne Hill",  "Amador",     -120.708, 38.301),
    ("Plymouth",        "Amador",     -120.844, 38.482),
    ("Drytown",         "Amador",     -120.853, 38.443),
    ("Jackson",         "Amador",     -120.774, 38.349),
    ("Murphys",         "Calaveras",  -120.460, 38.137),
    ("Angels Camp",     "Calaveras",  -120.557, 38.072),
    ("Carson Hill",     "Calaveras",  -120.546, 38.054),
    ("San Andreas",     "Calaveras",  -120.681, 38.196),
    ("Sonora",          "Tuolumne",   -120.382, 37.984),
    ("Columbia",        "Tuolumne",   -120.402, 38.038),
    ("Jamestown",       "Tuolumne",   -120.422, 37.948),
    ("Coulterville",    "Mariposa",   -120.197, 37.711),
    ("Mariposa town",   "Mariposa",   -119.965, 37.485),
    ("Hornitos",        "Mariposa",   -120.243, 37.502),
    # Northern deep-gravel for reference (not "southern" but in the weak counties)
    ("Cherokee",        "Butte",      -121.546, 39.626),
    ("Oroville",        "Butte",      -121.546, 39.514),
    ("Marysville",      "Yuba",       -121.591, 39.146),
    ("Smartsville",     "Yuba",       -121.305, 39.207),
]

WEAK_COUNTIES = ("Butte", "Yuba", "Amador", "Mariposa")
ANCHOR_RADIUS_M = 2000.0  # 2 km buffer matching v3.6 anchor convention


def load_raster(path: Path) -> tuple[np.ndarray, rasterio.transform.Affine, str, int, int]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.transform, str(ds.crs), ds.width, ds.height


def aoi_quantiles(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
    }


def rank_against(value: float, finite: np.ndarray) -> float:
    """Return the rank (0..1) of `value` in the sorted finite distribution."""
    if not np.isfinite(value):
        return float("nan")
    return float((finite <= value).mean())


def sample_buffer_max(arr: np.ndarray, src: rasterio.DatasetReader,
                      lon: float, lat: float, radius_m: float) -> tuple[float, int, int]:
    """Max raster value inside a square buffer of `radius_m` around (lon, lat).

    Approximates a circle with a square at 250 m grid resolution; cheap and
    good enough for ranking.
    """
    row, col = src.index(lon, lat)
    # Convert radius from meters to cells. At 38 deg N, 1 deg lat ~111 km,
    # 1 deg lon ~ 87.6 km. 250 m grid in 4326 has roughly 1/440 deg per cell
    # (varies with latitude). Conservative: convert via the raster's
    # transform pixel size and adjust for latitude in y.
    px_x = abs(src.transform.a) * (111_000.0 * np.cos(np.radians(lat)))  # meters per col
    px_y = abs(src.transform.e) * 111_000.0  # meters per row
    half_cols = int(np.ceil(radius_m / px_x))
    half_rows = int(np.ceil(radius_m / px_y))
    r_lo = max(0, row - half_rows)
    r_hi = min(src.height, row + half_rows + 1)
    c_lo = max(0, col - half_cols)
    c_hi = min(src.width, col + half_cols + 1)
    if r_hi <= r_lo or c_hi <= c_lo:
        return float("nan"), 0, 0
    sub = arr[r_lo:r_hi, c_lo:c_hi]
    finite = sub[np.isfinite(sub)]
    if len(finite) == 0:
        return float("nan"), 0, 0
    return float(finite.max()), (r_hi - r_lo), (c_hi - c_lo)


def find_mrds_gpkg() -> Path | None:
    for cand in MRDS_GPKG_CANDIDATES:
        if cand.exists():
            return cand
    return None


def main() -> int:
    if not FUSED_TIF.exists():
        print(f"ERROR: v3.7 fused raster not found at {FUSED_TIF}")
        print("       Training must complete first (H2.4).")
        return 2

    print(f"==> Loading rasters")
    print(f"    fused: {FUSED_TIF.name}")
    fused_arr, fused_xform, fused_crs, fused_w, fused_h = load_raster(FUSED_TIF)
    quat_arr = load_raster(QUAT_TIF)[0] if QUAT_TIF.exists() else None
    tert_arr = load_raster(TERT_TIF)[0] if TERT_TIF.exists() else None
    fused_finite = fused_arr[np.isfinite(fused_arr)]
    aoi_q = aoi_quantiles(fused_arr)
    print(f"    AOI quantiles: p50={aoi_q['p50']:.4f} p90={aoi_q['p90']:.4f} "
          f"p99={aoi_q['p99']:.4f} max={aoi_q['max']:.4f}")

    # === 1. Southern anchor capture ===
    print()
    print(f"==> Southern anchor held-out ({len(SOUTHERN_ANCHORS)} anchors, "
          f"{ANCHOR_RADIUS_M:.0f} m buffer)")
    rows = []
    with rasterio.open(FUSED_TIF) as src:
        for name, county, lon, lat in SOUTHERN_ANCHORS:
            p_fused, _, _ = sample_buffer_max(fused_arr, src, lon, lat, ANCHOR_RADIUS_M)
            p_quat = (sample_buffer_max(quat_arr, src, lon, lat, ANCHOR_RADIUS_M)[0]
                      if quat_arr is not None else float("nan"))
            p_tert = (sample_buffer_max(tert_arr, src, lon, lat, ANCHOR_RADIUS_M)[0]
                      if tert_arr is not None else float("nan"))
            rank = rank_against(p_fused, fused_finite)
            decile = int(np.floor(rank * 10)) if not np.isnan(rank) else -1
            top_decile = rank >= 0.9 if not np.isnan(rank) else False
            top_quintile = rank >= 0.8 if not np.isnan(rank) else False
            rows.append({
                "anchor": name,
                "county": county,
                "lon": lon,
                "lat": lat,
                "p_fused_max": p_fused,
                "p_quat_max": p_quat,
                "p_tert_max": p_tert,
                "rank_in_aoi": rank,
                "decile": decile,
                "top_decile": top_decile,
                "top_quintile": top_quintile,
            })
    df_anchor = pd.DataFrame(rows)
    df_anchor.to_parquet(OUT_ANCHOR, index=False)
    print(f"    wrote {OUT_ANCHOR}")
    print()
    print(df_anchor.to_string(index=False))

    # === 2. MRDS per-county held-out gate ===
    print()
    print(f"==> MRDS per-county held-out gate (weak counties: {WEAK_COUNTIES})")
    mrds_path = find_mrds_gpkg()
    if mrds_path is None:
        print(f"    MRDS gpkg not found in any of:")
        for cand in MRDS_GPKG_CANDIDATES:
            print(f"      {cand}")
        print(f"    Skipping per-county gate. Re-run after locating MRDS data.")
        county_summary = pd.DataFrame()
    else:
        print(f"    Loading MRDS placer-Au from {mrds_path}")
        mrds = gpd.read_file(mrds_path).to_crs("EPSG:4326")
        # Try to find a county column
        county_col = None
        for c in ("county", "COUNTY", "f_county", "geog_loc"):
            if c in mrds.columns:
                county_col = c
                break
        if county_col is None:
            print(f"    MRDS file has no county column (cols: {list(mrds.columns)[:10]})")
            print(f"    Falling back to spatial filter via known county bboxes.")
            # Filter by lat/lon bands corresponding to each county
            # rough bounding boxes:
            county_bbox = {
                "Butte":    (-121.85, 39.30, -121.20, 39.97),
                "Yuba":     (-121.55, 39.05, -121.10, 39.55),
                "Amador":   (-121.00, 38.25, -120.50, 38.60),
                "Mariposa": (-120.30, 37.40, -119.55, 37.85),
            }
            county_records = {}
            for cnty, (W, S, E, N) in county_bbox.items():
                mask = ((mrds.geometry.x >= W) & (mrds.geometry.x <= E)
                        & (mrds.geometry.y >= S) & (mrds.geometry.y <= N))
                county_records[cnty] = mrds[mask]
                print(f"      {cnty}: {int(mask.sum())} records via bbox")
        else:
            county_records = {
                cnty: mrds[mrds[county_col].astype(str).str.contains(cnty, case=False, na=False)]
                for cnty in WEAK_COUNTIES
            }
            for cnty, sub in county_records.items():
                print(f"      {cnty}: {len(sub)} records via {county_col}")

        county_rows = []
        for cnty, sub in county_records.items():
            if len(sub) == 0:
                county_rows.append({
                    "county": cnty, "n_records": 0,
                    "median_p": float("nan"), "p10_p": float("nan"),
                    "p90_p": float("nan"), "median_rank_in_aoi": float("nan"),
                    "gate": "no records",
                })
                continue
            probs = []
            with rasterio.open(FUSED_TIF) as src:
                for _, row in sub.iterrows():
                    if row.geometry is None or row.geometry.is_empty:
                        continue
                    r, c = src.index(row.geometry.x, row.geometry.y)
                    if 0 <= r < src.height and 0 <= c < src.width:
                        v = fused_arr[r, c]
                        if np.isfinite(v):
                            probs.append(float(v))
            if len(probs) == 0:
                gate = "no in-AOI records"
                med_p = p10 = p90 = med_rank = float("nan")
            else:
                probs_arr = np.array(probs)
                med_p = float(np.median(probs_arr))
                p10 = float(np.percentile(probs_arr, 10))
                p90 = float(np.percentile(probs_arr, 90))
                med_rank = rank_against(med_p, fused_finite)
                # Gate decision: median MRDS-cell probability vs AOI quantiles.
                # The AOI has a long zero-tail (mostly floor cells outside the
                # placer belt), so "median_rank >= 0.5" is a trivially-true bar.
                # The meaningful threshold is the AOI top decile: if the median
                # MRDS-known cell sits in the top 10% of the AOI distribution,
                # the model successfully identified these as candidate placer
                # cells. If it sits below the top quintile, the model isn't
                # seeing them and v3.7.0.1 augmentation should fire.
                if med_p >= aoi_q["p90"]:
                    gate = "PASS (top decile)"
                elif med_p >= aoi_q["p75"]:
                    gate = "MARGINAL (top quintile)"
                else:
                    gate = "FAIL (fires v3.7.0.1 augmentation)"
            county_rows.append({
                "county": cnty,
                "n_records": int(len(probs)),
                "median_p": med_p,
                "p10_p": p10,
                "p90_p": p90,
                "median_rank_in_aoi": med_rank,
                "gate": gate,
            })
        county_summary = pd.DataFrame(county_rows)
        county_summary.to_parquet(OUT_COUNTY, index=False)
        print(f"    wrote {OUT_COUNTY}")
        print()
        print(county_summary.to_string(index=False))

    # === 3. Write markdown report ===
    report_lines = [
        "# v3.7.0 H2.5 held-out validation",
        f"Raster: `{FUSED_TIF.name}`",
        "",
        "## AOI quantiles (fused)",
        f"- p50: {aoi_q['p50']:.4f}",
        f"- p90: {aoi_q['p90']:.4f}",
        f"- p99: {aoi_q['p99']:.4f}",
        f"- max: {aoi_q['max']:.4f}",
        "",
        "## Southern anchor capture",
        df_anchor[["anchor", "county", "p_fused_max", "rank_in_aoi",
                   "decile", "top_decile"]].to_markdown(index=False),
        "",
        "## MRDS per-county held-out gate",
    ]
    if len(county_summary) > 0:
        report_lines.append(county_summary.to_markdown(index=False))
        fail_counties = county_summary[county_summary["gate"].astype(str).str.startswith("FAIL")]
        if len(fail_counties) > 0:
            report_lines += [
                "",
                "### Decision: v3.7.0.1 augmentation patch fires",
                "Counties where median MRDS-known cell sits below AOI p50:",
            ]
            for _, r in fail_counties.iterrows():
                report_lines.append(f"- **{r['county']}**: median_rank={r['median_rank_in_aoi']:.3f}, n_records={int(r['n_records'])}")
            report_lines += [
                "",
                "Action: re-run assemble with `--placer-version v37 --quaternary-augment-mrds-counties Butte,Yuba,Amador,Mariposa` (TODO: add this flag), retrain Quaternary only for v3.7.0.1.",
            ]
        else:
            report_lines += [
                "",
                "### Decision: USMIN-only Quaternary ships clean",
                "All weak counties land at or above AOI p50; no augmentation needed.",
            ]
    else:
        report_lines.append("(skipped — MRDS data not found)")
    OUT_REPORT.write_text("\n".join(report_lines))
    print()
    print(f"==> Report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
