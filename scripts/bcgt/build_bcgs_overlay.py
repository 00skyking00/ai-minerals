"""Build pre/post-2010 BCGS drill+Cu overlay on the BCGT 500m grid.

Loads the full BCGS ARDH (~6,460 collars BC-wide), filters to the BCGT AOI
(~1,350 collars), joins per-hole max Cu assay, buckets to the 500 m feature
grid, and writes a per-cell overlay split by drill_start_dt < 2010 vs >= 2010.

Output: data/derived/bcgt/bcgs_pre_post_2010_overlay.parquet
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ARDH = REPO / "data/raw/bcgs_drillholes/bcgs_ardh.gpkg"
FEATURES = REPO / "data/derived/features_bcgt_500m.parquet"
OUT = REPO / "data/derived/bcgt/bcgs_pre_post_2010_overlay.parquet"

# BCGT AOI bounds (EPSG:4326) — see src/ai_minerals/regions/bcgt.py
MIN_LON, MAX_LON = -131.5, -129.5
MIN_LAT, MAX_LAT = 56.0, 58.0
CELL_M = 500.0

# BCGS analyte_code for Copper, from the code_analytes lookup table.
CU_ANALYTE_CODE = 75

# Cox-Singer porphyry-Cu cutoff in ppm (= 0.2 percent).
CU_POSITIVE_PPM = 2000.0


def _to_ppm(abundance, unit_code: int, unit_map: dict[int, str]) -> float:
    try:
        val = float(abundance)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(val):
        return np.nan
    unit = unit_map.get(unit_code, "").lower()
    if unit in ("%", "percent", "pct"):
        return val * 10000.0
    if unit == "ppb":
        return val / 1000.0
    return val   # ppm or unknown -> assume ppm


def main() -> int:
    print("[load] collars + spatial filter to BCGT AOI...")
    collars = pd.DataFrame(gpd.read_file(ARDH, layer="collar"))
    sub = collars[collars["epsg_srid"] == 3156].copy()   # only BCGT-region SRID
    gdf = gpd.GeoDataFrame(
        sub, geometry=gpd.points_from_xy(sub["easting"], sub["northing"]),
        crs="EPSG:3156",
    )
    gdf4326 = gdf.to_crs("EPSG:4326")
    in_aoi = (
        (gdf4326.geometry.x >= MIN_LON) & (gdf4326.geometry.x <= MAX_LON)
        & (gdf4326.geometry.y >= MIN_LAT) & (gdf4326.geometry.y <= MAX_LAT)
    )
    aoi_collars = gdf4326[in_aoi].copy()
    aoi_collars["start_year"] = pd.to_datetime(
        aoi_collars["drill_start_dt"], errors="coerce",
    ).dt.year
    n_pre = (aoi_collars["start_year"] < 2010).sum()
    n_post = (aoi_collars["start_year"] >= 2010).sum()
    print(f"[ok] AOI collars: {len(aoi_collars)}  (pre-2010 {n_pre}, post-2010 {n_post})")

    print("[load] Cu determinations (code=75)...")
    sample = pd.DataFrame(gpd.read_file(ARDH, layer="sample"))
    aoi_hole_ids = set(aoi_collars["hole_id"].dropna())
    sample_aoi = sample[sample["hole_id"].isin(aoi_hole_ids)][
        ["sample_id", "hole_id"]
    ].copy()
    aoi_sample_ids = set(sample_aoi["sample_id"])

    determ = pd.DataFrame(gpd.read_file(ARDH, layer="determin"))
    cu = determ[
        (determ["analyte_code"] == CU_ANALYTE_CODE)
        & (determ["sample_id"].isin(aoi_sample_ids))
    ].copy()
    units = pd.DataFrame(gpd.read_file(ARDH, layer="code_unit"))
    unit_map = dict(zip(units["unit_code"], units["unit_name"]))
    cu["cu_ppm"] = cu.apply(
        lambda r: _to_ppm(r["abundance"], r["unit_code"], unit_map),
        axis=1,
    )
    print(f"  Cu rows: {len(cu):,}; >=2000 ppm: {(cu['cu_ppm'] >= CU_POSITIVE_PPM).sum():,}")

    cu = cu.merge(sample_aoi, on="sample_id", how="left")
    hole_max_cu = (
        cu.dropna(subset=["hole_id"])
        .groupby("hole_id")["cu_ppm"]
        .max()
        .reset_index()
        .rename(columns={"cu_ppm": "max_cu_ppm"})
    )
    n_cu_pos_holes = (hole_max_cu["max_cu_ppm"] >= CU_POSITIVE_PPM).sum()
    print(f"  AOI holes with Cu data: {len(hole_max_cu)}; Cu+ holes: {n_cu_pos_holes}")

    aoi_collars = aoi_collars.merge(hole_max_cu, on="hole_id", how="left")

    # Bucket per 500m cell on the EPSG:3005 (BC Albers) feature grid.
    feat = pd.read_parquet(FEATURES)
    aoi_3005 = gpd.GeoDataFrame(
        aoi_collars, geometry=aoi_collars.geometry, crs="EPSG:4326",
    ).to_crs("EPSG:3005")
    aoi_3005["x_3005"] = aoi_3005.geometry.x
    aoi_3005["y_3005"] = aoi_3005.geometry.y
    x_min = feat["x"].min()
    y_min = feat["y"].min()
    aoi_3005["cell_col"] = (
        ((aoi_3005["x_3005"] - x_min) / CELL_M).round().astype("Int64")
    )
    aoi_3005["cell_row"] = (
        ((aoi_3005["y_3005"] - y_min) / CELL_M).round().astype("Int64")
    )
    in_grid = (
        (aoi_3005["cell_col"] >= feat["col"].min())
        & (aoi_3005["cell_col"] <= feat["col"].max())
        & (aoi_3005["cell_row"] >= feat["row"].min())
        & (aoi_3005["cell_row"] <= feat["row"].max())
    )
    aoi_in_grid = aoi_3005[in_grid].copy()
    aoi_in_grid["pre_2010"] = aoi_in_grid["start_year"] < 2010

    def _agg(g: pd.DataFrame) -> pd.Series:
        cu = g["max_cu_ppm"]
        return pd.Series({
            "n_holes": len(g),
            "cu_pos_n_holes": (cu >= CU_POSITIVE_PPM).sum(),
            "max_cu_ppm": cu.max() if cu.notna().any() else 0.0,
        })

    pre = (
        aoi_in_grid[aoi_in_grid["pre_2010"]]
        .groupby(["cell_row", "cell_col"], group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    pre.columns = [
        "row", "col",
        "pre_2010_n_holes", "pre_2010_cu_positive_n_holes", "pre_2010_max_cu_ppm",
    ]
    post = (
        aoi_in_grid[~aoi_in_grid["pre_2010"]]
        .groupby(["cell_row", "cell_col"], group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    post.columns = [
        "row", "col",
        "post_2010_n_holes", "post_2010_cu_positive_n_holes", "post_2010_max_cu_ppm",
    ]

    overlay = feat[["row", "col", "x", "y"]].copy()
    overlay = (
        overlay
        .merge(pre, on=["row", "col"], how="left")
        .merge(post, on=["row", "col"], how="left")
    )
    for c in (
        "pre_2010_n_holes", "pre_2010_cu_positive_n_holes",
        "post_2010_n_holes", "post_2010_cu_positive_n_holes",
    ):
        overlay[c] = overlay[c].fillna(0).astype(int)
    for c in ("pre_2010_max_cu_ppm", "post_2010_max_cu_ppm"):
        overlay[c] = overlay[c].fillna(0.0).astype(float)

    n_pre_drilled = (overlay["pre_2010_n_holes"] > 0).sum()
    n_pre_pos = (overlay["pre_2010_cu_positive_n_holes"] > 0).sum()
    n_post_drilled = (overlay["post_2010_n_holes"] > 0).sum()
    n_post_pos = (overlay["post_2010_cu_positive_n_holes"] > 0).sum()
    print(f"\n[output] overlay: {len(overlay):,} cells")
    print(f"  pre-2010 drilled cells:  {n_pre_drilled:5d}   Cu+ cells: {n_pre_pos}")
    print(f"  post-2010 drilled cells: {n_post_drilled:5d}   Cu+ cells: {n_post_pos}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_parquet(OUT)
    print(f"[wrote] {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
