"""Glue layer: assemble the full pixel × feature DataFrame for modeling."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.aoi import EASTERN_ALASKA
from ai_minerals.data._common import DATA_RAW
from ai_minerals.features.labels import assign_cells, porphyry_positives
from ai_minerals.features.geochem import (
    PATHFINDER_ELEMENTS,
    aggregate_in_radius,
    load_pathfinder_assays,
)
from ai_minerals.features.geology import assign_lithology, distance_to_fault
from ai_minerals.features.rasters import s2_indices, sample_raster, slope_and_tri
from ai_minerals.grid import Grid, build_grid


_GEO_GDB = "data/raw/geology_ak/sim3340/sim3340_gdb/AKgeol_web_gdb/geologic_data/AKStategeol.gdb"


def _mask_geophys_nodata(arr: np.ndarray) -> np.ndarray:
    """USGS national-composite grids have sentinel nodata (~±3.4e38 for float32)."""
    return np.where(np.abs(arr) > 1e10, np.nan, arr)


def build_feature_frame(grid: Grid | None = None, resolution_m: int = 500) -> pd.DataFrame:
    """Return a flat DataFrame with one row per grid cell and all features.

    Columns:
      row, col, x, y
      elevation, slope, tri
      s2_iron_oxide, s2_ferrous, s2_clay, s2_ndvi
      magnetic, gravity
      lithology_class
      distance_to_fault_m
      <el>_mean_5km, <el>_max_5km, <el>_count_5km  for each pathfinder
      is_porphyry (family), is_porphyry_strict (21a only)
      any_mineral_occurrence  (MRDS or ARDF with any Cu commodity, for neg exclusion)
    """
    if grid is None:
        grid = build_grid(EASTERN_ALASKA, resolution_m=resolution_m)

    # --- Raster features ---
    print("[assemble] DEM + derivatives")
    dem = sample_raster(DATA_RAW / "dem/dem_eastak.tif", grid)
    slope, tri = slope_and_tri(dem, grid.resolution_m)

    print("[assemble] Sentinel-2 indices")
    s2 = s2_indices(DATA_RAW / "sentinel2/s2_mosaic_eastak.tif", grid)

    print("[assemble] geophysics")
    mag = _mask_geophys_nodata(sample_raster(DATA_RAW / "geophysics/magnetic_eastak.tif", grid))
    grav = _mask_geophys_nodata(sample_raster(DATA_RAW / "geophysics/gravity_eastak.tif", grid))

    # --- Geology ---
    print("[assemble] geology polygons + faults")
    geo_poly = gpd.read_file(DATA_RAW / "geology_ak/geology_eastak.gpkg")
    lith, top_classes = assign_lithology(grid, geo_poly, top_n=10)

    aoi_mask = gpd.GeoSeries([EASTERN_ALASKA.polygon], crs=EASTERN_ALASKA.crs).to_crs(grid.crs).iloc[0]
    arc_gdf = gpd.read_file(_GEO_GDB, layer="AKStategeol_arc", mask=aoi_mask)
    fault_mask = arc_gdf["LINE_TYPE"].fillna("").str.contains("fault", case=False)
    faults = arc_gdf[fault_mask]
    print(f"  fault lines in AOI: {len(faults):,}")
    dist_fault = distance_to_fault(grid, faults)

    # --- Geochemistry ---
    print("[assemble] AGDB4 pathfinder aggregation (5 km)")
    samples = pd.read_parquet(DATA_RAW / "agdb4/agdb4_samples_eastak.parquet")
    joined = load_pathfinder_assays(samples, DATA_RAW / "agdb4/AGDB4_text.zip")
    samples_gdf = gpd.GeoDataFrame(
        joined,
        geometry=gpd.points_from_xy(joined.LONGITUDE, joined.LATITUDE),
        crs="EPSG:4326",
    )
    geochem = aggregate_in_radius(grid, samples_gdf, radius_m=5000.0)

    # --- Labels ---
    print("[assemble] labels")
    ardf = gpd.read_file(DATA_RAW / "ardf/ardf_eastak.gpkg")
    fam_cells = assign_cells(porphyry_positives(ardf, strict=False), grid)
    str_cells = assign_cells(porphyry_positives(ardf, strict=True), grid)

    is_fam = np.zeros(grid.shape, dtype=np.uint8)
    is_str = np.zeros(grid.shape, dtype=np.uint8)
    for _, row in fam_cells.iterrows():
        is_fam[int(row["row"]), int(row["col"])] = 1
    for _, row in str_cells.iterrows():
        is_str[int(row["row"]), int(row["col"])] = 1

    # Any mineral occurrence from ARDF + MRDS (used to define pseudo-neg exclusion buffer).
    occ_cells = assign_cells(ardf, grid)
    try:
        mrds = gpd.read_file(DATA_RAW / "mrds/mrds_eastak.geojson")
        occ_cells = pd.concat(
            [occ_cells, assign_cells(mrds, grid)[["row", "col"]]], ignore_index=True
        )
    except Exception:
        pass
    any_occ = np.zeros(grid.shape, dtype=np.uint8)
    for _, row in occ_cells[["row", "col"]].drop_duplicates().iterrows():
        any_occ[int(row["row"]), int(row["col"])] = 1

    # --- Assemble flat table ---
    xv, yv = np.meshgrid(grid.xs, grid.ys)
    cols: dict[str, np.ndarray] = {
        "row": np.repeat(np.arange(grid.shape[0]), grid.shape[1]).astype(np.int32),
        "col": np.tile(np.arange(grid.shape[1]), grid.shape[0]).astype(np.int32),
        "x": xv.ravel(),
        "y": yv.ravel(),
        "elevation": dem.ravel(),
        "slope": slope.ravel(),
        "tri": tri.ravel(),
        "magnetic": mag.ravel(),
        "gravity": grav.ravel(),
        "lithology_class": lith.ravel().astype(np.int32),
        "distance_to_fault_m": dist_fault.ravel(),
        "is_porphyry": is_fam.ravel(),
        "is_porphyry_strict": is_str.ravel(),
        "any_mineral_occurrence": any_occ.ravel(),
    }
    for name, arr in s2.items():
        cols[name] = arr.ravel()
    for name, arr in geochem.items():
        cols[name] = arr.ravel()

    df = pd.DataFrame(cols)

    # Clip to AOI polygon — cells in the bbox but outside the polygon are excluded.
    centroid_gdf = grid.centroid_gdf()
    inside = centroid_gdf.within(aoi_mask).to_numpy()
    df = df[inside].reset_index(drop=True)

    df.attrs["top_lithology_classes"] = top_classes
    print(f"[assemble] done: {len(df):,} cells × {len(df.columns)} columns  "
          f"({df['is_porphyry'].sum()} family positives, {df['is_porphyry_strict'].sum()} strict)")
    return df
