"""Glue layer: assemble the full pixel × feature DataFrame for modeling.

Takes a `Region` config, resolves per-source adapters from the registry,
produces a flat per-grid-cell feature DataFrame with `is_<depositclass>`
label columns for every semantic class the region declares.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.data.adapters import get_adapter
from ai_minerals.features.labels import assign_cells, deposit_positives
from ai_minerals.features.geochem import aggregate_in_radius
from ai_minerals.features.geology import assign_lithology, distance_to_fault
from ai_minerals.features.rasters import s2_indices, sample_raster, slope_and_tri
from ai_minerals.grid import Grid, build_grid


def build_feature_frame(region, resolution_m: int = 500) -> pd.DataFrame:
    """Return a flat DataFrame with one row per grid cell and all features.

    Columns:
      row, col, x, y
      elevation, slope, tri
      s2_iron_oxide, s2_ferrous, s2_clay, s2_ndvi
      magnetic, gravity
      lithology_class, lith_group (via assign_lithology on canonical geology)
      distance_to_fault_m
      <el>_{mean,max,count,has_data}_5km  for each pathfinder (per region config)
      is_<depositclass>      one per entry in region.deposit_classes
      any_mineral_occurrence (MRDS + ARDF with any commodity matching region filter)
    """
    grid = build_grid(region.aoi, resolution_m=resolution_m, working_crs=region.working_crs)

    # --- Raster features ---
    print("[assemble] DEM + derivatives")
    dem = sample_raster(region.raw_paths["dem"], grid)
    slope, tri = slope_and_tri(dem, grid.resolution_m)

    print("[assemble] Sentinel-2 indices")
    s2 = s2_indices(region.raw_paths["sentinel2"], grid)

    print("[assemble] geophysics")
    geophys_adapter = get_adapter("geophysics", region.geophysics_source)
    mag = geophys_adapter(sample_raster(region.raw_paths["magnetic"], grid))
    grav = geophys_adapter(sample_raster(region.raw_paths["gravity"], grid))

    # --- Geology (canonical schema: lith_class, lith_group) ---
    print("[assemble] geology polygons + faults")
    geo_adapter = get_adapter("geology", region.geology_source)
    geo_poly = geo_adapter(region.raw_paths["geology"], region.aoi)
    lith, top_classes = assign_lithology(grid, geo_poly, top_n=10, class_column="lith_class")

    aoi_mask = gpd.GeoSeries(
        [region.aoi.polygon], crs=region.aoi.crs
    ).to_crs(grid.crs).iloc[0]
    fault_layer = region.fault_layer if hasattr(region, "fault_layer") else None
    if fault_layer:
        # Alaska SGMC GDB — faults are in the `AKStategeol_arc` layer with a
        # LINE_TYPE discriminator, so we read + filter.
        arc_gdf = gpd.read_file(region.raw_paths["geology_arcs"], layer=fault_layer, mask=aoi_mask)
        fault_mask = arc_gdf["LINE_TYPE"].fillna("").str.contains("fault", case=False)
        faults = arc_gdf[fault_mask]
    else:
        # BC and other regions: faults come from the fetch layer as a dedicated
        # file with every row already a fault line.
        faults = gpd.read_file(region.raw_paths["geology_arcs"])
    print(f"  fault lines in AOI: {len(faults):,}")
    dist_fault = distance_to_fault(grid, faults)

    # --- Geochemistry (canonical) ---
    print(f"[assemble] geochem aggregation (5 km, {len(region.pathfinder_elements)} elements)")
    gc_adapter = get_adapter("geochem", region.geochem_source)
    gc_kwargs = {}
    if "geochem_bv_zip" in region.raw_paths:  # AGDB4-specific
        gc_kwargs["bv_zip"] = region.raw_paths["geochem_bv_zip"]
    gc_kwargs["elements"] = region.pathfinder_elements
    samples_gdf = gc_adapter(region.raw_paths["geochem"], region.aoi, **gc_kwargs)
    geochem = aggregate_in_radius(
        grid, samples_gdf, radius_m=5000.0,
        elements=region.pathfinder_elements,
    )

    # --- Labels ---
    print("[assemble] labels")
    occ_adapter = get_adapter("occurrences", region.occurrences_source)
    ardf_canon = occ_adapter(region.raw_paths["occurrences"], region.aoi)

    # Build is_<class> columns from region.deposit_classes.
    label_masks: dict[str, np.ndarray] = {}
    for class_name, codes in region.deposit_classes.items():
        class_cells = assign_cells(deposit_positives(ardf_canon, codes), grid)
        m = np.zeros(grid.shape, dtype=np.uint8)
        for _, row in class_cells.iterrows():
            m[int(row["row"]), int(row["col"])] = 1
        label_masks[f"is_{class_name}"] = m
        print(f"  is_{class_name}: {int(m.sum())} cells (codes={codes})")

    # any_mineral_occurrence mask: every ARDF record (any commodity) plus MRDS
    # if available. Feeds pseudo-neg exclusion — we want to exclude from the
    # negative class any cell near *any* known mineralization, regardless of
    # commodity.
    occ_cells = assign_cells(ardf_canon, grid)
    if "occurrences_mrds" in region.raw_paths:
        try:
            mrds_canon = get_adapter("occurrences", "mrds")(region.raw_paths["occurrences_mrds"], region.aoi)
            occ_cells = pd.concat(
                [occ_cells, assign_cells(mrds_canon, grid)[["row", "col"]]],
                ignore_index=True,
            )
        except Exception as e:
            print(f"  MRDS skipped: {e}")
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
        "any_mineral_occurrence": any_occ.ravel(),
    }
    for class_name, mask in label_masks.items():
        cols[class_name] = mask.ravel()
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
    pos_counts = {k: int(df[k].sum()) for k in label_masks}
    print(f"[assemble] done: {len(df):,} cells × {len(df.columns)} columns  positives={pos_counts}")
    return df
