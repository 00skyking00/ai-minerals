"""Geology features — lithology class (one-hot) and distance-to-fault."""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

from ai_minerals.grid import Grid


def assign_lithology(
    grid: Grid,
    geology: gpd.GeoDataFrame,
    *,
    top_n: int = 10,
    class_column: str = "CLASS",
) -> tuple[np.ndarray, list[int]]:
    """Spatial-join the lithology class at each grid centroid.

    Returns
    -------
    class_grid : 2-D array of shape grid.shape with the class code per cell
                 (or -1 if the cell falls outside any polygon).
    top_classes : the `top_n` most common class codes (in the AOI).
    """
    centroids = grid.centroid_gdf().to_crs(geology.crs)
    joined = gpd.sjoin(centroids, geology[[class_column, "geometry"]], how="left", predicate="within")
    # A centroid can fall on a boundary and match multiple polygons — keep the first.
    joined = joined.loc[~joined.index.duplicated(keep="first")]
    classes = joined[class_column].to_numpy()

    # Convert to int with -1 for NaN
    class_arr = np.full(grid.n_cells, -1, dtype=np.int64)
    valid = ~gpd.pd.isna(classes)
    class_arr[valid.to_numpy() if hasattr(valid, "to_numpy") else valid] = (
        classes[valid.to_numpy() if hasattr(valid, "to_numpy") else valid].astype(np.int64)
    )
    class_grid = class_arr.reshape(grid.shape)

    # Top-N most common
    codes, counts = np.unique(class_arr[class_arr >= 0], return_counts=True)
    order = np.argsort(-counts)
    top = codes[order[:top_n]].tolist()
    return class_grid, top


def distance_to_fault(
    grid: Grid,
    fault_lines: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float = 100.0,
    cap_m: float = 50_000.0,
) -> np.ndarray:
    """Distance from each grid cell to the nearest fault line (meters).

    Fault lines are densified to points at `sample_spacing_m`, and a KDTree
    over those points gives us a fast nearest-neighbor distance. Values are
    capped at `cap_m` (50 km default) to bound the feature range.
    """
    faults = fault_lines.to_crs(grid.crs)
    # Densify each line into a set of points.
    pts = []
    for geom in faults.geometry:
        if geom is None or geom.is_empty:
            continue
        length = geom.length
        n = max(2, int(np.ceil(length / sample_spacing_m)))
        ds = np.linspace(0, length, n)
        pts.extend([geom.interpolate(d) for d in ds])
    if not pts:
        return np.full(grid.shape, cap_m, dtype=np.float32)
    fault_xy = np.array([(p.x, p.y) for p in pts])
    tree = cKDTree(fault_xy)

    xv, yv = np.meshgrid(grid.xs, grid.ys)
    grid_xy = np.column_stack([xv.ravel(), yv.ravel()])
    dist, _ = tree.query(grid_xy)
    return np.clip(dist.reshape(grid.shape), 0, cap_m).astype(np.float32)
