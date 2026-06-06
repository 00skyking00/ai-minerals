"""Rasterize polygon labels onto the analysis grid with a soft edge buffer.

v3.6 Tertiary polygons (deep-gravel terrace patches digitized from Lindgren)
are larger than a single grid cell but small enough that hard centroid-in-
polygon labeling is noisy at the boundary. ``rasterize_polygon_positives``
gives full weight (1.0) to cells whose centroid sits inside any polygon,
partial weight (``edge_weight``, default 0.5) to a one-cell Chebyshev buffer
around the inside mask, and 0.0 elsewhere. Polygons smaller than one cell
still produce a single full-weight cell at their nearest centroid so no
target polygon silently disappears.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy.ndimage import binary_dilation

from ai_minerals.grid import Grid


def rasterize_polygon_positives(
    polys: gpd.GeoDataFrame,
    grid: Grid,
    *,
    edge_buffer_cells: int = 1,
    edge_weight: float = 0.5,
) -> np.ndarray:
    """Rasterize polygon footprints onto the grid as a soft-edge weight array.

    Cells whose centroid is inside any polygon get weight 1.0; cells within
    ``edge_buffer_cells`` of the inside mask (Chebyshev distance, via a 2-D
    binary dilation) get ``edge_weight``; all other cells get 0.0. A polygon
    smaller than one cell still rasterizes — the cell whose centroid is
    nearest the polygon's representative point is forced to weight 1.0 so we
    never drop a target polygon entirely.

    Returns a float32 array of shape ``(grid.n_cells,)`` in row-major order
    (matching ``grid.centroid_gdf()`` row ordering).
    """
    n_rows, n_cols = grid.shape
    inside_2d = np.zeros((n_rows, n_cols), dtype=bool)

    if len(polys) == 0:
        return np.zeros(grid.n_cells, dtype=np.float32)

    polys_proj = polys.to_crs(grid.crs)

    # Centroid-in-polygon test via spatial join. centroid_gdf() returns rows
    # in row-major order (row, col), so the join's index aligns 1:1 with
    # the flat output array.
    centroids = grid.centroid_gdf()
    joined = gpd.sjoin(
        centroids[["row", "col", "geometry"]],
        polys_proj[["geometry"]],
        how="inner",
        predicate="within",
    )
    if len(joined) > 0:
        inside_rows = joined["row"].to_numpy(dtype=np.int64)
        inside_cols = joined["col"].to_numpy(dtype=np.int64)
        inside_2d[inside_rows, inside_cols] = True

    # Sub-cell polygons can miss every centroid via the `within` predicate.
    # For each polygon that lit up zero inside cells, snap its representative
    # point to the nearest cell centroid and mark that cell as inside.
    inside_poly_idx = set()
    if len(joined) > 0:
        inside_poly_idx = set(joined["index_right"].to_numpy().tolist())
    missing_mask = ~polys_proj.index.isin(inside_poly_idx)
    if missing_mask.any():
        r = grid.resolution_m
        x0 = grid.xs[0] - r / 2
        y0 = grid.ys[0] - r / 2
        x1 = grid.xs[-1] + r / 2
        y1 = grid.ys[-1] + r / 2
        rep_pts = polys_proj.loc[missing_mask].geometry.representative_point()
        for pt in rep_pts:
            x, y = pt.x, pt.y
            if not (x0 <= x < x1 and y0 <= y < y1):
                # Polygon's representative point falls outside the grid extent;
                # nothing to snap to.
                continue
            col = int(np.floor((x - x0) / r))
            row = int(np.floor((y - y0) / r))
            inside_2d[row, col] = True

    if edge_buffer_cells > 0:
        # Chebyshev dilation: 3x3 ones structuring element grows the mask by
        # one cell in all 8 neighbours per iteration. Repeating `iterations=k`
        # gives a k-cell Chebyshev buffer.
        structure = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(
            inside_2d, structure=structure, iterations=edge_buffer_cells
        )
        edge_2d = dilated & ~inside_2d
    else:
        edge_2d = np.zeros_like(inside_2d)

    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    out[edge_2d] = np.float32(edge_weight)
    out[inside_2d] = np.float32(1.0)
    return out.ravel()
