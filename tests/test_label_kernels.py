"""Verify the v3.6 polygon-to-grid label rasterizer.

Three properties:
  1. A polygon covering known cells produces the expected interior + edge mask.
  2. Changing ``edge_weight`` only affects the buffer cells.
  3. A polygon smaller than one cell still produces a single full-weight cell.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from ai_minerals.features.label_kernels import rasterize_polygon_positives
from ai_minerals.grid import Grid


def _build_10x10_grid() -> Grid:
    """10x10 grid, 100 m cells, centers at 50..950 along both axes.

    Cell (row=i, col=j) has center (xs[j], ys[i]) and covers
    [xs[j]-50, xs[j]+50) x [ys[i]-50, ys[i]+50).
    """
    xs = np.arange(50.0, 1000.0, 100.0)
    ys = np.arange(50.0, 1000.0, 100.0)
    return Grid(xs=xs, ys=ys, resolution_m=100, crs="EPSG:3310")


def _grid_2d(arr: np.ndarray, grid: Grid) -> np.ndarray:
    return arr.reshape(grid.shape)


def test_single_polygon_covers_known_cells() -> None:
    """Polygon over rows 4-6 cols 4-6 gives 9 interior cells + 16 buffer cells."""
    grid = _build_10x10_grid()

    # Polygon edges aligned just inside the cell boundary so centroids of
    # rows 4-6 cols 4-6 (centers at 450..650 on both axes) fall inside.
    poly = Polygon([
        (400.5, 400.5),
        (700.5, 400.5),
        (700.5, 700.5),
        (400.5, 700.5),
    ])
    polys = gpd.GeoDataFrame({"id": [1], "geometry": [poly]}, crs="EPSG:3310")

    out = rasterize_polygon_positives(polys, grid)
    assert out.shape == (grid.n_cells,)
    assert out.dtype == np.float32

    out2d = _grid_2d(out, grid)

    # Interior: rows 4-6, cols 4-6 → 9 cells at 1.0.
    interior = out2d[4:7, 4:7]
    assert interior.shape == (3, 3)
    assert np.all(interior == np.float32(1.0))
    assert int((out == np.float32(1.0)).sum()) == 9

    # Edge buffer (Chebyshev distance 1): rows 3-7, cols 3-7 minus interior
    # → 25 - 9 = 16 cells at 0.5.
    buffer_block = out2d[3:8, 3:8].copy()
    buffer_block[1:4, 1:4] = np.nan  # mask out the interior we already checked
    edge_cells = buffer_block[~np.isnan(buffer_block)]
    assert edge_cells.size == 16
    assert np.all(edge_cells == np.float32(0.5))
    assert int((out == np.float32(0.5)).sum()) == 16

    # Everything else stays 0.
    n_zero = int((out == np.float32(0.0)).sum())
    assert n_zero == grid.n_cells - 9 - 16


def test_edge_weight_param() -> None:
    """Changing edge_weight from 0.5 to 0.3 only changes the buffer cells."""
    grid = _build_10x10_grid()
    poly = Polygon([
        (400.5, 400.5),
        (700.5, 400.5),
        (700.5, 700.5),
        (400.5, 700.5),
    ])
    polys = gpd.GeoDataFrame({"id": [1], "geometry": [poly]}, crs="EPSG:3310")

    out_default = rasterize_polygon_positives(polys, grid)
    out_low = rasterize_polygon_positives(polys, grid, edge_weight=0.3)

    # Interior cells (==1.0) and zero cells (==0.0) match exactly.
    interior_default = out_default == np.float32(1.0)
    interior_low = out_low == np.float32(1.0)
    np.testing.assert_array_equal(interior_default, interior_low)

    zero_default = out_default == np.float32(0.0)
    zero_low = out_low == np.float32(0.0)
    np.testing.assert_array_equal(zero_default, zero_low)

    # Cells that were 0.5 in the default run are now 0.3.
    edge_mask = out_default == np.float32(0.5)
    assert edge_mask.sum() == 16
    assert np.all(out_low[edge_mask] == np.float32(0.3))

    # Diff is confined to the buffer: cells outside the buffer are unchanged.
    diff = out_default != out_low
    np.testing.assert_array_equal(diff, edge_mask)


def test_small_polygon_does_not_disappear() -> None:
    """A polygon smaller than one cell still rasterizes to weight 1.0 at the
    nearest centroid."""
    grid = _build_10x10_grid()

    # A tiny 1 m square well inside cell (row=5, col=5), whose centroid is
    # at (550, 550). The polygon is far too small to contain any centroid.
    poly = Polygon([
        (552.0, 552.0),
        (553.0, 552.0),
        (553.0, 553.0),
        (552.0, 553.0),
    ])
    polys = gpd.GeoDataFrame({"id": [99], "geometry": [poly]}, crs="EPSG:3310")

    out = rasterize_polygon_positives(polys, grid)
    out2d = _grid_2d(out, grid)

    # Exactly one cell at full weight: row 5, col 5 (the nearest centroid).
    assert out2d[5, 5] == np.float32(1.0)
    assert int((out == np.float32(1.0)).sum()) == 1

    # The 8 Chebyshev neighbours get the edge weight.
    neighbours = [
        (r, c)
        for r in (4, 5, 6)
        for c in (4, 5, 6)
        if (r, c) != (5, 5)
    ]
    for r, c in neighbours:
        assert out2d[r, c] == pytest.approx(0.5), (
            f"expected edge weight at ({r},{c}), got {out2d[r, c]}"
        )
    assert int((out == np.float32(0.5)).sum()) == 8
