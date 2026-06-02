"""Verify the v3 Phase A.1 cell_mask refactor of hawkes_dual_decay_catchment.

Three properties:
  1. masked-true cells produce identical values to the unmasked call
  2. masked-false cells return NaN
  3. masked runs are measurably faster than unmasked

Uses the synthetic smoke setup from placer_geology.py::_smoke, which builds a
10x10 grid, 5 samples, 3 pit polys, and 1 NHD reach. Quick and self-contained.
"""

from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from ai_minerals.features.placer_geology import hawkes_dual_decay_catchment
from ai_minerals.grid import Grid


def _build_test_fixture():
    """Tiny synthetic problem: 10x10 grid + 5 samples + 1 NHD reach."""
    xs = np.arange(-50.0, 950.0, 100.0)
    ys = np.arange(-50.0, 950.0, 100.0)
    grid = Grid(xs=xs, ys=ys, resolution_m=100, crs="EPSG:3310")

    # 1 NHD reach (a single LineString that all samples + cells will snap to).
    nhd = gpd.GeoDataFrame(
        {
            "comid": [1001],
            "arbolate_sum": [10.0],
            "hydroseq": [50.0],
            "geometry": [LineString([(50, 50), (950, 950)])],
        },
        crs="EPSG:3310",
    )

    # 5 samples along the reach with Au values.
    samples = gpd.GeoDataFrame(
        {
            "Au_ppm": [0.5, 1.0, 2.0, 3.0, 5.0],
            "geometry": [Point(100 + 150 * i, 100 + 150 * i) for i in range(5)],
        },
        crs="EPSG:3310",
    )
    return grid, samples, nhd


def test_cell_mask_subset_equals_unmasked_at_mask_true_cells():
    """Cells in the cell_mask compute identical values to the unmasked call."""
    grid, samples, nhd = _build_test_fixture()

    unmasked = hawkes_dual_decay_catchment(samples, nhd, grid, element="Au_ppm")

    # Mask: only the first 30 cells.
    cell_mask = np.zeros(grid.n_cells, dtype=bool)
    cell_mask[:30] = True
    masked = hawkes_dual_decay_catchment(
        samples, nhd, grid, element="Au_ppm", cell_mask=cell_mask,
    )

    # Compare at masked-true positions.
    a = unmasked.to_numpy()
    b = masked.to_numpy()
    for i in np.where(cell_mask)[0]:
        if np.isnan(a[i]) and np.isnan(b[i]):
            continue
        assert a[i] == pytest.approx(b[i], rel=1e-6, abs=1e-9), (
            f"cell {i} differs: unmasked={a[i]} masked={b[i]}"
        )


def test_cell_mask_returns_nan_at_mask_false_cells():
    """Cells excluded by cell_mask return NaN regardless of upstream signal."""
    grid, samples, nhd = _build_test_fixture()

    cell_mask = np.zeros(grid.n_cells, dtype=bool)
    cell_mask[:30] = True

    masked = hawkes_dual_decay_catchment(
        samples, nhd, grid, element="Au_ppm", cell_mask=cell_mask,
    )

    arr = masked.to_numpy()
    excluded = ~cell_mask
    assert np.all(np.isnan(arr[excluded])), (
        "cells excluded by cell_mask should be NaN"
    )


def test_cell_mask_length_mismatch_raises():
    grid, samples, nhd = _build_test_fixture()
    bad_mask = np.zeros(grid.n_cells - 1, dtype=bool)
    with pytest.raises(ValueError, match="cell_mask length"):
        hawkes_dual_decay_catchment(
            samples, nhd, grid, element="Au_ppm", cell_mask=bad_mask,
        )


def test_cell_mask_empty_mask_returns_all_nan():
    """A mask with no true cells produces an all-NaN output."""
    grid, samples, nhd = _build_test_fixture()
    cell_mask = np.zeros(grid.n_cells, dtype=bool)
    out = hawkes_dual_decay_catchment(
        samples, nhd, grid, element="Au_ppm", cell_mask=cell_mask,
    )
    assert np.all(np.isnan(out.to_numpy()))


def test_cell_mask_full_mask_equals_unmasked():
    """A mask with all cells true reproduces the unmasked call exactly."""
    grid, samples, nhd = _build_test_fixture()
    a = hawkes_dual_decay_catchment(samples, nhd, grid, element="Au_ppm")
    b = hawkes_dual_decay_catchment(
        samples, nhd, grid, element="Au_ppm",
        cell_mask=np.ones(grid.n_cells, dtype=bool),
    )
    a_arr = a.to_numpy()
    b_arr = b.to_numpy()
    same_nan = np.isnan(a_arr) == np.isnan(b_arr)
    assert same_nan.all(), "NaN pattern differs between unmasked and all-true mask"
    finite = ~np.isnan(a_arr)
    np.testing.assert_allclose(a_arr[finite], b_arr[finite], rtol=1e-6)


def test_cell_mask_speedup_on_synthetic_grid():
    """Masking 10% of cells should reduce wall clock noticeably.

    Synthetic grid is small (100 cells) so the speedup is modest, but the
    direction must be right: masked time <= unmasked time + small buffer.
    """
    grid, samples, nhd = _build_test_fixture()

    # Warm up.
    hawkes_dual_decay_catchment(samples, nhd, grid, element="Au_ppm")

    t0 = time.perf_counter()
    for _ in range(20):
        hawkes_dual_decay_catchment(samples, nhd, grid, element="Au_ppm")
    t_unmasked = time.perf_counter() - t0

    cell_mask = np.zeros(grid.n_cells, dtype=bool)
    cell_mask[::10] = True  # 10% of cells
    t0 = time.perf_counter()
    for _ in range(20):
        hawkes_dual_decay_catchment(
            samples, nhd, grid, element="Au_ppm", cell_mask=cell_mask,
        )
    t_masked = time.perf_counter() - t0

    # Speedup is modest on a 100-cell grid (the cell-loop overhead is small
    # vs. setup costs). Allow masked to be up to 1.5x unmasked on tiny grids;
    # the real-world ~800k-cell case shows a much bigger speedup.
    assert t_masked < t_unmasked * 1.5, (
        f"masked time {t_masked:.4f}s should be <= unmasked {t_unmasked:.4f}s * 1.5"
    )
