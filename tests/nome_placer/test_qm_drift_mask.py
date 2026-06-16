"""Tests for the score_qm drift_mask integration (2026-06-16).

Verifies that:
- Passing a drift_mask replaces the elevation-band proxy.
- An all-zero drift_mask kills QM entirely.
- An all-one drift_mask reproduces score_qm = stream_membership alone.
- Backward compat: no drift_mask falls back to the elevation band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_minerals.aoi import AOI
from ai_minerals.features.coastal_scorer import score_qm
from ai_minerals.grid import build_grid


@pytest.fixture(scope="module")
def small_grid():
    aoi = AOI(
        name="qm_test",
        min_lon=-165.35, min_lat=64.52,
        max_lon=-165.33, max_lat=64.54,
    )
    return build_grid(aoi, resolution_m=200, working_crs="EPSG:3338")


@pytest.fixture(scope="module")
def synthetic_dem(small_grid):
    """A flat DEM at 100 m for the legacy elevation-band fallback."""
    import numpy as np
    import xarray as xr
    H, W = small_grid.shape
    da = xr.DataArray(
        np.full((H, W), 100.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": small_grid.ys, "x": small_grid.xs},
    )
    da = da.rio.write_crs(small_grid.crs)
    return da


@pytest.fixture(scope="module")
def zero_distance(small_grid):
    """All cells exactly on a stream → stream_membership = 1."""
    n = small_grid.n_cells
    return pd.Series(np.zeros(n, dtype=np.float32), name="distance_to_stream_m")


def test_zero_drift_mask_kills_qm(small_grid, synthetic_dem, zero_distance):
    n = small_grid.n_cells
    mask = pd.Series(np.zeros(n, dtype=np.float32), name="drift_mask")
    qm = score_qm(synthetic_dem, small_grid, zero_distance, drift_mask=mask)
    assert (qm.to_numpy() == 0.0).all()


def test_full_drift_mask_yields_stream_membership_alone(
    small_grid, synthetic_dem, zero_distance,
):
    """When drift mask covers every cell AND distance=0, QM = 1.0 everywhere."""
    n = small_grid.n_cells
    mask = pd.Series(np.ones(n, dtype=np.float32), name="drift_mask")
    qm = score_qm(synthetic_dem, small_grid, zero_distance, drift_mask=mask)
    assert np.allclose(qm.to_numpy(), 1.0)


def test_partial_drift_mask_passes_through(small_grid, synthetic_dem, zero_distance):
    """Cells in mask score 1.0; cells out of mask score 0.0."""
    n = small_grid.n_cells
    mask_arr = np.zeros(n, dtype=np.float32)
    mask_arr[:n // 2] = 1.0
    mask = pd.Series(mask_arr, name="drift_mask")
    qm = score_qm(synthetic_dem, small_grid, zero_distance, drift_mask=mask)
    arr = qm.to_numpy()
    assert (arr[:n // 2] == 1.0).all()
    assert (arr[n // 2:] == 0.0).all()


def test_no_drift_mask_falls_back_to_elevation_band(
    small_grid, synthetic_dem, zero_distance,
):
    """Without drift_mask, the elevation-band proxy fires.

    DEM is at 100 m, default band is [30, 250] m, so every cell is
    in-band and QM should be 1.0 everywhere when distance = 0.
    """
    qm = score_qm(synthetic_dem, small_grid, zero_distance)
    assert np.allclose(qm.to_numpy(), 1.0)


def test_no_drift_mask_outside_elevation_band(small_grid, zero_distance):
    """DEM at 500 m is outside the default 30-250 m band; QM should be 0."""
    import xarray as xr
    H, W = small_grid.shape
    da = xr.DataArray(
        np.full((H, W), 500.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": small_grid.ys, "x": small_grid.xs},
    )
    da = da.rio.write_crs(small_grid.crs)
    qm = score_qm(da, small_grid, zero_distance)
    assert (qm.to_numpy() == 0.0).all()
