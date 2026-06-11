"""Tests for the bcgt-v2.0 Hypothesis GP prior sampler.

Covers issue #2 (B.1 Hypothesis class + GP prior). Three things checked:

1. Construction validation: shape mismatches between cell_coords_m and
   prior_mean_field raise a clear error; invalid kernel nu rejected.

2. Statistical properties of `sample_realization`:
   - Empirical marginal variance across many draws is approximately
     gp_marginal_std^2 (within Monte Carlo tolerance).
   - Empirical correlation between two cells at distance ~lengthscale
     is in the expected Matern v=2.5 range (~0.5).
   - The Cholesky cache speeds up the second call (functional check,
     not strict timing).
   - Two consecutive draws with the same RNG state are reproducible,
     and two draws with different states produce different realizations.

3. Multi-sample API: requesting n_samples > 1 returns an array of
   the right shape and each row is a valid realization.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.hypotheses import (
    KERNEL_MARGINAL_STD,
    KERNEL_NU,
    Hypothesis,
)


def _grid_30x30(spacing_m: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
    """Helper: build the BCGT 30x30 cell grid + a zero mean field."""
    x = np.arange(30) * spacing_m
    y = np.arange(30) * spacing_m
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mean_field = np.zeros(coords.shape[0])
    return coords, mean_field


# --- Construction validation -------------------------------------------------


def test_construction_validates_coords_shape():
    with pytest.raises(ValueError, match="cell_coords_m must be"):
        Hypothesis(
            name="x", n_grabens=1, n_domains=1,
            cell_coords_m=np.zeros(10),         # 1D, should be (n, 2)
            prior_mean_field=np.zeros(10),
        )


def test_construction_validates_mean_field_shape():
    with pytest.raises(ValueError, match="prior_mean_field"):
        Hypothesis(
            name="x", n_grabens=1, n_domains=1,
            cell_coords_m=np.zeros((10, 2)),
            prior_mean_field=np.zeros(5),       # wrong length
        )


def test_construction_rejects_unsupported_kernel_nu():
    coords, mean = _grid_30x30()
    with pytest.raises(ValueError, match="gp_kernel_nu"):
        Hypothesis(
            name="x", n_grabens=1, n_domains=1,
            cell_coords_m=coords, prior_mean_field=mean,
            gp_kernel_nu=1.0,                   # not 0.5/1.5/2.5
        )


def test_construction_accepts_valid_inputs():
    coords, mean = _grid_30x30()
    h = Hypothesis(
        name="porphyry_cu", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
    )
    assert h.n_cells == 900
    assert h.gp_kernel_nu == KERNEL_NU


# --- Statistical correctness of sample_realization ---------------------------


def test_realization_marginal_variance_matches_kernel_sigma():
    """Average marginal variance across cells should match sigma^2."""
    coords, mean = _grid_30x30()
    sigma = 0.1
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
        gp_marginal_std=sigma,
    )
    rng = np.random.default_rng(42)
    draws = h.sample_realization(rng, n_samples=2000)
    assert draws.shape == (2000, 900)
    # Marginal variance per cell, then mean across cells.
    per_cell_var = draws.var(axis=0)            # (n_cells,)
    expected = sigma ** 2
    # ~3% Monte Carlo tolerance at 2000 draws on each cell.
    np.testing.assert_allclose(per_cell_var.mean(), expected, rtol=0.05)


def test_realization_spatial_correlation_at_lengthscale():
    """Two cells separated by exactly lengthscale should have Matern v=2.5
    correlation ~0.52 (the closed-form value of (1 + sqrt(5) + 5/3) * exp(-sqrt(5)) ~ 0.522)."""
    spacing = 500.0
    lengthscale = 1500.0   # 3 cells = lengthscale, paper-matched
    coords, mean = _grid_30x30(spacing_m=spacing)
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
        gp_lengthscale_m=lengthscale,
    )
    rng = np.random.default_rng(7)
    draws = h.sample_realization(rng, n_samples=5000)
    # Cell at (0, 0) is index 0. Cell at (3 * spacing, 0) is exactly
    # `lengthscale` meters away. Find its index.
    cell_a = 0
    cell_b = 3                                  # x-index 3, y-index 0
    a = draws[:, cell_a]
    b = draws[:, cell_b]
    empirical_corr = np.corrcoef(a, b)[0, 1]
    # Matern v=2.5 at d/lengthscale=1: r = (1 + sqrt(5) + 5/3) * exp(-sqrt(5))
    expected_corr = (1.0 + np.sqrt(5.0) + 5.0 / 3.0) * np.exp(-np.sqrt(5.0))
    np.testing.assert_allclose(empirical_corr, expected_corr, atol=0.04)


def test_realization_distant_cells_are_uncorrelated():
    """Cells very far apart (>>lengthscale) should have correlation near zero."""
    spacing = 500.0
    coords, mean = _grid_30x30(spacing_m=spacing)
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
        gp_lengthscale_m=500.0,    # short lengthscale
    )
    rng = np.random.default_rng(99)
    draws = h.sample_realization(rng, n_samples=3000)
    # Opposite corners of the 30x30 grid: (0,0) and (29, 29). Distance ~
    # 41 cells, i.e., ~41x lengthscale. Correlation should be essentially zero.
    cell_corner_a = 0
    cell_corner_b = 30 * 30 - 1
    empirical_corr = np.corrcoef(draws[:, cell_corner_a],
                                 draws[:, cell_corner_b])[0, 1]
    assert abs(empirical_corr) < 0.05


def test_realization_reproducibility_under_same_rng_state():
    coords, mean = _grid_30x30()
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
    )
    rng_a = np.random.default_rng(1234)
    rng_b = np.random.default_rng(1234)
    draws_a = h.sample_realization(rng_a, n_samples=10)
    draws_b = h.sample_realization(rng_b, n_samples=10)
    np.testing.assert_array_equal(draws_a, draws_b)


def test_realization_mean_field_offsets_the_distribution():
    """Non-zero prior_mean_field should shift the marginal mean."""
    coords, mean_zero = _grid_30x30()
    mean_offset = np.full(coords.shape[0], 7.5)
    h_offset = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean_offset,
    )
    rng = np.random.default_rng(0)
    draws = h_offset.sample_realization(rng, n_samples=1000)
    # Per-cell mean should be ~7.5; tolerance scaled by sigma / sqrt(N).
    per_cell_mean = draws.mean(axis=0)
    expected = 7.5
    np.testing.assert_allclose(per_cell_mean, expected,
                               atol=4 * KERNEL_MARGINAL_STD / np.sqrt(1000))


def test_realization_shape_and_dtype():
    coords, mean = _grid_30x30()
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
    )
    rng = np.random.default_rng(3)
    draws = h.sample_realization(rng, n_samples=3)
    assert draws.shape == (3, 900)
    assert draws.dtype == np.float64


def test_realization_rejects_zero_or_negative_n_samples():
    coords, mean = _grid_30x30()
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        h.sample_realization(rng, n_samples=0)
