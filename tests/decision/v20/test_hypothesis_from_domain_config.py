"""Tests for `Hypothesis.from_domain_config` — the bridge from the
structured-prior domain machinery in `domains.py` to a fully formed
Hypothesis instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.domains import (
    DOMAIN_MEAN_BOOST,
    GRABEN_MEAN_BOOST,
)
from ai_minerals.decision.v20.hypotheses import (
    KERNEL_MARGINAL_STD,
    KERNEL_NU,
    Hypothesis,
)


def test_from_domain_config_shape_and_grid_size() -> None:
    h = Hypothesis.from_domain_config(
        name="H_1_1", n_grabens=1, n_domains=1, grid_n=32,
        rng=np.random.default_rng(0),
    )
    assert h.n_cells == 32 * 32
    assert h.cell_coords_m.shape == (1024, 2)
    assert h.prior_mean_field.shape == (1024,)


def test_from_domain_config_metadata_passes_through() -> None:
    h = Hypothesis.from_domain_config(
        name="example", n_grabens=2, n_domains=1, grid_n=16,
        rng=np.random.default_rng(0),
    )
    assert h.name == "example"
    assert h.n_grabens == 2
    assert h.n_domains == 1


def test_from_domain_config_default_kernel_params_match_module_locks() -> None:
    h = Hypothesis.from_domain_config(
        name="H", n_grabens=1, n_domains=1, grid_n=16,
        rng=np.random.default_rng(0),
        cell_spacing_m=500.0,
    )
    assert h.gp_marginal_std == pytest.approx(KERNEL_MARGINAL_STD)
    assert h.gp_kernel_nu == pytest.approx(KERNEL_NU)
    # Default lengthscale is 3 * cell_spacing
    assert h.gp_lengthscale_m == pytest.approx(3.0 * 500.0)


def test_from_domain_config_explicit_lengthscale_overrides_default() -> None:
    h = Hypothesis.from_domain_config(
        name="H", n_grabens=1, n_domains=1, grid_n=16,
        rng=np.random.default_rng(0),
        cell_spacing_m=500.0,
        gp_lengthscale_m=750.0,
    )
    assert h.gp_lengthscale_m == pytest.approx(750.0)


def test_from_domain_config_prior_mean_range_reflects_boosts() -> None:
    rng = np.random.default_rng(0)
    h = Hypothesis.from_domain_config(
        name="H", n_grabens=2, n_domains=2, grid_n=32, rng=rng,
    )
    assert h.prior_mean_field.min() >= 0.0
    expected_max_possible = GRABEN_MEAN_BOOST + DOMAIN_MEAN_BOOST
    assert h.prior_mean_field.max() <= expected_max_possible + 1e-9
    # With n_grabens=2 and n_domains=2 there should be some non-zero cells
    assert h.prior_mean_field.max() > 0.0


def test_from_domain_config_deterministic_given_seed() -> None:
    h_a = Hypothesis.from_domain_config(
        name="H", n_grabens=1, n_domains=1, grid_n=16,
        rng=np.random.default_rng(42),
    )
    h_b = Hypothesis.from_domain_config(
        name="H", n_grabens=1, n_domains=1, grid_n=16,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_allclose(h_a.prior_mean_field, h_b.prior_mean_field)
    np.testing.assert_allclose(h_a.cell_coords_m, h_b.cell_coords_m)


def test_from_domain_config_with_zero_grabens_no_graben_contribution() -> None:
    """When n_grabens=0, no graben_boost should appear in the prior."""
    h = Hypothesis.from_domain_config(
        name="H_0_1", n_grabens=0, n_domains=1, grid_n=32,
        rng=np.random.default_rng(0),
    )
    # All non-zero contributions come from the domain only
    assert h.prior_mean_field.max() <= DOMAIN_MEAN_BOOST + 1e-9
    assert h.prior_mean_field.max() > 0.0


def test_from_domain_config_with_zero_domains_no_domain_contribution() -> None:
    h = Hypothesis.from_domain_config(
        name="H_1_0", n_grabens=1, n_domains=0, grid_n=32,
        rng=np.random.default_rng(0),
    )
    assert h.prior_mean_field.max() <= GRABEN_MEAN_BOOST + 1e-9
    assert h.prior_mean_field.max() > 0.0


def test_from_domain_config_sample_realization_yields_correlated_field() -> None:
    """A drawn realization should still have shape (n_cells,) and
    not blow up numerically."""
    rng = np.random.default_rng(0)
    h = Hypothesis.from_domain_config(
        name="H", n_grabens=1, n_domains=1, grid_n=16, rng=rng,
    )
    draws = h.sample_realization(np.random.default_rng(123), n_samples=3)
    assert draws.shape == (3, 16 * 16)
    assert np.isfinite(draws).all()
