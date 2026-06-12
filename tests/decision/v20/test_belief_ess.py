"""Tests for the bcgt-v2.0 Elliptical Slice Sampler (Murray 2010).

Covers C.2 part 1 issue #10.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.belief_ess import (
    DEFAULT_BURNIN,
    DEFAULT_N_ITERATIONS,
    DEFAULT_THIN,
    EllipticalSliceSampler,
    elliptical_slice_step,
    log_gaussian_observation_likelihood,
)
from ai_minerals.decision.v20.hypotheses import (
    KERNEL_LENGTHSCALE_M_BCGT, KERNEL_MARGINAL_STD, Hypothesis,
)


def _make_test_hypothesis(n_grid: int = 10) -> Hypothesis:
    spacing = 500.0
    x = np.arange(n_grid) * spacing
    y = np.arange(n_grid) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mean = np.full(coords.shape[0], 0.1)
    return Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
        gp_marginal_std=KERNEL_MARGINAL_STD,
        gp_lengthscale_m=KERNEL_LENGTHSCALE_M_BCGT,
    )


# --- log-likelihood helper --------------------------------------------------


def test_log_gaussian_observation_likelihood_no_observations_returns_zero():
    assert log_gaussian_observation_likelihood(
        np.zeros(10), observations=[], sensor_noise_sigma=0.05,
    ) == 0.0


def test_log_gaussian_observation_likelihood_penalizes_residuals():
    """Closer-to-observed fields should have higher log-likelihood."""
    f_near = np.full(10, 0.5)
    f_far = np.full(10, 0.1)
    obs = [(0, 0.5), (5, 0.5)]
    sigma = 0.05
    lik_near = log_gaussian_observation_likelihood(f_near, obs, sigma)
    lik_far = log_gaussian_observation_likelihood(f_far, obs, sigma)
    assert lik_near > lik_far


# --- single ESS step --------------------------------------------------------


def test_elliptical_slice_step_returns_field_of_correct_shape():
    h = _make_test_hypothesis()
    K_chol = h._cholesky()
    rng = np.random.default_rng(0)
    f0 = K_chol @ rng.standard_normal(K_chol.shape[0])

    def trivial_log_lik(f: np.ndarray) -> float:
        return 0.0   # uniform likelihood

    f_new, n_shrink = elliptical_slice_step(f0, trivial_log_lik, K_chol, rng)
    assert f_new.shape == f0.shape
    assert n_shrink >= 0


def test_elliptical_slice_step_under_uniform_likelihood_yields_prior_marginal():
    """With a constant (zero) log-likelihood, ESS samples should match the
    GP prior. Run 200 chain steps from a single particle; check the empirical
    marginal variance against the kernel marginal variance."""
    h = _make_test_hypothesis()
    K_chol = h._cholesky()
    rng = np.random.default_rng(7)
    n_cells = K_chol.shape[0]
    f = K_chol @ rng.standard_normal(n_cells)

    def log_lik_zero(_f: np.ndarray) -> float:
        return 0.0

    samples = []
    for _ in range(400):
        f, _ = elliptical_slice_step(f, log_lik_zero, K_chol, rng)
        samples.append(f.copy())
    arr = np.array(samples)
    # Marginal stdev at each cell should be ~ gp_marginal_std (0.1).
    per_cell_std = arr.std(axis=0)
    np.testing.assert_allclose(per_cell_std.mean(), 0.1, atol=0.025)


def test_elliptical_slice_step_concentrates_around_observation():
    """A sharp observation should pull the posterior toward the observed value
    at that cell."""
    h = _make_test_hypothesis()
    K_chol = h._cholesky()
    rng = np.random.default_rng(11)
    n_cells = K_chol.shape[0]
    f = K_chol @ rng.standard_normal(n_cells)

    obs = [(50, 0.5)]
    sigma = 0.01

    def log_lik(f_: np.ndarray) -> float:
        return log_gaussian_observation_likelihood(f_, obs, sigma)

    samples = []
    for _ in range(500):
        f, _ = elliptical_slice_step(f, log_lik, K_chol, rng)
        samples.append(f[50])
    arr = np.array(samples)
    # Post-burnin mean at the observed cell should land near 0.5
    np.testing.assert_allclose(arr[100:].mean(), 0.5, atol=0.05)


# --- EllipticalSliceSampler driver ------------------------------------------


def test_ess_sampler_defaults_match_module_constants():
    assert DEFAULT_N_ITERATIONS == 1000
    assert DEFAULT_BURNIN == 200
    assert DEFAULT_THIN == 10


def test_ess_sampler_construction_rejects_bad_args():
    h = _make_test_hypothesis()
    with pytest.raises(ValueError, match="n_iterations"):
        EllipticalSliceSampler(hypothesis=h, n_iterations=0)
    with pytest.raises(ValueError, match="burnin"):
        EllipticalSliceSampler(hypothesis=h, n_iterations=10, burnin=10)
    with pytest.raises(ValueError, match="thin"):
        EllipticalSliceSampler(hypothesis=h, thin=0)


def test_ess_sampler_returns_thinned_samples():
    h = _make_test_hypothesis()
    sampler = EllipticalSliceSampler(
        hypothesis=h, n_iterations=100, burnin=20, thin=5,
    )
    samples = sampler.sample_chain(
        observations=[(0, 0.4), (50, 0.3)],
        sensor_noise_sigma=0.05,
        rng=np.random.default_rng(0),
    )
    # Expected count: floor((100 - 20) / 5) = 16
    assert samples.shape[0] == 16
    assert samples.shape[1] == 100   # 10x10 grid


def test_ess_sampler_is_reproducible_with_same_rng():
    h = _make_test_hypothesis()
    sampler = EllipticalSliceSampler(
        hypothesis=h, n_iterations=60, burnin=10, thin=2,
    )
    a = sampler.sample_chain(
        observations=[(5, 0.4)], sensor_noise_sigma=0.05,
        rng=np.random.default_rng(42),
    )
    b = sampler.sample_chain(
        observations=[(5, 0.4)], sensor_noise_sigma=0.05,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_allclose(a, b)


def test_ess_sampler_posterior_mean_tracks_observations():
    """If we observe a cell at a strongly-positive value, the chain's
    sample-averaged field should be elevated at that cell relative to a
    distant cell."""
    h = _make_test_hypothesis()
    sampler = EllipticalSliceSampler(
        hypothesis=h, n_iterations=500, burnin=100, thin=5,
    )
    samples = sampler.sample_chain(
        observations=[(50, 0.5)], sensor_noise_sigma=0.02,
        rng=np.random.default_rng(99),
    )
    mean_at_obs = samples[:, 50].mean()
    mean_far = samples[:, 0].mean()    # cell 0 is far from cell 50 in the 10x10
    assert mean_at_obs > mean_far + 0.1, (
        f"posterior mean should be higher at observed cell: "
        f"obs_cell={mean_at_obs:.3f}, far_cell={mean_far:.3f}"
    )
