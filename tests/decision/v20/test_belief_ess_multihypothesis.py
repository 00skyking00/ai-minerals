"""Tests for the C.3 hardening fix: MultiHypothesisESSParticleFilter.

The particle filter replaces ``BcgtScaleSARSOPPolicy``'s
canonical-realization shortcut with a proper marginalization over GP
fields per hypothesis. These tests cover initialization, the categorical
posterior update on informative observations, marginal-probability
queries, and a small convergence check that the posterior shifts toward
the truth hypothesis after several informative observations.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.belief_ess import MultiHypothesisESSParticleFilter
from ai_minerals.decision.v20.hypotheses import (
    Hypothesis,
    HypothesisSet,
    NullHypothesis,
)


def _make_2hypothesis_set_with_disjoint_peaks(grid_n: int = 8) -> HypothesisSet:
    """Two GP-correlated hypotheses with peaks on opposite quadrants and
    a null. Small enough that the particle filter runs in milliseconds."""
    rng = np.random.default_rng(0)
    h_nw = Hypothesis.from_domain_config(
        name="H_NW", n_grabens=1, n_domains=1, grid_n=grid_n, rng=rng,
    )
    rng = np.random.default_rng(1)
    h_se = Hypothesis.from_domain_config(
        name="H_SE", n_grabens=1, n_domains=1, grid_n=grid_n, rng=rng,
    )
    return HypothesisSet(
        hypotheses=(h_nw, h_se),
        null=NullHypothesis(marginal_std=0.1),
        include_null=True,
    )


def test_initialize_sets_uniform_prior_belief() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=20, ess_refresh_steps=1,
    )
    pf.initialize(np.random.default_rng(0))
    np.testing.assert_allclose(
        pf.categorical_belief, [1/3, 1/3, 1/3], atol=1e-9,
    )


def test_initialize_creates_particles_per_paper_hypothesis() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=20, ess_refresh_steps=1,
    )
    pf.initialize(np.random.default_rng(0))
    particles = pf.particles
    # Only paper hypotheses get particle ensembles
    assert set(particles.keys()) == {0, 1}
    for hypothesis_index in (0, 1):
        assert particles[hypothesis_index].shape == (20, 16 * 16)


def test_marginal_probability_above_cutoff_shape_and_range() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=20, ess_refresh_steps=1,
    )
    pf.initialize(np.random.default_rng(0))
    marg = pf.marginal_probability_above_cutoff(cutoff=0.05)
    assert marg.shape == (3, 16 * 16)
    assert (marg >= 0.0).all()
    assert (marg <= 1.0).all()


def test_update_advances_observation_count() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=10, ess_refresh_steps=1,
    )
    pf.initialize(np.random.default_rng(0))
    assert pf.observation_count == 0
    pf.update(
        cell_idx=0, observation=0.15, sensor_noise_sigma=0.05,
        rng=np.random.default_rng(1),
    )
    assert pf.observation_count == 1
    pf.update(
        cell_idx=1, observation=0.0, sensor_noise_sigma=0.05,
        rng=np.random.default_rng(2),
    )
    assert pf.observation_count == 2


def test_categorical_belief_sums_to_one_after_updates() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=15, ess_refresh_steps=1,
    )
    pf.initialize(np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for cell_idx, obs in [(0, 0.15), (10, 0.0), (50, 0.08), (100, 0.01)]:
        pf.update(cell_idx=cell_idx, observation=obs,
                  sensor_noise_sigma=0.05, rng=rng)
    assert pf.categorical_belief.sum() == pytest.approx(1.0)


def test_categorical_belief_shifts_toward_truth_with_informative_obs() -> None:
    """Sample a truth realization from H_NW, feed positive observations from
    cells where H_NW's prior mean is high, check the categorical posterior
    on H_NW rises above its initial uniform value."""
    grid_n = 16
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=grid_n)
    h_nw = hset.hypotheses[0]
    # Pick the 5 cells with the highest H_NW prior mean
    high_cells = np.argsort(-h_nw.prior_mean_field)[:5]

    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=30, ess_refresh_steps=2,
    )
    pf.initialize(np.random.default_rng(0))
    initial_p_nw = pf.categorical_belief[0]

    rng = np.random.default_rng(42)
    truth = h_nw.sample_realization(rng=np.random.default_rng(7), n_samples=1)[0]
    for cell_idx in high_cells:
        cell_idx = int(cell_idx)
        obs = float(truth[cell_idx]) + 0.005 * rng.standard_normal()
        pf.update(
            cell_idx=cell_idx, observation=obs,
            sensor_noise_sigma=0.02, rng=rng,
        )
    final_p_nw = pf.categorical_belief[0]
    # We expect the posterior on H_NW to be MEANINGFULLY above 1/3 after
    # the 5 informative observations (not necessarily > 0.7 yet because
    # the H_SE prior may have non-negligible values at some of these cells
    # depending on the polygon layout)
    assert final_p_nw > initial_p_nw + 0.05


def test_rejects_zero_particles() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    with pytest.raises(ValueError, match="n_particles"):
        MultiHypothesisESSParticleFilter(
            hypothesis_set=hset, n_particles=0,
        )


def test_rejects_negative_ess_refresh_steps() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    with pytest.raises(ValueError, match="ess_refresh_steps"):
        MultiHypothesisESSParticleFilter(
            hypothesis_set=hset, n_particles=10, ess_refresh_steps=-1,
        )


def test_query_before_initialize_raises() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=10,
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = pf.categorical_belief


def test_update_before_initialize_raises() -> None:
    hset = _make_2hypothesis_set_with_disjoint_peaks(grid_n=16)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=10,
    )
    with pytest.raises(RuntimeError, match="initialize"):
        pf.update(
            cell_idx=0, observation=0.1,
            sensor_noise_sigma=0.05, rng=np.random.default_rng(0),
        )
