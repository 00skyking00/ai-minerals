"""Tests for the C.2 part 2 falsification check.

Covers GitHub issue #12.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.hypotheses import (
    Hypothesis, HypothesisSet, NullHypothesis,
)
from ai_minerals.decision.v20.policies import MultiHypothesisFalsificationPolicy
from ai_minerals.decision.v20.pomdp import (
    MultiHypothesisDrillingProblem,
    SensorModel,
)


def _make_hypothesis(name: str, mean_value: float, n_cells: int = 30) -> Hypothesis:
    coords = np.column_stack([np.arange(n_cells) * 500.0, np.zeros(n_cells)])
    return Hypothesis(
        name=name, n_grabens=1, n_domains=1,
        cell_coords_m=coords,
        prior_mean_field=np.full(n_cells, mean_value),
        gp_marginal_std=0.05,
    )


def _make_policy(
    *,
    paper_mean: float = 0.3,
    null_std: float = 0.1,
    include_null: bool = True,
) -> MultiHypothesisFalsificationPolicy:
    h_a = _make_hypothesis("A", paper_mean)
    h_b = _make_hypothesis("B", paper_mean)
    hs = HypothesisSet(
        hypotheses=(h_a, h_b),
        null=NullHypothesis(marginal_std=null_std),
        include_null=include_null,
    )
    # Minimal MultiHypothesisDrillingProblem stub; the falsification check
    # doesn't drill the problem, it just tracks posterior over hypotheses.
    coords = h_a.cell_coords_m
    problem = MultiHypothesisDrillingProblem(
        hypotheses=hs, true_hypothesis_idx=0,
        x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=np.zeros(coords.shape[0]),
    )
    return MultiHypothesisFalsificationPolicy(
        problem=problem, hypothesis_set=hs,
        sensor_noise_sigma=0.05,
        falsification_threshold_likelihood=0.5,
    )


# --- reset + posterior tracking --------------------------------------------


def test_reset_initializes_posterior_to_uniform():
    p = _make_policy()
    p.reset(np.random.default_rng(0))
    expected = np.full(3, 1 / 3)
    np.testing.assert_allclose(p._hypothesis_posterior, expected)


def test_step_posterior_shifts_with_consistent_observation():
    """A run of observations consistent with the paper hypotheses (mean 0.3)
    should shrink the null's posterior weight."""
    p = _make_policy()
    p.reset(np.random.default_rng(0))
    initial_null = p._hypothesis_posterior[2]
    for cell in range(8):
        p.step_posterior(cell_idx=cell, observation=0.3)
    final_null = p._hypothesis_posterior[2]
    assert final_null < initial_null


def test_step_posterior_boosts_null_with_inconsistent_observations():
    """A run of observations contradicting the paper hypotheses should boost
    the null's posterior."""
    p = _make_policy(paper_mean=0.3, null_std=0.1)
    p.reset(np.random.default_rng(0))
    initial_null = p._hypothesis_posterior[2]
    for cell in range(20):
        # Observations centered at 0 contradict mean-0.3 paper hypotheses
        # and match the null's zero-mean prior.
        p.step_posterior(cell_idx=cell, observation=0.0)
    final_null = p._hypothesis_posterior[2]
    assert final_null > initial_null + 0.3, (
        f"null should rise sharply with contradictory observations; "
        f"initial={initial_null:.3f}, final={final_null:.3f}"
    )


# --- falsification trigger --------------------------------------------------


def test_falsification_does_not_fire_at_initial_uniform():
    """Uniform prior 1/(N+1) is below the 0.5 likelihood threshold."""
    p = _make_policy()
    p.reset(np.random.default_rng(0))
    assert p.falsification_fired() is False


def test_falsification_fires_after_strong_contradiction():
    """After enough observations contradicting both paper hypotheses, the null
    should dominate enough to trigger the falsification flag."""
    p = _make_policy()
    p.reset(np.random.default_rng(0))
    for cell in range(20):
        p.step_posterior(cell_idx=cell, observation=0.0)
    assert p.falsification_fired() is True


def test_falsification_does_not_fire_when_paper_hypothesis_dominates():
    """If observations match a paper hypothesis the falsification flag stays
    down even after many observations."""
    p = _make_policy()
    p.reset(np.random.default_rng(0))
    for cell in range(20):
        p.step_posterior(cell_idx=cell, observation=0.3)
    assert p.falsification_fired() is False


def test_falsification_returns_false_when_null_disabled():
    """If the HypothesisSet excludes the null, falsification never fires."""
    p = _make_policy(include_null=False)
    p.reset(np.random.default_rng(0))
    for cell in range(20):
        p.step_posterior(cell_idx=cell, observation=0.0)
    assert p.falsification_fired() is False


def test_step_posterior_rejects_uninitialized_call():
    p = _make_policy()
    with pytest.raises(RuntimeError, match="not reset"):
        p.step_posterior(cell_idx=0, observation=0.0)
