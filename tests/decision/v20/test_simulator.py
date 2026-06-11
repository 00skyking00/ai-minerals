"""Tests for the bcgt-v2.0 SyntheticMonteCarloSimulator.

Covers issue #5 (B.1 simulator + Random / GreedyMean baselines).
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.hypotheses import Hypothesis
from ai_minerals.decision.v20.pomdp import (
    CorrelatedDrillingProblem,
    SensorModel,
)
from ai_minerals.decision.v20.policies import (
    GreedyMeanPolicy,
    RandomPolicy,
)
from ai_minerals.decision.v20.simulator import (
    PAPER_DRILL_BUDGET,
    PAPER_N_GROUND_TRUTHS,
    SyntheticMonteCarloSimulator,
)


def _make_template(
    *,
    spacing_m: float = 500.0,
    mean_offset: float = 0.0,
    sensor_sigma: float = 0.01,
    cutoff_grade: float = 0.05,
) -> CorrelatedDrillingProblem:
    x = np.arange(10) * spacing_m
    y = np.arange(10) * spacing_m
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mean = np.full(coords.shape[0], mean_offset)
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
    )
    n = coords.shape[0]
    return CorrelatedDrillingProblem(
        hypothesis=h,
        x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=np.zeros(n),     # gets overwritten per-episode
        sensor_model=SensorModel.GAUSSIAN_CONTINUOUS,
        sensor_noise_sigma=sensor_sigma,
        cutoff_grade=cutoff_grade,
        drill_cost=1.0,
        discovery_value=50.0,
    )


# --- Simulator structure ----------------------------------------------------


def test_run_returns_one_episode_per_ground_truth():
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"random": RandomPolicy(), "greedy": GreedyMeanPolicy()},
        n_ground_truths=5,
        drill_budget=4,
    )
    episodes = sim.run(np.random.default_rng(0))
    assert len(episodes) == 5
    for ep in episodes:
        assert ep.true_grade_field.shape == (100,)
        assert set(ep.policy_trajectories.keys()) == {"random", "greedy"}
        for traj in ep.policy_trajectories.values():
            assert len(traj) == 4
            assert all(0 <= c < 100 for c in traj)


def test_default_constants_match_paper_p20():
    """Paper p.20 says 17 ground truths, ~9 holes per episode."""
    assert PAPER_N_GROUND_TRUTHS == 17
    assert PAPER_DRILL_BUDGET == 9


def test_simulator_reproducible_under_same_rng():
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"r": RandomPolicy()},
        n_ground_truths=3, drill_budget=4,
    )
    a = sim.run(np.random.default_rng(42))
    b = sim.run(np.random.default_rng(42))
    for ep_a, ep_b in zip(a, b):
        np.testing.assert_array_equal(
            ep_a.policy_trajectories["r"], ep_b.policy_trajectories["r"]
        )


def test_simulator_different_seeds_produce_different_episodes():
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"r": RandomPolicy()},
        n_ground_truths=3, drill_budget=4,
    )
    a = sim.run(np.random.default_rng(1))
    b = sim.run(np.random.default_rng(2))
    # At least one episode should differ in its trajectory.
    diffs = sum(
        ep_a.policy_trajectories["r"] != ep_b.policy_trajectories["r"]
        for ep_a, ep_b in zip(a, b)
    )
    assert diffs >= 1


# --- Policy behavior on real episodes ----------------------------------------


def test_random_policy_does_not_repeat_cells():
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"r": RandomPolicy()},
        n_ground_truths=3, drill_budget=6,
    )
    episodes = sim.run(np.random.default_rng(11))
    for ep in episodes:
        traj = ep.policy_trajectories["r"]
        assert len(set(traj)) == len(traj), (
            f"RandomPolicy repeated a cell: {traj}"
        )


def test_greedy_policy_picks_highest_prior_mean_first():
    """If prior_mean is rank-ordered (one cell with high mean, others 0),
    GreedyMeanPolicy should pick the high-mean cell first."""
    template = _make_template()
    # Inject a per-cell prior mean that has one dominant cell.
    pm = np.zeros(100)
    pm[55] = 5.0
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=template.hypothesis.cell_coords_m,
        prior_mean_field=pm,
    )
    template2 = CorrelatedDrillingProblem(
        hypothesis=h,
        x_m=template.x_m, y_m=template.y_m,
        true_grade=np.zeros(100),
        sensor_model=template.sensor_model,
        sensor_noise_sigma=template.sensor_noise_sigma,
        cutoff_grade=template.cutoff_grade,
        drill_cost=template.drill_cost,
        discovery_value=template.discovery_value,
    )
    sim = SyntheticMonteCarloSimulator(
        problem_template=template2,
        policies={"g": GreedyMeanPolicy()},
        n_ground_truths=3, drill_budget=4,
    )
    episodes = sim.run(np.random.default_rng(99))
    for ep in episodes:
        assert ep.policy_trajectories["g"][0] == 55


def test_discovery_rate_in_unit_interval():
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"r": RandomPolicy(), "g": GreedyMeanPolicy()},
        n_ground_truths=8, drill_budget=6,
    )
    episodes = sim.run(np.random.default_rng(7))
    for ep in episodes:
        for name in ("r", "g"):
            dr = ep.policy_discovery_rates[name]
            assert 0.0 <= dr <= 1.0


def test_regret_is_non_negative():
    """The optimal-drill-the-top-K-by-true-grade baseline is the
    realizable upper bound; any actual trajectory regrets >= 0."""
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"r": RandomPolicy(), "g": GreedyMeanPolicy()},
        n_ground_truths=10, drill_budget=5,
    )
    episodes = sim.run(np.random.default_rng(33))
    for ep in episodes:
        for name in ("r", "g"):
            assert ep.policy_regrets[name] >= -1e-9   # FP slack


# --- Aggregation ------------------------------------------------------------


def test_aggregate_returns_per_policy_metrics():
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"r": RandomPolicy(), "g": GreedyMeanPolicy()},
        n_ground_truths=8, drill_budget=4,
    )
    episodes = sim.run(np.random.default_rng(0))
    agg = sim.aggregate(episodes)
    assert set(agg.keys()) == {"r", "g"}
    for name in ("r", "g"):
        metrics = agg[name]
        assert "discovery_rate_mean" in metrics
        assert "discovery_rate_median" in metrics
        assert "regret_mean" in metrics
        assert "regret_median" in metrics
        assert metrics["n_episodes"] == 8


def test_aggregate_handles_empty_episode_list():
    template = _make_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"r": RandomPolicy()},
        n_ground_truths=0, drill_budget=4,
    )
    assert sim.aggregate([]) == {}
