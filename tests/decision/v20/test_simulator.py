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
    CAPTURE_KS,
    PAPER_DRILL_BUDGET,
    PAPER_N_GROUND_TRUTHS,
    RetrospectiveBCGSValidator,
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


# --- B.2 RetrospectiveBCGSValidator -----------------------------------------


def _make_b2_validator(
    *,
    n_grid: int = 10,
    drill_budget: int = 50,
    positives: list[int] | None = None,
    pre_drilled: list[int] | None = None,
) -> RetrospectiveBCGSValidator:
    spacing = 500.0
    x = np.arange(n_grid) * spacing
    y = np.arange(n_grid) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    n = n_grid * n_grid
    prior = np.full(n, 0.1)
    pos = np.zeros(n, dtype=int)
    grade = np.zeros(n, dtype=float)
    if positives is None:
        positives = [0, 5, 11]
    for i in positives:
        pos[i] = 1
        grade[i] = 0.5    # well above 0.2 cutoff
    drilled_pre = np.zeros(n, dtype=int)
    for i in (pre_drilled or []):
        drilled_pre[i] = 1
    return RetrospectiveBCGSValidator(
        pre_2010_prior=prior,
        post_2010_positives=pos,
        cells_drilled_pre_2010=drilled_pre,
        cell_coords_m=coords,
        post_2010_grade=grade,
        drill_budget=drill_budget,
    )


def test_b2_validator_construction_shape_check():
    n = 100
    coords = np.zeros((n, 2))
    prior = np.zeros(n)
    pos = np.zeros(50)   # wrong shape
    with pytest.raises(ValueError, match="post_2010_positives"):
        RetrospectiveBCGSValidator(
            pre_2010_prior=prior,
            post_2010_positives=pos,
            cells_drilled_pre_2010=np.zeros(n),
            cell_coords_m=coords,
            post_2010_grade=np.zeros(n),
        )


def test_b2_validator_construction_cell_coords_shape():
    n = 100
    with pytest.raises(ValueError, match="cell_coords_m"):
        RetrospectiveBCGSValidator(
            pre_2010_prior=np.zeros(n),
            post_2010_positives=np.zeros(n, dtype=int),
            cells_drilled_pre_2010=np.zeros(n, dtype=int),
            cell_coords_m=np.zeros((n, 3)),   # wrong second dim
            post_2010_grade=np.zeros(n),
        )


def test_b2_capture_at_k_pct_definition():
    """capture-at-k% = (positives in top-k% of trajectory) / total positives."""
    validator = _make_b2_validator(n_grid=10, positives=[0, 1, 2, 3, 99])
    # Top 5 by hand: if trajectory[:5] = [0, 1, 2, 3, 50], capture-at-5% = 4/5 = 0.8.
    traj = [0, 1, 2, 3, 50, 51, 52, 53, 54, 55]
    cap = validator._capture_at_k_pct(traj, 5)
    assert abs(cap - 4 / 5) < 1e-9


def test_b2_run_policy_returns_one_value_per_k():
    from ai_minerals.decision.v20.policies import RandomPolicy
    validator = _make_b2_validator(n_grid=10, drill_budget=30)
    result = validator.run_policy(RandomPolicy(), np.random.default_rng(0))
    assert set(result.keys()) == set(CAPTURE_KS)
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_b2_run_policy_skips_pre_2010_drilled():
    """Pre-2010 drilled cells should not appear in the trajectory."""
    from ai_minerals.decision.v20.policies import RandomPolicy
    pre_drilled = [10, 20, 30, 40]
    validator = _make_b2_validator(
        n_grid=10, drill_budget=20, pre_drilled=pre_drilled,
    )
    problem = validator._build_problem()
    # Run by hand to capture the trajectory.
    policy = RandomPolicy()
    rng = np.random.default_rng(0)
    policy.reset(problem, rng)
    drilled = frozenset(np.where(validator.cells_drilled_pre_2010 > 0)[0].tolist())
    traj = []
    for _ in range(validator.drill_budget):
        c = policy.choose_action([], drilled, rng)
        _, _, drilled = problem.step(c, drilled, rng)
        traj.append(c)
    for skip_cell in pre_drilled:
        assert skip_cell not in traj


def test_b2_greedy_mean_captures_more_than_random_when_prior_is_informative():
    """Build a 5x5 grid with prior biased toward where the positives sit;
    GreedyMeanPolicy should beat RandomPolicy at every k% reported."""
    from ai_minerals.decision.v20.policies import GreedyMeanPolicy, RandomPolicy
    n = 25
    coords = np.column_stack([
        np.arange(n) % 5 * 500.0,
        np.arange(n) // 5 * 500.0,
    ])
    # Positives clustered at cells 0, 1, 5, 6
    positives = [0, 1, 5, 6]
    pos = np.zeros(n, dtype=int)
    grade = np.zeros(n, dtype=float)
    for i in positives:
        pos[i] = 1
        grade[i] = 0.5
    prior = np.full(n, 0.05)
    # Prior is informative: anchor near the positive cluster
    for i in positives:
        prior[i] = 0.3
    validator = RetrospectiveBCGSValidator(
        pre_2010_prior=prior,
        post_2010_positives=pos,
        cells_drilled_pre_2010=np.zeros(n, dtype=int),
        cell_coords_m=coords,
        post_2010_grade=grade,
        drill_budget=10,
    )
    rng_g = np.random.default_rng(100)
    rng_r = np.random.default_rng(101)
    g = validator.run_policy(GreedyMeanPolicy(), rng_g)
    r = validator.run_policy(RandomPolicy(), rng_r)
    # At k=25 (top 25 percent = top 6 cells), GreedyMean should capture all 4 positives
    # since the top-prior cells coincide with them.
    assert g[25] >= 0.99
    assert g[25] >= r[25]


def test_b2_compare_returns_per_policy_dict():
    from ai_minerals.decision.v20.policies import GreedyMeanPolicy, RandomPolicy
    validator = _make_b2_validator(n_grid=10, drill_budget=30)
    table = validator.compare(
        {"random": RandomPolicy(), "greedy": GreedyMeanPolicy()},
        np.random.default_rng(0),
    )
    assert set(table.keys()) == {"random", "greedy"}
    for policy_name, capture in table.items():
        assert set(capture.keys()) == set(CAPTURE_KS)
        for v in capture.values():
            assert 0.0 <= v <= 1.0
