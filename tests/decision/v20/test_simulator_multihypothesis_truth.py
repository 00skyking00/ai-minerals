"""Tests for ``SyntheticMonteCarloSimulator.run_multihypothesis``.

The new method samples ground truth from one of N paper hypotheses
per episode, used in the Mern 2024 4-hypothesis reproduction
benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.hypotheses import (
    Hypothesis,
    HypothesisSet,
    make_mern_2x2_hypothesis_set,
)
from ai_minerals.decision.v20.policies import (
    GreedyMeanPolicy,
    GridDrillingPolicy,
    RandomPolicy,
)
from ai_minerals.decision.v20.pomdp import (
    CorrelatedDrillingProblem,
    SensorModel,
)
from ai_minerals.decision.v20.simulator import SyntheticMonteCarloSimulator


def _make_template_problem(grid_n: int = 16) -> CorrelatedDrillingProblem:
    """Build a small template problem the simulator can drive policies on."""
    rng = np.random.default_rng(0)
    h = Hypothesis.from_domain_config(
        name="template", n_grabens=1, n_domains=1, grid_n=grid_n, rng=rng,
    )
    n_cells = grid_n * grid_n
    return CorrelatedDrillingProblem(
        hypothesis=h,
        x_m=h.cell_coords_m[:, 0],
        y_m=h.cell_coords_m[:, 1],
        true_grade=np.zeros(n_cells),  # placeholder; sim replaces per episode
        sensor_model=SensorModel.GAUSSIAN_CONTINUOUS,
        sensor_noise_sigma=0.001,
        cutoff_grade=0.05,
        drill_cost=1.0,
        discovery_value=50.0,
    )


def test_run_multihypothesis_produces_one_episode_per_request() -> None:
    template = _make_template_problem(grid_n=16)
    truth_set = make_mern_2x2_hypothesis_set(grid_n=16, seed=0, include_null=False)
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"random": RandomPolicy()},
        n_ground_truths=8,
        drill_budget=4,
    )
    episodes = sim.run_multihypothesis(
        rng=np.random.default_rng(0),
        truth_hypothesis_set=truth_set,
    )
    assert len(episodes) == 8


def test_run_multihypothesis_n_episodes_per_truth_distributes_evenly() -> None:
    template = _make_template_problem(grid_n=16)
    truth_set = make_mern_2x2_hypothesis_set(grid_n=16, seed=0, include_null=False)
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"random": RandomPolicy()},
        drill_budget=4,
    )
    # 4 hypotheses x 3 per = 12 episodes
    episodes = sim.run_multihypothesis(
        rng=np.random.default_rng(0),
        truth_hypothesis_set=truth_set,
        n_episodes_per_truth=3,
    )
    assert len(episodes) == 12


def test_run_multihypothesis_truths_vary_across_episodes() -> None:
    """Different truth hypotheses should yield distinct true_grade fields
    most of the time (probabilistic; we check that not all 17 fields are
    identical)."""
    template = _make_template_problem(grid_n=16)
    truth_set = make_mern_2x2_hypothesis_set(grid_n=16, seed=0, include_null=False)
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"random": RandomPolicy()},
        n_ground_truths=17,
        drill_budget=4,
    )
    episodes = sim.run_multihypothesis(
        rng=np.random.default_rng(42),
        truth_hypothesis_set=truth_set,
    )
    fields = [ep.true_grade_field for ep in episodes]
    distinct_pairs = 0
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            if not np.allclose(fields[i], fields[j]):
                distinct_pairs += 1
    # With 17 episodes there are 136 pairs; we expect almost all of them distinct
    assert distinct_pairs > 100


def test_run_multihypothesis_accepts_multiple_policies() -> None:
    template = _make_template_problem(grid_n=16)
    truth_set = make_mern_2x2_hypothesis_set(grid_n=16, seed=0, include_null=False)
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={
            "random": RandomPolicy(),
            "greedy": GreedyMeanPolicy(),
            "grid": GridDrillingPolicy(n_per_side=2, margin=1),
        },
        n_ground_truths=4,
        drill_budget=4,
    )
    episodes = sim.run_multihypothesis(
        rng=np.random.default_rng(0),
        truth_hypothesis_set=truth_set,
    )
    for ep in episodes:
        assert set(ep.policy_trajectories.keys()) == {"random", "greedy", "grid"}
        for trajectory in ep.policy_trajectories.values():
            assert len(trajectory) == 4


def test_run_multihypothesis_rejects_empty_hypothesis_set() -> None:
    template = _make_template_problem(grid_n=16)
    empty_set = HypothesisSet(hypotheses=(), include_null=False)
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"random": RandomPolicy()},
        n_ground_truths=2,
        drill_budget=4,
    )
    with pytest.raises(ValueError, match="at least one paper hypothesis"):
        sim.run_multihypothesis(
            rng=np.random.default_rng(0),
            truth_hypothesis_set=empty_set,
        )


def test_run_multihypothesis_legacy_run_method_still_works() -> None:
    """Backward compat: the existing single-hypothesis ``run`` path should
    not be affected by the new method."""
    template = _make_template_problem(grid_n=16)
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={"random": RandomPolicy()},
        n_ground_truths=3,
        drill_budget=4,
    )
    episodes = sim.run(np.random.default_rng(0))
    assert len(episodes) == 3
