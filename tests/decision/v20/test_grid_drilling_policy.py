"""Tests for the GridDrillingPolicy baseline.

Mern et al. 2024's headline POMDP-vs-grid comparison uses a 6x6
regular sub-grid drilling 36 boreholes over a 32x32 working area.
This policy reproduces the baseline so the same SyntheticMonteCarloSimulator
loop can drive it alongside Random / BayesianGreedy / POMCP / SARSOP.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.hypotheses import Hypothesis
from ai_minerals.decision.v20.policies import GridDrillingPolicy
from ai_minerals.decision.v20.pomdp import (
    CorrelatedDrillingProblem,
    SensorModel,
)


def _make_problem(grid_n: int) -> CorrelatedDrillingProblem:
    """Build a minimal CorrelatedDrillingProblem the policy can reset against."""
    rng = np.random.default_rng(0)
    h = Hypothesis.from_domain_config(
        name="H", n_grabens=1, n_domains=1, grid_n=grid_n, rng=rng,
    )
    n_cells = grid_n * grid_n
    return CorrelatedDrillingProblem(
        hypothesis=h,
        x_m=h.cell_coords_m[:, 0],
        y_m=h.cell_coords_m[:, 1],
        true_grade=np.zeros(n_cells),
        sensor_model=SensorModel.GAUSSIAN_CONTINUOUS,
        sensor_noise_sigma=0.001,
    )


def test_grid_drilling_default_produces_36_unique_cells_on_32x32() -> None:
    policy = GridDrillingPolicy()
    problem = _make_problem(grid_n=32)
    policy.reset(problem, np.random.default_rng(0))
    drilled: set[int] = set()
    for _ in range(36):
        cell = policy.choose_action([], frozenset(drilled), np.random.default_rng(0))
        assert cell not in drilled, "GridDrillingPolicy returned a duplicate cell"
        drilled.add(cell)
    assert len(drilled) == 36


def test_grid_drilling_n_per_side_3_produces_9_unique_cells() -> None:
    policy = GridDrillingPolicy(n_per_side=3, margin=1)
    problem = _make_problem(grid_n=16)
    policy.reset(problem, np.random.default_rng(0))
    drilled: set[int] = set()
    for _ in range(9):
        cell = policy.choose_action([], frozenset(drilled), np.random.default_rng(0))
        drilled.add(cell)
    assert len(drilled) == 9


def test_grid_drilling_inferred_grid_size_from_problem_n_cells() -> None:
    policy = GridDrillingPolicy(n_per_side=4)
    problem = _make_problem(grid_n=16)
    policy.reset(problem, np.random.default_rng(0))
    # Should infer grid_n = 16 from 256-cell working area
    cell = policy.choose_action([], frozenset(), np.random.default_rng(0))
    row = cell // 16
    col = cell % 16
    # First cell should be at (margin, margin) = (2, 2) for default margin
    assert (row, col) == (2, 2)


def test_grid_drilling_explicit_grid_n_overrides_inference() -> None:
    """The user can pass grid_n explicitly when the cell count is not a
    perfect square or when they want a different working-area shape."""
    policy = GridDrillingPolicy(n_per_side=4, grid_n=16)
    problem = _make_problem(grid_n=16)
    policy.reset(problem, np.random.default_rng(0))
    # First cell should be at (2, 2) given margin=2
    cell = policy.choose_action([], frozenset(), np.random.default_rng(0))
    assert (cell // 16, cell % 16) == (2, 2)


def test_grid_drilling_skips_pre_drilled_cells_via_grid_position_advancement() -> None:
    """If a cell at a planned grid position has already been drilled
    (e.g., by a pre-2010 set in B.2), the policy advances to the next
    planned position rather than returning the duplicate."""
    policy = GridDrillingPolicy(n_per_side=6)
    problem = _make_problem(grid_n=32)
    policy.reset(problem, np.random.default_rng(0))
    # Pre-fill the first planned grid cell as "already drilled"
    first_planned = policy._ordered_indices[0]
    drilled = frozenset({first_planned})
    cell = policy.choose_action([], drilled, np.random.default_rng(0))
    # Should advance to second planned position
    assert cell == policy._ordered_indices[1]


def test_grid_drilling_falls_back_to_random_after_grid_exhausted() -> None:
    """Past the 36-drill budget on the 32x32 grid, the policy falls
    back to random uniform pick from unvisited cells."""
    policy = GridDrillingPolicy(n_per_side=6)
    problem = _make_problem(grid_n=32)
    policy.reset(problem, np.random.default_rng(0))
    drilled: set[int] = set()
    # Drill the entire 6x6 grid
    for _ in range(36):
        cell = policy.choose_action([], frozenset(drilled), np.random.default_rng(0))
        drilled.add(cell)
    # 37th call should produce a fallback cell from the unvisited remainder
    cell = policy.choose_action([], frozenset(drilled), np.random.default_rng(0))
    assert cell not in drilled


def test_grid_drilling_rejects_non_square_when_grid_n_not_provided() -> None:
    """If problem.n_cells is not a perfect square and grid_n is unset,
    the policy raises rather than guess."""
    rng = np.random.default_rng(0)
    coords = np.zeros((30, 2))  # not a perfect square count
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords,
        prior_mean_field=np.zeros(30),
    )
    problem = CorrelatedDrillingProblem(
        hypothesis=h, x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=np.zeros(30),
    )
    policy = GridDrillingPolicy()
    with pytest.raises(ValueError, match="square grid"):
        policy.reset(problem, rng)


def test_grid_drilling_rejects_invalid_n_per_side() -> None:
    policy = GridDrillingPolicy(n_per_side=0)
    problem = _make_problem(grid_n=16)
    with pytest.raises(ValueError, match="n_per_side"):
        policy.reset(problem, np.random.default_rng(0))


def test_grid_drilling_rejects_grid_too_small_for_margin() -> None:
    """Margin > grid_n/2 leaves no valid drill positions; policy should
    reject this at reset time."""
    policy = GridDrillingPolicy(margin=20)  # bigger than grid_n=16/2
    problem = _make_problem(grid_n=16)
    with pytest.raises(ValueError, match="too small for margin"):
        policy.reset(problem, np.random.default_rng(0))
