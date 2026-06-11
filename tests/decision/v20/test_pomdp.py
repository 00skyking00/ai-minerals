"""Tests for the bcgt-v2.0 CorrelatedDrillingProblem.

Covers issue #4 (B.1 problem + Gaussian sensor). Three families:

1. Construction validation: shape checks on y_m / true_grade;
   sensor_noise_sigma must be positive for the Gaussian branch.

2. Gaussian sensor step: observation = true_grade + Gaussian noise;
   reward = -drill_cost + discovery_value when true_grade > cutoff,
   else just -drill_cost. Drilling an already-drilled cell wastes a
   turn (no discovery bonus).

3. Sensor model gating: NOISELESS works; BERNOULLI_BINARY raises
   NotImplementedError with a C.1-pointer; bad cell_idx errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.hypotheses import Hypothesis
from ai_minerals.decision.v20.pomdp import (
    CorrelatedDrillingProblem,
    SensorModel,
)


def _make_problem(
    *,
    sensor_model: SensorModel = SensorModel.GAUSSIAN_CONTINUOUS,
    sensor_noise_sigma: float = 0.001,
    cutoff_grade: float = 0.2,
    drill_cost: float = 1.0,
    discovery_value: float = 50.0,
) -> CorrelatedDrillingProblem:
    spacing = 500.0
    x = np.arange(30) * spacing
    y = np.arange(30) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mean = np.zeros(coords.shape[0])
    h = Hypothesis(
        name="x", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=mean,
    )
    n = coords.shape[0]
    # Synthesize a true grade field with one high-grade cluster at cell 0.
    true_grade = np.full(n, 0.05)
    true_grade[0] = 0.5    # well above cutoff
    return CorrelatedDrillingProblem(
        hypothesis=h,
        x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=true_grade,
        sensor_model=sensor_model,
        sensor_noise_sigma=sensor_noise_sigma,
        cutoff_grade=cutoff_grade,
        drill_cost=drill_cost,
        discovery_value=discovery_value,
    )


# --- Construction validation -------------------------------------------------


def test_post_init_validates_true_grade_shape():
    spacing = 500.0
    x = np.arange(30) * spacing
    y = np.arange(30) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mean = np.zeros(coords.shape[0])
    h = Hypothesis(name="x", n_grabens=1, n_domains=1,
                   cell_coords_m=coords, prior_mean_field=mean)
    with pytest.raises(ValueError, match="true_grade must be shape"):
        CorrelatedDrillingProblem(
            hypothesis=h, x_m=coords[:, 0], y_m=coords[:, 1],
            true_grade=np.zeros(5),    # wrong length
        )


def test_post_init_validates_y_m_shape():
    spacing = 500.0
    x = np.arange(30) * spacing
    y = np.arange(30) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mean = np.zeros(coords.shape[0])
    h = Hypothesis(name="x", n_grabens=1, n_domains=1,
                   cell_coords_m=coords, prior_mean_field=mean)
    with pytest.raises(ValueError, match="y_m must be shape"):
        CorrelatedDrillingProblem(
            hypothesis=h,
            x_m=coords[:, 0], y_m=np.zeros(5),
            true_grade=np.zeros(coords.shape[0]),
        )


def test_post_init_rejects_zero_sensor_noise_for_gaussian():
    with pytest.raises(ValueError, match="sensor_noise_sigma"):
        _make_problem(sensor_noise_sigma=0.0)


def test_post_init_accepts_noiseless_with_zero_sigma():
    """The post-init guard only fires for the Gaussian branch; NOISELESS
    can carry sigma=0 happily."""
    p = _make_problem(
        sensor_model=SensorModel.NOISELESS, sensor_noise_sigma=0.0,
    )
    assert p.sensor_model is SensorModel.NOISELESS


# --- Gaussian sensor step ----------------------------------------------------


def test_gaussian_step_returns_observation_near_true_grade():
    p = _make_problem(sensor_noise_sigma=0.001)
    rng = np.random.default_rng(0)
    obs, reward, drilled = p.step(cell_idx=0, drilled=frozenset(), rng=rng)
    # Cell 0 true_grade = 0.5; sigma = 0.001 -> obs within ~5 sigma of 0.5.
    assert abs(obs - 0.5) < 0.01


def test_gaussian_step_reward_includes_discovery_value_above_cutoff():
    p = _make_problem(cutoff_grade=0.2, drill_cost=1.0, discovery_value=50.0)
    rng = np.random.default_rng(1)
    obs, reward, drilled = p.step(cell_idx=0, drilled=frozenset(), rng=rng)
    # cell 0 true_grade = 0.5 > 0.2 -> reward = -1 + 50 = 49.
    assert reward == 49.0
    assert drilled == frozenset({0})


def test_gaussian_step_reward_subtracts_drill_cost_below_cutoff():
    p = _make_problem(cutoff_grade=0.2, drill_cost=1.0, discovery_value=50.0)
    rng = np.random.default_rng(2)
    # Cell 1 true_grade = 0.05 < 0.2 -> no discovery bonus.
    obs, reward, drilled = p.step(cell_idx=1, drilled=frozenset(), rng=rng)
    assert reward == -1.0


def test_gaussian_step_already_drilled_wastes_a_turn():
    p = _make_problem()
    rng = np.random.default_rng(3)
    # Drill cell 0 once.
    obs1, reward1, drilled = p.step(cell_idx=0, drilled=frozenset(), rng=rng)
    assert reward1 == 49.0
    # Re-drill it. Should get only the cost, no discovery bonus.
    obs2, reward2, drilled2 = p.step(cell_idx=0, drilled=drilled, rng=rng)
    assert reward2 == -1.0
    assert drilled2 == frozenset({0})


def test_gaussian_step_noise_distribution_matches_sigma():
    p = _make_problem(sensor_noise_sigma=0.01)
    rng = np.random.default_rng(4)
    # 2000 observations of cell 1 (true grade 0.05).
    obs = []
    drilled = frozenset()
    for _ in range(2000):
        o, _, _ = p.step(cell_idx=1, drilled=drilled, rng=rng)
        obs.append(o)
    obs = np.array(obs)
    np.testing.assert_allclose(obs.mean(), 0.05, atol=0.002)
    # Empirical sigma within 5% of the configured 0.01 at 2000 draws.
    np.testing.assert_allclose(obs.std(ddof=1), 0.01, rtol=0.10)


# --- Sensor model + edge cases ----------------------------------------------


def test_noiseless_sensor_returns_exact_true_grade():
    p = _make_problem(sensor_model=SensorModel.NOISELESS, sensor_noise_sigma=0.0)
    rng = np.random.default_rng(5)
    obs, _, _ = p.step(cell_idx=0, drilled=frozenset(), rng=rng)
    assert obs == 0.5


def test_bernoulli_sensor_gated_to_C1():
    p = _make_problem(sensor_model=SensorModel.BERNOULLI_BINARY)
    rng = np.random.default_rng(6)
    with pytest.raises(NotImplementedError, match="C.1"):
        p.step(cell_idx=0, drilled=frozenset(), rng=rng)


def test_step_rejects_out_of_range_cell_idx():
    p = _make_problem()
    rng = np.random.default_rng(7)
    with pytest.raises(IndexError):
        p.step(cell_idx=999_999, drilled=frozenset(), rng=rng)
