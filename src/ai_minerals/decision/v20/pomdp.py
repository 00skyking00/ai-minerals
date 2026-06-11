"""POMDP problem extension for bcgt-v2.0.

Extends `src/ai_minerals/decision/pomdp.py`'s `DrillingProblem` (v1.0,
iid Bernoulli + noiseless sensor) with:

- correlated draws via Hypothesis.sample_realization (B.1)
- noisy Bernoulli sensor (B.2 + C.1) or Gaussian sensor (B.1)
- multi-hypothesis state with categorical hypothesis index (C.2)

Locked parameters per the spec:
    Discount factor gamma:    0.99 (default; not stated in paper)
    Sensor noise (B.1):       Gaussian sigma=0.001 (matches paper)
    Sensor noise (B.2 + C.1): Bernoulli alpha=0.05, beta=0.10
    Sensitivity sweep (C.1):  3x3 over (alpha, beta)

B.1 IMPLEMENTATION STATUS (2026-06-11):
    CorrelatedDrillingProblem.step (Gaussian branch): DONE - GH issue #4
    Bernoulli branch:                                  NOT YET - C.1 milestone
    MultiHypothesisDrillingProblem.step:               NOT YET - C.2 milestone
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .hypotheses import Hypothesis, HypothesisSet


DISCOUNT_FACTOR = 0.99  # gamma; not specified by paper; chosen for 17-step horizon
DEFAULT_BERNOULLI_ALPHA = 0.05  # false positive rate (per spec lock 2026-06-10)
DEFAULT_BERNOULLI_BETA = 0.10   # false negative rate


class SensorModel(Enum):
    """Three sensor models for B.1 / B.2 / C.1 experiments."""
    NOISELESS = "noiseless"               # v1.0 behavior; for sanity tests
    GAUSSIAN_CONTINUOUS = "gaussian"      # B.1 synthetic, matches Mern 2024
    BERNOULLI_BINARY = "bernoulli"        # B.2 + C.1 BCGS binary indicator


@dataclass
class CorrelatedDrillingProblem:
    """v2.0 single-hypothesis POMDP with GP-correlated state + noisy sensor.

    Differs from v1.0 `DrillingProblem`:
    - `p_prior` replaced by `Hypothesis` (GP prior over continuous grade)
    - `true_label` sampled from Hypothesis.sample_realization rather than
      independent Bernoulli draws
    - `step()` returns a noisy observation per the SensorModel

    For B.2 retrospective BCGS validation, true_label is the actual
    BCGS post-2010 drill outcome (assay-Cu >= 0.2% Cox-Singer cutoff =
    1, else 0), not a synthetic draw.
    """
    hypothesis: Hypothesis
    x_m: np.ndarray  # per-cell x coordinate in working CRS, length N
    y_m: np.ndarray  # per-cell y coordinate in working CRS, length N
    true_grade: np.ndarray  # per-cell true ore grade, length N (B.1)
                            # OR true binary deposit indicator (B.2 + C.1)
    sensor_model: SensorModel = SensorModel.GAUSSIAN_CONTINUOUS
    sensor_noise_sigma: float = 0.001  # B.1 Gaussian
    sensor_alpha: float = DEFAULT_BERNOULLI_ALPHA  # B.2/C.1 Bernoulli FP rate
    sensor_beta: float = DEFAULT_BERNOULLI_BETA    # B.2/C.1 Bernoulli FN rate
    drill_cost: float = 1.0
    discovery_value: float = 50.0  # for binary; for continuous see step()
    cutoff_grade: float = 0.2     # B.1 Cox-Singer porphyry-Cu cutoff

    @property
    def n_cells(self) -> int:
        return int(len(self.x_m))

    def __post_init__(self) -> None:
        n = self.n_cells
        if self.true_grade.shape != (n,):
            raise ValueError(
                f"true_grade must be shape ({n},); got {self.true_grade.shape}"
            )
        if self.y_m.shape != (n,):
            raise ValueError(
                f"y_m must be shape ({n},); got {self.y_m.shape}"
            )
        if self.sensor_noise_sigma <= 0 and self.sensor_model is SensorModel.GAUSSIAN_CONTINUOUS:
            raise ValueError(
                f"sensor_noise_sigma must be > 0 for Gaussian sensor; "
                f"got {self.sensor_noise_sigma}"
            )

    def step(
        self,
        cell_idx: int,
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[float | int, float, frozenset[int]]:
        """Apply a drill action; return (observation, reward, next_drilled).

        For SensorModel.GAUSSIAN_CONTINUOUS:
            obs    = true_grade[cell_idx] + N(0, sensor_noise_sigma^2)
            reward = -drill_cost + discovery_value * indicator(
                                                  true_grade > cutoff_grade)
        For SensorModel.BERNOULLI_BINARY (C.1 milestone):
            true   = int(true_grade[cell_idx] >= cutoff_grade)
            obs    = noisy_bernoulli(true, alpha, beta, rng)
            reward = -drill_cost + discovery_value * true
        For SensorModel.NOISELESS:
            obs    = true_grade[cell_idx]
            reward = same as Gaussian.

        Drilling an already-drilled cell wastes a turn (returns reward
        = -drill_cost and the same observation; matches v1.0 semantics).
        """
        if not (0 <= cell_idx < self.n_cells):
            raise IndexError(
                f"cell_idx {cell_idx} out of range for {self.n_cells} cells"
            )

        already_drilled = cell_idx in drilled
        true_value = float(self.true_grade[cell_idx])
        is_discovery = true_value > self.cutoff_grade

        if self.sensor_model is SensorModel.GAUSSIAN_CONTINUOUS:
            noise = float(rng.normal(0.0, self.sensor_noise_sigma))
            obs: float | int = true_value + noise
        elif self.sensor_model is SensorModel.NOISELESS:
            obs = true_value
        elif self.sensor_model is SensorModel.BERNOULLI_BINARY:
            # C.1 milestone; gated to keep B.1 surface narrow.
            raise NotImplementedError(
                "Bernoulli sensor model lands in C.1 (issue #8)"
            )
        else:  # pragma: no cover  - enum exhaustiveness
            raise ValueError(f"Unknown SensorModel: {self.sensor_model!r}")

        if already_drilled:
            reward = -self.drill_cost
        else:
            reward = -self.drill_cost + (
                self.discovery_value if is_discovery else 0.0
            )

        next_drilled = drilled | {cell_idx}
        return obs, reward, next_drilled


@dataclass
class MultiHypothesisDrillingProblem:
    """C.2 multi-hypothesis POMDP with N paper hypotheses + null.

    State = (true_grade, hypothesis_index). Belief = (particle filter
    per hypothesis, Dirichlet posterior over hypothesis indices).

    The categorical posterior over hypothesis indices is the C.2
    contribution: drilling reduces both per-hypothesis grade
    uncertainty AND the uncertainty about which hypothesis is correct.

    NOT IMPLEMENTED YET.
    """
    hypotheses: HypothesisSet
    true_hypothesis_idx: int  # ground truth h*; -1 means h_0 (null)
    x_m: np.ndarray
    y_m: np.ndarray
    true_grade: np.ndarray
    sensor_model: SensorModel = SensorModel.GAUSSIAN_CONTINUOUS
    sensor_noise_sigma: float = 0.001
    sensor_alpha: float = DEFAULT_BERNOULLI_ALPHA
    sensor_beta: float = DEFAULT_BERNOULLI_BETA
    drill_cost: float = 1.0
    discovery_value: float = 50.0

    def step(
        self,
        cell_idx: int,
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[float | int, float, frozenset[int]]:
        """Same step semantics as CorrelatedDrillingProblem; ground truth
        comes from true_grade (which was drawn from hypotheses[true_hypothesis_idx]
        if true_hypothesis_idx >= 0, or from h_0 if -1).

        TODO C.2: implement.
        """
        raise NotImplementedError("C.2 milestone")
