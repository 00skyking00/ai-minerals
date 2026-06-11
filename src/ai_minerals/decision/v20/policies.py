"""Policies for bcgt-v2.0.

Extends `src/ai_minerals/decision/policies.py` (v1.0 random + greedy +
POMCP + EOI on Bernoulli prior) with:

- B.1 (this file): RandomPolicy + GreedyMeanPolicy + (skeleton)
  CorrelatedPriorPOMCPPolicy. Random + greedy are concrete baselines
  the SyntheticMonteCarloSimulator can drive end-to-end without the
  pomdp_py wrapper. POMCP integration is the chapter-update milestone.
- B.2: same Random / Greedy / POMCP triplet, fed BCGS pre-2010 prior +
  post-2010 ground truth.
- C.1: NoisyObservationPOMCPPolicy — adds the Bernoulli sensor model to
  the belief update; otherwise identical to B.1.
- C.2: MultiHypothesisFalsificationPolicy — maintains posterior over
  hypothesis set, fires falsification flag when h_0 likelihood dominates
  per the Mern 2024 protocol.

Locked decision (2026-06-10): use POMCP for B.1 through C.1 (matches our
existing Chapter 7 v1.0 infrastructure). Gate-check at C.1 exit: if
POMCP re-planning takes more than 5 sec/step, switch to PyJulia bridge
calling Mern's SARSOP. See research/pomdp_v20_implementation_plan.md
section Q5.

B.1 IMPLEMENTATION STATUS (2026-06-11):
    RandomPolicy, GreedyMeanPolicy:      DONE - GH issue #5 prelude
    CorrelatedPriorPOMCPPolicy:           NOT YET - GH issue #6 chapter
    MultiHypothesisFalsificationPolicy:   NOT YET - C.2 milestone
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .belief_pf import ParticleFilter
from .hypotheses import Hypothesis, HypothesisSet
from .pomdp import (
    CorrelatedDrillingProblem,
    MultiHypothesisDrillingProblem,
    SensorModel,
)

# pomdp_py POMCP defaults; we may tune these per experiment.
POMCP_N_SIMS_DEFAULT = 1000   # MCTS rollouts per planning step
POMCP_C_EXPLORATION = 1.41    # UCB constant; sqrt(2) is the textbook default
POMCP_MAX_DEPTH = 20          # truncates rollouts; should exceed drill budget


# --- Policy interface --------------------------------------------------------


class Policy(Protocol):
    """The policy contract the B.1 simulator drives.

    A policy is reset at the start of every episode (so any internal
    state doesn't leak across realizations) and queried once per drill
    step. The simulator passes the observed history so far; the policy
    returns the next cell to drill. Policies should NOT inspect the
    problem's true_grade field; the simulator type-checks neither, so
    discipline is on the implementation.
    """

    def reset(self, problem: CorrelatedDrillingProblem,
              rng: np.random.Generator) -> None: ...

    def choose_action(
        self,
        history: list[tuple[int, float]],
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> int: ...


# --- Concrete B.1 baselines --------------------------------------------------


@dataclass
class RandomPolicy:
    """Uniform random pick from unvisited cells. Baseline for any
    discovery-rate comparison."""

    _n_cells: int = 0

    def reset(self, problem: CorrelatedDrillingProblem,
              rng: np.random.Generator) -> None:
        self._n_cells = problem.n_cells

    def choose_action(
        self,
        history: list[tuple[int, float]],
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> int:
        if self._n_cells == 0:
            raise RuntimeError("RandomPolicy not reset; call reset() first")
        unvisited = np.array(
            [i for i in range(self._n_cells) if i not in drilled],
            dtype=np.int64,
        )
        if unvisited.size == 0:
            raise RuntimeError("All cells drilled; no action available")
        return int(rng.choice(unvisited))


@dataclass
class GreedyMeanPolicy:
    """Pick the highest prior-mean cell among unvisited.

    Doesn't incorporate observations into a belief update; just uses the
    static GP prior mean as its ranking. Ties are broken by RNG (uniform
    over tied indices). The simulator's POMCP baseline (issue #6) will
    replace this with a particle-filter-driven greedy step.
    """

    _prior_mean: np.ndarray | None = None

    def reset(self, problem: CorrelatedDrillingProblem,
              rng: np.random.Generator) -> None:
        self._prior_mean = problem.hypothesis.prior_mean_field.copy()

    def choose_action(
        self,
        history: list[tuple[int, float]],
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> int:
        if self._prior_mean is None:
            raise RuntimeError("GreedyMeanPolicy not reset; call reset() first")
        mask = np.ones(self._prior_mean.size, dtype=bool)
        for idx in drilled:
            mask[idx] = False
        if not mask.any():
            raise RuntimeError("All cells drilled; no action available")
        # Mask out drilled cells by setting their value to -inf, then argmax
        # with random tie-break.
        scores = np.where(mask, self._prior_mean, -np.inf)
        top_value = scores.max()
        candidates = np.flatnonzero(scores >= top_value - 1e-12)
        return int(rng.choice(candidates))


@dataclass
class CorrelatedPriorPOMCPPolicy:
    """B.1 policy: POMCP over correlated-prior particle filter belief.

    Each step:
    1. Use particle filter posterior mean as the current per-cell belief.
    2. Run POMCP rollouts from the current belief; each rollout simulates
       drill -> noisy obs -> particle update -> next-step decision.
    3. Pick the action with the highest UCB-tuned Q-value.
    4. Apply action to the environment; update particle filter with the
       observed (cell_idx, observation) pair.

    NOT IMPLEMENTED YET. Will wrap pomdp_py.algorithms.pomcp.POMCP.
    """
    problem: CorrelatedDrillingProblem
    particle_filter: ParticleFilter
    n_sims: int = POMCP_N_SIMS_DEFAULT
    c_exploration: float = POMCP_C_EXPLORATION
    max_depth: int = POMCP_MAX_DEPTH

    def plan(self, rng: np.random.Generator) -> int:
        """Run one POMCP planning step; return the chosen cell_idx.

        TODO B.1: implement.
        """
        raise NotImplementedError("B.1 milestone")

    def step_and_update(
        self,
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[int, float, frozenset[int]]:
        """Plan -> act -> update particle filter; return (cell, reward, drilled).

        TODO B.1: implement.
        """
        raise NotImplementedError("B.1 milestone")


@dataclass
class MultiHypothesisFalsificationPolicy:
    """C.2 policy: POMCP + multi-hypothesis posterior + falsification check.

    Each step:
    1. Get current per-hypothesis particle filter posteriors + current
       categorical posterior over hypothesis indices.
    2. Run POMCP rollouts where the rollout's hypothesis is sampled from
       the categorical posterior at the start of each rollout. This is
       the standard "hierarchical belief" rollout.
    3. Pick action; apply; update both the per-hypothesis particle filter
       and the categorical posterior.
    4. Falsification check: if h_0 (null) categorical posterior is the
       maximum after K observations (paper uses K=7-10 boreholes), flag
       "human priors falsified" and continue drilling per a fallback
       policy (or stop early per the experimenter's choice).

    NOT IMPLEMENTED YET.
    """
    problem: MultiHypothesisDrillingProblem
    hypothesis_set: HypothesisSet
    particle_filters: dict[int, ParticleFilter]  # h_idx -> per-h PF
    hypothesis_posterior: np.ndarray  # current categorical posterior
    n_sims: int = POMCP_N_SIMS_DEFAULT
    c_exploration: float = POMCP_C_EXPLORATION
    max_depth: int = POMCP_MAX_DEPTH
    falsification_threshold_argmax: bool = True  # h_0 = argmax fires it
    falsification_threshold_likelihood: float = 0.5  # alternative: h_0 > 0.5

    def plan(self, rng: np.random.Generator) -> int:
        raise NotImplementedError("C.2 milestone")

    def step_and_update(
        self,
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> tuple[int, float, frozenset[int], bool]:
        """Returns (cell, reward, drilled, falsification_fired)."""
        raise NotImplementedError("C.2 milestone")

    def check_falsification(self) -> bool:
        """True if the null hypothesis dominates the categorical posterior."""
        raise NotImplementedError("C.2 milestone")
