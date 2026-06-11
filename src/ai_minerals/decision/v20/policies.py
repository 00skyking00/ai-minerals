"""Policies for bcgt-v2.0.

Extends `src/ai_minerals/decision/policies.py` (v1.0 random + greedy +
POMCP + EOI on Bernoulli prior) with:

- B.1: CorrelatedPriorPOMCPPolicy — POMCP over the particle-filter belief
  state; per-step replanning uses the current PF posterior mean as the
  rollout prior.
- B.2: same policy class, fed BCGS pre-2010 prior + post-2010 ground truth.
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

NOT IMPLEMENTED YET. Skeletons only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief_pf import ParticleFilter
from .hypotheses import HypothesisSet
from .pomdp import (
    CorrelatedDrillingProblem,
    MultiHypothesisDrillingProblem,
    SensorModel,
)

# pomdp_py POMCP defaults; we may tune these per experiment.
POMCP_N_SIMS_DEFAULT = 1000   # MCTS rollouts per planning step
POMCP_C_EXPLORATION = 1.41    # UCB constant; sqrt(2) is the textbook default
POMCP_MAX_DEPTH = 20          # truncates rollouts; should exceed drill budget


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
