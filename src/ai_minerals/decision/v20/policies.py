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
    over tied indices). Compare to BayesianGreedyPolicy which uses the
    particle-filter posterior, and CorrelatedPriorPOMCPPolicy which
    plans multiple steps ahead.
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
        scores = np.where(mask, self._prior_mean, -np.inf)
        top_value = scores.max()
        candidates = np.flatnonzero(scores >= top_value - 1e-12)
        return int(rng.choice(candidates))


@dataclass
class BayesianGreedyPolicy:
    """Pick the highest PF posterior-mean cell among unvisited.

    Updates the particle filter after each observation, then picks the
    cell with the highest current posterior mean. The interesting
    middle baseline between GreedyMeanPolicy (uses only the static
    prior) and CorrelatedPriorPOMCPPolicy (uses the PF and plans
    multiple steps ahead).

    The 'Bayesian' is because the cell selection is now conditioned on
    the entire observation history through the PF's importance-weighted
    posterior; the 'Greedy' is because we still pick one step at a time
    without multi-step lookahead.
    """

    n_particles: int = 500
    sensor_noise_sigma: float = 0.001

    _pf: ParticleFilter | None = None
    _last_history_len: int = 0

    def reset(self, problem: CorrelatedDrillingProblem,
              rng: np.random.Generator) -> None:
        self._pf = ParticleFilter(
            hypothesis=problem.hypothesis,
            n_particles=self.n_particles,
            rng=rng,
        )
        self._pf.initialize()
        self._last_history_len = 0
        self.sensor_noise_sigma = problem.sensor_noise_sigma

    def choose_action(
        self,
        history: list[tuple[int, float]],
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> int:
        if self._pf is None:
            raise RuntimeError("BayesianGreedyPolicy not reset; call reset() first")
        # Catch up the PF with any new observations since the last call.
        while self._last_history_len < len(history):
            cell, obs = history[self._last_history_len]
            self._pf.update(
                cell_idx=cell, observation=obs,
                sensor_noise_sigma=self.sensor_noise_sigma,
            )
            self._last_history_len += 1

        post_mean = self._pf.posterior_mean()
        mask = np.ones(post_mean.size, dtype=bool)
        for idx in drilled:
            mask[idx] = False
        if not mask.any():
            raise RuntimeError("All cells drilled; no action available")
        scores = np.where(mask, post_mean, -np.inf)
        top_value = scores.max()
        candidates = np.flatnonzero(scores >= top_value - 1e-12)
        return int(rng.choice(candidates))


@dataclass
class CorrelatedPriorPOMCPPolicy:
    """Particle-rollout Monte Carlo planning (POMCP-style) over the PF belief.

    For each candidate unvisited cell, run `n_rollouts` simulated episodes
    where a particle (sampled from the current PF belief) plays the role of
    hidden ground truth. The rollout drills the candidate cell first, then
    follows a greedy-on-particle rollout policy for the remaining horizon,
    summing discounted rewards. The candidate with the best mean rollout
    return is the chosen action.

    This is not the full UCB-based MCTS tree search of textbook POMCP; it's
    the "particle filter + Monte Carlo rollouts" core of POMCP without the
    progressive widening / UCT tree, which captures the multi-step planning
    benefit at a fraction of the implementation cost. The pomdp_py
    library-wrapped version (`pomdp_py.algorithms.pomcp`) is the production
    target if this baseline is not enough; for B.1 the simple version is
    enough to show whether multi-step planning beats Bayesian greedy on
    correlated terrain.
    """

    n_particles: int = 500
    n_rollouts: int = 60
    planning_horizon: int = 9       # match drill_budget by default
    discount: float = 0.95
    sensor_noise_sigma: float = 0.001

    _pf: ParticleFilter | None = None
    _last_history_len: int = 0
    _problem: CorrelatedDrillingProblem | None = None

    def reset(self, problem: CorrelatedDrillingProblem,
              rng: np.random.Generator) -> None:
        self._pf = ParticleFilter(
            hypothesis=problem.hypothesis,
            n_particles=self.n_particles,
            rng=rng,
        )
        self._pf.initialize()
        self._last_history_len = 0
        self._problem = problem
        self.sensor_noise_sigma = problem.sensor_noise_sigma

    def _greedy_rollout_from_particle(
        self,
        particle: np.ndarray,
        first_action: int,
        already_drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> float:
        """One full-horizon rollout. The rollout policy after the first
        action is greedy-on-particle: pick the highest unvisited cell of
        the particle. Returns the discounted cumulative reward."""
        problem = self._problem
        cutoff = problem.cutoff_grade
        discovery = problem.discovery_value
        drill_cost = problem.drill_cost

        drilled = set(already_drilled)
        cumulative = 0.0
        discount = 1.0

        # Step 1: the candidate first_action.
        if first_action in drilled:
            cumulative += -drill_cost   # repeat drill: cost only
        else:
            r = -drill_cost + (
                discovery if particle[first_action] > cutoff else 0.0
            )
            cumulative += discount * r
            drilled.add(first_action)
        discount *= self.discount

        # Remaining steps: greedy on the (known to the rollout) particle.
        mask = np.ones(particle.size, dtype=bool)
        for idx in drilled:
            mask[idx] = False
        scores = np.where(mask, particle, -np.inf)
        # Pre-sort once; pop from the top.
        order = np.argsort(-scores)

        steps_left = self.planning_horizon - 1
        for cell in order[:steps_left]:
            if cell in drilled:
                continue
            r = -drill_cost + (
                discovery if particle[cell] > cutoff else 0.0
            )
            cumulative += discount * r
            drilled.add(int(cell))
            discount *= self.discount
        return cumulative

    def choose_action(
        self,
        history: list[tuple[int, float]],
        drilled: frozenset[int],
        rng: np.random.Generator,
    ) -> int:
        if self._pf is None or self._problem is None:
            raise RuntimeError(
                "CorrelatedPriorPOMCPPolicy not reset; call reset() first"
            )
        # Catch up the PF with new observations since last call.
        while self._last_history_len < len(history):
            cell, obs = history[self._last_history_len]
            self._pf.update(
                cell_idx=cell, observation=obs,
                sensor_noise_sigma=self.sensor_noise_sigma,
            )
            self._last_history_len += 1

        # Candidate actions: unvisited cells.
        n_cells = self._problem.n_cells
        candidates = np.array(
            [i for i in range(n_cells) if i not in drilled], dtype=np.int64,
        )
        if candidates.size == 0:
            raise RuntimeError("All cells drilled; no action available")

        # Restrict the candidate search to the top-K cells by PF posterior
        # mean so we don't waste rollouts on obviously-bad actions. This is
        # the same shortcut full POMCP gets from UCT.
        post_mean = self._pf.posterior_mean()
        cand_scores = post_mean[candidates]
        top_k = min(15, candidates.size)
        top_idxs = np.argpartition(-cand_scores, top_k - 1)[:top_k]
        candidates = candidates[top_idxs]

        # Sample particle indices weighted by PF weights once per rollout.
        weights = self._pf._normalized_weights()
        particle_indices = rng.choice(
            self._pf.n_particles,
            size=self.n_rollouts,
            replace=True,
            p=weights,
        )

        # Score each candidate via mean rollout return.
        Q = np.full(candidates.size, -np.inf)
        for i, cand in enumerate(candidates):
            returns = np.empty(self.n_rollouts)
            for j, p_idx in enumerate(particle_indices):
                returns[j] = self._greedy_rollout_from_particle(
                    self._pf.particles[p_idx],
                    int(cand),
                    drilled,
                    rng,
                )
            Q[i] = returns.mean()

        best_idx = int(np.argmax(Q))
        return int(candidates[best_idx])


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
