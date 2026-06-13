"""SARSOP-backed multi-hypothesis policy for bcgt-v2.0 C.2.

Wraps `pomdp_py.utils.interfaces.solvers.sarsop` over a small discretized
multi-hypothesis POMDP. The discretization choices match the Mern 2024
recipe:

- state space: one discrete state per hypothesis (including the null);
  each hypothesis carries a single deterministic deposit-cell footprint
  for this small demo.
- action space: one drill action per grid cell.
- observation space: binary {0, 1} (Bernoulli sensor).

This is the demonstration-scale C.2 wiring. The grid is intentionally
small (5x5 = 25 cells) so SARSOP converges in seconds; the goal is to
show that the multi-hypothesis pre-computed alpha-vector policy works
and beats POMCP's online tree search on the same problem, which is the
argument the Mern paper makes (POMCP's value-function variance grows
with the hypothesis-set size).

See `research/pomdp_v20_implementation_plan.md` for the full design
rationale.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pomdp_py


DEFAULT_DISCOUNT = 0.95
DEFAULT_BERNOULLI_ALPHA = 0.05
DEFAULT_BERNOULLI_BETA = 0.10
DEFAULT_DRILL_COST = 1.0
DEFAULT_DISCOVERY_VALUE = 50.0


class HypothesisState(pomdp_py.State):
    """State = which hypothesis is the ground truth (including null)."""

    def __init__(self, idx: int, name: str):
        self.idx = int(idx)
        self.name = str(name)

    def __hash__(self) -> int:
        return hash(self.idx)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HypothesisState) and self.idx == other.idx

    def __repr__(self) -> str:
        return f"H_{self.name}"


class CellAction(pomdp_py.Action):
    """Action = drill the cell at this flat index."""

    def __init__(self, cell_idx: int):
        self.cell_idx = int(cell_idx)

    def __hash__(self) -> int:
        return hash(self.cell_idx)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CellAction) and self.cell_idx == other.cell_idx

    def __repr__(self) -> str:
        return f"Drill_{self.cell_idx}"


class BinaryObs(pomdp_py.Observation):
    """Observation = sensor reading, 0 or 1 (Bernoulli)."""

    def __init__(self, value: int):
        self.value = int(value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BinaryObs) and self.value == other.value

    def __repr__(self) -> str:
        return f"Obs_{self.value}"


class _StaticTransitionModel(pomdp_py.TransitionModel):
    """Drilling never changes the hypothesis."""

    def __init__(self, states: list[HypothesisState]):
        self._states = states

    def probability(self, next_state, state, action) -> float:
        return 1.0 if next_state == state else 0.0

    def sample(self, state, action):
        return state

    def get_all_states(self):
        return self._states


class _BernoulliObservationModel(pomdp_py.ObservationModel):
    """P(obs=1 | hypothesis h, drill cell c) = (1 - beta) if c is the
    deposit cell for h, else alpha.
    """

    def __init__(
        self,
        deposit_cell_by_hypothesis: dict[int, int | None],
        alpha: float,
        beta: float,
    ):
        self._deposit_cell = dict(deposit_cell_by_hypothesis)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._all_obs = [BinaryObs(0), BinaryObs(1)]

    def _p_one(self, hypothesis_idx: int, cell_idx: int) -> float:
        deposit_cell = self._deposit_cell.get(hypothesis_idx)
        if deposit_cell is not None and deposit_cell == cell_idx:
            return 1.0 - self._beta
        return self._alpha

    def probability(self, observation, next_state, action) -> float:
        p1 = self._p_one(next_state.idx, action.cell_idx)
        return p1 if observation.value == 1 else (1.0 - p1)

    def sample(self, next_state, action):
        p1 = self._p_one(next_state.idx, action.cell_idx)
        return BinaryObs(1) if np.random.random() < p1 else BinaryObs(0)

    def get_all_observations(self):
        return self._all_obs


class _DepositRewardModel(pomdp_py.RewardModel):
    """Reward = -drill_cost + discovery_value if drilling the deposit cell."""

    def __init__(
        self,
        deposit_cell_by_hypothesis: dict[int, int | None],
        drill_cost: float,
        discovery_value: float,
    ):
        self._deposit_cell = dict(deposit_cell_by_hypothesis)
        self._drill_cost = float(drill_cost)
        self._discovery_value = float(discovery_value)

    def _reward(self, state, action) -> float:
        deposit_cell = self._deposit_cell.get(state.idx)
        if deposit_cell is not None and deposit_cell == action.cell_idx:
            return -self._drill_cost + self._discovery_value
        return -self._drill_cost

    def sample(self, state, action, next_state):
        return self._reward(state, action)


class _EnumerablePolicyModel(pomdp_py.RolloutPolicy):
    """Action space = drill any cell. Used as a rollout policy fallback."""

    def __init__(self, actions: list[CellAction]):
        self._actions = actions

    def sample(self, state):
        return self._actions[np.random.randint(len(self._actions))]

    def rollout(self, state, history=None):
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return self._actions


@dataclass
class MultiHypothesisSmallGridPOMDP:
    """Wraps the discretized multi-hypothesis POMDP and the pomdp_py.Agent
    needed to feed SARSOP.

    deposit_cell_by_hypothesis maps hypothesis_idx -> deposit cell flat index,
    or None for the null hypothesis (no deposit anywhere). Hypothesis indices
    are 0..N-1; the order is preserved across `states`, `actions`, and the
    `initial_belief` dict.
    """
    n_cells: int
    hypothesis_names: list[str]
    deposit_cell_by_hypothesis: dict[int, int | None]
    initial_prior: np.ndarray
    alpha_fp: float = DEFAULT_BERNOULLI_ALPHA
    beta_fn: float = DEFAULT_BERNOULLI_BETA
    drill_cost: float = DEFAULT_DRILL_COST
    discovery_value: float = DEFAULT_DISCOVERY_VALUE

    states: list[HypothesisState] = field(init=False)
    actions: list[CellAction] = field(init=False)
    observations: list[BinaryObs] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.hypothesis_names) != len(self.initial_prior):
            raise ValueError(
                "hypothesis_names and initial_prior must have equal length"
            )
        prior_sum = float(self.initial_prior.sum())
        if abs(prior_sum - 1.0) > 1e-6:
            raise ValueError(f"initial_prior must sum to 1.0; got {prior_sum}")
        self.states = [
            HypothesisState(i, name)
            for i, name in enumerate(self.hypothesis_names)
        ]
        self.actions = [CellAction(c) for c in range(self.n_cells)]
        self.observations = [BinaryObs(0), BinaryObs(1)]

    def build_agent(self, belief: np.ndarray | None = None) -> pomdp_py.Agent:
        """Returns a pomdp_py.Agent ready to hand to sarsop()."""
        if belief is None:
            belief = self.initial_prior
        if belief.shape != self.initial_prior.shape:
            raise ValueError("belief shape must match initial_prior shape")
        belief_dict = {self.states[i]: float(belief[i]) for i in range(len(self.states))}
        init_belief = pomdp_py.Histogram(belief_dict)

        transition = _StaticTransitionModel(self.states)
        observation = _BernoulliObservationModel(
            self.deposit_cell_by_hypothesis,
            alpha=self.alpha_fp, beta=self.beta_fn,
        )
        reward = _DepositRewardModel(
            self.deposit_cell_by_hypothesis,
            drill_cost=self.drill_cost,
            discovery_value=self.discovery_value,
        )
        policy = _EnumerablePolicyModel(self.actions)

        agent = pomdp_py.Agent(
            init_belief,
            policy,
            transition,
            observation,
            reward,
        )
        return agent

    def update_belief(
        self,
        belief: np.ndarray,
        cell_idx: int,
        observation: int,
    ) -> np.ndarray:
        """Bayesian categorical update given one binary observation."""
        likelihoods = np.empty(len(self.states))
        for i in range(len(self.states)):
            p1 = (
                1.0 - self.beta_fn
                if self.deposit_cell_by_hypothesis.get(i) == cell_idx
                else self.alpha_fp
            )
            likelihoods[i] = p1 if observation == 1 else (1.0 - p1)
        unnorm = belief * likelihoods
        s = unnorm.sum()
        if s <= 0:
            return belief.copy()
        return unnorm / s


def solve_sarsop(
    pomdp: MultiHypothesisSmallGridPOMDP,
    pomdpsol_path: str | Path,
    discount: float = DEFAULT_DISCOUNT,
    timeout_sec: int = 30,
    memory_mb: int = 200,
    precision: float = 0.5,
    work_dir: str | Path | None = None,
) -> pomdp_py.AlphaVectorPolicy:
    """Build the agent, run SARSOP, return the AlphaVectorPolicy.

    The pomdpsol binary is built from AdaCompNUS/sarsop via
    scripts/build_pomdpsol.sh. We run it from a temp working directory so
    the .pomdp/.policy files don't litter the repo root.
    """
    from pomdp_py.utils.interfaces.solvers import sarsop

    pomdpsol_path = str(Path(pomdpsol_path).resolve())
    if not Path(pomdpsol_path).exists():
        raise FileNotFoundError(
            f"pomdpsol binary not found at {pomdpsol_path}; "
            f"run scripts/build_pomdpsol.sh first"
        )

    agent = pomdp.build_agent()

    if work_dir is None:
        work_dir = Path(".") / ".sarsop_tmp"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cwd = Path.cwd()
    try:
        import os
        os.chdir(work_dir)
        policy = sarsop(
            agent,
            pomdpsol_path=pomdpsol_path,
            discount_factor=discount,
            timeout=timeout_sec,
            memory=memory_mb,
            precision=precision,
            pomdp_name="bcgt_v20_c2",
            remove_generated_files=True,
        )
    finally:
        import os
        os.chdir(cwd)

    return policy


@dataclass
class MultiHypothesisSARSOPPolicy:
    """C.2 SARSOP-backed policy. Solve once on a small grid; query the
    pre-computed alpha-vector policy at each step against the current
    categorical belief.

    Compatible with the SyntheticMonteCarloSimulator policy interface
    when used on a small-grid problem: `reset()` and `choose_action()`.
    """
    pomdp: MultiHypothesisSmallGridPOMDP
    alpha_policy: pomdp_py.AlphaVectorPolicy
    _belief: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._belief = self.pomdp.initial_prior.copy()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._belief = self.pomdp.initial_prior.copy()

    @property
    def belief(self) -> np.ndarray:
        return self._belief.copy()

    def choose_action(
        self,
        history: list[tuple[int, float]] | None = None,
        drilled: frozenset[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Pick the alpha-vector-policy's action under the current belief.

        history and drilled are accepted to match the simulator interface
        but not used directly: the belief is maintained inside this policy.
        """
        belief_dict = {
            self.pomdp.states[i]: float(self._belief[i])
            for i in range(len(self.pomdp.states))
        }

        class _StubAgent:
            def __init__(self, belief_map):
                self.belief = belief_map

        action = self.alpha_policy.plan(_StubAgent(belief_dict))
        return int(action.cell_idx)

    def observe(self, cell_idx: int, observation: int) -> None:
        """Update belief given a new (action, observation) pair."""
        self._belief = self.pomdp.update_belief(
            self._belief, cell_idx, observation,
        )
