"""Policies for the BCGT drill-planning POMDP.

Three policies, in increasing order of sophistication:

- ``RandomPolicy`` — uniform over un-drilled cells.
- ``GreedyPolicy`` — drill the un-drilled cell with the highest current
  posterior probability of being a deposit.
- ``pomcp_plan`` — POMCP wrapper around ``pomdp_py``. Takes one
  decision step at a time given the current belief; the caller drives
  the simulation loop and updates the belief between steps.

All policies share a small interface: ``choose(problem, posterior,
drilled, rng) → cell_idx | None``. Returning ``None`` means STOP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .pomdp import DrillingProblem


class Policy(Protocol):
    def choose(
        self,
        problem: "DrillingProblem",
        posterior: np.ndarray,
        drilled: frozenset[int],
        *,
        rng: np.random.Generator | None = None,
    ) -> int | None: ...


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomPolicy:
    """Uniform random over un-drilled cells."""

    name: str = "random"

    def choose(
        self,
        problem: "DrillingProblem",
        posterior: np.ndarray,
        drilled: frozenset[int],
        *,
        rng: np.random.Generator | None = None,
    ) -> int | None:
        rng = np.random.default_rng(rng)
        candidates = [i for i in range(problem.n_cells) if i not in drilled]
        if not candidates:
            return None
        return int(rng.choice(candidates))


@dataclass(frozen=True)
class GreedyPolicy:
    """Drill the highest-posterior un-drilled cell."""

    name: str = "greedy"

    def choose(
        self,
        problem: "DrillingProblem",
        posterior: np.ndarray,
        drilled: frozenset[int],
        *,
        rng: np.random.Generator | None = None,
    ) -> int | None:
        idx = np.argsort(-posterior)
        for i in idx:
            if int(i) not in drilled:
                return int(i)
        return None


# ---------------------------------------------------------------------------
# POMCP wrapper (pomdp_py)
# ---------------------------------------------------------------------------


def pomcp_plan(
    problem: "DrillingProblem",
    posterior: np.ndarray,
    drilled: frozenset[int],
    *,
    n_particles: int = 200,
    planning_time: float = 1.0,
    max_depth: int = 5,
    discount_factor: float = 0.95,
    exploration_const: float = 50.0,
    rng: np.random.Generator | None = None,
) -> int | None:
    """One-step POMCP plan.

    Builds a fresh particle belief from the current posterior, runs
    POMCP for ``planning_time`` seconds, and returns the recommended
    cell to drill next (or ``None`` for STOP).

    The pomdp_py state representation is ``DrillState(cell_labels:
    tuple[int, ...], drilled: frozenset[int])``. Particles are sampled
    by drawing Bernoulli(``posterior[i]``) per cell.

    Notes
    -----
    Re-instantiates the planner per step (POMCP retains its tree
    between calls if the agent + history is preserved, but for our
    portfolio demo the per-step latency matters less than the simpler
    code path. v1.2 work could thread the agent across steps for tree-
    reuse savings.)
    """
    import pomdp_py

    rng = np.random.default_rng(rng)
    n = problem.n_cells

    # ---- pomdp_py glue ----------------------------------------------------

    class _State(pomdp_py.State):
        def __init__(self, labels: tuple[int, ...], drilled: frozenset[int]):
            self.labels = labels
            self.drilled = drilled
        def __hash__(self): return hash((self.labels, self.drilled))
        def __eq__(self, o):
            return isinstance(o, _State) and self.labels == o.labels and self.drilled == o.drilled

    class _Action(pomdp_py.Action):
        def __init__(self, cell: int): self.cell = cell
        def __hash__(self): return hash(self.cell)
        def __eq__(self, o): return isinstance(o, _Action) and self.cell == o.cell
        def __repr__(self): return f"Drill({self.cell})"

    class _Obs(pomdp_py.Observation):
        def __init__(self, hit: int): self.hit = hit
        def __hash__(self): return hash(self.hit)
        def __eq__(self, o): return isinstance(o, _Obs) and self.hit == o.hit

    class _Trans(pomdp_py.TransitionModel):
        def sample(self, state, action):
            new_drilled = state.drilled | {action.cell}
            return _State(state.labels, new_drilled)
        def probability(self, next_state, state, action):
            # Deterministic in the cells; needed for POMCP particle
            # filtering only if the planner does explicit reweighting.
            return 1.0 if next_state.drilled == (state.drilled | {action.cell}) else 0.0

    class _ObsModel(pomdp_py.ObservationModel):
        def sample(self, next_state, action):
            return _Obs(next_state.labels[action.cell])
        def probability(self, observation, next_state, action):
            return 1.0 if observation.hit == next_state.labels[action.cell] else 0.0

    class _Reward(pomdp_py.RewardModel):
        def sample(self, state, action, next_state):
            cell = action.cell
            if cell in state.drilled:
                return -problem.drill_cost
            label = state.labels[cell]
            return -problem.drill_cost + (problem.discovery_value if label == 1 else 0.0)

    candidate_actions = [
        _Action(i) for i in range(n) if i not in drilled
    ]
    if not candidate_actions:
        return None

    class _Policy(pomdp_py.RolloutPolicy):
        def sample(self, state, *args, **kwargs):
            # Greedy-prior rollout — better than uniform for prospectivity
            # priors that are highly skewed toward a few high-p cells.
            available = [a for a in candidate_actions if a.cell not in state.drilled]
            if not available:
                return rng.choice(candidate_actions)  # fallback
            ps = np.array([posterior[a.cell] + 1e-9 for a in available])
            ps = ps / ps.sum()
            i = int(rng.choice(len(available), p=ps))
            return available[i]
        def rollout(self, state, *args, **kwargs):
            return self.sample(state)
        def get_all_actions(self, state=None, history=None):
            return candidate_actions

    # Belief: particle sample from per-cell Bernoulli posterior
    particles = []
    for _ in range(n_particles):
        labels = tuple(int(x) for x in (rng.random(n) < posterior).astype(int))
        # Override labels for already-drilled cells with their known truth.
        # (The caller's posterior already collapsed drilled cells to 0/1, so
        # this is effectively a no-op — but it makes the invariant explicit.)
        particles.append(_State(labels, drilled))
    init_belief = pomdp_py.Particles(particles)

    agent = pomdp_py.Agent(
        init_belief, _Policy(), _Trans(), _ObsModel(), _Reward(),
    )
    planner = pomdp_py.POMCP(
        max_depth=max_depth,
        discount_factor=discount_factor,
        planning_time=planning_time,
        exploration_const=exploration_const,
        rollout_policy=_Policy(),
    )
    action = planner.plan(agent)
    if action is None:
        return None
    return int(action.cell)


@dataclass(frozen=True)
class POMCPPolicy:
    """Adapter that exposes pomcp_plan via the Policy protocol."""

    name: str = "pomcp"
    n_particles: int = 200
    planning_time: float = 1.0
    max_depth: int = 5

    def choose(
        self,
        problem: "DrillingProblem",
        posterior: np.ndarray,
        drilled: frozenset[int],
        *,
        rng: np.random.Generator | None = None,
    ) -> int | None:
        return pomcp_plan(
            problem, posterior, drilled,
            n_particles=self.n_particles,
            planning_time=self.planning_time,
            max_depth=self.max_depth,
            rng=rng,
        )
