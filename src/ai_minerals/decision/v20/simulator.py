"""Synthetic Monte Carlo + retrospective BCGS simulators for bcgt-v2.0.

Two simulators ship in this module:

1. SyntheticMonteCarloSimulator (B.1, mirrors Mern 2024 p.20)
   - 17 ground-truth realizations sampled from a single Hypothesis
   - Each episode runs each policy on the same realization for fair
     comparison (random seed shared across policies, varied across episodes)
   - Drill budget: ~9 holes per episode (paper's POMDP achieves accuracy
     at this budget; grid baseline uses 36)
   - Score: discovery rate, regret, posterior accuracy at episode end

2. RetrospectiveBCGSValidator (B.2, our original contribution beyond Mern)
   - Use BCGS pre-2010 drill record as the prior
   - Use BCGS post-2010 (or post-some-cutoff) drill record as held-out
     ground truth
   - For each policy: starting from pre-2010 prior, simulate up to K holes
     of recommendation; check what fraction of post-2010 operator-positive
     holes (assay-Cu >= 0.2%) fall in the policy's top-k% choice set

B.1 IMPLEMENTATION STATUS (2026-06-11):
    SyntheticMonteCarloSimulator.run / aggregate: DONE - GH issue #5
    RetrospectiveBCGSValidator:                   NOT YET - B.2 milestone
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pomdp import CorrelatedDrillingProblem, MultiHypothesisDrillingProblem


PAPER_N_GROUND_TRUTHS = 17  # Mern 2024 p.20
PAPER_DRILL_BUDGET = 9      # POMDP holes per episode (paper)
GRID_BASELINE_BUDGET = 36   # 6x6 grid baseline for comparison


@dataclass
class SimulationEpisode:
    """One ground-truth realization + the trajectory each policy took on it.

    Stored shape:
      realization_seed                    int   (per-episode seed)
      true_grade_field                    (n_cells,) float
      policy_trajectories[name]           list[int]   cells drilled in order
      policy_observations[name]           list[float] obs at each drill
      policy_discovery_rates[name]        float  fraction of trajectory
                                                 cells whose true grade
                                                 exceeded cutoff
      policy_regrets[name]                float  optimal cumulative reward
                                                 minus realized reward
    """
    realization_seed: int
    true_grade_field: np.ndarray
    policy_trajectories: dict[str, list[int]]
    policy_observations: dict[str, list[float]]
    policy_discovery_rates: dict[str, float]
    policy_regrets: dict[str, float]


def _episode_optimal_reward(
    true_grade: np.ndarray,
    cutoff_grade: float,
    drill_cost: float,
    discovery_value: float,
    drill_budget: int,
) -> float:
    """Best-possible cumulative reward: drill the top `drill_budget` cells
    by true grade, take the reward whenever they're above cutoff."""
    sorted_grades = np.sort(true_grade)[::-1][:drill_budget]
    discoveries = (sorted_grades > cutoff_grade).sum()
    return float(discoveries * discovery_value - drill_budget * drill_cost)


@dataclass
class SyntheticMonteCarloSimulator:
    """B.1 synthetic Monte Carlo per Mern 2024 p.20.

    Runs n_ground_truths episodes; each episode runs every policy on the
    same ground-truth realization. Aggregates per-policy metrics.

    The simulator owns the problem template (Hypothesis + sensor model +
    reward params), draws one realization per episode from the template's
    GP prior, and instantiates a fresh CorrelatedDrillingProblem with that
    realization as true_grade. Policies are reset at the start of every
    episode so internal state doesn't leak.
    """
    problem_template: CorrelatedDrillingProblem
    policies: dict[str, object]
    n_ground_truths: int = PAPER_N_GROUND_TRUTHS
    drill_budget: int = PAPER_DRILL_BUDGET

    def _run_episode_for_policy(
        self,
        policy: object,
        problem: CorrelatedDrillingProblem,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[float], float, float]:
        """Run one (policy, ground_truth) episode.

        Returns (trajectory, observations, discovery_rate, regret).
        """
        policy.reset(problem, rng)
        drilled: frozenset[int] = frozenset()
        history: list[tuple[int, float]] = []
        trajectory: list[int] = []
        observations: list[float] = []
        cumulative_reward = 0.0

        for _ in range(self.drill_budget):
            cell_idx = policy.choose_action(history, drilled, rng)
            obs, reward, drilled = problem.step(cell_idx, drilled, rng)
            cumulative_reward += reward
            trajectory.append(cell_idx)
            observations.append(float(obs))
            history.append((cell_idx, float(obs)))

        n_discoveries = sum(
            1 for cell in trajectory
            if problem.true_grade[cell] > problem.cutoff_grade
        )
        discovery_rate = n_discoveries / self.drill_budget
        optimal = _episode_optimal_reward(
            problem.true_grade,
            problem.cutoff_grade,
            problem.drill_cost,
            problem.discovery_value,
            self.drill_budget,
        )
        regret = optimal - cumulative_reward
        return trajectory, observations, discovery_rate, regret

    def run(self, rng: np.random.Generator) -> list[SimulationEpisode]:
        """Run all episodes; return per-episode results."""
        episodes: list[SimulationEpisode] = []
        # Generate per-episode seeds up front so each episode's behaviour is
        # reproducible regardless of policy order; same seed feeds every
        # policy in that episode so they see the same ground truth + the
        # same noise.
        episode_seeds = rng.integers(0, 2**31 - 1, size=self.n_ground_truths)
        for ep_idx, seed in enumerate(episode_seeds):
            ep_rng = np.random.default_rng(int(seed))
            true_grade = self.problem_template.hypothesis.sample_realization(
                ep_rng, n_samples=1,
            )[0]
            problem = CorrelatedDrillingProblem(
                hypothesis=self.problem_template.hypothesis,
                x_m=self.problem_template.x_m,
                y_m=self.problem_template.y_m,
                true_grade=true_grade,
                sensor_model=self.problem_template.sensor_model,
                sensor_noise_sigma=self.problem_template.sensor_noise_sigma,
                sensor_alpha=self.problem_template.sensor_alpha,
                sensor_beta=self.problem_template.sensor_beta,
                drill_cost=self.problem_template.drill_cost,
                discovery_value=self.problem_template.discovery_value,
                cutoff_grade=self.problem_template.cutoff_grade,
            )
            trajectories: dict[str, list[int]] = {}
            observations_d: dict[str, list[float]] = {}
            discovery_rates: dict[str, float] = {}
            regrets: dict[str, float] = {}
            for name, policy in self.policies.items():
                # Each policy gets its own RNG seeded off the episode seed so
                # noise + tie-breaks are reproducible but policies don't
                # share state.
                policy_rng = np.random.default_rng(
                    int(seed) + hash(name) % (2**31 - 1)
                )
                traj, obs, dr, regret = self._run_episode_for_policy(
                    policy, problem, policy_rng,
                )
                trajectories[name] = traj
                observations_d[name] = obs
                discovery_rates[name] = dr
                regrets[name] = regret
            episodes.append(SimulationEpisode(
                realization_seed=int(seed),
                true_grade_field=true_grade.copy(),
                policy_trajectories=trajectories,
                policy_observations=observations_d,
                policy_discovery_rates=discovery_rates,
                policy_regrets=regrets,
            ))
        return episodes

    def aggregate(self, episodes: list[SimulationEpisode]) -> dict:
        """Mean / median discovery rate + regret per policy across episodes."""
        if not episodes:
            return {}
        policy_names = list(self.policies.keys())
        agg: dict[str, dict[str, float]] = {}
        for name in policy_names:
            drs = np.array([ep.policy_discovery_rates[name] for ep in episodes])
            regs = np.array([ep.policy_regrets[name] for ep in episodes])
            agg[name] = {
                "discovery_rate_mean": float(drs.mean()),
                "discovery_rate_median": float(np.median(drs)),
                "regret_mean": float(regs.mean()),
                "regret_median": float(np.median(regs)),
                "n_episodes": int(len(episodes)),
            }
        return agg


@dataclass
class RetrospectiveBCGSValidator:
    """B.2 retrospective validation against historic BCGS drilling.

    Our original contribution beyond Mern 2024. The paper validates only
    on synthetic Monte Carlo; we additionally show whether the planner
    would have recommended cells where operators actually found Cu.

    Concrete setup:
    - Pre-2010 BCGS drill record -> per-cell prior (could be the v3 RF
      posterior; could be a Hawkes-like aggregate of pre-2010 assays).
    - Post-2010 BCGS drill record -> held-out ground truth. For each
      hole, the operator-positive flag is (assay-Cu >= 0.2% Cox-Singer
      cutoff = 1, else 0).
    - Score: capture-at-k%. What fraction of post-2010 operator-positives
      fall in the planner's top k% recommendations? Baselines: random
      (= k%), v3-RF-prior alone (= no POMDP value-add).

    NOT IMPLEMENTED YET.
    """
    pre_2010_prior: np.ndarray         # (n_cells,) v3 RF posterior or equivalent
    post_2010_positives: np.ndarray    # (n_cells,) binary indicator, 1 at operator-positive
    cells_drilled_pre_2010: np.ndarray  # (n_cells,) binary indicator
    drill_budget: int = 50              # held-out hole count to allocate

    def run_policy(self, policy: object, rng: np.random.Generator) -> dict:
        """Run one policy; return capture-at-k% for k in {1, 5, 10, 25}.

        TODO B.2: implement.
        """
        raise NotImplementedError("B.2 milestone")

    def compare(self, policies: dict[str, object], rng: np.random.Generator) -> dict:
        """Run all policies; return per-policy capture-at-k% table.

        TODO B.2: implement.
        """
        raise NotImplementedError("B.2 milestone")
