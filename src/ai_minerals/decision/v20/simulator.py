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


CAPTURE_KS = (1, 5, 10, 25)   # percent of cells to evaluate at


@dataclass
class RetrospectiveBCGSValidator:
    """B.2 retrospective validation against historic BCGS drilling.

    Original contribution beyond Mern 2024. The paper validates only on
    synthetic Monte Carlo; this validator additionally scores whether the
    planner would have recommended cells where operators later found Cu.

    Setup:
    - Pre-2010 BCGS drill record -> per-cell prior (smoothed mineral-occurrence
      surface or a separately-supplied v3 RF posterior).
    - Post-2010 BCGS drill record -> held-out ground truth. For each cell,
      `post_2010_positive` = 1 iff max Cu assay across post-2010 holes in that
      cell hit the Cox-Singer 0.2 percent porphyry-Cu cutoff.
    - Score: capture-at-k%. Of the post-2010 positive cells, what fraction
      sit in the planner's top k% recommendations for k in {1, 5, 10, 25}?

    Each policy is driven sequentially for `drill_budget` steps starting from
    a `drilled` set initialized to the pre-2010 drilled cells. The cell order
    in the trajectory IS the policy's ranked recommendation list. Capture-at-k%
    is computed by taking the first k% of cells from the trajectory (rounded
    up to at least 1 cell) and counting hits against `post_2010_positives`.

    Cells without post-2010 drill data are assumed barren (the standard
    retrospective-scoring assumption). The planner-visible observation for a
    cell drilled during the trajectory follows the CorrelatedDrillingProblem's
    sensor model; barren cells produce near-zero observations with the
    configured sensor noise.
    """
    pre_2010_prior: np.ndarray         # (n_cells,)
    post_2010_positives: np.ndarray    # (n_cells,) binary
    cells_drilled_pre_2010: np.ndarray  # (n_cells,) binary
    cell_coords_m: np.ndarray          # (n_cells, 2)
    post_2010_grade: np.ndarray        # (n_cells,) Cu percent, 0 for barren / unknown
    sensor_noise_sigma: float = 0.05
    cutoff_grade: float = 0.2
    drill_budget: int = 250            # top-25 percent of 1000 cells = 250
    gp_marginal_std: float = 0.1
    gp_lengthscale_m: float = 1500.0

    def __post_init__(self) -> None:
        n = len(self.pre_2010_prior)
        if self.post_2010_positives.shape != (n,):
            raise ValueError("post_2010_positives shape mismatch")
        if self.cells_drilled_pre_2010.shape != (n,):
            raise ValueError("cells_drilled_pre_2010 shape mismatch")
        if self.cell_coords_m.shape != (n, 2):
            raise ValueError(
                f"cell_coords_m must be ({n}, 2); got {self.cell_coords_m.shape}"
            )
        if self.post_2010_grade.shape != (n,):
            raise ValueError("post_2010_grade shape mismatch")

    def _build_problem(self) -> CorrelatedDrillingProblem:
        """Build the CorrelatedDrillingProblem the policies will see."""
        # Lazy import to avoid circular imports at module load.
        from .hypotheses import Hypothesis
        from .pomdp import SensorModel

        h = Hypothesis(
            name="bcgt_retro",
            n_grabens=1, n_domains=1,
            cell_coords_m=self.cell_coords_m,
            prior_mean_field=self.pre_2010_prior,
            gp_marginal_std=self.gp_marginal_std,
            gp_lengthscale_m=self.gp_lengthscale_m,
        )
        return CorrelatedDrillingProblem(
            hypothesis=h,
            x_m=self.cell_coords_m[:, 0],
            y_m=self.cell_coords_m[:, 1],
            true_grade=self.post_2010_grade,
            sensor_model=SensorModel.GAUSSIAN_CONTINUOUS,
            sensor_noise_sigma=self.sensor_noise_sigma,
            cutoff_grade=self.cutoff_grade,
            drill_cost=1.0,
            discovery_value=50.0,
        )

    def _capture_at_k_pct(
        self, trajectory: list[int], k_percent: int,
    ) -> float:
        """Fraction of post-2010 positives captured in the top k% of the trajectory."""
        n_cells = len(self.pre_2010_prior)
        k_cells = max(1, int(round(k_percent / 100.0 * n_cells)))
        top = trajectory[:k_cells]
        total_pos = int(self.post_2010_positives.sum())
        if total_pos == 0:
            return 0.0
        hits = int(self.post_2010_positives[top].sum())
        return hits / total_pos

    def run_policy(
        self,
        policy: object,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        """Drive `policy` for `drill_budget` steps; return capture-at-k% dict."""
        problem = self._build_problem()
        policy.reset(problem, rng)
        drilled = frozenset(np.where(self.cells_drilled_pre_2010 > 0)[0].tolist())
        history: list[tuple[int, float]] = []
        trajectory: list[int] = []
        for _ in range(self.drill_budget):
            cell_idx = policy.choose_action(history, drilled, rng)
            obs, _, drilled = problem.step(cell_idx, drilled, rng)
            history.append((cell_idx, float(obs)))
            trajectory.append(int(cell_idx))
        return {k: self._capture_at_k_pct(trajectory, k) for k in CAPTURE_KS}

    def compare(
        self,
        policies: dict[str, object],
        rng: np.random.Generator,
    ) -> dict[str, dict[int, float]]:
        """Run every policy; return per-policy capture-at-k% table.

        Each policy gets its own RNG seeded off the master generator so noise
        and tie-breaks are reproducible across policies that share state.
        """
        seeds = rng.integers(0, 2**31 - 1, size=len(policies))
        return {
            name: self.run_policy(p, np.random.default_rng(int(seeds[i])))
            for i, (name, p) in enumerate(policies.items())
        }
