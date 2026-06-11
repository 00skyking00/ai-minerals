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

NOT IMPLEMENTED YET. Skeletons only.
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
    """One ground-truth realization + the trajectory each policy took on it."""
    realization_seed: int
    true_grade_field: np.ndarray            # shape (n_cells,)
    policy_trajectories: dict[str, list[int]]  # policy_name -> list of cell_idxs drilled
    policy_discovery_rates: dict[str, float]
    policy_regrets: dict[str, float]


@dataclass
class SyntheticMonteCarloSimulator:
    """B.1 synthetic Monte Carlo per Mern 2024 p.20.

    Runs N_GROUND_TRUTHS=17 episodes; each episode runs every policy on
    the same realization. Aggregates per-policy metrics.

    NOT IMPLEMENTED YET.
    """
    problem_template: CorrelatedDrillingProblem
    policies: dict[str, object]   # policy_name -> policy callable / class instance
    n_ground_truths: int = PAPER_N_GROUND_TRUTHS
    drill_budget: int = PAPER_DRILL_BUDGET

    def run(self, rng: np.random.Generator) -> list[SimulationEpisode]:
        """Run all episodes; return per-episode results.

        TODO B.1: implement; outer loop over realizations, inner loop over
        policies, shared random state per realization for fair comparison.
        """
        raise NotImplementedError("B.1 milestone")

    def aggregate(self, episodes: list[SimulationEpisode]) -> dict:
        """Mean / median discovery rate + regret per policy across episodes.

        TODO B.1: implement; straightforward numpy aggregation.
        """
        raise NotImplementedError("B.1 milestone")


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
