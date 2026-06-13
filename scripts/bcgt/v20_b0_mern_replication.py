"""Mern 2024 reproduction benchmark: structured prior + 9-hole POMDP vs 36-hole grid.

Replicates the headline experiment from
*Intelligent Prospector v2.0* (Mern et al., arXiv 2410.10610, 2024).
The paper's key claim is that a POMDP planner achieves the same
discovery accuracy with fewer than half the boreholes a regular
grid pattern would need.

Setup (matching the paper's single-hypothesis base case):

- 32x32 working grid
- One structured Mern hypothesis ((n_grabens=2, n_domains=2)) used
  as BOTH the planner's prior and the source of episode ground truth.
  Each episode samples a fresh GP realization from this hypothesis
  and uses it as the true deposit field.
- Gaussian-continuous sensor at sigma=0.001 (paper p.28)
- POMDP policies (Random, GreedyMean, BayesianGreedy, POMCP) at
  9-hole budget
- GridDrilling baseline at 36-hole budget (6x6 sub-grid)

The 4-hypothesis multi-hypothesis variant lives in the C.2 chapter
sections; this script is the paper's base reproduction.

Output:
  data/derived/bcgt/fig_v20_b0_mern_replication.png
  data/derived/bcgt/v20_b0_mern_replication_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_minerals.decision.v20.hypotheses import Hypothesis
from ai_minerals.decision.v20.policies import (
    BayesianGreedyPolicy,
    CorrelatedPriorPOMCPPolicy,
    GreedyMeanPolicy,
    GridDrillingPolicy,
    RandomPolicy,
)
from ai_minerals.decision.v20.pomdp import (
    CorrelatedDrillingProblem,
    SensorModel,
)
from ai_minerals.decision.v20.simulator import SyntheticMonteCarloSimulator

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_b0_mern_replication.png"
OUT_JSON = REPO / "data/derived/bcgt/v20_b0_mern_replication_results.json"

GRID_N = 32
SENSOR_NOISE_SIGMA = 0.001  # paper p.28
CUTOFF_GRADE = 0.15         # ~ peak prior mean (0.16) so deposits are sparse;
                            # tighter cutoff keeps Random from saturating early
DRILL_COST = 1.0
DISCOVERY_VALUE = 50.0
POMDP_BUDGET = 9            # paper p.20 POMDP budget
GRID_BUDGET = 36            # 6x6 grid baseline = 36 holes
N_EPISODES_PER_TRUTH = 8    # 32 total; tighter SEM than the initial 20-run

POMCP_N_PARTICLES = 400
POMCP_N_ROLLOUTS = 120      # higher than initial 40; gives POMCP a fair shake

N_TOTAL_EPISODES = N_EPISODES_PER_TRUTH * 4


def make_paper_base_hypothesis() -> Hypothesis:
    """The structured Mern 2024 hypothesis the planner sees as its
    prior. We pick the (2 grabens, 2 domains) configuration since it
    has the richest spatial structure of the 2x2 grid and gives the
    correlation-aware planners the most to work with.
    """
    rng = np.random.default_rng(2024)
    return Hypothesis.from_domain_config(
        name="H_2_2_paper_base",
        n_grabens=2, n_domains=2,
        grid_n=GRID_N, rng=rng,
    )


def build_template_problem(
    canonical_hypothesis: Hypothesis,
) -> CorrelatedDrillingProblem:
    """The CorrelatedDrillingProblem template the simulator drives."""
    n_cells = canonical_hypothesis.n_cells
    return CorrelatedDrillingProblem(
        hypothesis=canonical_hypothesis,
        x_m=canonical_hypothesis.cell_coords_m[:, 0],
        y_m=canonical_hypothesis.cell_coords_m[:, 1],
        true_grade=np.zeros(n_cells),  # placeholder; sim replaces per episode
        sensor_model=SensorModel.GAUSSIAN_CONTINUOUS,
        sensor_noise_sigma=SENSOR_NOISE_SIGMA,
        cutoff_grade=CUTOFF_GRADE,
        drill_cost=DRILL_COST,
        discovery_value=DISCOVERY_VALUE,
    )


def discovery_curve_for_policy(
    episodes: list,
    policy_name: str,
    cutoff_grade: float,
    budget: int,
) -> np.ndarray:
    """Per-drill discovery curve: at each drill step t in 0..budget,
    fraction of episodes that have at least one true positive in the
    policy's first t+1 drilled cells.
    """
    curve = np.zeros(budget, dtype=np.float64)
    for ep in episodes:
        trajectory = ep.policy_trajectories[policy_name]
        true_grade = ep.true_grade_field
        found_yet = False
        for t in range(budget):
            if t >= len(trajectory):
                if found_yet:
                    curve[t] += 1.0
                continue
            cell = trajectory[t]
            if true_grade[cell] > cutoff_grade:
                found_yet = True
            if found_yet:
                curve[t] += 1.0
    curve /= len(episodes)
    return curve


def main() -> int:
    paper_h = make_paper_base_hypothesis()
    print(f"paper-base hypothesis built on {GRID_N}x{GRID_N} grid")
    print(f"  prior_mean range: [{paper_h.prior_mean_field.min():.4f}, "
          f"{paper_h.prior_mean_field.max():.4f}]")
    print(f"  GP kernel: Matern nu={paper_h.gp_kernel_nu}, "
          f"sigma={paper_h.gp_marginal_std}, "
          f"ell={paper_h.gp_lengthscale_m}")

    template_problem = build_template_problem(paper_h)

    pomdp_policies = {
        "random": RandomPolicy(),
        "greedy_mean": GreedyMeanPolicy(),
        "bayes_greedy": BayesianGreedyPolicy(
            n_particles=POMCP_N_PARTICLES,
            sensor_noise_sigma=SENSOR_NOISE_SIGMA,
        ),
        "pomcp": CorrelatedPriorPOMCPPolicy(
            n_particles=POMCP_N_PARTICLES,
            n_rollouts=POMCP_N_ROLLOUTS,
        ),
    }
    grid_policies = {
        "grid_drilling": GridDrillingPolicy(n_per_side=6, grid_n=GRID_N),
    }

    # POMDP policies run at the 9-hole budget
    pomdp_sim = SyntheticMonteCarloSimulator(
        problem_template=template_problem,
        policies=pomdp_policies,
        n_ground_truths=N_TOTAL_EPISODES,
        drill_budget=POMDP_BUDGET,
    )
    # Grid baseline runs at the 36-hole budget on the SAME ground truths
    grid_sim = SyntheticMonteCarloSimulator(
        problem_template=template_problem,
        policies=grid_policies,
        n_ground_truths=N_TOTAL_EPISODES,
        drill_budget=GRID_BUDGET,
    )

    print(f"\nRunning POMDP policies (budget = {POMDP_BUDGET})...")
    t0 = time.perf_counter()
    pomdp_episodes = pomdp_sim.run(rng=np.random.default_rng(20260613))
    pomdp_elapsed = time.perf_counter() - t0
    print(f"  done in {pomdp_elapsed:.1f}s "
          f"({len(pomdp_episodes)} episodes)")

    print(f"\nRunning grid baseline (budget = {GRID_BUDGET})...")
    t0 = time.perf_counter()
    grid_episodes = grid_sim.run(rng=np.random.default_rng(20260613))
    grid_elapsed = time.perf_counter() - t0
    print(f"  done in {grid_elapsed:.1f}s")

    pomdp_curves: dict[str, list[float]] = {}
    for name in pomdp_policies:
        curve = discovery_curve_for_policy(
            pomdp_episodes, name, CUTOFF_GRADE, POMDP_BUDGET,
        )
        pomdp_curves[name] = [float(v) for v in curve]
        final = curve[-1] * 100
        print(f"  POMDP curve {name:>14s}  final discovery rate = {final:5.1f}%")

    grid_curves: dict[str, list[float]] = {}
    for name in grid_policies:
        curve = discovery_curve_for_policy(
            grid_episodes, name, CUTOFF_GRADE, GRID_BUDGET,
        )
        grid_curves[name] = [float(v) for v in curve]
        final = curve[-1] * 100
        print(f"  GRID  curve {name:>14s}  final discovery rate = {final:5.1f}%")

    # Chart
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {
        "random": "#888888",
        "greedy_mean": "#d97b00",
        "bayes_greedy": "#1f77b4",
        "pomcp": "#2c7c2c",
        "grid_drilling": "#a5256e",
    }
    labels = {
        "random": "Random (9-hole budget)",
        "greedy_mean": "GreedyMean (9-hole)",
        "bayes_greedy": "BayesianGreedy (9-hole)",
        "pomcp": "POMCP (9-hole)",
        "grid_drilling": "GridDrilling (36-hole)",
    }
    for name, curve in pomdp_curves.items():
        x = np.arange(1, POMDP_BUDGET + 1)
        ax.plot(x, np.array(curve) * 100,
                marker="o", color=palette[name], linewidth=1.6,
                label=labels[name])
    for name, curve in grid_curves.items():
        x = np.arange(1, GRID_BUDGET + 1)
        ax.plot(x, np.array(curve) * 100,
                marker="s", color=palette[name], linewidth=1.6,
                label=labels[name])

    ax.set_xlabel("Drill step")
    ax.set_ylabel("Fraction of episodes with at least one true deposit found (%)")
    ax.set_title(
        f"Reproducing Mern 2024: discovery rate by drill step\n"
        f"{GRID_N}x{GRID_N} grid, structured (2 graben, 2 domain) prior, "
        f"Gaussian sensor sigma={SENSOR_NOISE_SIGMA}, "
        f"{N_TOTAL_EPISODES} episodes"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, GRID_BUDGET + 1)
    ax.set_ylim(0, 105)
    ax.axvline(POMDP_BUDGET, color="black", linestyle=":", linewidth=0.7,
               label=None)
    ax.text(POMDP_BUDGET + 0.3, 5, f"POMDP budget = {POMDP_BUDGET}",
            fontsize=8, color="black")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT_PNG}")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "grid_n": GRID_N,
            "sensor_noise_sigma": SENSOR_NOISE_SIGMA,
            "cutoff_grade": CUTOFF_GRADE,
            "drill_cost": DRILL_COST,
            "discovery_value": DISCOVERY_VALUE,
            "pomdp_budget": POMDP_BUDGET,
            "grid_budget": GRID_BUDGET,
            "n_episodes_per_truth": N_EPISODES_PER_TRUTH,
            "n_episodes_total": N_TOTAL_EPISODES,
            "pomcp_n_particles": POMCP_N_PARTICLES,
            "pomcp_n_rollouts": POMCP_N_ROLLOUTS,
            "pomdp_curves": pomdp_curves,
            "grid_curves": grid_curves,
            "pomdp_elapsed_sec": pomdp_elapsed,
            "grid_elapsed_sec": grid_elapsed,
        }, f, indent=2)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
