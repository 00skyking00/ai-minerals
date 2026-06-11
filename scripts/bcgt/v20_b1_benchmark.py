"""bcgt-v2.0 B.1 baseline benchmark: 17-realization MC, Random vs GreedyMean.

Runs the SyntheticMonteCarloSimulator at paper-matched settings (17
ground truths, 9 holes per episode) with the two baseline policies
shipped in v1.2.0 milestone:

    RandomPolicy        uniform pick from unvisited
    GreedyMeanPolicy    argmax prior_mean over unvisited

The CorrelatedPriorPOMCPPolicy (true POMCP over the particle-filter
belief) is the next bcgt-v2.0 milestone (issue queued); this benchmark
is the validation harness it will plug into.

Synthetic terrain: 30x30 BCGT subarea, Matern v=2.5 kernel, sigma=0.1,
lengthscale 1500 m, prior mean field shaped as a single anomaly bump
centered at cell (15, 15) with peak 0.3 and Gaussian falloff matching
the GP lengthscale. The cutoff grade is 0.2 (paper-matched Cox-Singer
porphyry-Cu).

Output: data/derived/bcgt/fig_v20_b1_baseline_benchmark.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_minerals.decision.v20.hypotheses import (
    KERNEL_LENGTHSCALE_M_BCGT,
    KERNEL_MARGINAL_STD,
    Hypothesis,
)
from ai_minerals.decision.v20.pomdp import (
    CorrelatedDrillingProblem,
    SensorModel,
)
from ai_minerals.decision.v20.policies import (
    GreedyMeanPolicy,
    RandomPolicy,
)
from ai_minerals.decision.v20.simulator import (
    PAPER_DRILL_BUDGET,
    PAPER_N_GROUND_TRUTHS,
    SyntheticMonteCarloSimulator,
)

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_b1_baseline_benchmark.png"


def _make_problem_template() -> CorrelatedDrillingProblem:
    spacing = 500.0
    x = np.arange(30) * spacing
    y = np.arange(30) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Prior mean: Gaussian bump at the grid center with peak 0.3 and
    # spread matching the GP lengthscale. This stands in for the v3 RF
    # posterior surface that the production BCGT version will use.
    center = np.array([15 * spacing, 15 * spacing])
    distances = np.linalg.norm(coords - center, axis=1)
    prior_mean = 0.3 * np.exp(-0.5 * (distances / KERNEL_LENGTHSCALE_M_BCGT) ** 2)

    h = Hypothesis(
        name="porphyry_cu", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=prior_mean,
        gp_marginal_std=KERNEL_MARGINAL_STD,
        gp_lengthscale_m=KERNEL_LENGTHSCALE_M_BCGT,
    )
    return CorrelatedDrillingProblem(
        hypothesis=h,
        x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=np.zeros(coords.shape[0]),  # gets overwritten per-episode
        sensor_model=SensorModel.GAUSSIAN_CONTINUOUS,
        sensor_noise_sigma=0.001,
        cutoff_grade=0.2,
        drill_cost=1.0,
        discovery_value=50.0,
    )


def main() -> int:
    template = _make_problem_template()
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={
            "random": RandomPolicy(),
            "greedy_mean": GreedyMeanPolicy(),
        },
        n_ground_truths=PAPER_N_GROUND_TRUTHS,
        drill_budget=PAPER_DRILL_BUDGET,
    )
    rng = np.random.default_rng(20260611)
    print(f"==> Running {PAPER_N_GROUND_TRUTHS} episodes x "
          f"{PAPER_DRILL_BUDGET} holes per episode...")
    episodes = sim.run(rng)
    agg = sim.aggregate(episodes)

    print()
    print("--- per-policy aggregates ---")
    for name, metrics in agg.items():
        print(f"  {name:14s} discovery_rate: mean={metrics['discovery_rate_mean']:.3f}  "
              f"median={metrics['discovery_rate_median']:.3f}")
        print(f"  {' ':14s} regret:         mean={metrics['regret_mean']:.2f}  "
              f"median={metrics['regret_median']:.2f}")
        print()

    # Discovery curve: per step, fraction of episodes in which by-now-drilled
    # cells include at least one above-cutoff.
    n_steps = PAPER_DRILL_BUDGET
    discovery_curves: dict[str, np.ndarray] = {}
    for name in sim.policies:
        curve = np.zeros(n_steps)
        for ep in episodes:
            traj = ep.policy_trajectories[name]
            tg = ep.true_grade_field
            found_so_far = False
            for step in range(n_steps):
                if tg[traj[step]] > template.cutoff_grade:
                    found_so_far = True
                curve[step] += float(found_so_far)
        curve /= len(episodes)
        discovery_curves[name] = curve

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    steps = np.arange(1, n_steps + 1)
    for name, curve in discovery_curves.items():
        ax_curve.plot(steps, curve, marker="o", label=name)
    ax_curve.set_xlabel("Drill step")
    ax_curve.set_ylabel("Fraction of episodes with >=1 discovery")
    ax_curve.set_title(
        f"Discovery curve over {PAPER_N_GROUND_TRUTHS} episodes\n"
        f"(30x30 BCGT subarea, Matern v=2.5, sigma_along=1500 m)"
    )
    ax_curve.set_ylim(0, 1.05)
    ax_curve.legend()
    ax_curve.grid(alpha=0.3)

    names = list(agg.keys())
    means = [agg[n]["discovery_rate_mean"] for n in names]
    ax_bar.bar(names, means, color=["#888", "#2c7"])
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("Mean discovery rate (across trajectory)")
    ax_bar.set_title("Per-policy mean discovery rate")
    for i, v in enumerate(means):
        ax_bar.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
