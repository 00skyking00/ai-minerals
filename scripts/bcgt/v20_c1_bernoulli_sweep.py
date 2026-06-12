"""bcgt-v2.0 C.1: Bernoulli sensor + 3x3 (alpha, beta) sensitivity sweep.

Runs the four B.1 policies through SyntheticMonteCarloSimulator with the
Bernoulli sensor enabled across a 3x3 grid of (alpha, beta) values, scoring
mean discovery rate per (policy, alpha, beta) cell. The chart is a heatmap
matrix: one heatmap per policy, with alpha on x and beta on y.

Output: data/derived/bcgt/fig_v20_c1_bernoulli_sweep.png
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
    BayesianGreedyPolicy,
    CorrelatedPriorPOMCPPolicy,
    GreedyMeanPolicy,
    RandomPolicy,
)
from ai_minerals.decision.v20.simulator import (
    PAPER_DRILL_BUDGET,
    PAPER_N_GROUND_TRUTHS,
    SyntheticMonteCarloSimulator,
)

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_c1_bernoulli_sweep.png"

ALPHAS = [0.02, 0.05, 0.10]   # false-positive rate
BETAS = [0.05, 0.10, 0.20]    # false-negative rate

ANOMALY_PEAK = 0.18
ANOMALY_SPREAD_M = 3000.0


def _make_template(alpha: float, beta: float) -> CorrelatedDrillingProblem:
    spacing = 500.0
    x = np.arange(30) * spacing
    y = np.arange(30) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    center = np.array([15 * spacing, 15 * spacing])
    distances = np.linalg.norm(coords - center, axis=1)
    prior_mean = ANOMALY_PEAK * np.exp(
        -0.5 * (distances / ANOMALY_SPREAD_M) ** 2
    )
    h = Hypothesis(
        name="porphyry_cu", n_grabens=1, n_domains=1,
        cell_coords_m=coords, prior_mean_field=prior_mean,
        gp_marginal_std=KERNEL_MARGINAL_STD,
        gp_lengthscale_m=KERNEL_LENGTHSCALE_M_BCGT,
    )
    return CorrelatedDrillingProblem(
        hypothesis=h,
        x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=np.zeros(coords.shape[0]),
        sensor_model=SensorModel.BERNOULLI_BINARY,
        sensor_alpha=alpha, sensor_beta=beta,
        cutoff_grade=0.2,
        drill_cost=1.0, discovery_value=50.0,
    )


def _run_cell(alpha: float, beta: float) -> dict[str, float]:
    template = _make_template(alpha, beta)
    sim = SyntheticMonteCarloSimulator(
        problem_template=template,
        policies={
            "random": RandomPolicy(),
            "greedy_mean": GreedyMeanPolicy(),
            "bayes_greedy": BayesianGreedyPolicy(n_particles=400),
            "pomcp": CorrelatedPriorPOMCPPolicy(
                n_particles=400, n_rollouts=80,
            ),
        },
        n_ground_truths=PAPER_N_GROUND_TRUTHS,
        drill_budget=PAPER_DRILL_BUDGET,
    )
    rng = np.random.default_rng(20260612)
    episodes = sim.run(rng)
    agg = sim.aggregate(episodes)
    return {name: m["discovery_rate_mean"] for name, m in agg.items()}


def main() -> int:
    print(f"==> 3x3 (alpha, beta) sweep across "
          f"{len(ALPHAS) * len(BETAS)} cells x 4 policies x "
          f"{PAPER_N_GROUND_TRUTHS} episodes...")
    # results[policy][alpha_idx][beta_idx] = mean discovery rate
    policy_names = ["random", "greedy_mean", "bayes_greedy", "pomcp"]
    results: dict[str, np.ndarray] = {
        n: np.zeros((len(ALPHAS), len(BETAS))) for n in policy_names
    }
    for ai, alpha in enumerate(ALPHAS):
        for bi, beta in enumerate(BETAS):
            print(f"  alpha={alpha} beta={beta}...", flush=True)
            cell = _run_cell(alpha, beta)
            for name in policy_names:
                results[name][ai, bi] = cell[name]

    # Plot: 1x4 heatmap row, one per policy
    labels = {
        "random": "Random",
        "greedy_mean": "GreedyMean",
        "bayes_greedy": "BayesianGreedy",
        "pomcp": "POMCP",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    vmax = max(r.max() for r in results.values())
    for ax, name in zip(axes, policy_names):
        m = results[name]
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(BETAS)))
        ax.set_xticklabels([f"{b}" for b in BETAS])
        ax.set_yticks(range(len(ALPHAS)))
        ax.set_yticklabels([f"{a}" for a in ALPHAS])
        ax.set_xlabel("beta (false-negative rate)")
        ax.set_ylabel("alpha (false-positive rate)")
        ax.set_title(labels[name])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                color = "white" if v < vmax * 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=10)
    fig.suptitle(
        f"BCGT C.1 Bernoulli sensor sweep: mean discovery rate over "
        f"{PAPER_N_GROUND_TRUTHS} episodes (9-hole budget), "
        f"30x30 synthetic subarea matching the B.1 setup",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Mean discovery rate")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)")

    # Plain-text summary
    print()
    print("--- mean discovery rate (rows alpha, cols beta) ---")
    for name in policy_names:
        print(f"  {name}:")
        for ai, alpha in enumerate(ALPHAS):
            cells = [f"{results[name][ai, bi]:.3f}" for bi in range(len(BETAS))]
            print(f"    alpha={alpha}: " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
