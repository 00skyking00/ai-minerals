"""bcgt-v2.0 B.2 retrospective benchmark across three prior variants.

Drives the four policies through a RetrospectiveBCGSValidator built on the same
50x50 BCGT subarea (~25 km x 25 km) centered on the largest BCGS post-2010 Cu+
cluster (KSM area), scoring capture-at-k% against post-2010 operator-Cu+ holes.

Three priors compared side-by-side:
  informative  smoothed MINFILE any_mineral_occurrence (likely temporally
               contaminated since MINFILE is a current snapshot)
  pre2010_only smoothed pre-2010 BCGS Cu+ rate (leak-free, very sparse)
  uniform      constant 0.1 (planner starts blind)

The contrast is the chapter narrative: with the contaminated prior the static
recommendation dominates; with a leak-free or uninformative prior the v2.0
Bayesian update + multi-step planning earn their value.

Output: data/derived/bcgt/fig_v20_b2_retrospective_capture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_minerals.decision.v20.policies import (
    BayesianGreedyPolicy,
    CorrelatedPriorPOMCPPolicy,
    GreedyMeanPolicy,
    RandomPolicy,
)
from ai_minerals.decision.v20.simulator import (
    CAPTURE_KS,
    RetrospectiveBCGSValidator,
)

REPO = Path(__file__).resolve().parents[2]
INPUTS = {
    "informative": REPO / "data/derived/bcgt/b2_inputs.npz",
    "pre2010_only": REPO / "data/derived/bcgt/b2_inputs_pre2010_only.npz",
    "uniform": REPO / "data/derived/bcgt/b2_inputs_uniform.npz",
}
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_b2_retrospective_capture.png"

SENSOR_NOISE_SIGMA = 0.05
DRILL_BUDGET = 625    # top 25 percent of 2500 cells
N_SEEDS = 1           # single seed; multi-seed averaging is too slow at scale

PANEL_TITLES = {
    "informative": "Informative prior (smoothed MINFILE)",
    "pre2010_only": "Pre-2010 leak-free prior",
    "uniform": "Uniform prior (planner blind)",
}

POLICY_ORDER = ["random", "static_prior", "bayes_greedy", "pomcp"]
POLICY_LABELS = {
    "random": "Random",
    "static_prior": "Static prior",
    "bayes_greedy": "BayesianGreedy",
    "pomcp": "POMCP",
}
POLICY_COLORS = {
    "random": "#888888",
    "static_prior": "#d97b00",
    "bayes_greedy": "#1f77b4",
    "pomcp": "#2c7c2c",
}


def _run_variant(npz_path: Path) -> tuple[dict[str, dict[int, float]], dict]:
    """Run the four policies on this variant N_SEEDS times; return mean
    capture-at-k% per policy plus run metadata."""
    d = np.load(npz_path)
    coords = d["cell_coords_m"]
    prior = d["prior_mean"]
    post_pos = d["post_2010_positive"]
    post_grade = d["post_2010_grade"]
    pre_drilled = d["pre_2010_drilled"]
    n_cells = len(prior)
    n_pos = int(post_pos.sum())
    n_pre = int(pre_drilled.sum())

    validator = RetrospectiveBCGSValidator(
        pre_2010_prior=prior,
        post_2010_positives=post_pos,
        cells_drilled_pre_2010=pre_drilled,
        cell_coords_m=coords,
        post_2010_grade=post_grade,
        sensor_noise_sigma=SENSOR_NOISE_SIGMA,
        drill_budget=DRILL_BUDGET,
    )

    seed_tables: list[dict[str, dict[int, float]]] = []
    for seed_idx in range(N_SEEDS):
        policies = {
            "random": RandomPolicy(),
            "static_prior": GreedyMeanPolicy(),
            "bayes_greedy": BayesianGreedyPolicy(n_particles=400),
            "pomcp": CorrelatedPriorPOMCPPolicy(
                n_particles=400, n_rollouts=60,
            ),
        }
        master = np.random.default_rng(20260612 + seed_idx * 7919)
        table = validator.compare(policies, master)
        seed_tables.append(table)

    # Average across seeds.
    policy_names = list(seed_tables[0].keys())
    mean_table: dict[str, dict[int, float]] = {}
    for pname in policy_names:
        mean_table[pname] = {}
        for k in CAPTURE_KS:
            mean_table[pname][k] = float(np.mean(
                [t[pname][k] for t in seed_tables]
            ))
    return mean_table, {
        "n_cells": n_cells, "n_pos": n_pos, "n_pre": n_pre,
        "n_seeds": N_SEEDS,
    }


def main() -> int:
    results: dict[str, dict] = {}
    for name, path in INPUTS.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        print(f"\n[run] {name}: {path}")
        table, meta = _run_variant(path)
        results[name] = {"table": table, "meta": meta}
        print(f"       cells={meta['n_cells']}  "
              f"post-2010 Cu+={meta['n_pos']}  "
              f"pre-2010 drilled={meta['n_pre']}")
        print(f"       capture-at-k% (k = {list(CAPTURE_KS)})")
        for pname, caps in table.items():
            values = [f"{caps[k] * 100:5.1f}%" for k in CAPTURE_KS]
            print(f"         {pname:14s}  " + "  ".join(values))

    # Plot: one panel per variant, grouped bars per policy
    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    x = np.arange(len(CAPTURE_KS))
    width = 0.2
    k_labels = [f"top {k}%" for k in CAPTURE_KS]

    for ax, (variant_name, rec) in zip(axes, results.items()):
        table = rec["table"]
        meta = rec["meta"]
        for i, policy_name in enumerate(POLICY_ORDER):
            values = [table[policy_name][k] * 100 for k in CAPTURE_KS]
            offset = (i - len(POLICY_ORDER) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, values, width,
                label=POLICY_LABELS[policy_name],
                color=POLICY_COLORS[policy_name],
            )
            for b, v in zip(bars, values):
                ax.text(
                    b.get_x() + b.get_width() / 2, v + 1.5,
                    f"{v:.0f}", ha="center", fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(k_labels)
        ax.set_title(PANEL_TITLES[variant_name])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Percent of post-2010 Cu+ cells captured")
    axes[0].legend(loc="upper left")
    n_pos = list(results.values())[0]["meta"]["n_pos"]
    n_pre = list(results.values())[0]["meta"]["n_pre"]
    n_seeds = list(results.values())[0]["meta"]["n_seeds"]
    fig.suptitle(
        f"BCGT B.2 retrospective: capture of {n_pos} post-2010 Cu+ cells in the "
        f"top-k% recommendations\n50 x 50 cell working area around the KSM "
        f"cluster; {n_pre} pre-2010 drilled cells excluded; "
        f"averaged across {n_seeds} master seeds per variant",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
