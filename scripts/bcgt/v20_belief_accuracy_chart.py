"""Generate a side-by-side chart of final truth-belief by policy.

D.1.C (synthetic priors) and D.1.D (real priors) both report the same
all-tied result on cumulative reward. The truth-belief metric is where
SARSOP earns its keep on both. This chart plots the final posterior
probability the planner assigns to the true hypothesis at episode end,
two panels side by side: synthetic priors on the left, real priors on
the right. The visual gap between SARSOP and the other policies is the
finding the chapter prose mentions but does not plot.

Output: data/derived/bcgt/fig_v20_d1_belief_accuracy.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SYN = REPO / "data/derived/bcgt/fig_v20_d1_benchmark.json"
REAL = REPO / "data/derived/bcgt/fig_v20_d1d_benchmark.json"
OUT = REPO / "data/derived/bcgt/fig_v20_d1_belief_accuracy.png"


def main() -> int:
    syn = json.load(open(SYN))
    real = json.load(open(REAL))

    syn_order = ["random", "greedy_MAP", "pomcp_topK", "sarsop_topK", "sarsop_pf_topK"]
    real_order = ["random", "greedy_MAP", "pomcp_topK", "sarsop_topK"]
    colors = {
        "random": "#9aa0a6",
        "greedy_MAP": "#1f77b4",
        "pomcp_topK": "#ff7f0e",
        "sarsop_topK": "#2ca02c",
        "sarsop_pf_topK": "#9467bd",
    }
    display = {
        "random": "random",
        "greedy_MAP": "greedy",
        "pomcp_topK": "POMCP",
        "sarsop_topK": "SARSOP",
        "sarsop_pf_topK": "SARSOP+PF",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, summary, order, title in (
        (axes[0], syn, syn_order, "D.1.C: synthetic NW/SE priors (100 episodes)"),
        (axes[1], real, real_order, "D.1.D: real BCGS deposit-type priors (30 episodes)"),
    ):
        names = [display[n] for n in order]
        beliefs = [summary[n]["mean_final_belief_truth"] for n in order]
        bars = ax.bar(
            names, beliefs, color=[colors[n] for n in order],
            edgecolor="black", linewidth=0.6,
        )
        for bar, val in zip(bars, beliefs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0 / len(order), color="gray", linewidth=0.7,
                   linestyle="--", label="uniform prior")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Final posterior probability on the true hypothesis", fontsize=10)
    fig.suptitle(
        "Final truth-belief by policy. Higher means the planner's "
        "categorical posterior at episode end places more mass on the "
        "actual ground-truth hypothesis.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}: {OUT.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
