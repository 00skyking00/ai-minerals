"""Generate v3 ablation AUC drop figure.

Grouped horizontal bar chart comparing v3 Tertiary base learners (RF, XGB) with
and without the hydraulic_pit_proximity_m_buffered feature.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/sky/src/learning/ai-minerals")
INPUT = REPO / "data/derived/northern_sierra_placer/ablation_no_pit_proximity.json"
OUTPUT = REPO / "data/derived/northern_sierra_placer/v3_ablation_no_pit_proximity.png"


def main() -> None:
    data = json.loads(INPUT.read_text())

    rf_base = float(data["baseline"]["rf_auc_mean_v3"])
    xgb_base = float(data["baseline"]["xgb_auc_mean_v3"])
    rf_abl = float(data["rf"]["plain_mean_auc"])
    xgb_abl = float(data["xgb"]["plain_mean_auc"])

    rf_drop_pp = (rf_base - rf_abl) * 100.0
    xgb_drop_pp = (xgb_base - xgb_abl) * 100.0

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    fig, ax = plt.subplots(figsize=(8.1, 4.05), dpi=150)
    # Slightly oversized so bbox_inches='tight' lands near 1200x600 after cropping.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    models = ["RF", "XGB"]
    y = np.arange(len(models))
    bar_h = 0.34

    base_vals = [rf_base, xgb_base]
    abl_vals = [rf_abl, xgb_abl]
    drops = [rf_drop_pp, xgb_drop_pp]

    color_base = "#1f4e79"
    color_abl = "#6fa8dc"

    bars_base = ax.barh(
        y - bar_h / 2,
        base_vals,
        height=bar_h,
        color=color_base,
        label="baseline (with hydraulic_pit_proximity_m_buffered)",
        edgecolor="white",
    )
    bars_abl = ax.barh(
        y + bar_h / 2,
        abl_vals,
        height=bar_h,
        color=color_abl,
        label="ablation (feature removed)",
        edgecolor="white",
    )

    for i, (b, a, d) in enumerate(zip(base_vals, abl_vals, drops)):
        ax.text(
            b + 0.006,
            y[i] - bar_h / 2,
            f"{b:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )
        ax.text(
            a + 0.006,
            y[i] + bar_h / 2,
            f"{a:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )
        # drop annotation: place to the right of the baseline value, on the group's center line
        ax.text(
            b + 0.065,
            y[i],
            f"-{d:.1f} pp",
            va="center",
            ha="left",
            fontsize=10.5,
            color="#8b0000",
            fontweight="bold",
        )

    ax.axvline(0.5, color="#888888", linewidth=1.0, linestyle="--", zorder=1)
    # "chance" annotation: anchored in axes-fraction coords so invert_yaxis doesn't flip it
    ax.text(
        0.005,
        0.04,
        "chance (0.5)",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        ha="left",
        va="bottom",
    )

    ax.set_xlim(0.5, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Spatial-block CV ROC-AUC", fontsize=10)
    ax.invert_yaxis()

    ax.grid(axis="x", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "v3 Tertiary base learners with and without hydraulic_pit_proximity_m_buffered",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.98,
    )
    fig.text(
        0.02,
        0.91,
        "Ablation measures the marginal contribution of the feature to each model's spatial-block CV AUC.",
        fontsize=9.5,
        color="#444444",
        ha="left",
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.14)

    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {OUTPUT}")
    print(f"rf_base={rf_base:.4f} rf_abl={rf_abl:.4f} drop={rf_drop_pp:.2f} pp")
    print(f"xgb_base={xgb_base:.4f} xgb_abl={xgb_abl:.4f} drop={xgb_drop_pp:.2f} pp")


if __name__ == "__main__":
    main()
