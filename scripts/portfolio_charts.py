"""Generate the two portfolio hero charts for Chapters 4 and 5.

* `lawley_waterfall.png`  — Chapter 4 (reproductions/audits).
  Where Lawley 2022's published AUC 0.983 comes from. Four bars:
  published (with label leak) → leak corrected → 2-D blocked CV →
  cross-continent transfer.

* `cross_region_top1.png` — Chapter 5 (cross-region experiments).
  Grouped bar chart: top-1% capture for RF/GBM vs DevNet on each of
  five regions. Four out of five regions are RF-favored; only the
  Curnamona tuning dataset favors DevNet.

Numbers come from `data/derived/lawley/path1*.json`,
`data/derived/lawley/path2_tightened_eval_metrics.json`,
`data/derived/{eastak,arizona,us_carbonatite_ree}/...json`, and the
verbatim Curnamona row in cross_region.qmd.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("data/derived/portfolio_charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def lawley_waterfall() -> None:
    """Stacked-style waterfall: published AUC 0.983 → 0.71."""
    labels = [
        "Published\n(GBM, 1-D CV,\nlabel leak in)",
        "Label-leak\nremoved",
        "+ 2-D spatial\nblocking",
        "+ Cross-\ncontinent\ntransfer",
    ]
    values = [0.983, 0.959, 0.868, 0.709]
    drops = [None, -0.024, -0.091, -0.159]
    colors = ["#2c7bb6", "#5a8fbd", "#9da9b3", "#d7301f"]

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=160)
    bars = ax.bar(labels, values, color=colors, width=0.62,
                  edgecolor="#222", linewidth=0.6)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("AUC (test)", fontsize=11)
    ax.set_title(
        "Lawley 2022 continental Zn-Pb prospectivity:\n"
        "where the published AUC 0.983 comes from",
        fontsize=12.5, pad=14, loc="left",
    )

    # Annotate each bar with value
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate drops between adjacent bars
    for i, drop in enumerate(drops):
        if drop is None:
            continue
        x0 = bars[i - 1].get_x() + bars[i - 1].get_width()
        x1 = bars[i].get_x()
        y0 = values[i - 1]
        y1 = values[i]
        # Drop arrow (going down)
        ax.annotate(
            "",
            xy=((x0 + x1) / 2, y1), xytext=((x0 + x1) / 2, y0),
            arrowprops=dict(arrowstyle="->", color="#7d0202", lw=1.3),
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            f"{drop * 100:+.1f}\npp",
            ha="center", va="center", fontsize=9.5, color="#7d0202",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#7d0202",
                       lw=0.8),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    out = OUT_DIR / "lawley_waterfall.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def cross_region_top1() -> None:
    """Grouped bar: RF/GBM vs DevNet top-1% capture per region."""
    # Verbatim from cross_region.qmd's scorecard table.
    regions = [
        "Curnamona REE\n(7 deposits, LOO,\nDEEP-SEAM tuning)",
        "US Western\nCarbonatite Belt\n(14, LOO)",
        "Tanacross\nporphyry-Cu\n(45, OOF)",
        "Arizona\nporphyry-Cu\n(191, OOF)",
        "Lawley\ncontinental MVT\n(2,027, 2-D blocked OOF)",
    ]
    rf_gbm = [0.0, 85.7, 22.2, 23.0, 28.4]
    devnet = [43.0, 71.4, 13.3, 8.9, 0.0]

    x = np.arange(len(regions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    bars_rf = ax.bar(
        x - width / 2, rf_gbm, width,
        label="Random Forest / Gradient Boosting",
        color="#2ecc40", edgecolor="#1f7a23", linewidth=0.6,
    )
    bars_dn = ax.bar(
        x + width / 2, devnet, width,
        label="DEEP-SEAM (DevNet)",
        color="#d7301f", edgecolor="#7d0202", linewidth=0.6,
    )

    ax.set_ylabel("Top-1% capture rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "DEEP-SEAM transferability test across 5 regions\n"
        "Tree-based methods win 4 of 5; DevNet wins only its tuning dataset",
        fontsize=12.5, pad=14, loc="left",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=9)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    # Value labels above each bar
    for bars in (bars_rf, bars_dn):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1.2,
                f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    # Highlight the one region where DevNet wins
    ax.axvspan(-0.5, 0.5, color="#fdebd0", alpha=0.5, zorder=0)
    ax.text(
        0, 98, "DevNet wins",
        ha="center", va="top", fontsize=9, color="#a04000",
        style="italic",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(regions) - 0.5)

    plt.tight_layout()
    out = OUT_DIR / "cross_region_top1.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    lawley_waterfall()
    cross_region_top1()
