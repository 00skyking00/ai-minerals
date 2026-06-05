"""Per-fold ROC-AUC scatter for the v3 Quaternary spatial-block CV.

One column of dots per base learner (RF, LGBM, XGB), with a black bar at
the model mean. The stack OOF AUC (0.832) is plotted as a separate solid
black bar at the right.

Source: data/derived/northern_sierra_placer/pop_fold_metrics_placer_quaternary.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "derived" / "northern_sierra_placer"
METRICS_CSV = DATA / "pop_fold_metrics_placer_quaternary.csv"
OUT_PATH = DATA / "v3_q_per_fold_auc.png"

MODELS = ["rf", "lgbm", "xgb"]
MODEL_LABELS = {"rf": "RF", "lgbm": "LGBM", "xgb": "XGB"}
MODEL_COLOR = {"rf": "#3a6ea5", "lgbm": "#7a3aa5", "xgb": "#a55a3a"}


def main() -> None:
    df = pd.read_csv(METRICS_CSV)
    folds = df[df["fold_id"] != -1].copy()
    stack_oof = float(df[df["model"] == "stack"].iloc[0]["roc_auc"])

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rng = np.random.default_rng(7)
    x_positions = [0, 1, 2, 3]  # rf, lgbm, xgb, stack
    means: dict[str, float] = {}
    for i, m in enumerate(MODELS):
        sub = folds[folds["model"] == m]
        aucs = sub["roc_auc"].to_numpy()
        means[m] = float(aucs.mean())
        # Jitter horizontally so dots don't stack.
        jitter = rng.uniform(-0.18, 0.18, size=len(aucs))
        ax.scatter(
            np.full_like(aucs, x_positions[i], dtype=float) + jitter,
            aucs,
            s=18, alpha=0.55,
            color=MODEL_COLOR[m],
            edgecolors="white", linewidths=0.4,
            zorder=3,
        )
        # Mean bar.
        ax.hlines(
            means[m], x_positions[i] - 0.28, x_positions[i] + 0.28,
            colors="black", linewidth=2.0, zorder=4,
        )
        ax.text(
            x_positions[i], means[m] + 0.018,
            f"mean {means[m]:.3f}",
            ha="center", va="bottom", fontsize=8.5, color="black",
            zorder=5,
        )

    # Stack OOF as a solid black bar (single value, not per-fold).
    ax.hlines(
        stack_oof, x_positions[3] - 0.28, x_positions[3] + 0.28,
        colors="black", linewidth=3.5, zorder=4,
    )
    ax.scatter(
        [x_positions[3]], [stack_oof],
        s=70, color="black", marker="s", zorder=5,
        edgecolors="white", linewidths=0.6,
    )
    ax.text(
        x_positions[3], stack_oof + 0.018,
        f"OOF {stack_oof:.3f}",
        ha="center", va="bottom", fontsize=8.5, color="black",
        fontweight="bold",
        zorder=5,
    )

    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(
        3.45, 0.5, "chance",
        fontsize=8, color="#888888", ha="right", va="bottom",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS] + ["Stack (OOF)"])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.set_title(
        "v3 Quaternary spatial-block CV per-fold AUC by model",
        fontsize=11, loc="left", pad=20,
    )
    subtitle = (
        f"RF mean {means['rf']:.3f}, LGBM {means['lgbm']:.3f}, "
        f"XGB {means['xgb']:.3f}. Stacking OOF: {stack_oof:.3f}."
    )
    ax.text(
        0.0, 1.02, subtitle,
        transform=ax.transAxes, fontsize=9, color="#444444",
    )

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    print(f"  rf folds: {(folds['model']=='rf').sum()}, mean={means['rf']:.4f}")
    print(f"  lgbm folds: {(folds['model']=='lgbm').sum()}, mean={means['lgbm']:.4f}")
    print(f"  xgb folds: {(folds['model']=='xgb').sum()}, mean={means['xgb']:.4f}")
    print(f"  stack OOF: {stack_oof:.4f}")


if __name__ == "__main__":
    main()
