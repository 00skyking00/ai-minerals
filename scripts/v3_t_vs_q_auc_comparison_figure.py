"""Grouped horizontal bar: T vs Q per-base-learner pos-weighted AUC.

T values come from the v3 audit (pos-weighted ROC-AUC across spatial-block
CV folds, weighted by positives-per-fold):
    RF=0.809, LGBM=0.655, XGB=0.826, Stack OOF=0.970

Q values are computed from pop_fold_metrics_placer_quaternary.csv at
runtime so the script is reproducible end-to-end; the stack OOF row is
read directly from the same CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "derived" / "northern_sierra_placer"
Q_METRICS = DATA / "pop_fold_metrics_placer_quaternary.csv"
OUT_PATH = DATA / "v3_t_vs_q_auc_comparison.png"

# T pos-weighted AUCs (from F.2 audit of pop_fold_metrics_placer_tertiary.csv).
T_AUCS = {
    "RF":    0.809,
    "LGBM":  0.655,
    "XGB":   0.826,
    "Stack": 0.970,
}


def _pos_weighted(folds: pd.DataFrame, model: str) -> float:
    sub = folds[folds["model"] == model]
    w = sub["n_test_pos"].to_numpy()
    a = sub["roc_auc"].to_numpy()
    if w.sum() == 0:
        return float("nan")
    return float((a * w).sum() / w.sum())


def main() -> None:
    df = pd.read_csv(Q_METRICS)
    folds = df[df["fold_id"] != -1].copy()
    q_aucs = {
        "RF":    _pos_weighted(folds, "rf"),
        "LGBM":  _pos_weighted(folds, "lgbm"),
        "XGB":   _pos_weighted(folds, "xgb"),
        "Stack": float(df[df["model"] == "stack"].iloc[0]["roc_auc"]),
    }
    print("T (pos-wtd):", T_AUCS)
    print("Q (pos-wtd from CSV):", {k: round(v, 4) for k, v in q_aucs.items()})

    labels = ["RF", "LGBM", "XGB", "Stack"]
    t_vals = [T_AUCS[k] for k in labels]
    q_vals = [q_aucs[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(labels))
    height = 0.38
    t_color = "#b8741a"  # inferno-ish (T)
    q_color = "#1f8f5c"  # viridis-ish (Q)

    bars_t = ax.barh(
        y - height / 2, t_vals, height=height,
        color=t_color, edgecolor="white", linewidth=0.6,
        label="Tertiary",
    )
    bars_q = ax.barh(
        y + height / 2, q_vals, height=height,
        color=q_color, edgecolor="white", linewidth=0.6,
        label="Quaternary",
    )

    for bar, v in zip(bars_t, t_vals):
        ax.text(
            v + 0.008, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", ha="left",
            fontsize=8.5, color=t_color,
        )
    for bar, v in zip(bars_q, q_vals):
        ax.text(
            v + 0.008, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", ha="left",
            fontsize=8.5, color=q_color,
        )

    ax.axvline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(
        0.505, len(labels) - 0.5, "chance",
        fontsize=8, color="#888888", ha="left", va="top",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("ROC-AUC (positives-weighted across spatial-block CV folds)")

    ax.set_title(
        "v3 Tertiary vs Quaternary by model",
        fontsize=11, loc="left", pad=20,
    )
    ax.text(
        0.0, 1.02,
        "Quaternary is materially harder than Tertiary; stacking partly "
        "closes the gap.",
        transform=ax.transAxes, fontsize=9, color="#444444",
    )

    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.legend(loc="lower right", frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
