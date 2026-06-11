"""Generate the spatial-block-size adjacency diagnostic chart.

Reads block_size_ablation_diagnostic.csv and produces a two-panel figure
showing 8-neighbour adjacency correlation and fold-count / zero-positive
trade-off vs. block size for the northern Sierra placer v3.5 work.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DERIVED = Path(
    "/home/sky/src/learning/ai-minerals/data/derived/northern_sierra_placer"
)
CSV_PATH = DERIVED / "block_size_ablation_diagnostic.csv"
OUT_PATH = DERIVED / "block_size_adjacency_diagnostic.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH).sort_values("block_size_m").reset_index(drop=True)
    df["block_size_km"] = df["block_size_m"] / 1000.0

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5), facecolor="white"
    )

    # Left panel: adjacency correlation vs block size
    ax_left.plot(
        df["block_size_km"],
        df["adjacency_pos_corr"],
        color="crimson",
        marker="o",
        markersize=8,
        linewidth=2,
        zorder=3,
    )
    ax_left.axhline(
        0.5,
        color="grey",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )
    ax_left.text(
        df["block_size_km"].min() + 0.3,
        0.51,
        "acceptable threshold",
        color="grey",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    ax_left.set_xlabel("Spatial-CV block size (km)")
    ax_left.set_ylabel("8-neighbour adjacency correlation\nof positive counts")
    ax_left.set_title("Adjacency correlation of positives", fontsize=11)
    ax_left.set_xticks(df["block_size_km"].tolist())
    ax_left.set_ylim(0.35, 0.95)
    ax_left.grid(True, color="lightgrey", linewidth=0.6, zorder=1)
    ax_left.set_facecolor("white")
    for spine in ax_left.spines.values():
        spine.set_color("grey")

    # Annotations on 20 km and 25 km points
    row_20 = df.loc[df["block_size_m"] == 20000].iloc[0]
    row_25 = df.loc[df["block_size_m"] == 25000].iloc[0]
    ax_left.annotate(
        f"v3 default ({row_20['adjacency_pos_corr']:.2f})",
        xy=(row_20["block_size_km"], row_20["adjacency_pos_corr"]),
        xytext=(row_20["block_size_km"] - 4.5, row_20["adjacency_pos_corr"] + 0.08),
        fontsize=9,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ax_left.annotate(
        f"v3.5 candidate ({row_25['adjacency_pos_corr']:.2f})",
        xy=(row_25["block_size_km"], row_25["adjacency_pos_corr"]),
        xytext=(row_25["block_size_km"] - 6.0, row_25["adjacency_pos_corr"] - 0.08),
        fontsize=9,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    # Right panel: n_folds + blocks_with_zero_pos vs block size
    ax_folds = ax_right
    ax_folds.plot(
        df["block_size_km"],
        df["n_folds"],
        color="steelblue",
        marker="o",
        markersize=7,
        linewidth=2,
        label="n_folds",
        zorder=3,
    )
    ax_folds.set_xlabel("Spatial-CV block size (km)")
    ax_folds.set_ylabel("Number of folds", color="steelblue")
    ax_folds.tick_params(axis="y", labelcolor="steelblue")
    ax_folds.set_xticks(df["block_size_km"].tolist())
    ax_folds.grid(True, color="lightgrey", linewidth=0.6, zorder=1)
    ax_folds.set_facecolor("white")
    for spine in ax_folds.spines.values():
        spine.set_color("grey")

    ax_zero = ax_folds.twinx()
    ax_zero.plot(
        df["block_size_km"],
        df["blocks_with_zero_pos"],
        color="darkorange",
        marker="s",
        markersize=7,
        linewidth=2,
        label="blocks_with_zero_pos",
        zorder=3,
    )
    ax_zero.set_ylabel("Blocks with zero positives", color="darkorange")
    ax_zero.tick_params(axis="y", labelcolor="darkorange")
    for spine in ax_zero.spines.values():
        spine.set_color("grey")

    ax_folds.set_title("Folds vs. zero-positive blocks", fontsize=11)

    # Titles
    fig.suptitle(
        "Spatial-CV block size: the empirical trade-off",
        fontsize=13,
        y=1.02,
    )
    fig.text(
        0.5,
        0.96,
        "Smaller blocks have more folds but adjacent blocks correlate, defeating "
        "spatial CV's point.\nLarger blocks decorrelate but waste statistical "
        "power on zero-positive blocks.",
        ha="center",
        va="top",
        fontsize=10,
        color="dimgrey",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
