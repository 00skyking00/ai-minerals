"""Regenerate the Mother Lode prospectivity figure with cleaned labels.

Two changes from `motherlode_cover_figures.py`:

1. Predictions: read from `model_predictions_motherlode_cleaned.parquet`
   (RF refit on the Cox-Singer-cleaned positive set, n=6,149).
2. Overlay: plot the cleaned positive cells from the feature frame
   (cells with is_orogenic_gold == 1), not all 13,305 commodity-Au
   MRDS records. The cleaned cells are the actual ground truth the
   v3.1 model was trained against.

Output: data/derived/motherlode/fig_prospectivity_motherlode_cleaned.png
(side-by-side comparable to the existing fig_prospectivity_motherlode.png).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
OUT = DATA_DERIVED / "motherlode" / "fig_prospectivity_motherlode_cleaned.png"


def main() -> None:
    pred = pd.read_parquet(
        DATA_DERIVED / "motherlode" / "model_predictions_motherlode_cleaned.parquet"
    )
    print(f"cleaned predictions: {pred.shape}")

    # Pivot to a 2D map for imshow.
    n_rows = pred["row"].max() + 1
    n_cols = pred["col"].max() + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grid[pred["row"].to_numpy(), pred["col"].to_numpy()] = pred["p_rf_no_count"].to_numpy()

    x_min, x_max = pred["x"].min(), pred["x"].max()
    y_min, y_max = pred["y"].min(), pred["y"].max()

    # Cleaned positive cells from the feature frame — these are the actual
    # ground truth the v3.1 model was trained on.
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    pos = df[df["is_orogenic_gold"] == 1]
    print(f"cleaned positive cells: {len(pos):,}")

    fig, ax = plt.subplots(figsize=(9, 10))
    extent = (x_min, x_max, y_min, y_max)
    im = ax.imshow(
        grid,
        extent=extent,
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Random Forest prospectivity (no count features)")

    ax.scatter(
        pos["x"],
        pos["y"],
        s=4,
        c="white",
        edgecolor="black",
        linewidth=0.2,
        alpha=0.7,
        label=f"Cleaned orogenic-Au cells (n={len(pos):,})",
    )

    ax.set_title("Mother Lode v3.1: RF prospectivity with Cox-Singer-cleaned orogenic-Au cells")
    ax.set_xlabel("EPSG:3310 easting (m)")
    ax.set_ylabel("EPSG:3310 northing (m)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
