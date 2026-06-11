"""Generate the cover-page hero figure for Mother Lode v3.

A prospectivity heatmap (RF no-count scores) with MRDS Au-bearing
records overlaid. The map should visibly show the Mother Lode Belt as
a high-prospectivity strip running N-S through the foothills.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_minerals.regions.motherlode import MOTHERLODE
from ai_minerals.data._common import DATA_RAW

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/motherlode/fig_prospectivity_motherlode.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    pred = pd.read_parquet(DATA_DERIVED / "motherlode" / "model_predictions_motherlode.parquet")
    print(f"predictions: {pred.shape}")

    # Pivot to a 2D map for imshow.
    n_rows = pred["row"].max() + 1
    n_cols = pred["col"].max() + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grid[pred["row"].to_numpy(), pred["col"].to_numpy()] = pred["p_rf_no_count"].to_numpy()

    # Cell size 500 m, working CRS is EPSG:3310 (CA Albers). Use the actual
    # x/y from pred to set extent.
    x_min, x_max = pred["x"].min(), pred["x"].max()
    y_min, y_max = pred["y"].min(), pred["y"].max()

    # Load the MRDS Au records for overlay.
    mrds_path = DATA_RAW / "mrds" / "mrds_motherlode.gpkg"
    mrds = gpd.read_file(mrds_path)
    mrds = mrds.to_crs("EPSG:3310")
    au_mask = mrds["commodity"].str.contains("au|gold", case=False, regex=True, na=False)
    au_pts = mrds[au_mask]
    print(f"Au-bearing MRDS points: {len(au_pts):,}")

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
        au_pts.geometry.x,
        au_pts.geometry.y,
        s=4,
        c="white",
        edgecolor="black",
        linewidth=0.2,
        alpha=0.7,
        label=f"MRDS Au records (n={len(au_pts):,})",
    )

    ax.set_title("Mother Lode v3 — RF prospectivity, MRDS Au-bearing records overlay")
    ax.set_xlabel("EPSG:3310 easting (m)")
    ax.set_ylabel("EPSG:3310 northing (m)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
