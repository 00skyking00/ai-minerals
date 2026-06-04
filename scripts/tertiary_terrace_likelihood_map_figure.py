"""Tertiary terrace likelihood feature map for data_overview.qmd.

Builds a 1400x1000 px PNG showing the tertiary_terrace_likelihood feature
(geometric mean of TPI-high, slope-low, not-Quaternary-alluvium) over the
northern Sierra placer AOI, with hydraulic-mine-pit polygons outlined.

Output: data/derived/northern_sierra_placer/tertiary_terrace_likelihood_map.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from ai_minerals.grid import build_grid
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
WCRS = REGION.working_crs

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PARQUET = REPO_ROOT / "data" / "derived" / "features_northern_sierra_placer_250m.parquet"
OUT_PATH = REPO_ROOT / "data" / "derived" / "northern_sierra_placer" / "tertiary_terrace_likelihood_map.png"


def main() -> Path:
    df = pd.read_parquet(
        FEATURES_PARQUET,
        columns=["row", "col", "x", "y", "tertiary_terrace_likelihood"],
    )

    grid = build_grid(REGION.aoi, resolution_m=250, working_crs=WCRS)
    nrows, ncols = grid.shape

    raster = np.full((nrows, ncols), np.nan, dtype=np.float32)
    raster[df["row"].to_numpy(), df["col"].to_numpy()] = (
        df["tertiary_terrace_likelihood"].to_numpy()
    )

    # Working-CRS bounds (EPSG:3310 meters), then reproject to EPSG:4326 for extent.
    xmin_m, ymin_m, xmax_m, ymax_m = grid.bounds
    to_4326 = Transformer.from_crs(WCRS, "EPSG:4326", always_xy=True)
    # Reproject the four corners and take the lon/lat bounding box. Equal-area to
    # geographic is non-linear, so this is an approximation; for a 2.5 deg wide
    # AOI it's close enough to read as a map.
    corners_lon, corners_lat = to_4326.transform(
        [xmin_m, xmax_m, xmin_m, xmax_m],
        [ymin_m, ymin_m, ymax_m, ymax_m],
    )
    lon_min, lon_max = min(corners_lon), max(corners_lon)
    lat_min, lat_max = min(corners_lat), max(corners_lat)

    fig, ax = plt.subplots(figsize=(1400 / 150, 1000 / 150), dpi=150)
    im = ax.imshow(
        raster,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="upper",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )

    pits = gpd.read_file(REGION.raw_paths["hydraulic_pits"]).to_crs("EPSG:4326")
    pits.boundary.plot(ax=ax, color="#0b3d91", linewidth=0.6, alpha=0.9, zorder=5)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(
        "Tertiary terrace likelihood: TPI-high ∧ slope-low ∧ "
        "not-Quaternary-alluvium geometric mean",
        fontsize=12, y=0.98,
    )
    ax.set_title(
        "The hot bands match the hydraulic-pit cluster, which is the design "
        "intent. The geometric mean zeros out anywhere any single signature "
        "is missing.",
        fontsize=9, color="#333333", loc="center", pad=8,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("tertiary_terrace_likelihood (0 to 1)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
