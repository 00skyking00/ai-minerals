"""Build the Phase 1 knowledge-driven prospectivity index figure.

Renders the weighted-sum Phase 1 score (11 features) on a percentile-rank
stretch, with the 158 Orlando 2016 hydraulic-mine pit polygons outlined in
cyan and the 7 anchor districts marked as red dots.

Output: data/derived/northern_sierra_placer/phase1_score_map.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS


REGION = NORTHERN_SIERRA_PLACER
WCRS = REGION.working_crs  # EPSG:3310 California Albers

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "derived" / "northern_sierra_placer"
PHASE1_PARQUET = DATA / "phase1_index_250m.parquet"
OUT_PNG = DATA / "phase1_score_map.png"


_ANCHOR_LABEL_OFFSETS = {
    "Malakoff Diggins / North Bloomfield": (6, 8),
    "North San Juan":                      (-95, 4),
    "Dutch Flat":                          (6, -12),
    "You Bet":                             (-50, 10),
    "Iowa Hill":                           (-65, 0),
    "Forest Hill":                         (6, -12),
    "Michigan Bluff":                      (6, 6),
}


def _anchor_xy() -> pd.DataFrame:
    pts = gpd.GeoDataFrame(
        {"name": list(ANCHOR_DISTRICTS.keys())},
        geometry=[Point(lon, lat) for (lon, lat) in ANCHOR_DISTRICTS.values()],
        crs="EPSG:4326",
    ).to_crs(WCRS)
    return pd.DataFrame({
        "name": pts["name"].to_list(),
        "x": pts.geometry.x.to_numpy(),
        "y": pts.geometry.y.to_numpy(),
    })


def _label_anchors(ax, anchor_xy: pd.DataFrame, *, fontsize: int = 9) -> None:
    for _, row in anchor_xy.iterrows():
        ax.plot(row["x"], row["y"], "o", color="crimson", markersize=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=10)
        short = row["name"].split(" / ")[0]
        dx, dy = _ANCHOR_LABEL_OFFSETS.get(row["name"], (6, 6))
        ax.annotate(short, (row["x"], row["y"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=fontsize,
                    color="black", fontweight="semibold",
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
                    zorder=11)


def build_figure() -> Path:
    df = pd.read_parquet(PHASE1_PARQUET)

    # Percentile-rank stretch: ties get the same rank, mapped to [0, 1].
    df["pct_rank"] = df["phase1_score"].rank(method="min", pct=True).astype(np.float32)

    nrows = int(df["row"].max() + 1)
    ncols = int(df["col"].max() + 1)
    grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    grid[df["row"].to_numpy(), df["col"].to_numpy()] = df["pct_rank"].to_numpy()

    # Cell-size half-step for a clean extent.
    rx_min = df["x"].min() - 125
    rx_max = df["x"].max() + 125
    ry_min = df["y"].min() - 125
    ry_max = df["y"].max() + 125

    anchor_xy = _anchor_xy()

    # Target ~1400x1000 px at dpi 150 -> ~9.33 x 6.67 in. The AOI itself
    # is portrait (aspect ~0.65); place it on the left and keep room on
    # the right for the colorbar and breathing space for the title.
    fig = plt.figure(figsize=(10.0, 7.5), facecolor="white")
    ax = fig.add_axes([0.04, 0.04, 0.55, 0.82])
    cax = fig.add_axes([0.64, 0.10, 0.022, 0.70])

    im = ax.imshow(
        grid,
        extent=(rx_min, rx_max, ry_min, ry_max),
        origin="upper",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Phase 1 score (percentile rank)", fontsize=10)
    cbar.set_ticks([0.0, 0.5, 0.9, 0.99, 1.0])
    cbar.set_ticklabels(["0", "median", "p90", "p99", "max"])

    # Hydraulic-mine pit polygons in cyan, outlines only.
    pits = gpd.read_file(REGION.raw_paths["hydraulic_pits"]).to_crs(WCRS)
    pits.boundary.plot(ax=ax, color="cyan", linewidth=0.6, alpha=0.95, zorder=4)

    _label_anchors(ax, anchor_xy)

    ax.set_xlim(rx_min, rx_max)
    ax.set_ylim(ry_min, ry_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#9ca3af")
        spine.set_linewidth(0.5)

    fig.text(
        0.04, 0.945,
        "Phase 1 knowledge-driven placer-Au prospectivity index",
        fontsize=14, fontweight="semibold", ha="left", va="bottom",
    )
    fig.text(
        0.04, 0.905,
        "Weighted-sum composite of 11 features. All seven anchor districts "
        "land in the top decile;",
        fontsize=10, color="#374151", ha="left", va="top",
    )
    fig.text(
        0.04, 0.882,
        "the index is the gate v2 and v3 supervised passes have to clear.",
        fontsize=10, color="#374151", ha="left", va="top",
    )

    # Side panel: short legend keys for pit outlines and anchor markers.
    fig.text(
        0.68, 0.84,
        "Map elements",
        fontsize=10, fontweight="semibold", color="#111827",
    )
    fig.text(
        0.68, 0.815,
        "Cyan outlines: 158 hydraulic-\nmine pit polygons\n(Orlando 2016).",
        fontsize=9, color="#374151", va="top",
    )
    fig.text(
        0.68, 0.74,
        "Red dots: 7 anchor districts\nused as the held-out\nvalidation gate.",
        fontsize=9, color="#374151", va="top",
    )

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT_PNG


if __name__ == "__main__":
    out = build_figure()
    print(f"Wrote {out}")
