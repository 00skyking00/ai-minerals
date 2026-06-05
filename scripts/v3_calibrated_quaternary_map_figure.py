"""Generate the v3 Quaternary calibrated probability map figure.

Displays the calibrated p_quaternary raster on a percentile-rank stretch,
overlays NHD-HR flowlines (Quaternary placers are channel-aligned), and
marks the five Lindgren Quaternary-named secondary diggings as orange
dots with text labels.

Different colormap from the Tertiary map (viridis here, inferno there) so
the two figures read as different populations at a glance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patheffects import withStroke


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RASTER_PATH = (
    REPO_ROOT
    / "data" / "derived" / "northern_sierra_placer"
    / "prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"
)
NHD_PATH = REPO_ROOT / "data" / "raw" / "nhd_hr" / "nhd_flowlines_northern_sierra.gpkg"
OUT_PATH = (
    REPO_ROOT
    / "data" / "derived" / "northern_sierra_placer"
    / "v3_calibrated_quaternary_map.png"
)

# The 5 Q-named held-out districts to highlight on the map.
# Coordinates taken from src/ai_minerals/regions/_northern_sierra_lindgren_secondaries.py.
Q_DISTRICTS: dict[str, tuple[float, float]] = {
    "Downieville":   (-120.825, 39.560),
    "Goodyears Bar": (-120.890, 39.531),
    "Sierra City":   (-120.629, 39.566),
    "Brandy City":   (-120.971, 39.523),
    "Camptonville":  (-121.045, 39.452),
}


def percentile_rank(values: np.ndarray) -> np.ndarray:
    n = values.size
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return ranks


def main() -> None:
    with rasterio.open(RASTER_PATH) as src:
        arr = src.read(1).astype(np.float32)
        bounds = src.bounds

    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]

    ranks = percentile_rank(finite_vals)
    rank_img = np.full(arr.shape, np.nan, dtype=np.float32)
    rank_img[finite_mask] = ranks

    # Load NHD flowlines, clip to raster bounds for speed.
    nhd = gpd.read_file(NHD_PATH, bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top))
    if nhd.crs is None or nhd.crs.to_epsg() != 4326:
        nhd = nhd.to_crs(epsg=4326)
    print(f"Loaded {len(nhd)} NHD flowline segments inside raster bbox")

    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    im = ax.imshow(
        rank_img,
        extent=extent,
        origin="upper",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # NHD flowlines: darker blue, thin.
    nhd.plot(ax=ax, color="#0b3d91", linewidth=0.35, alpha=0.85, zorder=4)

    # Q districts: orange dots + white-outlined labels.
    label_offsets: dict[str, tuple[float, float, str]] = {
        "Downieville":   (0.020, 0.018, "left"),
        "Goodyears Bar": (-0.020, -0.020, "right"),
        "Sierra City":   (0.020, 0.018, "left"),
        "Brandy City":   (-0.020, 0.018, "right"),
        "Camptonville":  (-0.020, -0.020, "right"),
    }
    label_stroke = [withStroke(linewidth=2.0, foreground="black")]
    for name, (lon, lat) in Q_DISTRICTS.items():
        ax.plot(
            lon, lat,
            marker="o", markersize=7,
            markerfacecolor="#ff8a00",
            markeredgecolor="white",
            markeredgewidth=0.9,
            linestyle="none",
            zorder=6,
        )
        dx, dy, ha = label_offsets.get(name, (0.012, 0.008, "left"))
        ax.text(
            lon + dx, lat + dy, name,
            color="white", fontsize=9, fontweight="bold",
            ha=ha, va="center",
            path_effects=label_stroke,
            zorder=7,
        )

    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=8)
    ax.grid(True, color="lightgrey", linewidth=0.3, alpha=0.5)
    ax.set_aspect("equal")

    fig.suptitle(
        "v3 Quaternary calibrated probability, percentile rank across AOI",
        fontsize=13, fontweight="bold",
        x=0.06, y=0.965, ha="left",
    )
    fig.text(
        0.06, 0.940,
        "NHD flowlines overlaid; Lindgren PP73 Quaternary-named held-out "
        "districts in orange.",
        fontsize=10, color="#333333", ha="left", va="top",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(
        "Quaternary calibrated probability (percentile rank)", fontsize=9
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    cbar.set_ticklabels(["0", "p25", "p50", "p75", "p90", "max"])

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")

    # Sanity: report Q-district decile ranks.
    with rasterio.open(RASTER_PATH) as src:
        for name, (lon, lat) in Q_DISTRICTS.items():
            row, col = src.index(lon, lat)
            v = arr[row, col]
            if not np.isfinite(v):
                print(f"  {name}: NaN cell")
                continue
            r = rank_img[row, col]
            decile = int((1.0 - r) * 10)
            print(f"  {name}: rank={r:.4f} -> decile {decile}")

    print(
        f"raster: {arr.shape}, finite cells: {finite_mask.sum()}, "
        f"zero cells: {(finite_vals == 0).sum()}, "
        f"max calibrated p = {finite_vals.max():.5f}"
    )


if __name__ == "__main__":
    main()
