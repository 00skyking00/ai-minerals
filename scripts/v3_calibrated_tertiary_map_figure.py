"""Generate the v3 Tertiary calibrated probability map figure.

Displays the calibrated p_tertiary raster on a percentile-rank stretch (so the
long zero tail of the calibrated probability doesn't wash out the high-value
cells), overlays the Orlando 2016 hydraulic-mine pit polygons that supplied
the training labels, and marks the seven anchor districts as red dots with
text labels.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patheffects import withStroke

from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS


RASTER_PATH = Path(
    "/home/sky/src/learning/ai-minerals/data/derived/northern_sierra_placer/"
    "prospectivity_placer_placer_tertiary_250m_calibrated_4326.tif"
)
PITS_PATH = Path(
    "/home/sky/src/learning/ai-minerals/data/raw/hydraulic_pits/"
    "hydraulic_mine_pits_ca.gpkg"
)
OUT_PATH = Path(
    "/home/sky/src/learning/ai-minerals/data/derived/northern_sierra_placer/"
    "v3_calibrated_tertiary_map.png"
)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map each finite value to its percentile rank in [0, 1].

    Uses argsort indexing; ties resolved by sort position (close enough for a
    display stretch, no need for full average-rank).
    """
    n = values.size
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return ranks


def main() -> None:
    # Load raster.
    with rasterio.open(RASTER_PATH) as src:
        arr = src.read(1).astype(np.float32)
        bounds = src.bounds
        crs = src.crs

    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]

    # Percentile-rank stretch over the finite cells.
    ranks = percentile_rank(finite_vals)
    rank_img = np.full(arr.shape, np.nan, dtype=np.float32)
    rank_img[finite_mask] = ranks

    # Load pit polygons.
    pits = gpd.read_file(PITS_PATH)
    if pits.crs is None or pits.crs.to_epsg() != 4326:
        pits = pits.to_crs(epsg=4326)
    print(f"Loaded {len(pits)} pit polygons from {PITS_PATH.name}")

    # Tight viewport: union bounds of pit polygons + 0.15 deg pad on all sides.
    # The raster spans a much larger AOI; without cropping we get a big zoomed-
    # out frame plus a band of invalid/empty cells on the east edge.
    pit_minx, pit_miny, pit_maxx, pit_maxy = pits.total_bounds
    pad = 0.15
    view_lon_min = pit_minx - pad
    view_lon_max = pit_maxx + pad
    view_lat_min = pit_miny - pad
    view_lat_max = pit_maxy + pad
    print(
        f"Pit union bounds: lon [{pit_minx:.4f}, {pit_maxx:.4f}], "
        f"lat [{pit_miny:.4f}, {pit_maxy:.4f}]"
    )
    print(
        f"View extent (+{pad} pad): lon [{view_lon_min:.4f}, "
        f"{view_lon_max:.4f}], lat [{view_lat_min:.4f}, {view_lat_max:.4f}]"
    )

    # Figure.
    # Figure is sized so bbox_inches="tight" trims to roughly the target
    # ~1400x1000. The AOI itself is taller than it is wide, so the final png
    # ends up taller than 1000 px even with width capped near 1400.
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    im = ax.imshow(
        rank_img,
        extent=extent,
        origin="upper",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # Pit polygon outlines in cyan, no fill.
    pits.boundary.plot(ax=ax, color="#00e5ff", linewidth=0.55, alpha=0.95)

    # Anchor districts: red dots + labels.
    # Per-anchor label offsets (dx, dy in degrees) + horizontal alignment to
    # keep labels off the dot and off neighbouring anchors.
    label_offsets: dict[str, tuple[float, float, str]] = {
        "Malakoff Diggins / North Bloomfield": (0.020, 0.020, "left"),
        "North San Juan":                      (-0.020, 0.020, "right"),
        "Dutch Flat":                          (0.020, -0.018, "left"),
        "You Bet":                             (-0.020, 0.010, "right"),
        "Iowa Hill":                           (-0.020, 0.010, "right"),
        "Forest Hill":                         (-0.020, -0.022, "right"),
        "Michigan Bluff":                      (0.020, 0.010, "left"),
    }
    label_stroke = [withStroke(linewidth=2.0, foreground="black")]
    for name, (lon, lat) in ANCHOR_DISTRICTS.items():
        ax.plot(
            lon,
            lat,
            marker="o",
            markersize=7,
            markerfacecolor="#ff2d2d",
            markeredgecolor="white",
            markeredgewidth=0.9,
            linestyle="none",
            zorder=5,
        )
        dx, dy, ha = label_offsets.get(name, (0.012, 0.008, "left"))
        label = name.split(" / ")[0]
        ax.text(
            lon + dx,
            lat + dy,
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            ha=ha,
            va="center",
            path_effects=label_stroke,
            zorder=6,
        )

    # Crop to the pit-cluster viewport so the action fills the frame and the
    # dead band on the east edge of the raster falls outside the axes.
    ax.set_xlim(view_lon_min, view_lon_max)
    ax.set_ylim(view_lat_min, view_lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=8)
    ax.grid(True, color="lightgrey", linewidth=0.3, alpha=0.5)
    ax.set_aspect("equal")

    # Use figure-level supertitle + subtitle so they don't collide with the
    # axes colorbar's "max" tick label.
    fig.suptitle(
        "v3 Tertiary calibrated probability, percentile rank across AOI",
        fontsize=13,
        fontweight="bold",
        x=0.06,
        y=0.965,
        ha="left",
    )
    fig.text(
        0.06,
        0.940,
        "Anchors (red) all land in top decile; pit polygons (cyan) outline "
        "training labels.",
        fontsize=10,
        color="#333333",
        ha="left",
        va="top",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(
        "Tertiary calibrated probability (percentile rank)", fontsize=9
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    cbar.set_ticklabels(["0", "p25", "p50", "p75", "p90", "max"])

    fig.savefig(
        OUT_PATH,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")

    # Sanity: report anchor decile ranks.
    # Map anchor lon/lat to nearest cell in the raster, then look up rank.
    with rasterio.open(RASTER_PATH) as src:
        for name, (lon, lat) in ANCHOR_DISTRICTS.items():
            row, col = src.index(lon, lat)
            v = arr[row, col]
            if not np.isfinite(v):
                print(f"  {name}: NaN cell")
                continue
            r = rank_img[row, col]
            decile = int((1.0 - r) * 10)  # decile 0 = top
            print(f"  {name}: rank={r:.4f} -> decile {decile}")

    # Coverage sanity.
    print(
        f"raster: {arr.shape}, finite cells: {finite_mask.sum()}, "
        f"zero cells: {(finite_vals == 0).sum()}, "
        f"max calibrated p = {finite_vals.max():.5f}"
    )


if __name__ == "__main__":
    main()
