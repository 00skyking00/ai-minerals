"""v3.6 vs v3.7 Quaternary calibrated-raster comparison figure.

Two-pane plate for the v3.7.0 chapter section: v3.6 (left) and v3.7
(right), both shown at percentile rank against the full AOI distribution
so the long zero-tail doesn't wash out the southern signal. Southern
Mother Lode anchor districts (Mokelumne Hill, Murphys, Sonora,
Mariposa town) are circled at 2 km radius matching the H2.5 buffer
convention; the four below-gate counties (Butte, Yuba, Amador,
Mariposa) are outlined.

If v3.7 raster doesn't exist yet (training still running), draws only
the v3.6 panel + a placeholder right pane.

Inputs:
  data/derived/northern_sierra_placer/_v36/prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif
  data/derived/northern_sierra_placer/prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif

Output:
  data/derived/portfolio_charts/v37_vs_v36_quaternary_comparison.png

Usage:
    .venv/bin/python scripts/v37_vs_v36_quaternary_comparison_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parent.parent
DERIVED = REPO / "data/derived/northern_sierra_placer"
V36_Q = DERIVED / "_v36" / "prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"
V37_Q = DERIVED / "prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"
OUT_PNG = REPO / "data/derived/portfolio_charts/v37_vs_v36_quaternary_comparison.png"

# Southern Mother Lode anchor districts from the H2.5 script (same coords).
SOUTHERN_ANCHORS = [
    ("Mokelumne Hill",  -120.708, 38.301),
    ("Murphys",         -120.460, 38.137),
    ("Sonora",          -120.382, 37.984),
    ("Mariposa town",   -119.965, 37.485),
    ("Columbia",        -120.402, 38.038),
    ("Carson Hill",     -120.546, 38.054),
]
ANCHOR_RADIUS_DEG = 0.02  # ~2 km buffer at 38 N

# Four weak counties; bbox in (W, S, E, N) for overlay rectangle.
WEAK_COUNTY_BBOXES = {
    "Butte":    (-121.85, 39.30, -121.20, 39.97),
    "Yuba":     (-121.55, 39.05, -121.10, 39.55),
    "Amador":   (-121.00, 38.25, -120.50, 38.60),
    "Mariposa": (-120.30, 37.40, -119.55, 37.85),
}


def percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Convert calibrated probability to per-cell rank in [0, 1].

    The long zero-tail dominates the raw probability distribution; percentile
    rank shows spatial structure that linear color stretching would wash out.
    """
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return out
    sorted_finite = np.sort(finite)
    mask = np.isfinite(arr)
    # searchsorted returns rank as integer index; divide by length for [0,1]
    out[mask] = np.searchsorted(sorted_finite, arr[mask], side="right") / len(sorted_finite)
    return out


def setup_panel(ax, raster_path: Path | None, title: str, extent_4326: tuple,
                placeholder_text: str | None = None) -> None:
    ax.set_xlim(extent_4326[0], extent_4326[2])
    ax.set_ylim(extent_4326[1], extent_4326[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if raster_path is None or not raster_path.exists():
        ax.text(0.5, 0.5, placeholder_text or "(raster not yet available)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="gray")
        return

    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        rank = percentile_rank(arr)
        b = src.bounds
        ax.imshow(rank, extent=(b.left, b.right, b.bottom, b.top),
                  origin="upper", cmap="magma", vmin=0.5, vmax=1.0)

    # Weak-county rectangles
    for name, (W, S, E, N) in WEAK_COUNTY_BBOXES.items():
        ax.add_patch(mpatches.Rectangle(
            (W, S), E - W, N - S, fill=False, edgecolor="cyan",
            linewidth=1.2, alpha=0.85, linestyle="--",
        ))
        ax.text(W + 0.02, S + 0.04, name, color="cyan", fontsize=8,
                weight="bold", alpha=0.9)

    # Southern anchor circles
    for name, lon, lat in SOUTHERN_ANCHORS:
        ax.add_patch(Circle((lon, lat), ANCHOR_RADIUS_DEG,
                            fill=False, edgecolor="white", linewidth=1.3))
        ax.text(lon + ANCHOR_RADIUS_DEG, lat - 0.025, name,
                color="white", fontsize=8, weight="bold")


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    plt.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.06,
                        wspace=0.10)

    # Use v3.6 bounds as the canonical extent
    with rasterio.open(V36_Q) as src:
        b = src.bounds
        extent = (b.left, b.bottom, b.right, b.top)

    setup_panel(
        axes[0], V36_Q,
        "v3.6 Quaternary (MRDS-derived labels, n=437)",
        extent,
    )
    v37_ok = V37_Q.exists() and V37_Q.stat().st_size > 1000
    setup_panel(
        axes[1], V37_Q if v37_ok else None,
        ("v3.7.0 Quaternary (USMIN channel-kernel, n=8,333 binary)"
         if v37_ok else "v3.7.0 Quaternary — training in progress"),
        extent,
        placeholder_text=("(v3.7 raster pending; rerun this script\n"
                          "after training completes)"),
    )

    fig.suptitle(
        "Quaternary placer-Au calibrated probability, percentile rank against AOI distribution\n"
        "Cyan dashed rectangles = below-gate counties (Butte/Yuba/Amador/Mariposa); "
        "white circles = southern anchor districts (2 km buffer)",
        fontsize=11,
    )

    sm = plt.cm.ScalarMappable(cmap="magma",
                               norm=plt.Normalize(vmin=0.5, vmax=1.0))
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.07, shrink=0.8)
    cbar.set_label("Percentile rank in AOI (p50–p100; lower than p50 not shown)")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)")
    if not v37_ok:
        print("NOTE: v3.7 raster not yet available; right pane shows placeholder.")
        print("Re-run this script after training completes to regenerate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
