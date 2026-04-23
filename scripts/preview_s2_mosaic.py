"""Render a visual-inspection preview of the Sentinel-2 mosaic.

Produces two PNGs you can open in any image viewer:

  1. data/derived/s2_mosaic_rgb_preview.png
       Natural-color RGB (B04/B03/B02), 2-98 percentile stretch per band.
       Overlays ARDF porphyry positives as cyan stars.
       Use this to sanity-check that the mosaic "looks right" — clouds,
       snow, seams, holes should all be visible to the eye.

  2. data/derived/s2_mosaic_valid_mask.png
       Greyscale: white = valid pixel, black = NaN.
       Makes coverage gaps obvious at a glance.

    uv run python scripts/preview_s2_mosaic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import geopandas as gpd

from ai_minerals.aoi import EASTERN_ALASKA, WORKING_CRS
from ai_minerals.data._common import DATA_DERIVED, DATA_RAW

AOI = EASTERN_ALASKA
MOSAIC_PATH = DATA_RAW / "sentinel2" / f"s2_mosaic_{AOI.name.lower()}.tif"


def main() -> int:
    if not MOSAIC_PATH.exists():
        print(f"Mosaic not found at {MOSAIC_PATH}; run sentinel2_mosaic.py first.")
        return 1

    s2 = rioxarray.open_rasterio(MOSAIC_PATH, masked=True)
    print(f"Loaded {MOSAIC_PATH.name}: shape={s2.shape}, CRS={s2.rio.crs}")

    # Band order matches sentinel2_mosaic.BANDS: [B02, B03, B04, B08, B11, B12].
    # RGB = [B04 red, B03 green, B02 blue] = indices 2, 1, 0 in our stack.
    rgb = s2.sel(band=[3, 2, 1])  # 1-indexed band coords from rioxarray

    # Per-band percentile stretch to 0-1 for display.
    vis = np.zeros(rgb.shape, dtype=np.float32)
    for i in range(3):
        band = rgb.values[i]
        p2, p98 = np.nanpercentile(band, [2, 98])
        vis[i] = np.clip((band - p2) / (p98 - p2 + 1e-9), 0, 1)
    vis = np.nan_to_num(vis, nan=0.0)  # NaN → black

    # Load porphyry labels to overlay.
    ardf = gpd.read_file(DATA_RAW / "ardf" / f"ardf_{AOI.name.lower()}.gpkg")
    import re
    porphyry_codes = ("17", "20c", "21a", "21b")
    pat = r"\b(?:" + "|".join(porphyry_codes) + r")\b"
    por = ardf[ardf["model_code"].fillna("").str.contains(pat, case=False, regex=True)]
    por_proj = por.to_crs(WORKING_CRS)

    DATA_DERIVED.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: natural color + porphyry stars ---
    bounds = s2.rio.bounds()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(
        vis.transpose(1, 2, 0),
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        origin="upper",
    )
    ax.scatter(
        por_proj.geometry.x, por_proj.geometry.y,
        s=80, marker="*", facecolor="cyan", edgecolor="black", linewidth=0.8,
        label=f"porphyry family (N={len(por_proj)})",
    )
    ax.set_title(
        f"Sentinel-2 natural-color mosaic — {AOI.name} AOI\n"
        f"{len(por_proj)} porphyry-family positives overlaid"
    )
    ax.set_xlabel("Easting (m, EPSG:3338)")
    ax.set_ylabel("Northing (m, EPSG:3338)")
    ax.legend(loc="lower left")
    ax.set_aspect("equal")
    plt.tight_layout()
    rgb_path = DATA_DERIVED / "s2_mosaic_rgb_preview.png"
    plt.savefig(rgb_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {rgb_path} ({rgb_path.stat().st_size // 1024} KB)")

    # --- Plot 2: valid-pixel mask ---
    valid = np.isfinite(s2.values).any(axis=0).astype(np.uint8) * 255
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(
        valid,
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        origin="upper",
        cmap="Greys_r",
        vmin=0, vmax=255,
    )
    ax.scatter(
        por_proj.geometry.x, por_proj.geometry.y,
        s=60, marker="*", facecolor="red", edgecolor="black", linewidth=0.6,
        label="porphyry family",
    )
    coverage_pct = 100 * (valid > 0).mean()
    ax.set_title(
        f"Sentinel-2 valid-pixel mask — {AOI.name} AOI\n"
        f"{coverage_pct:.1f}% of raster bbox has data (white=valid, black=NaN)"
    )
    ax.set_xlabel("Easting (m, EPSG:3338)")
    ax.set_ylabel("Northing (m, EPSG:3338)")
    ax.legend(loc="lower left")
    ax.set_aspect("equal")
    plt.tight_layout()
    mask_path = DATA_DERIVED / "s2_mosaic_valid_mask.png"
    plt.savefig(mask_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {mask_path} ({mask_path.stat().st_size // 1024} KB)")

    # Summary of porphyry coverage
    from rasterio.transform import rowcol
    transform = s2.rio.transform()
    valid_mask_arr = np.isfinite(s2.values).any(axis=0)
    missing = []
    for _, row in por_proj.iterrows():
        r, c = rowcol(transform, row.geometry.x, row.geometry.y)
        if 0 <= r < valid_mask_arr.shape[0] and 0 <= c < valid_mask_arr.shape[1]:
            if not valid_mask_arr[r, c]:
                missing.append(row["site"])
    print(f"\nPorphyry coverage check:")
    print(f"  {len(por_proj)} porphyry-family positives")
    print(f"  {len(missing)} in S2 gaps ({100*len(missing)/len(por_proj):.1f}%)")
    if missing:
        for name in missing:
            print(f"    MISSING: {name}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
