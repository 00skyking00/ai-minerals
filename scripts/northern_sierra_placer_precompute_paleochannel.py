"""Precompute the paleochannel-likelihood raster for the northern Sierra.

Runs once before scripts/northern_sierra_placer_assemble_250m.py (Phase E)
so the feature stack can sample the resulting raster as a Tertiary
deep-gravel feature. Phase 1 substitutes hydraulic-pit proximity as a
proxy when this raster is absent.

Inputs:
  data/raw/3dep_lidar/3dep_1m_northern_sierra.tif  (from threedep fetcher;
      can be 10 m fallback if 1 m LiDAR not flown)
  GRASS r.geomorphon on PATH (used by features/hydrology.py)
  whitebox-workflows + riverrem + scikit-image installed
      (`uv sync --extra paleochannel`)

Outputs:
  data/raw/3dep_lidar/paleochannel_likelihood_northern_sierra.tif
  data/raw/3dep_lidar/paleochannel_likelihood_northern_sierra_meta.json
      (records the input DEM resolution + composite weights for the model card)

Usage:
  .venv/bin/python scripts/northern_sierra_placer_precompute_paleochannel.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import rasterio

from ai_minerals.features.paleochannel import build_paleochannel_likelihood
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER

IN_DEM = REGION.raw_paths["lidar_dem"]
NHD_FLOWLINES = REGION.raw_paths.get("nhd_flowlines")
OUT_TIF = REGION.raw_paths["paleochannel_likelihood"]
OUT_META = OUT_TIF.with_suffix(".meta.json")

REM_RADIUS_M = 200.0
LRM_KERNEL_M = 100.0
COMPOSITE_WEIGHTS = (0.45, 0.20, 0.35)
# Internal processing resolution. The morphometric operators are 80-250 m
# low-pass filters feeding a 250 m model grid, so native 10 m is wasted
# precision and won't fit in RAM over this footprint (~1.4 Gcell). 30 m keeps
# every kernel well-resolved (finest is ~80 m) and fits in <1 GB.
INTERNAL_RES_M = 30.0
# REM channel source. "nhd" rasterizes NHDPlus HR flowlines from K.1 (USGS
# authoritative high-res hydrography, far better headwater coverage than OSM,
# no network dependency at runtime) — the right default for any US AOI. "flow"
# derives the channel network from the DEM's own D-infinity flow accumulation
# (works anywhere, no external data) and is the fallback when the NHD GPKG is
# missing. "osm" uses riverrem/Overpass and is retained for ROW parity testing;
# rate-limits aggressively and falls back to "flow" on failure. Override per-run
# with REM_SOURCE=flow or =osm.
REM_SOURCE = os.environ.get("REM_SOURCE", "nhd")


def main() -> int:
    if not IN_DEM.exists():
        print(f"ERROR: input DEM not found at {IN_DEM}.\n"
              f"Run `python -m ai_minerals.data.threedep` (or "
              f"scripts/northern_sierra_placer_fetch_all.py --only threedep) first.",
              file=sys.stderr)
        return 2

    with rasterio.open(IN_DEM) as src:
        res_x = abs(src.transform.a)
        crs = str(src.crs)
        width, height = src.width, src.height

    print(f"==> Building paleochannel-likelihood raster")
    print(f"    input DEM: {IN_DEM}")
    print(f"    {width}x{height} cells, ~{res_x:.2f} units/cell, CRS={crs}")
    print(f"    internal resolution: ~{INTERNAL_RES_M:.0f} m  (downsampled from native)")
    print(f"    REM source: {REM_SOURCE}")
    print(f"    weights: REM={COMPOSITE_WEIGHTS[0]} LRM={COMPOSITE_WEIGHTS[1]} GMI={COMPOSITE_WEIGHTS[2]}")

    if REM_SOURCE == "nhd" and (NHD_FLOWLINES is None or not NHD_FLOWLINES.exists()):
        print(f"WARNING: REM_SOURCE=nhd but NHD flowlines missing at {NHD_FLOWLINES};"
              f" will fall back to flow-REM inside the dispatcher.", file=sys.stderr)

    t0 = time.monotonic()
    out_path = build_paleochannel_likelihood(
        IN_DEM,
        OUT_TIF,
        rem_radius_m=REM_RADIUS_M,
        lrm_kernel_m=LRM_KERNEL_M,
        weights=COMPOSITE_WEIGHTS,
        downsample_to_m=INTERNAL_RES_M,
        rem_source=REM_SOURCE,
        nhd_path=NHD_FLOWLINES,
    )
    elapsed = time.monotonic() - t0

    OUT_META.write_text(json.dumps({
        "input_dem": str(IN_DEM),
        "input_dem_resolution": res_x,
        "input_dem_crs": crs,
        "internal_resolution_m": INTERNAL_RES_M,
        "rem_source": REM_SOURCE,
        "nhd_flowlines": str(NHD_FLOWLINES) if REM_SOURCE == "nhd" else None,
        "output_raster": str(out_path),
        "rem_radius_m": REM_RADIUS_M,
        "lrm_kernel_m": LRM_KERNEL_M,
        "composite_weights": {
            "rem": COMPOSITE_WEIGHTS[0],
            "lrm": COMPOSITE_WEIGHTS[1],
            "gmi": COMPOSITE_WEIGHTS[2],
        },
        "elapsed_seconds": round(elapsed, 1),
    }, indent=2))

    print(f"==> Wrote {out_path}  ({elapsed:.1f}s)")
    print(f"    metadata: {OUT_META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
