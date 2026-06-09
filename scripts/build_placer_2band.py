"""Two-band GeoTIFF for the placer-Au calibrated raster.

The shipped fused raster
(`prospectivity_placer_northern_sierra_250m_calibrated_4326.tif`) is
`np.maximum(P_tertiary, P_quaternary)` per cell, which loses the
per-population provenance. Goldbug's 2026-06-08 handoff asked to see
Tertiary and Quaternary surfaced separately so a downstream consumer
can tell which population drives a given cell.

Builds a 2-band float32 GeoTIFF at the same grid as the fused raster:
  Band 1 = Tertiary calibrated probability
  Band 2 = Quaternary calibrated probability

Per-population rasters already exist under
`data/derived/northern_sierra_placer/`; this script just stacks them
into a single file at `data/ml/` for goldbug to consume. The fused
single-band file remains the primary deliverable for backward
compatibility.

Usage:
    .venv/bin/python scripts/build_placer_2band.py

References:
    Plan H1.4 in ~/.claude/plans/hazy-humming-lynx.md
    Handoff: handoff/inbox/2026-06-08-ai-minerals-placer-coverage-by-region.md (ask 2)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
DERIVED = REPO / "data/derived/northern_sierra_placer"
TERT_TIF = DERIVED / "prospectivity_placer_placer_tertiary_250m_calibrated_4326.tif"
QUAT_TIF = DERIVED / "prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"
OUT_TIF = REPO / "data/ml/prospectivity_placer_northern_sierra_250m_calibrated_4326_2band.tif"
OUT_META = OUT_TIF.with_suffix(".meta.json")


def main() -> int:
    print(f"[2band] reading {TERT_TIF.name}")
    with rasterio.open(TERT_TIF) as src_t:
        tert = src_t.read(1)
        meta = src_t.meta.copy()
    print(f"[2band] reading {QUAT_TIF.name}")
    with rasterio.open(QUAT_TIF) as src_q:
        quat = src_q.read(1)
        if (src_q.width, src_q.height) != (meta["width"], meta["height"]):
            print(f"ERROR: T and Q rasters have different shapes "
                  f"({(meta['width'], meta['height'])} vs "
                  f"{(src_q.width, src_q.height)})")
            return 2

    meta.update(count=2, compress="deflate", tiled=True)
    OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_TIF, "w", **meta) as dst:
        dst.write(tert, 1)
        dst.write(quat, 2)
        dst.set_band_description(1, "placer_tertiary_calibrated_probability")
        dst.set_band_description(2, "placer_quaternary_calibrated_probability")
    print(f"[2band] wrote {OUT_TIF}")

    OUT_META.write_text(json.dumps({
        "model_version": "placer-v3.6.0",
        "release": "ai-minerals-v1.0.1",
        "source_tertiary": str(TERT_TIF.relative_to(REPO)),
        "source_quaternary": str(QUAT_TIF.relative_to(REPO)),
        "bands": {
            "1": "placer_tertiary_calibrated_probability",
            "2": "placer_quaternary_calibrated_probability",
        },
        "stats": {
            "tertiary": {
                "min": float(np.nanmin(tert)),
                "max": float(np.nanmax(tert)),
                "mean": float(np.nanmean(tert)),
                "median": float(np.nanmedian(tert)),
            },
            "quaternary": {
                "min": float(np.nanmin(quat)),
                "max": float(np.nanmax(quat)),
                "mean": float(np.nanmean(quat)),
                "median": float(np.nanmedian(quat)),
            },
        },
        "fusion_note": (
            "The single-band fused raster at "
            "prospectivity_placer_northern_sierra_250m_calibrated_4326.tif "
            "remains the primary deliverable. It is computed per-cell as "
            "np.maximum(band1, band2) from this 2-band file."
        ),
    }, indent=2))
    print(f"[2band] wrote {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
