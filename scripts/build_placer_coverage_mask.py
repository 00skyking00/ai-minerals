"""Coverage-mask sidecar for the placer-Au calibrated raster.

Builds a 1-bit raster matching the calibrated placer raster's grid where
1 marks cells the v3.6 model produces a non-floor probability for, and
0 marks cells that sit at the raster floor (the model has no signal
there). Goldbug reads this to grey out the Placer column in regions
where the v3.6 model can't see modern-channel signal — primarily the
southern Mother Lode (Tuolumne, Mariposa, Calaveras) which sit at the
~0.0004 floor in the current raster.

Why not "convex hull of training positives"? The v3 Quaternary label
set spans the full Mother Lode (USMIN/MRDS placer-Au records south to
lat 37.5), so the convex hull captures the entire AOI even though the
model's predictions are flat-zero across the southern districts. The
honest coverage signal is the raster itself: cells where the calibrated
probability exceeds the noise floor are where the model "knows
something"; cells at the floor are where it doesn't.

Threshold default is 0.01 (the v3.6 raster's noise floor sits at
~0.0004; the southern Mother Lode districts have p99 in the
0.003-0.005 range, so a 0.01 cutoff cleanly separates real signal
from the near-floor noise that arises just because the model
extrapolates slightly above the floor outside its training belt).
Adjustable via --threshold.

Usage:
    .venv/bin/python scripts/build_placer_coverage_mask.py
    .venv/bin/python scripts/build_placer_coverage_mask.py --threshold 0.005

References:
    Plan H1.2 in ~/.claude/plans/hazy-humming-lynx.md
    Handoff: handoff/inbox/2026-06-08-ai-minerals-placer-coverage-by-region.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
FUSED_TIF = REPO / "data/derived/northern_sierra_placer/prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
OUT_TIF = REPO / "data/ml/prospectivity_placer_northern_sierra_250m_coverage_mask_4326.tif"
OUT_META = OUT_TIF.with_suffix(".meta.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Calibrated-probability threshold above which a cell is marked "
             "as in-coverage (default: 0.001).",
    )
    args = parser.parse_args()

    print(f"[coverage_mask] reading fused raster: {FUSED_TIF}")
    with rasterio.open(FUSED_TIF) as src:
        p = src.read(1)
        meta = src.meta.copy()
        height, width = src.height, src.width

    finite = np.isfinite(p)
    mask = ((p > args.threshold) & finite).astype(np.uint8)
    in_cov = int(mask.sum())
    total = int(mask.size)
    pct = 100.0 * in_cov / total
    print(f"[coverage_mask] threshold = {args.threshold}")
    print(f"[coverage_mask] raster stats: min={float(p[finite].min()):.6f} "
          f"max={float(p[finite].max()):.6f} "
          f"median={float(np.median(p[finite])):.6f}")
    print(f"[coverage_mask] coverage = {in_cov:,} / {total:,} cells ({pct:.1f}%)")

    meta.update(dtype="uint8", count=1, nodata=None, compress="deflate", tiled=True)
    OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_TIF, "w", **meta) as dst:
        dst.write(mask, 1)
    print(f"[coverage_mask] wrote {OUT_TIF}")

    OUT_META.write_text(json.dumps({
        "source_raster": str(FUSED_TIF.relative_to(REPO)),
        "threshold": args.threshold,
        "n_cells_total": total,
        "n_cells_in_coverage": in_cov,
        "pct_coverage": round(pct, 2),
        "schema": {
            "dtype": "uint8",
            "values": {"0": "outside training-signal coverage; mask placer score",
                       "1": "inside training-signal coverage; placer score is meaningful"},
        },
        "definition": (
            f"Cells where the calibrated fused placer probability "
            f"exceeds {args.threshold}. The v3.6 raster's empirical "
            "floor is ~0.0004 (per goldbug 2026-06-08); cells at the "
            "floor are regions where the model has no learned signal "
            "and goldbug should not show a placer score."
        ),
    }, indent=2))
    print(f"[coverage_mask] wrote {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
