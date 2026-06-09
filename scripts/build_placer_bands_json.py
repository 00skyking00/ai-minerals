"""Bands / calibration JSON sidecar for the placer-Au calibrated raster.

Emits per-region probability quantiles + the calibration method used to
produce the raster, so goldbug (or any downstream consumer) can pick
bucket cutoffs without reverse-engineering them.

Goldbug today uses absolute bands (high=0.05, moderate=0.01, weak=0.001)
that were tuned against the northern-Sierra distribution. The handoff
2026-06-08-ai-minerals-placer-coverage-by-region.md asked for per-region
quantiles so the same scoring logic could be re-banded in one config edit.

Outputs a JSON document next to the calibrated raster with three blocks:
- `fused`: quantiles over the whole AOI
- `per_population.tertiary` and `per_population.quaternary`: quantiles
  over each per-population raster
- `per_region`: quantiles over each goldbug-known bbox for the fused
  raster (the most directly actionable section for re-banding)

Usage:
    .venv/bin/python scripts/build_placer_bands_json.py

References:
    Plan H1.3 in ~/.claude/plans/hazy-humming-lynx.md
    Handoff: handoff/inbox/2026-06-08-ai-minerals-placer-coverage-by-region.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
DERIVED = REPO / "data/derived/northern_sierra_placer"
FUSED_TIF = DERIVED / "prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
TERT_TIF = DERIVED / "prospectivity_placer_placer_tertiary_250m_calibrated_4326.tif"
QUAT_TIF = DERIVED / "prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"
OUT_JSON = REPO / "data/ml/prospectivity_placer_northern_sierra_250m_calibrated_4326.bands.json"

# Goldbug region bboxes as (W, S, E, N) in EPSG:4326. Source:
# handoff/inbox/2026-06-08-ai-minerals-placer-coverage-by-region.md
REGIONS = [
    ("northern_sierra",  -121.15, 39.00, -120.70, 39.45),
    ("el_dorado_placer", -121.15, 38.65, -120.60, 39.00),
    ("amador",           -121.00, 38.25, -120.55, 38.55),
    ("tuolumne",         -120.90, 37.75, -120.10, 38.15),
    ("mariposa",         -120.20, 37.50, -119.65, 37.80),
    ("calaveras",        -120.70, 37.90, -120.35, 38.20),
]
QUANTILE_LIST = [50, 75, 90, 95, 99, 99.9]
EXTRA_STATS = ("min", "max", "mean", "median")


def stats_for(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {"n_finite": 0}
    out = {
        "n_finite": int(len(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
    }
    for q in QUANTILE_LIST:
        out[f"p{str(q).replace('.', 'p')}"] = float(np.percentile(finite, q))
    out["pct_gt_0p001"] = float(100.0 * (finite > 0.001).mean())
    out["pct_gt_0p01"] = float(100.0 * (finite > 0.01).mean())
    out["pct_gt_0p05"] = float(100.0 * (finite > 0.05).mean())
    return out


def window_for_bbox(src: rasterio.DatasetReader, w: float, s: float, e: float, n: float) -> np.ndarray:
    r1, c1 = src.index(w, n)
    r2, c2 = src.index(e, s)
    rmin, rmax = sorted([r1, r2])
    cmin, cmax = sorted([c1, c2])
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, src.height - 1)
    cmax = min(cmax, src.width - 1)
    return src.read(1)[rmin:rmax + 1, cmin:cmax + 1]


def main() -> int:
    print(f"[bands_json] reading {FUSED_TIF.name}")
    with rasterio.open(FUSED_TIF) as src_fused:
        fused = src_fused.read(1)
        fused_stats = stats_for(fused)
        per_region = {}
        for name, w, s, e, n in REGIONS:
            sub = window_for_bbox(src_fused, w, s, e, n)
            per_region[name] = {
                "bbox_wsen": [w, s, e, n],
                "stats": stats_for(sub),
            }

    print(f"[bands_json] reading {TERT_TIF.name}")
    with rasterio.open(TERT_TIF) as src_t:
        tert_stats = stats_for(src_t.read(1))
    print(f"[bands_json] reading {QUAT_TIF.name}")
    with rasterio.open(QUAT_TIF) as src_q:
        quat_stats = stats_for(src_q.read(1))

    out = {
        "model_version": "placer-v3.6.0",
        "release": "ai-minerals-v1.0.1",
        "calibration_method": "isotonic_via_CalibratedClassifierCV",
        "fusion_method": "np.maximum(P_tertiary, P_quaternary)",
        "crs": "EPSG:4326",
        "quantile_keys_used": [f"p{str(q).replace('.', 'p')}" for q in QUANTILE_LIST],
        "extra_keys_used": list(EXTRA_STATS) + ["pct_gt_0p001", "pct_gt_0p01", "pct_gt_0p05"],
        "fused": fused_stats,
        "per_population": {
            "tertiary": tert_stats,
            "quaternary": quat_stats,
        },
        "per_region": per_region,
        "recommended_bands": {
            "description": (
                "Cutoffs derived from the fused raster's distribution. "
                "Goldbug's prior absolute bands (high=0.05, moderate=0.01, "
                "weak=0.001) map cleanly onto these quantiles: high cuts at "
                "the AOI p99, moderate at p95, weak at p90. Per-region "
                "cutoffs should generally use the northern_sierra region's "
                "values rather than per-region p99, since the model's "
                "training signal is northern-Sierra-anchored."
            ),
            "high": float(np.percentile(fused[np.isfinite(fused)], 99)),
            "moderate": float(np.percentile(fused[np.isfinite(fused)], 95)),
            "weak": float(np.percentile(fused[np.isfinite(fused)], 90)),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[bands_json] wrote {OUT_JSON}")

    print()
    print("Summary:")
    print(f"  fused AOI: p90={fused_stats['p90']:.4f} p95={fused_stats['p95']:.4f} "
          f"p99={fused_stats['p99']:.4f} max={fused_stats['max']:.4f}")
    for name, w, s, e, n in REGIONS:
        s_ = per_region[name]["stats"]
        print(f"  {name:18s} p90={s_['p90']:.4f} p99={s_['p99']:.4f} max={s_['max']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
