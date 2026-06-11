"""Goldbug-facing sidecar builders for the placer calibrated raster.

Three functions, one per sidecar:
- `build_coverage_mask`: 1-bit GeoTIFF where 1 = cells with non-floor
  signal (calibrated P > threshold).
- `build_bands_json`: JSON sidecar with per-AOI + per-region + per-
  population quantiles + the calibration method + recommended high /
  moderate / weak bucket cutoffs.
- `build_two_band_geotiff`: 2-band float32 GeoTIFF with Tertiary on
  band 1 and Quaternary on band 2 (for goldbug if it wants
  per-population reasoning per parcel).

Each function returns the meta dict it would have written; the script
wrapper at `scripts/build_placer_sidecars.py` is the thin CLI that
calls into these and writes the .meta.json siblings.

Why this module exists at all: when we shipped placer-v3.6.1 + v3.7.0
the three sidecars each had their own ~100-line CLI script that
duplicated raster loading, quantile-computing, and meta-writing. The
common machinery lives here; the scripts shrink to argparse + call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


QUANTILE_LIST = (50, 75, 90, 95, 99, 99.9)
EXTRA_THRESHOLDS = (0.001, 0.01, 0.05)


def _key_for_quantile(q: float) -> str:
    return f"p{str(q).replace('.', 'p')}"


def _stats_for(arr: np.ndarray) -> dict:
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
        out[_key_for_quantile(q)] = float(np.percentile(finite, q))
    for t in EXTRA_THRESHOLDS:
        out[f"pct_gt_{str(t).replace('.', 'p')}"] = float(100.0 * (finite > t).mean())
    return out


def _window_for_bbox(
    src: rasterio.DatasetReader,
    w: float, s: float, e: float, n: float,
) -> np.ndarray:
    r1, c1 = src.index(w, n)
    r2, c2 = src.index(e, s)
    rmin, rmax = sorted([r1, r2])
    cmin, cmax = sorted([c1, c2])
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, src.height - 1)
    cmax = min(cmax, src.width - 1)
    return src.read(1)[rmin:rmax + 1, cmin:cmax + 1]


# --- Coverage mask -----------------------------------------------------------


def build_coverage_mask(
    fused_path: Path,
    out_path: Path,
    *,
    threshold: float = 0.01,
) -> dict:
    """Write a 1-bit coverage-mask GeoTIFF and return its meta dict.

    Cells where the calibrated probability exceeds `threshold` are
    flagged 1; everything else is 0. Threshold 0.01 cleanly separates
    real signal from near-floor extrapolation outside the model's
    training belt at v3.6 / v3.7 calibration magnitudes.
    """
    with rasterio.open(fused_path) as src:
        p = src.read(1)
        meta = src.meta.copy()
    finite = np.isfinite(p)
    mask = ((p > threshold) & finite).astype(np.uint8)
    n_in_coverage = int(mask.sum())
    n_total = int(mask.size)
    pct = 100.0 * n_in_coverage / n_total

    meta.update(dtype="uint8", count=1, nodata=None, compress="deflate", tiled=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask, 1)

    return {
        "source_raster": str(fused_path),
        "threshold": threshold,
        "n_cells_total": n_total,
        "n_cells_in_coverage": n_in_coverage,
        "pct_coverage": round(pct, 2),
        "schema": {
            "dtype": "uint8",
            "values": {
                "0": "outside training-signal coverage; mask placer score",
                "1": "inside training-signal coverage; placer score is meaningful",
            },
        },
    }


# --- Bands JSON --------------------------------------------------------------


def build_bands_json(
    fused_path: Path,
    tertiary_path: Path,
    quaternary_path: Path,
    out_path: Path,
    *,
    model_version: str,
    release: str,
    calibration_method: str = "isotonic_via_CalibratedClassifierCV",
    fusion_method: str = "np.maximum(P_tertiary, P_quaternary)",
    per_region_bboxes: Iterable[tuple[str, float, float, float, float]] = (),
) -> dict:
    """Write the bands.json sidecar and return its document.

    `per_region_bboxes` is an iterable of (name, W, S, E, N) tuples for
    the regions goldbug wants per-region quantiles for (the six
    Mother Lode regions it covers as of 2026-06-11).
    """
    with rasterio.open(fused_path) as src_fused:
        fused = src_fused.read(1)
        fused_stats = _stats_for(fused)
        per_region = {}
        for name, w, s, e, n in per_region_bboxes:
            sub = _window_for_bbox(src_fused, w, s, e, n)
            per_region[name] = {
                "bbox_wsen": [w, s, e, n],
                "stats": _stats_for(sub),
            }

    with rasterio.open(tertiary_path) as src_t:
        tert_stats = _stats_for(src_t.read(1))
    with rasterio.open(quaternary_path) as src_q:
        quat_stats = _stats_for(src_q.read(1))

    fused_finite = fused[np.isfinite(fused)]

    doc = {
        "model_version": model_version,
        "release": release,
        "calibration_method": calibration_method,
        "fusion_method": fusion_method,
        "crs": "EPSG:4326",
        "quantile_keys_used": [_key_for_quantile(q) for q in QUANTILE_LIST],
        "extra_keys_used": ["min", "max", "mean", "median"] + [
            f"pct_gt_{str(t).replace('.', 'p')}" for t in EXTRA_THRESHOLDS
        ],
        "fused": fused_stats,
        "per_population": {"tertiary": tert_stats, "quaternary": quat_stats},
        "per_region": per_region,
        "recommended_bands": {
            "description": (
                "Cutoffs derived from the fused raster's distribution. "
                "Goldbug's prior absolute bands (high=0.05, moderate=0.01, "
                "weak=0.001) map cleanly onto these quantiles: high cuts "
                "at the AOI p99, moderate at p95, weak at p90."
            ),
            "high": float(np.percentile(fused_finite, 99)),
            "moderate": float(np.percentile(fused_finite, 95)),
            "weak": float(np.percentile(fused_finite, 90)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    return doc


# --- 2-band GeoTIFF ----------------------------------------------------------


def build_two_band_geotiff(
    tertiary_path: Path,
    quaternary_path: Path,
    out_path: Path,
    *,
    model_version: str,
    release: str,
) -> dict:
    """Write a 2-band float32 GeoTIFF + return its meta dict.

    Band 1 = Tertiary calibrated P. Band 2 = Quaternary calibrated P.
    Shape and CRS must match between the two source rasters.
    """
    with rasterio.open(tertiary_path) as src_t:
        tert = src_t.read(1)
        meta = src_t.meta.copy()
    with rasterio.open(quaternary_path) as src_q:
        quat = src_q.read(1)
        if (src_q.width, src_q.height) != (meta["width"], meta["height"]):
            raise ValueError(
                f"Tertiary and Quaternary rasters have different shapes: "
                f"{(meta['width'], meta['height'])} vs "
                f"{(src_q.width, src_q.height)}"
            )

    meta.update(count=2, compress="deflate", tiled=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(tert, 1)
        dst.write(quat, 2)
        dst.set_band_description(1, "placer_tertiary_calibrated_probability")
        dst.set_band_description(2, "placer_quaternary_calibrated_probability")

    return {
        "model_version": model_version,
        "release": release,
        "source_tertiary": str(tertiary_path),
        "source_quaternary": str(quaternary_path),
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
            "The single-band fused raster (np.maximum of these two bands) "
            "remains the primary deliverable; this 2-band file is a sibling "
            "for goldbug if it wants per-population reasoning per parcel."
        ),
    }
