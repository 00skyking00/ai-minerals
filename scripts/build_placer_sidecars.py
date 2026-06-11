"""Build the goldbug-facing placer sidecars (coverage mask, bands JSON, 2-band).

Thin CLI on top of `ai_minerals.io.sidecars`. The actual machinery
(raster IO, quantile computation, meta-writing) lives in the library;
this script wires the three placer-specific input paths in and writes
the meta sidecars.

Three subcommands:

    coverage_mask    (default threshold=0.01)
    bands_json       (per-population stats + per-region quantiles)
    two_band         (Tertiary band 1, Quaternary band 2)
    all              (runs all three in order)

Default paths are the placer Northern Sierra deliverable. Override via
`--fused`, `--tertiary`, `--quaternary`, `--out-dir`.

Usage:
    .venv/bin/python scripts/build_placer_sidecars.py all
    .venv/bin/python scripts/build_placer_sidecars.py coverage_mask --threshold 0.005

References:
    Plan H1.2-H1.4 in ~/.claude/plans/hazy-humming-lynx.md (placer-v3.6.1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_minerals.io.sidecars import (
    build_bands_json,
    build_coverage_mask,
    build_two_band_geotiff,
)

REPO = Path(__file__).resolve().parent.parent
DERIVED = REPO / "data/derived/northern_sierra_placer"
DATA_ML = REPO / "data/ml"
DEFAULT_FUSED = DERIVED / "prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
DEFAULT_TERT = DERIVED / "prospectivity_placer_placer_tertiary_250m_calibrated_4326.tif"
DEFAULT_QUAT = DERIVED / "prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"

# Per-region goldbug bboxes (W, S, E, N) in EPSG:4326. Source: handoff
# from goldbug 2026-06-08.
REGION_BBOXES = (
    ("northern_sierra",  -121.15, 39.00, -120.70, 39.45),
    ("el_dorado_placer", -121.15, 38.65, -120.60, 39.00),
    ("amador",           -121.00, 38.25, -120.55, 38.55),
    ("tuolumne",         -120.90, 37.75, -120.10, 38.15),
    ("mariposa",         -120.20, 37.50, -119.65, 37.80),
    ("calaveras",        -120.70, 37.90, -120.35, 38.20),
)


def _write_meta_json(meta: dict, out_tif: Path) -> Path:
    meta_path = out_tif.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def cmd_coverage_mask(args: argparse.Namespace) -> int:
    out_tif = args.out_dir / "prospectivity_placer_northern_sierra_250m_coverage_mask_4326.tif"
    print(f"[coverage_mask] reading {args.fused}")
    meta = build_coverage_mask(args.fused, out_tif, threshold=args.threshold)
    meta["threshold"] = args.threshold
    meta["definition"] = (
        f"Cells where the calibrated fused placer probability exceeds "
        f"{args.threshold}. Cells at the raster floor (~0.0004) are regions "
        "where the model has no learned signal."
    )
    meta_json = _write_meta_json(meta, out_tif)
    print(f"[coverage_mask] coverage = {meta['n_cells_in_coverage']:,} / "
          f"{meta['n_cells_total']:,} cells ({meta['pct_coverage']}%)")
    print(f"[coverage_mask] wrote {out_tif}")
    print(f"[coverage_mask] wrote {meta_json}")
    return 0


def cmd_bands_json(args: argparse.Namespace) -> int:
    out_json = args.out_dir / "prospectivity_placer_northern_sierra_250m_calibrated_4326.bands.json"
    print(f"[bands_json] reading {args.fused.name} + per-population")
    doc = build_bands_json(
        args.fused, args.tertiary, args.quaternary, out_json,
        model_version=args.model_version,
        release=args.release,
        per_region_bboxes=REGION_BBOXES,
    )
    print(f"[bands_json] wrote {out_json}")
    fs = doc["fused"]
    print(f"  fused AOI: p90={fs['p90']:.4f} p95={fs['p95']:.4f} "
          f"p99={fs['p99']:.4f} max={fs['max']:.4f}")
    for name, _, _, _, _ in REGION_BBOXES:
        rs = doc["per_region"][name]["stats"]
        print(f"  {name:18s} p90={rs['p90']:.4f} p99={rs['p99']:.4f} "
              f"max={rs['max']:.4f}")
    return 0


def cmd_two_band(args: argparse.Namespace) -> int:
    out_tif = args.out_dir / "prospectivity_placer_northern_sierra_250m_calibrated_4326_2band.tif"
    meta = build_two_band_geotiff(
        args.tertiary, args.quaternary, out_tif,
        model_version=args.model_version,
        release=args.release,
    )
    meta_json = _write_meta_json(meta, out_tif)
    print(f"[2band] wrote {out_tif}")
    print(f"[2band] wrote {meta_json}")
    print(f"  Band 1 (Tertiary):   max={meta['stats']['tertiary']['max']:.4f}")
    print(f"  Band 2 (Quaternary): max={meta['stats']['quaternary']['max']:.4f}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_coverage_mask(args)
    if rc: return rc
    print()
    rc = cmd_bands_json(args)
    if rc: return rc
    print()
    rc = cmd_two_band(args)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fused", type=Path, default=DEFAULT_FUSED)
    parser.add_argument("--tertiary", type=Path, default=DEFAULT_TERT)
    parser.add_argument("--quaternary", type=Path, default=DEFAULT_QUAT)
    parser.add_argument("--out-dir", type=Path, default=DATA_ML)
    parser.add_argument("--model-version", default="placer-v3.7.0")
    parser.add_argument("--release", default="ai-minerals-v1.1.0")

    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cov = sub.add_parser("coverage_mask")
    p_cov.add_argument("--threshold", type=float, default=0.01)
    p_cov.set_defaults(func=cmd_coverage_mask)

    p_bands = sub.add_parser("bands_json")
    p_bands.set_defaults(func=cmd_bands_json)

    p_2band = sub.add_parser("two_band")
    p_2band.set_defaults(func=cmd_two_band)

    p_all = sub.add_parser("all")
    p_all.add_argument("--threshold", type=float, default=0.01)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
