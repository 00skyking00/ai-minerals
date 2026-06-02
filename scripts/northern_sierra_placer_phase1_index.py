"""Phase 1 knowledge-driven placer-Au index for the northern Sierra.

Driver script that:

  1. Loads the per-cell feature DataFrame
     (data/derived/features_northern_sierra_placer_250m.parquet).
  2. Runs the USGS-Alaska-style weighted scorer.
  3. Writes per-cell parquet + dual-CRS GeoTIFF deliverables.
  4. Maps the 7 anchor districts (WGS84) to their cells and reports the
     decile rank of each. With --check-anchors-top-decile, exits 0 iff
     every anchor lands in the top decile, else exits 1.

The Phase 1 gate (every named district in the top decile) must pass
before Phase 2 supervised training wires up. If it fails, iterate
weights/features in `scorers/usgs_alaska_placer.py::DEFAULT_WEIGHTS`
rather than papering over the issue with Phase 2's larger feature count.

Usage:
    .venv/bin/python scripts/northern_sierra_placer_phase1_index.py
    .venv/bin/python scripts/northern_sierra_placer_phase1_index.py --check-anchors-top-decile
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from ai_minerals.io.geotiff import write_geotiff_dual_crs
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER
from ai_minerals.scorers.usgs_alaska_placer import (
    anchor_decile_check,
    usgs_alaska_placer_index,
)


REGION = NORTHERN_SIERRA_PLACER
RES_M = 250.0

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
IN_FEATURES = DATA_DERIVED / f"features_{REGION.data_prefix}_250m.parquet"
OUT_DIR = DATA_DERIVED / REGION.data_prefix
OUT_PARQUET = OUT_DIR / "phase1_index_250m.parquet"
OUT_TIF_3310 = OUT_DIR / "phase1_index_250m_3310.tif"
OUT_TIF_4326 = OUT_DIR / "phase1_index_250m_4326.tif"
OUT_ANCHOR_REPORT = OUT_DIR / "phase1_anchor_decile_report.csv"


def _anchor_cell_indices(df: pd.DataFrame) -> pd.Series:
    """Snap each anchor district's (lon, lat) to the nearest grid cell row index."""
    transformer = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    names, idxs = [], []
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    for name, (lon, lat) in ANCHOR_DISTRICTS.items():
        ax, ay = transformer.transform(lon, lat)
        d2 = (xs - ax) ** 2 + (ys - ay) ** 2
        cell = int(np.argmin(d2))
        snap_dist_m = float(np.sqrt((xs[cell] - ax) ** 2 + (ys[cell] - ay) ** 2))
        if snap_dist_m > 250:
            warnings.warn(f"Anchor {name!r} snapped {snap_dist_m:.0f}m to nearest cell")
        names.append(name)
        # Positional index into the score array (downstream anchor_decile_check
        # treats this as a positional row into `score`, which assumes
        # df.index is RangeIndex(0, n). Using `cell` directly is correct
        # regardless of the underlying df.index.
        idxs.append(cell)
    return pd.Series(idxs, index=names, name="cell_idx")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-anchors-top-decile",
        action="store_true",
        help="After computing the index, exit 0 iff every anchor district lands in the top decile; exit 1 otherwise.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=IN_FEATURES,
        help=f"Path to features parquet (default: {IN_FEATURES}).",
    )
    args = parser.parse_args(argv)

    if not args.features.exists():
        print(f"ERROR: features parquet not found at {args.features}.\n"
              f"Run scripts/northern_sierra_placer_assemble_250m.py first (Phase E).",
              file=sys.stderr)
        return 2

    print(f"==> Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    print(f"    cells: {len(df):,}  columns: {len(df.columns)}")

    print("==> Computing Phase 1 index")
    score = usgs_alaska_placer_index(df).rename("phase1_score")
    print(f"    score: min={score.min():.3f}  mean={score.mean():.3f}  "
          f"max={score.max():.3f}  NaN={int(score.isna().sum())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "row": df["row"].values,
        "col": df["col"].values,
        "x": df["x"].values,
        "y": df["y"].values,
        "phase1_score": score.values,
    })
    out_df.to_parquet(OUT_PARQUET, index=False)
    print(f"    wrote {OUT_PARQUET}")

    print(f"==> Writing GeoTIFFs ({REGION.working_crs} + EPSG:4326)")
    write_geotiff_dual_crs(
        score.values,
        df[["x", "y"]],
        resolution_m=RES_M,
        src_crs=REGION.working_crs,
        out_src=OUT_TIF_3310,
        out_4326=OUT_TIF_4326,
    )
    print(f"    wrote {OUT_TIF_3310}")
    print(f"    wrote {OUT_TIF_4326}")

    print("==> Anchor-district validation")
    anchors = _anchor_cell_indices(df)
    report = anchor_decile_check(score, anchors)
    report.to_csv(OUT_ANCHOR_REPORT, index=False)
    print(report.to_string(index=False))
    print(f"    wrote {OUT_ANCHOR_REPORT}")

    all_in_top = bool(report["in_top_decile"].all())
    n_pass = int(report["in_top_decile"].sum())
    n_total = len(report)
    print(f"\n    {n_pass}/{n_total} anchor districts in top decile.")
    if all_in_top:
        print("    Phase 1 gate: PASS")
    else:
        misses = report.loc[~report["in_top_decile"], ["district", "decile"]]
        print("    Phase 1 gate: FAIL")
        print(f"    Misses: {misses.to_dict('records')}")
        print("    Iterate scorers/usgs_alaska_placer.DEFAULT_WEIGHTS before Phase 2.")

    if args.check_anchors_top_decile:
        return 0 if all_in_top else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
