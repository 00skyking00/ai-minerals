"""Phase F: fuse the per-population calibrated rasters into the deliverable.

Sierra placer divides cleanly into two populations (Tertiary deep-gravel +
Quaternary modern-channel) with different geomorphic signatures. Phase E
trains and calibrates one classifier per population and writes a calibrated
parquet for each. This script fuses them via per-cell `np.maximum`, writes
the deliverable GeoTIFF that the gldbg sibling repo samples, and keeps the
per-population calibrated rasters as sidecars.

Inputs (from Phase E):
  data/derived/northern_sierra_placer/pop_calibrated_placer_tertiary_250m.parquet
  data/derived/northern_sierra_placer/pop_calibrated_placer_quaternary_250m.parquet

Outputs (under data/derived/northern_sierra_placer/):
  prospectivity_placer_northern_sierra_250m_calibrated_3310.tif
  prospectivity_placer_northern_sierra_250m_calibrated_4326.tif  ← the deliverable
  fusion_meta.json  (per-cell summary: count where Tertiary wins, Quaternary
                     wins, ties, NaN; counts where each branch goes above the
                     calibration cutoff. Goes into the model card.)

The deliverable filename matches the convention agreed with gldbg
(`PLACER-NOTES.md` and `config/regions/northern_sierra_ca.yaml`):
`prospectivity_placer_<region>_<res>_calibrated_<epsg>.tif`.

Usage:
    .venv/bin/python scripts/northern_sierra_placer_calibrate_and_fuse.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.io.geotiff import write_geotiff_dual_crs
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
RES_M = 250.0

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
OUT_DIR = DATA_DERIVED / REGION.data_prefix

IN_TERTIARY = OUT_DIR / "pop_calibrated_placer_tertiary_250m.parquet"
IN_QUATERNARY = OUT_DIR / "pop_calibrated_placer_quaternary_250m.parquet"

OUT_TIF_3310 = OUT_DIR / "prospectivity_placer_northern_sierra_250m_calibrated_3310.tif"
OUT_TIF_4326 = OUT_DIR / "prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
OUT_FUSION_PARQUET = OUT_DIR / "prospectivity_placer_northern_sierra_250m_fused.parquet"
OUT_META = OUT_DIR / "fusion_meta.json"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run scripts/northern_sierra_placer_train_predict_250m.py "
            f"--population {'placer_tertiary' if 'tertiary' in path.name else 'placer_quaternary'} first."
        )
    df = pd.read_parquet(path)
    required = {"row", "col", "x", "y", "p_cal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _per_cell_max(t_df: pd.DataFrame, q_df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell max() of the two calibrated rasters; aligned on (row, col)."""
    t = t_df.set_index(["row", "col"])[["x", "y", "p_cal"]].rename(columns={"p_cal": "p_tertiary"})
    q = q_df.set_index(["row", "col"])[["p_cal"]].rename(columns={"p_cal": "p_quaternary"})
    fused = t.join(q, how="outer")
    fused["p_fused"] = np.fmax(
        fused["p_tertiary"].fillna(-np.inf), fused["p_quaternary"].fillna(-np.inf)
    )
    # Where both NaN, both fillna(-inf) → np.fmax returns -inf; convert to NaN.
    fused.loc[~np.isfinite(fused["p_fused"]), "p_fused"] = np.nan
    return fused.reset_index()


def _fusion_summary(fused: pd.DataFrame, *, decile_cutoff: float = 0.5) -> dict:
    n = len(fused)
    t = fused["p_tertiary"]
    q = fused["p_quaternary"]
    f = fused["p_fused"]

    t_only = (t.notna() & q.isna()).sum()
    q_only = (q.notna() & t.isna()).sum()
    both = (t.notna() & q.notna()).sum()
    neither = (t.isna() & q.isna()).sum()

    # Where both exist, who wins?
    diff = t.fillna(-np.inf) - q.fillna(-np.inf)
    t_wins = ((diff > 0) & t.notna() & q.notna()).sum()
    q_wins = ((diff < 0) & t.notna() & q.notna()).sum()
    ties = ((diff == 0) & t.notna() & q.notna()).sum()

    return {
        "n_cells_total": int(n),
        "n_cells_finite_fused": int(f.notna().sum()),
        "n_cells_tertiary_only": int(t_only),
        "n_cells_quaternary_only": int(q_only),
        "n_cells_both": int(both),
        "n_cells_neither": int(neither),
        "n_cells_tertiary_wins": int(t_wins),
        "n_cells_quaternary_wins": int(q_wins),
        "n_cells_ties": int(ties),
        "fused_min": float(np.nanmin(f)),
        "fused_mean": float(np.nanmean(f)),
        "fused_max": float(np.nanmax(f)),
        "fused_p_above_cutoff": int((f >= decile_cutoff).sum()),
        "decile_cutoff": decile_cutoff,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="Calibrated-probability cutoff for the 'above-cutoff' diagnostic count (default 0.5).",
    )
    args = parser.parse_args(argv)

    print(f"==> Loading Tertiary: {IN_TERTIARY}")
    t_df = _load(IN_TERTIARY)
    print(f"    {len(t_df):,} cells, p_cal finite={int(t_df['p_cal'].notna().sum()):,}")

    print(f"==> Loading Quaternary: {IN_QUATERNARY}")
    q_df = _load(IN_QUATERNARY)
    print(f"    {len(q_df):,} cells, p_cal finite={int(q_df['p_cal'].notna().sum()):,}")

    print("==> Per-cell max() fusion")
    fused = _per_cell_max(t_df, q_df)
    print(f"    fused: {len(fused):,} cells, p_fused finite={int(fused['p_fused'].notna().sum()):,}")

    summary = _fusion_summary(fused, decile_cutoff=args.cutoff)
    print(f"    Tertiary wins: {summary['n_cells_tertiary_wins']:,}")
    print(f"    Quaternary wins: {summary['n_cells_quaternary_wins']:,}")
    print(f"    Ties: {summary['n_cells_ties']:,}")
    print(f"    Above cutoff {args.cutoff}: {summary['fused_p_above_cutoff']:,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_FUSION_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(OUT_FUSION_PARQUET, index=False)
    print(f"==> Wrote fused parquet: {OUT_FUSION_PARQUET}")

    print(f"==> Writing GeoTIFFs ({REGION.working_crs} + EPSG:4326)")
    write_geotiff_dual_crs(
        fused["p_fused"].values,
        fused[["x", "y"]],
        resolution_m=RES_M,
        src_crs=REGION.working_crs,
        out_src=OUT_TIF_3310,
        out_4326=OUT_TIF_4326,
    )
    print(f"    wrote {OUT_TIF_3310}")
    print(f"    wrote {OUT_TIF_4326}  ← deliverable for gldbg")

    OUT_META.write_text(json.dumps({
        "region": REGION.slug,
        "resolution_m": RES_M,
        "compute_crs": REGION.working_crs,
        "deliverable_crs": "EPSG:4326",
        "inputs": {
            "tertiary": str(IN_TERTIARY),
            "quaternary": str(IN_QUATERNARY),
        },
        "outputs": {
            "deliverable_4326_tif": str(OUT_TIF_4326),
            "compute_3310_tif": str(OUT_TIF_3310),
            "fused_parquet": str(OUT_FUSION_PARQUET),
        },
        "fusion": "per-cell np.maximum on calibrated probabilities",
        "summary": summary,
    }, indent=2))
    print(f"==> Wrote fusion metadata: {OUT_META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
