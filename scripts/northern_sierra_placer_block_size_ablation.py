"""Phase E.6 (diagnostic pre-flight): spatial block size empirical justification.

The v2 plan inherited a 20 km spatial block size for the SpatialBlockCV
without explicit grounding. Phase E.6 in `~/.claude/plans/hazy-humming-lynx.md`
calls for an empirical study across 10 / 15 / 20 / 25 / 30 km blocks to pick
the size that maximizes information per fold without crossing the leakage
threshold.

The FULL ablation (running the v3 train_predict pipeline at all five block
sizes) is queued for Phase F.6 and only triggers if Phase F.1's results
suggest the block size needs revision. Each full retrain is multi-hour and
out of budget for a workflow agent.

This script does the cheap DIAGNOSTIC pre-flight instead. For each block
size it computes:

    n_folds                     = unique block ids
    mean / std cells per fold   = fold-size distribution
    tert / quat positives       = total per-population positive cells covered
    blocks_with_zero_pos        = folds that contribute no test positives
    blocks_with_only_one_pos    = folds with a single positive (high variance)

These statistics are sufficient to flag obviously broken block sizes (too
small -> many singleton folds; too large -> few folds with most positives
concentrated in one block) before spending compute on the full retrain.

Inputs:
  data/derived/features_northern_sierra_placer_250m.parquet

Outputs:
  data/derived/northern_sierra_placer/block_size_ablation_diagnostic.csv

Usage:
    .venv/bin/python scripts/northern_sierra_placer_block_size_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the canonical _block_ids helper so this diagnostic mirrors exactly
# what the train pipeline does. Importing from the train script keeps the
# two in lockstep; if SpatialBlockCV ever changes, both move together.
from northern_sierra_placer_train_predict_250m import _block_ids  # noqa: E402

DATA_DERIVED = REPO_ROOT / "data" / "derived"
IN_FEATURES = DATA_DERIVED / "features_northern_sierra_placer_250m.parquet"
OUT_DIR = DATA_DERIVED / "northern_sierra_placer"
OUT_CSV = OUT_DIR / "block_size_ablation_diagnostic.csv"

BLOCK_SIZES_M = (10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0)
POPULATIONS = ("placer_tertiary", "placer_quaternary")


def _block_xy(block_ids: np.ndarray, df_xy: pd.DataFrame,
              block_size_m: float) -> dict[int, tuple[int, int]]:
    """Map each block id back to its (bx, by) integer grid coordinates.

    Used to count adjacency between blocks (8-neighbour Chebyshev).
    """
    bx = (df_xy["x"].to_numpy() // block_size_m).astype(int)
    by = (df_xy["y"].to_numpy() // block_size_m).astype(int)
    out: dict[int, tuple[int, int]] = {}
    for i, bid in enumerate(block_ids):
        if bid not in out:
            out[int(bid)] = (int(bx[i]), int(by[i]))
    return out


def diagnose_one_block_size(df: pd.DataFrame, block_size_m: float) -> dict:
    block_ids = _block_ids(df[["x", "y"]], block_size_m)
    unique_blocks, counts = np.unique(block_ids, return_counts=True)
    n_folds = int(unique_blocks.size)

    tert = df["is_placer_tertiary"].to_numpy(dtype=np.int64)
    quat = df["is_placer_quaternary"].to_numpy(dtype=np.int64)

    # Per-block positive counts via groupby on the block-id array.
    by_block = pd.DataFrame({
        "block_id": block_ids,
        "tert": tert,
        "quat": quat,
    }).groupby("block_id", sort=True).sum()

    pos_total = by_block["tert"].to_numpy() + by_block["quat"].to_numpy()
    blocks_zero = int((pos_total == 0).sum())
    blocks_one = int((pos_total == 1).sum())

    tert_pos = int(by_block["tert"].sum())
    quat_pos = int(by_block["quat"].sum())

    # Adjacency correlation: for each block, count positives in 8-connected
    # neighbours. High correlation between own-positives and neighbour-
    # positives is the classic spatial autocorrelation tell -- folds with
    # heavily-positive neighbours risk leaking signal across the split.
    block_xy = _block_xy(block_ids, df[["x", "y"]], block_size_m)
    xy_to_id = {v: k for k, v in block_xy.items()}
    pos_by_id = dict(zip(by_block.index.tolist(), pos_total.tolist()))

    own_pos: list[int] = []
    neigh_pos: list[int] = []
    for bid, (bxi, byi) in block_xy.items():
        own_pos.append(int(pos_by_id.get(bid, 0)))
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nbid = xy_to_id.get((bxi + dx, byi + dy))
                if nbid is None:
                    continue
                n += int(pos_by_id.get(nbid, 0))
        neigh_pos.append(n)

    own_arr = np.asarray(own_pos, dtype=np.float64)
    neigh_arr = np.asarray(neigh_pos, dtype=np.float64)
    if own_arr.std() > 0 and neigh_arr.std() > 0:
        adjacency_corr = float(np.corrcoef(own_arr, neigh_arr)[0, 1])
    else:
        adjacency_corr = float("nan")

    return {
        "block_size_m": int(block_size_m),
        "n_folds": n_folds,
        "mean_cells_per_fold": float(counts.mean()),
        "std_cells_per_fold": float(counts.std(ddof=0)),
        "tert_pos_in_blocks": tert_pos,
        "quat_pos_in_blocks": quat_pos,
        "blocks_with_zero_pos": blocks_zero,
        "blocks_with_only_one_pos": blocks_one,
        "adjacency_pos_corr": adjacency_corr,
    }


def main() -> int:
    if not IN_FEATURES.exists():
        print(f"ERROR: feature parquet not found at {IN_FEATURES}", file=sys.stderr)
        return 1

    df = pd.read_parquet(IN_FEATURES, columns=[
        "x", "y", "is_placer_tertiary", "is_placer_quaternary",
    ])
    print(f"loaded {len(df):,} cells from {IN_FEATURES.name}", flush=True)
    print(f"  tert positives: {int(df['is_placer_tertiary'].sum())}", flush=True)
    print(f"  quat positives: {int(df['is_placer_quaternary'].sum())}", flush=True)

    rows = []
    for bs in BLOCK_SIZES_M:
        row = diagnose_one_block_size(df, bs)
        print(
            f"  block_size={int(bs/1000)}km: "
            f"n_folds={row['n_folds']:3d} "
            f"mean_cells={row['mean_cells_per_fold']:8.0f} "
            f"zero_pos={row['blocks_with_zero_pos']:3d} "
            f"singleton_pos={row['blocks_with_only_one_pos']:3d} "
            f"adj_corr={row['adjacency_pos_corr']:.3f}",
            flush=True,
        )
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
