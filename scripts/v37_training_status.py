"""v3.7 Quaternary training live status.

Walks the per-fold joblib checkpoints under
data/derived/northern_sierra_placer/_k5_checkpoints/ and prints a v3-style
summary: per-stage progress, per-stage median ROC-AUC + PR-AUC, anchor
capture rates, and time-to-finish estimate based on fold cadence.

Stages tracked:
  __pu              nnPU (Quaternary only) or PU-bagging (Tertiary)
  __rf_cv__fold_N   RF per-fold spatial-block CV
  __lgbm_cv__fold_N LightGBM per-fold CV
  __xgb_cv__fold_N  XGBoost per-fold CV
  __stack_oof       Stacking out-of-fold meta-learner
  __cal             Isotonic calibration

Usage:
    .venv/bin/python scripts/v37_training_status.py
    .venv/bin/python scripts/v37_training_status.py --pop placer_tertiary
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO / "data/derived/northern_sierra_placer/_k5_checkpoints"
CKPT_DIR_V36 = REPO / "data/derived/northern_sierra_placer/_v36/_k5_checkpoints_v36"
LOG_PATH = REPO / "data/derived/northern_sierra_placer/v37_training.log"

STAGE_ORDER = ["__pu", "__rf_cv", "__lgbm_cv", "__xgb_cv", "__stack_oof", "__cal"]
STAGE_LABEL = {
    "__pu": "nnPU / PU bag",
    "__rf_cv": "RF spatial-block CV",
    "__lgbm_cv": "LightGBM CV",
    "__xgb_cv": "XGBoost CV",
    "__stack_oof": "stacking meta-LR",
    "__cal": "isotonic calibration",
}


def v36_max_block_id(pop: str, stage: str) -> int | None:
    """Max spatial-block id seen in v3.6 for the same (pop, stage).

    SpatialBlockCV walks the same range of block ids in both versions
    (same AOI grid, same block size), so v3.6's max block id is a real
    upper bound on the iteration: when v3.7's max_block_id == v3.6's,
    the loop is finished.

    Fold count differs between versions: v3.7's channel-aligned kernel
    spreads positives across more blocks than v3.6's MRDS-derived
    labels, so v3.7 produces more checkpoint files even though the
    block-id loop is the same.
    """
    if not CKPT_DIR_V36.exists():
        return None
    ids = []
    pat = re.compile(rf"{re.escape(pop)}{re.escape(stage)}__fold_(\d+)\.joblib$")
    for p in CKPT_DIR_V36.glob(f"{pop}{stage}__fold_*.joblib"):
        m = pat.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return max(ids) if ids else None


def current_max_block_id(folds: list[Path], pop: str, stage: str) -> int | None:
    pat = re.compile(rf"{re.escape(pop)}{re.escape(stage)}__fold_(\d+)\.joblib$")
    ids = []
    for p in folds:
        m = pat.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return max(ids) if ids else None


def fold_files(pop: str, stage: str) -> list[Path]:
    """Per-fold joblib checkpoints for a given (pop, stage)."""
    pat = re.compile(rf"{re.escape(pop)}{re.escape(stage)}__fold_(\d+)\.joblib$")
    out = []
    for p in CKPT_DIR.glob(f"{pop}{stage}__fold_*.joblib"):
        m = pat.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return [p for _, p in out]


def stage_present(pop: str, stage: str) -> Path | None:
    """Single (non-fold) checkpoint for a stage if it exists."""
    p = CKPT_DIR / f"{pop}{stage}.joblib"
    return p if p.exists() else None


def fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def fmt_auc(arr) -> str:
    arr = [a for a in arr if a is not None and not np.isnan(a)]
    if not arr:
        return " n/a "
    return f"{np.median(arr):.3f}"


def summarize_folds(folds: list[Path]) -> dict:
    rocs, prs, c1, c5, c10, enr1 = [], [], [], [], [], []
    for f in folds:
        try:
            d = joblib.load(f)
        except Exception:
            continue
        rocs.append(d.get("roc_auc"))
        prs.append(d.get("pr_auc"))
        c1.append(d.get("capture_at_1pct"))
        c5.append(d.get("capture_at_5pct"))
        c10.append(d.get("capture_at_10pct"))
        enr1.append(d.get("enrichment_at_1pct"))
    return {
        "n": len(folds),
        "roc_med": fmt_auc(rocs),
        "pr_med": fmt_auc(prs),
        "c1_med": fmt_auc(c1),
        "c5_med": fmt_auc(c5),
        "c10_med": fmt_auc(c10),
        "enr1_med": np.median([e for e in enr1 if e is not None and not np.isnan(e)]) if enr1 else float("nan"),
    }


def fold_pace_eta(folds: list[Path], target: int | None) -> str:
    """Estimate seconds-per-fold from the timestamps of the last K fold files."""
    if len(folds) < 2:
        return "ETA n/a"
    mtimes = sorted(f.stat().st_mtime for f in folds)
    deltas = np.diff(mtimes[-min(10, len(mtimes)):])
    if len(deltas) == 0:
        return "ETA n/a"
    s_per_fold = float(np.median(deltas))
    if target is None:
        return f"{s_per_fold/60:.1f} min/fold, target unknown"
    remaining = max(0, target - len(folds))
    eta_min = remaining * s_per_fold / 60.0
    eta_hr = eta_min / 60.0
    eta_str = f"~{eta_min:.0f} min" if eta_min < 90 else f"~{eta_hr:.1f} h"
    return (f"{s_per_fold/60:.1f} min/fold, "
            f"{remaining}/{target} folds left, "
            f"{eta_str} remaining in this stage")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", default="placer_quaternary",
                        choices=("placer_quaternary", "placer_tertiary"))
    args = parser.parse_args()

    pop = args.pop
    print(f"==> v3.7 training status — {pop}")
    print(f"    checkpoints: {CKPT_DIR}")
    print(f"    log:         {LOG_PATH}")

    # Header
    print()
    print(f"  {'Stage':22s} {'progress':<20s} {'ROC-AUC':>8s} {'PR-AUC':>8s} "
          f"{'cap@1%':>7s} {'cap@5%':>7s} {'cap@10%':>8s} {'pace / ETA':<55s}")
    print(f"  {'-'*22} {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*55}")

    completed_stages = 0
    fold_stages_count = 0
    fold_stages_done = 0
    for stage in STAGE_ORDER:
        label = STAGE_LABEL[stage]
        is_fold_stage = stage in ("__rf_cv", "__lgbm_cv", "__xgb_cv")
        if is_fold_stage:
            folds = fold_files(pop, stage)
            fold_stages_count += 1
            s = summarize_folds(folds)
            # Track progress by max block_id (the SpatialBlockCV upper bound),
            # not by checkpoint count (which depends on positive distribution).
            v36_max = v36_max_block_id(pop, stage)
            v37_max = current_max_block_id(folds, pop, stage)
            target_str = f"blk≤{v36_max}" if v36_max is not None else "??"
            progress_str = (f"{s['n']} files, blk={v37_max}/{v36_max}"
                            if v37_max is not None and v36_max is not None
                            else f"{s['n']} files")
            is_done = (v37_max is not None and v36_max is not None
                       and v37_max >= v36_max)
            if not is_done and s["n"] > 0:
                # Estimate remaining via block-id distance, treating each
                # remaining block_id as worth ~the same wall-clock as recent ones.
                remaining_blocks = max(0, (v36_max or 0) - (v37_max or 0))
                if len(folds) >= 2:
                    mtimes = sorted(f.stat().st_mtime for f in folds)
                    deltas = np.diff(mtimes[-min(10, len(mtimes)):])
                    if len(deltas) > 0:
                        s_per_blk = float(np.median(deltas))
                        eta_h = remaining_blocks * s_per_blk / 3600.0
                        pace = (f"{s_per_blk/60:.1f} min/blk, "
                                f"{remaining_blocks} blks left, "
                                f"~{eta_h:.1f}h remaining")
                    else:
                        pace = "running"
                else:
                    pace = "running"
            elif is_done:
                pace = "complete"
                fold_stages_done += 1
                completed_stages += 1
            else:
                pace = "not started"
            print(f"  {label:22s} {progress_str:<20s} "
                  f"{s['roc_med']:>8s} {s['pr_med']:>8s} "
                  f"{s['c1_med']:>7s} {s['c5_med']:>7s} {s['c10_med']:>8s} {pace:<55s}")
        else:
            p = stage_present(pop, stage)
            if p:
                completed_stages += 1
                print(f"  {label:22s} {'DONE':>7s} {'':>8s} {'':>8s} "
                      f"{'':>7s} {'':>7s} {'':>8s} {p.name:<55s}")
            else:
                print(f"  {label:22s} {'wait':>7s} {'-':>8s} {'-':>8s} "
                      f"{'-':>7s} {'-':>7s} {'-':>8s} {'not started':<55s}")

    print()
    # Overall ETA based on the in-flight stage's pace
    print(f"  Stages complete: {completed_stages}/{len(STAGE_ORDER)}")
    if fold_stages_count:
        print(f"  Fold stages: {fold_stages_done}/{fold_stages_count} complete")
    # Tail of log
    if LOG_PATH.exists():
        log_text = LOG_PATH.read_text().splitlines()
        print(f"  Log tail ({len(log_text)} lines):")
        for line in log_text[-8:]:
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
