"""v3.7 training fold-watcher.

Prints every per-fold metric row as it lands in the checkpoint directory,
followed by a running weighted average across folds completed so far. Polls
for new checkpoints and prints them as they arrive. Ctrl-C exits without
disturbing the running training process.

Metric per fold: ROC-AUC, PR-AUC, capture@1%/5%/10%, enrichment@1%, n_test_pos.
Weighted average uses n_test_pos as weights (so folds with more positives
contribute more, matching how the stacking OOF aggregator sees them).

Usage:
    .venv/bin/python scripts/v37_training_watch.py
    .venv/bin/python scripts/v37_training_watch.py --pop placer_tertiary
    .venv/bin/python scripts/v37_training_watch.py --poll-seconds 30
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

STAGES = ("__rf_cv", "__lgbm_cv", "__xgb_cv")
STAGE_LABEL = {"__rf_cv": "RF", "__lgbm_cv": "LGBM", "__xgb_cv": "XGB"}


def fold_id(path: Path, pop: str, stage: str) -> int | None:
    pat = re.compile(rf"{re.escape(pop)}{re.escape(stage)}__fold_(\d+)\.joblib$")
    m = pat.search(path.name)
    return int(m.group(1)) if m else None


def discover(pop: str) -> list[tuple[str, int, Path]]:
    """Return (stage, fold_id, path), sorted by mtime so playback matches arrival."""
    rows = []
    for stage in STAGES:
        for p in CKPT_DIR.glob(f"{pop}{stage}__fold_*.joblib"):
            fid = fold_id(p, pop, stage)
            if fid is not None:
                rows.append((stage, fid, p))
    rows.sort(key=lambda t: t[2].stat().st_mtime)
    return rows


def load(p: Path) -> dict | None:
    try:
        return joblib.load(p)
    except Exception:
        return None


def fmt_metric(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " n/a "
    return f"{v:.3f}"


def fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{int(v):d}"


def print_row(seq: int, stage: str, fid: int, d: dict | None, running: dict | None) -> None:
    stage_lbl = STAGE_LABEL.get(stage, stage)
    if d is None:
        print(f"  [{seq:>4d}] {stage_lbl:<5s} blk={fid:>3d}  (failed to load)")
        return
    print(
        f"  [{seq:>4d}] {stage_lbl:<5s} blk={fid:>3d}  "
        f"ROC={fmt_metric(d.get('roc_auc'))} "
        f"PR={fmt_metric(d.get('pr_auc'))} "
        f"c@1={fmt_metric(d.get('capture_at_1pct'))} "
        f"c@5={fmt_metric(d.get('capture_at_5pct'))} "
        f"c@10={fmt_metric(d.get('capture_at_10pct'))} "
        f"npos={fmt_int(d.get('n_test_pos')):>4s}  "
        + (f"| {STAGE_LABEL.get(stage, stage)} avg: "
           f"ROC_w={running['roc_w']:.3f} "
           f"PR_w={running['pr_w']:.3f} "
           f"(n={running['n_folds']})"
           if running is not None else "")
    )


def update_running(state: dict, stage: str, d: dict) -> dict:
    s = state.setdefault(stage, {
        "roc_num": 0.0, "pr_num": 0.0, "w_sum": 0.0, "n": 0,
        "c1_num": 0.0, "c5_num": 0.0, "c10_num": 0.0,
    })
    w = float(d.get("n_test_pos") or 0.0)
    roc = d.get("roc_auc")
    pr = d.get("pr_auc")
    c1 = d.get("capture_at_1pct")
    c5 = d.get("capture_at_5pct")
    c10 = d.get("capture_at_10pct")
    if w > 0 and roc is not None and not np.isnan(roc):
        s["roc_num"] += w * roc
        s["pr_num"] += w * (pr if pr is not None and not np.isnan(pr) else 0.0)
        s["c1_num"] += w * (c1 if c1 is not None and not np.isnan(c1) else 0.0)
        s["c5_num"] += w * (c5 if c5 is not None and not np.isnan(c5) else 0.0)
        s["c10_num"] += w * (c10 if c10 is not None and not np.isnan(c10) else 0.0)
        s["w_sum"] += w
    s["n"] += 1
    return {
        "roc_w": (s["roc_num"] / s["w_sum"]) if s["w_sum"] > 0 else float("nan"),
        "pr_w": (s["pr_num"] / s["w_sum"]) if s["w_sum"] > 0 else float("nan"),
        "c1_w": (s["c1_num"] / s["w_sum"]) if s["w_sum"] > 0 else float("nan"),
        "c5_w": (s["c5_num"] / s["w_sum"]) if s["w_sum"] > 0 else float("nan"),
        "c10_w": (s["c10_num"] / s["w_sum"]) if s["w_sum"] > 0 else float("nan"),
        "n_folds": s["n"],
    }


def print_stage_summary(state: dict) -> None:
    print()
    print("  --- weighted averages so far (weights = n_test_pos per fold) ---")
    for stage in STAGES:
        if stage not in state:
            continue
        s = state[stage]
        if s["w_sum"] == 0:
            continue
        print(
            f"    {STAGE_LABEL[stage]:<5s} n={s['n']:>3d}  "
            f"ROC_w={s['roc_num']/s['w_sum']:.3f}  "
            f"PR_w={s['pr_num']/s['w_sum']:.3f}  "
            f"c@1={s['c1_num']/s['w_sum']:.3f}  "
            f"c@5={s['c5_num']/s['w_sum']:.3f}  "
            f"c@10={s['c10_num']/s['w_sum']:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", default="placer_quaternary",
                        choices=("placer_quaternary", "placer_tertiary"))
    parser.add_argument("--poll-seconds", type=float, default=15.0,
                        help="Polling interval when waiting for new folds.")
    parser.add_argument("--once", action="store_true",
                        help="Print backlog and exit without watching.")
    args = parser.parse_args()

    pop = args.pop
    print(f"==> v3.7 fold watcher — {pop}")
    print(f"    checkpoint dir: {CKPT_DIR}")
    print(f"    weights = n_test_pos per fold (matches stacking OOF aggregator).")
    print()
    print(f"  {'#':>4s}  {'stage':<5s} {'blk':<5s} "
          f"{'ROC':<7s} {'PR':<7s} {'c@1':<7s} {'c@5':<7s} {'c@10':<7s} "
          f"{'npos':<5s} running")
    print(f"  {'-'*4}  {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*40}")

    seen: dict[Path, float] = {}
    state: dict = {}
    seq = 0

    # Initial backlog
    for stage, fid, p in discover(pop):
        d = load(p)
        running = update_running(state, stage, d) if d else None
        seq += 1
        print_row(seq, stage, fid, d, running)
        seen[p] = p.stat().st_mtime

    print_stage_summary(state)

    if args.once:
        return 0

    # Watch loop
    print()
    print(f"  ...waiting for new folds (Ctrl-C to exit, polling every {args.poll_seconds:.0f}s)...")
    try:
        while True:
            time.sleep(args.poll_seconds)
            new_rows = []
            for stage, fid, p in discover(pop):
                mtime = p.stat().st_mtime
                if p not in seen or seen[p] != mtime:
                    new_rows.append((stage, fid, p))
                    seen[p] = mtime
            if new_rows:
                for stage, fid, p in new_rows:
                    d = load(p)
                    running = update_running(state, stage, d) if d else None
                    seq += 1
                    print_row(seq, stage, fid, d, running)
                print_stage_summary(state)
    except KeyboardInterrupt:
        print()
        print("  exit (training process unaffected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
