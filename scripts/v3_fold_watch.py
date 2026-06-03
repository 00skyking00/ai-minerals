"""Watch v3 per-fold checkpoints as they land; report running mean AUC.

Usage:
    .venv/bin/python scripts/v3_fold_watch.py

Polls data/derived/northern_sierra_placer/_k5_checkpoints/ every 30 s. Each
time a new fold checkpoint appears, prints one line for that fold and an
updated running aggregate (simple mean AUC, positive-count-weighted mean
AUC restricted to folds with n_pos >= 2, total positives seen, fold count).

Single-positive folds (n_test_pos == 1) are degenerate: any rank order has
AUC near 0.5 by construction. The script flags those folds inline so they
don't drag the running mean visually; the weighted aggregate already
excludes them.

Stop with Ctrl-C.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
CK_DIR = REPO_ROOT / "data" / "derived" / "northern_sierra_placer" / "_k5_checkpoints"
POLL_SECONDS = 30
STAGES_TO_WATCH = ["rf_cv", "lgbm_cv", "stack_oof"]


class FoldMetric(NamedTuple):
    pop: str
    stage: str
    fold_id: int
    auc: float
    pr_auc: float
    n_pos: int
    n_test: int
    capture_5: float


def _pop_short(pop: str) -> str:
    return {
        "placer_tertiary": "T",
        "placer_quaternary": "Q",
        "ablation_no_pit": "Abl",
    }.get(pop, pop)


def _parse_checkpoint(p: Path) -> FoldMetric | None:
    # filename pattern: placer_<pop>__<stage>__fold_<id>.joblib
    name = p.stem
    parts = name.split("__")
    if len(parts) != 3 or not parts[2].startswith("fold_"):
        return None
    pop, stage, fold_part = parts
    try:
        fold_id = int(fold_part.split("_")[-1])
    except ValueError:
        return None
    if stage not in STAGES_TO_WATCH:
        return None
    try:
        d = joblib.load(p)
    except Exception as exc:
        print(f"  [skip] could not load {p.name}: {exc}", file=sys.stderr)
        return None
    return FoldMetric(
        pop=pop,
        stage=stage,
        fold_id=fold_id,
        auc=float(d.get("roc_auc", float("nan"))),
        pr_auc=float(d.get("pr_auc", float("nan"))),
        n_pos=int(d.get("n_test_pos", 0)),
        n_test=int(d.get("n_test", 0)),
        capture_5=float(d.get("capture_at_5pct", float("nan"))),
    )


def _aggregate(metrics: list[FoldMetric]) -> tuple[float, float, int, int]:
    if not metrics:
        return float("nan"), float("nan"), 0, 0
    n_folds = len(metrics)
    aucs = [m.auc for m in metrics]
    simple_mean = sum(aucs) / n_folds
    multi = [m for m in metrics if m.n_pos >= 2]
    total_pos = sum(m.n_pos for m in metrics)
    if not multi:
        return simple_mean, float("nan"), n_folds, total_pos
    weights = [m.n_pos for m in multi]
    weighted_mean = sum(m.auc * w for m, w in zip(multi, weights)) / sum(weights)
    return simple_mean, weighted_mean, n_folds, total_pos


def main() -> None:
    if not CK_DIR.exists():
        print(f"checkpoint dir not present: {CK_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"watching {CK_DIR}")
    print(f"polling every {POLL_SECONDS}s; Ctrl-C to stop\n")
    print(f"{'time':<10} {'fold':<14} {'AUC':>6}  {'cap@5%':>7}  {'n_pos':>5}    "
          f"{'running mean (n_pos>=2)':<28}  {'(plain mean)':<14}  {'pos seen':<8}")
    print("-" * 120)

    seen: set[Path] = set()
    metrics_by_group: dict[tuple[str, str], list[FoldMetric]] = defaultdict(list)

    # On first tick, load any existing checkpoints silently so we don't spam
    # the screen replaying history. After that, only print newly-landed ones.
    first_pass = True
    while True:
        try:
            current = set(CK_DIR.glob("*__*_cv__fold_*.joblib")) \
                | set(CK_DIR.glob("*__stack_oof__fold_*.joblib"))
            new = sorted(current - seen, key=lambda p: p.stat().st_mtime)
            for p in new:
                m = _parse_checkpoint(p)
                seen.add(p)
                if m is None:
                    continue
                key = (m.pop, m.stage)
                metrics_by_group[key].append(m)
                if first_pass:
                    continue
                ts = time.strftime("%H:%M:%S")
                tag = f"{_pop_short(m.pop)}/{m.stage} f{m.fold_id:>2}"
                degen = "  (single-pos: degenerate)" if m.n_pos < 2 else ""
                simple, weighted, n_folds, total_pos = _aggregate(metrics_by_group[key])
                weighted_s = (f"{weighted:.3f} ({n_folds} folds)" if weighted == weighted
                              else "(no multi-pos folds yet)")
                print(f"{ts:<10} {tag:<14} {m.auc:>6.3f}  {m.capture_5:>7.2f}  "
                      f"{m.n_pos:>5}    {weighted_s:<28}  "
                      f"{simple:>6.3f}        {total_pos:<8}{degen}")
            if first_pass and new:
                # After silent backfill, print a baseline summary per group.
                first_pass = False
                ts = time.strftime("%H:%M:%S")
                print(f"{ts} backfill: {len(new)} existing checkpoints loaded silently.")
                for (pop, stage), ms in sorted(metrics_by_group.items()):
                    simple, weighted, n_folds, total_pos = _aggregate(ms)
                    short = _pop_short(pop)
                    if weighted == weighted:
                        print(f"  {short}/{stage}: {n_folds} folds; "
                              f"plain mean AUC={simple:.3f}; "
                              f"pos-weighted (n>=2) AUC={weighted:.3f}; "
                              f"total pos={total_pos}")
                    else:
                        print(f"  {short}/{stage}: {n_folds} folds; "
                              f"plain mean AUC={simple:.3f}; "
                              f"no multi-pos folds yet; total pos={total_pos}")
                print()
            elif first_pass:
                first_pass = False
                print(f"{time.strftime('%H:%M:%S')} no existing checkpoints; "
                      f"waiting for first fold...")
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("\nstopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
