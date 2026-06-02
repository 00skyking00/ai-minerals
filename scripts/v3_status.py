"""v3 placer training status dashboard.

Usage:
    python scripts/v3_status.py

For auto-refresh:
    watch -n 60 python scripts/v3_status.py

For SSH dashboard:
    ssh user@host watch -n 60 'cd ai-minerals && python scripts/v3_status.py'

Returns 0 always (this is a status report, not a check).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DERIVED = REPO_ROOT / "data" / "derived" / "northern_sierra_placer"

POPS = ["placer_tertiary", "placer_quaternary"]
STAGES = ["pu", "rf_cv", "lgbm_cv", "stack_oof", "fullfit", "calibrate"]
PER_FOLD_STAGES = {"rf_cv", "lgbm_cv", "stack_oof"}

# Historical v2 timing (minutes per stage per population).
V2_TIMING_MIN = {
    "pu": 2.5,
    "rf_cv": 80.0,
    "lgbm_cv": 40.0,
    "stack_oof": 440.0,
    "fullfit": 5.0,
    "calibrate": 1.0,
}

EXPECTED_FOLDS = 53  # from v2 quaternary run

LOG_CANDIDATES = [
    Path("/tmp/v3_train.log"),
    Path("/tmp/k5_train.log"),
    DERIVED / "_v2_killed_train.log",
]

CKPT_CANDIDATES = [
    DERIVED / "_k5_checkpoints",
    DERIVED / "_k5_checkpoints_a",
    DERIVED / "_k5_checkpoints_b",
    DERIVED / "_v2_checkpoints_killed",
]


# ----- color -----

def _color_enabled() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _wrap(code: str, text: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def red(s: str) -> str: return _wrap("31", s)
def green(s: str) -> str: return _wrap("32", s)
def yellow(s: str) -> str: return _wrap("33", s)
def cyan(s: str) -> str: return _wrap("36", s)
def dim(s: str) -> str: return _wrap("2", s)
def bold(s: str) -> str: return _wrap("1", s)


# ----- log discovery + parsing -----

def find_log() -> Path | None:
    for p in LOG_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


@dataclass
class StageRecord:
    started: bool = False
    finished: bool = False
    cache_hit: bool = False
    minutes: float | None = None
    folds: int | None = None
    auc: float | None = None
    prauc: float | None = None


@dataclass
class PopRecord:
    name: str
    seen: bool = False
    stages: dict[str, StageRecord] = field(default_factory=dict)

    def stage(self, name: str) -> StageRecord:
        return self.stages.setdefault(name, StageRecord())


def _maybe_float(tok: str) -> float | None:
    try:
        return float(tok)
    except ValueError:
        return None


def _parse_done_min(line: str) -> float | None:
    # ".. done in 4.4 min .." → 4.4
    parts = line.split()
    for i, tok in enumerate(parts):
        if tok == "in" and i + 2 < len(parts) and parts[i + 2].startswith("min"):
            v = _maybe_float(parts[i + 1])
            if v is not None:
                return v
    return None


def _parse_kv(line: str, key: str) -> str | None:
    # "folds=53" → "53"
    for tok in line.replace(",", " ").split():
        if tok.startswith(key + "="):
            return tok.split("=", 1)[1]
    return None


def parse_log(path: Path) -> dict[str, PopRecord]:
    pops = {p: PopRecord(name=p) for p in POPS}
    current: str | None = None
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # population header
            if line.startswith("=========") and "placer_" in line:
                for p in POPS:
                    if p in line:
                        current = p
                        pops[p].seen = True
                        break
                continue

            if current is None:
                continue
            rec = pops[current]

            if "[cache hit]" in line:
                for st in STAGES:
                    tag = f"{current}__{st}"
                    if tag in line:
                        s = rec.stage(st)
                        s.cache_hit = True
                        s.started = True
                        s.finished = True
                continue

            if "PU bagging" in line:
                rec.stage("pu").started = True
            elif "PU train done" in line:
                s = rec.stage("pu")
                s.started = True
                s.finished = True
                s.minutes = _parse_done_min(line)
            elif "RF spatial-block CV" in line:
                rec.stage("rf_cv").started = True
            elif "RF CV done" in line:
                s = rec.stage("rf_cv")
                s.started = True
                s.finished = True
                s.minutes = _parse_done_min(line)
                folds = _parse_kv(line, "folds")
                if folds is not None:
                    s.folds = int(folds)
                auc = _parse_kv(line, "AUC mean") or _parse_kv(line, "mean")
                if auc is not None:
                    s.auc = _maybe_float(auc)
            elif "LightGBM spatial-block CV" in line:
                rec.stage("lgbm_cv").started = True
            elif "LGBM CV done" in line:
                s = rec.stage("lgbm_cv")
                s.started = True
                s.finished = True
                s.minutes = _parse_done_min(line)
                folds = _parse_kv(line, "folds")
                if folds is not None:
                    s.folds = int(folds)
                auc = _parse_kv(line, "AUC mean") or _parse_kv(line, "mean")
                if auc is not None:
                    s.auc = _maybe_float(auc)
            elif "stacking: spatial-block OOF" in line:
                rec.stage("stack_oof").started = True
            elif "stacking OOF done" in line:
                s = rec.stage("stack_oof")
                s.started = True
                s.finished = True
                s.minutes = _parse_done_min(line)
            elif "stacking OOF AUC" in line:
                s = rec.stage("stack_oof")
                s.started = True
                s.finished = True
                auc = _parse_kv(line, "AUC")
                if auc is not None:
                    s.auc = _maybe_float(auc)
                prauc = _parse_kv(line, "PR-AUC")
                if prauc is not None:
                    s.prauc = _maybe_float(prauc)
            elif "full-data refits" in line:
                rec.stage("fullfit").started = True
            elif "refit + predict done" in line:
                s = rec.stage("fullfit")
                s.started = True
                s.finished = True
                s.minutes = _parse_done_min(line)
            elif "calibration (method=" in line:
                rec.stage("calibrate").started = True
            elif "calibration done" in line:
                s = rec.stage("calibrate")
                s.started = True
                s.finished = True
                s.minutes = _parse_done_min(line)

    return pops


# ----- checkpoint counting -----

@dataclass
class FoldCounts:
    fold_files: dict[tuple[str, str], int] = field(default_factory=dict)
    finalized: set[tuple[str, str]] = field(default_factory=set)


def count_checkpoints() -> tuple[FoldCounts, list[Path]]:
    counts = FoldCounts()
    used: list[Path] = []
    for d in CKPT_CANDIDATES:
        if not d.exists():
            continue
        any_in_dir = False
        for f in d.iterdir():
            if f.suffix != ".joblib":
                continue
            stem = f.stem  # "<pop>__<stage>[__fold_N]"
            parts = stem.split("__")
            if len(parts) < 2:
                continue
            pop = parts[0]
            stage = parts[1]
            key = (pop, stage)
            if len(parts) >= 3 and parts[2].startswith("fold_"):
                counts.fold_files[key] = counts.fold_files.get(key, 0) + 1
            else:
                counts.finalized.add(key)
            any_in_dir = True
        if any_in_dir:
            used.append(d)
    return counts, used


# ----- process check -----

def find_running_process() -> tuple[str, str] | None:
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,time,cmd"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if "northern_sierra_placer_train_predict" in line and "grep" not in line:
            parts = line.split(None, 3)
            if len(parts) >= 3:
                pid, etime, cpu_time = parts[0], parts[1], parts[2]
                return pid, f"elapsed={etime} cpu={cpu_time}"
    return None


# ----- deliverables -----

DELIV_TEMPLATES = [
    "pop_predictions_{pop}_250m.parquet",
    "pop_calibrated_{pop}_250m.parquet",
    "pop_fold_metrics_{pop}.csv",
    "prospectivity_placer_{pop}_250m_calibrated_3310.tif",
    "prospectivity_placer_{pop}_250m_calibrated_4326.tif",
]


def _human_size(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def deliverable_status(pop: str) -> list[tuple[str, bool, int]]:
    rows = []
    for tpl in DELIV_TEMPLATES:
        name = tpl.format(pop=pop)
        path = DERIVED / name
        if path.exists():
            rows.append((name, True, path.stat().st_size))
        else:
            rows.append((name, False, 0))
    return rows


# ----- progress + ETA -----

def estimate_progress(pops: dict[str, PopRecord]) -> tuple[float, float]:
    total = 0.0
    done = 0.0
    for pop in POPS:
        rec = pops[pop]
        for st in STAGES:
            budget = V2_TIMING_MIN[st]
            total += budget
            s = rec.stages.get(st)
            if s is None:
                continue
            if s.finished:
                done += budget
            elif s.started:
                # in progress: credit half
                done += budget * 0.5
    if total <= 0:
        return 0.0, 0.0
    pct = 100.0 * done / total
    remaining = max(0.0, total - done)
    return pct, remaining


# ----- last log lines -----

_NOISE_SUBSTRINGS = (
    "UserWarning",
    "warnings.warn",
    "valid feature names",
)


def _is_noise(line: str) -> bool:
    return any(sub in line for sub in _NOISE_SUBSTRINGS)


def tail_lines(path: Path, n: int = 5) -> list[str]:
    # Avoid loading huge logs entirely.
    try:
        size = path.stat().st_size
    except OSError:
        return []
    chunk = min(size, 256 * 1024)
    with path.open("rb") as f:
        f.seek(max(0, size - chunk))
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not _is_noise(ln)]
    return lines[-n:]


# ----- rendering -----

def _stage_marker(s: StageRecord) -> str:
    if s.cache_hit:
        return cyan("CACHE")
    if s.finished:
        return green(" DONE")
    if s.started:
        return yellow(" RUN ")
    return dim(" --- ")


def _stage_detail(stage_name: str, s: StageRecord) -> str:
    bits = []
    if s.minutes is not None:
        bits.append(f"{s.minutes:.1f}m")
    if s.folds is not None:
        bits.append(f"folds={s.folds}")
    if s.auc is not None:
        bits.append(f"AUC={s.auc:.3f}")
    if s.prauc is not None:
        bits.append(f"PR={s.prauc:.3f}")
    return "  ".join(bits)


def render(log_path: Path | None, pops: dict[str, PopRecord],
           ckpt: FoldCounts, ckpt_dirs: list[Path],
           proc: tuple[str, str] | None) -> str:
    out: list[str] = []
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    out.append(bold(f"v3 placer training status  {now}"))
    out.append("=" * 72)

    if log_path is None:
        out.append(red("no log found"))
        out.append("  searched: " + ", ".join(str(p) for p in LOG_CANDIDATES))
        return "\n".join(out)

    try:
        mtime = time.strftime("%H:%M:%S", time.localtime(log_path.stat().st_mtime))
    except OSError:
        mtime = "?"
    out.append(f"log: {log_path}  (last write {mtime})")

    if proc is None:
        out.append("process: " + red("not running"))
    else:
        pid, info = proc
        out.append("process: " + green(f"pid={pid}  {info}"))

    if ckpt_dirs:
        out.append("checkpoints: " + ", ".join(str(p.relative_to(DERIVED.parent)) for p in ckpt_dirs))
    else:
        out.append("checkpoints: " + dim("none"))

    out.append("")
    out.append(bold("stages"))
    out.append(f"  {'population':<22} {'stage':<10} {'status':<8} {'folds':<14} {'detail'}")
    for pop in POPS:
        rec = pops[pop]
        for st in STAGES:
            s = rec.stages.get(st, StageRecord())
            fold_str = ""
            if st in PER_FOLD_STAGES and not s.cache_hit:
                n = ckpt.fold_files.get((pop, st), 0)
                if s.folds is not None:
                    fold_str = f"{s.folds}/{s.folds}"
                else:
                    fold_str = f"{n}/{EXPECTED_FOLDS}"
            out.append(
                f"  {pop:<22} {st:<10} {_stage_marker(s)}  {fold_str:<14} {_stage_detail(st, s)}"
            )
        out.append("")

    out.append(bold("deliverables"))
    for pop in POPS:
        out.append(f"  {pop}:")
        for name, present, size in deliverable_status(pop):
            if present:
                out.append(f"    {green('OK ')} {name}  {_human_size(size)}")
            else:
                out.append(f"    {red('-- ')} {name}")

    pct, remain_min = estimate_progress(pops)
    out.append("")
    eta_str = f"~{remain_min:.0f} min" if remain_min > 0 else "~0 min"
    if remain_min >= 60:
        eta_str += f" ({remain_min / 60:.1f} h)"
    color_pct = green if pct >= 90 else yellow if pct >= 40 else red
    out.append(bold("progress: ") + color_pct(f"{pct:.0f}%") + f"   remaining: {eta_str}")
    out.append(dim("  (v2 fallback budgets; in-progress stages credited 50%)"))

    out.append("")
    out.append(bold("last 5 log lines"))
    for ln in tail_lines(log_path, 5):
        if len(ln) > 200:
            ln = ln[:200] + "..."
        out.append("  " + dim(ln))

    return "\n".join(out)


def main() -> int:
    log_path = find_log()
    pops = parse_log(log_path) if log_path else {p: PopRecord(name=p) for p in POPS}
    ckpt, ckpt_dirs = count_checkpoints()
    proc = find_running_process()
    print(render(log_path, pops, ckpt, ckpt_dirs, proc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
