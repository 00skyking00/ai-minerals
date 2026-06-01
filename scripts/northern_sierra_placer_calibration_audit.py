"""Phase I: per-population calibration audit for the northern-Sierra placer raster.

For each population (placer_tertiary, placer_quaternary), compare the
pre-isotonic stack probability (`p_stack`) against the post-isotonic
calibrated probability (`p_cal`) on the anchor-district positive set.
Writes a CSV with per-bin reliability and ECE, a PNG reliability plot,
and prints a pass/marginal/disclaim classification per the Phase L
decision matrix in `~/.claude/plans/hazy-humming-lynx.md`.

Pass thresholds:
  ECE <= 0.05:                  clean
  0.05 < ECE <= 0.10:           marginal
  ECE > 0.10:                   disclaim

Inputs (under data/derived/northern_sierra_placer/):
  pop_predictions_<pop>_250m.parquet     (columns: row, col, x, y, p_stack, ...)
  pop_calibrated_<pop>_250m.parquet      (columns: row, col, x, y, p_cal)

Outputs:
  calibration_audit_<pop>.csv
  calibration_reliability_<pop>.png

Usage:
    .venv/bin/python scripts/northern_sierra_placer_calibration_audit.py
    .venv/bin/python scripts/northern_sierra_placer_calibration_audit.py --population placer_tertiary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from ai_minerals.metrics.calibration import (
    expected_calibration_error,
    reliability_table,
)
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
OUT_DIR = DATA_DERIVED / REGION.data_prefix

POPULATIONS = ("placer_tertiary", "placer_quaternary")
N_BINS = 10

CLEAN_THRESHOLD = 0.05
MARGINAL_THRESHOLD = 0.10


def _anchor_cell_indices(df: pd.DataFrame) -> pd.Series:
    """Snap each anchor district's (lon, lat) to the nearest grid-cell DataFrame index."""
    transformer = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    names, idxs = [], []
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    for name, (lon, lat) in ANCHOR_DISTRICTS.items():
        ax, ay = transformer.transform(lon, lat)
        d2 = (xs - ax) ** 2 + (ys - ay) ** 2
        cell = int(np.argmin(d2))
        names.append(name)
        idxs.append(df.index[cell])
    return pd.Series(idxs, index=names, name="cell_idx")


def _classify(ece: float) -> str:
    if ece <= CLEAN_THRESHOLD:
        return "clean"
    if ece <= MARGINAL_THRESHOLD:
        return "marginal"
    return "disclaim"


def _print_binned_table(label: str, table: pd.DataFrame) -> None:
    print(f"=== {label} ===")
    for _, row in table.iterrows():
        lo = float(row["bin_left"])
        hi = float(row["bin_right"])
        n = int(row["count"])
        rate = float(row["pos_rate"]) if pd.notna(row["pos_rate"]) else float("nan")
        print(f"  P=[{lo:.1f},{hi:.1f}): n={n:7d}  pos_rate={rate:.4f}")


def _plot_reliability(
    *,
    out_path: Path,
    pre: pd.DataFrame,
    post: pd.DataFrame,
    ece_pre: float,
    ece_post: float,
    pop: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="perfect")

    def _series(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mid = ((table["bin_left"] + table["bin_right"]) / 2.0).to_numpy()
        rate = table["pos_rate"].to_numpy(dtype=float)
        count = table["count"].to_numpy(dtype=int)
        return mid, rate, count

    mid_pre, rate_pre, cnt_pre = _series(pre)
    mid_post, rate_post, cnt_post = _series(post)

    ax.plot(mid_pre, rate_pre, color="C1", marker="o", linewidth=2,
            label=f"pre-isotonic stack: ECE={ece_pre:.3f}")
    ax.plot(mid_post, rate_post, color="C0", marker="s", linewidth=2,
            label=f"post-isotonic cal: ECE={ece_post:.3f}")

    for x, y, n in zip(mid_pre, rate_pre, cnt_pre):
        if np.isfinite(y) and n > 0:
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color="C1")
    for x, y, n in zip(mid_post, rate_post, cnt_post):
        if np.isfinite(y) and n > 0:
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7, color="C0")

    ax.set_xlabel("Predicted P (bin midpoint)")
    ax.set_ylabel("Observed positive rate (anchor cells)")
    ax.set_title(f"Reliability — {pop}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _audit_population(pop: str) -> dict[str, float | str] | None:
    pred_path = OUT_DIR / f"pop_predictions_{pop}_250m.parquet"
    cal_path = OUT_DIR / f"pop_calibrated_{pop}_250m.parquet"
    if not pred_path.exists() or not cal_path.exists():
        print(
            f"ERROR: missing input parquet(s) for {pop}:\n"
            f"  predictions: {pred_path}  exists={pred_path.exists()}\n"
            f"  calibrated:  {cal_path}  exists={cal_path.exists()}\n"
            f"Run scripts/northern_sierra_placer_train_predict_250m.py first.",
            file=sys.stderr,
        )
        return None

    preds = pd.read_parquet(pred_path)
    cal = pd.read_parquet(cal_path)

    if "p_stack" not in preds.columns:
        print(f"ERROR: {pred_path} missing column 'p_stack'.", file=sys.stderr)
        return None
    if "p_cal" not in cal.columns:
        print(f"ERROR: {cal_path} missing column 'p_cal'.", file=sys.stderr)
        return None

    merged = preds[["row", "col", "x", "y", "p_stack"]].merge(
        cal[["row", "col", "p_cal"]], on=["row", "col"], how="inner"
    ).reset_index(drop=True)

    anchor_idxs = _anchor_cell_indices(merged)
    y = np.zeros(len(merged), dtype=int)
    y[anchor_idxs.values] = 1

    p_pre = merged["p_stack"].to_numpy(dtype=float)
    p_post = merged["p_cal"].to_numpy(dtype=float)

    finite_pre = np.isfinite(p_pre)
    finite_post = np.isfinite(p_post)

    table_pre = reliability_table(p_pre[finite_pre], y[finite_pre], n_bins=N_BINS)
    table_post = reliability_table(p_post[finite_post], y[finite_post], n_bins=N_BINS)
    ece_pre = float(expected_calibration_error(p_pre[finite_pre], y[finite_pre], n_bins=N_BINS))
    ece_post = float(expected_calibration_error(p_post[finite_post], y[finite_post], n_bins=N_BINS))

    _print_binned_table(f"{pop}: pre-isotonic stack", table_pre)
    _print_binned_table(f"{pop}: post-isotonic cal", table_post)

    out_csv = OUT_DIR / f"calibration_audit_{pop}.csv"
    out_png = OUT_DIR / f"calibration_reliability_{pop}.png"

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    pre_by_left = {round(float(r["bin_left"]), 6): r for _, r in table_pre.iterrows()}
    post_by_left = {round(float(r["bin_left"]), 6): r for _, r in table_post.iterrows()}

    rows: list[dict[str, object]] = []
    for i in range(N_BINS):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        pre_row = pre_by_left.get(round(lo, 6))
        post_row = post_by_left.get(round(lo, 6))
        rec: dict[str, object] = {
            "bin": i,
            "bin_lo": lo,
            "bin_hi": hi,
            "count_pre": int(pre_row["count"]) if pre_row is not None else 0,
            "mean_pred_pre": (float(pre_row["mean_pred"])
                              if pre_row is not None and pd.notna(pre_row["mean_pred"])
                              else np.nan),
            "pos_rate_pre": (float(pre_row["pos_rate"])
                             if pre_row is not None and pd.notna(pre_row["pos_rate"])
                             else np.nan),
            "count_post": int(post_row["count"]) if post_row is not None else 0,
            "mean_pred_post": (float(post_row["mean_pred"])
                               if post_row is not None and pd.notna(post_row["mean_pred"])
                               else np.nan),
            "pos_rate_post": (float(post_row["pos_rate"])
                              if post_row is not None and pd.notna(post_row["pos_rate"])
                              else np.nan),
            "ece_pre": ece_pre,
            "ece_post": ece_post,
        }
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"    wrote {out_csv}")

    _plot_reliability(
        out_path=out_png, pre=table_pre, post=table_post,
        ece_pre=ece_pre, ece_post=ece_post, pop=pop,
    )
    print(f"    wrote {out_png}")

    cls = _classify(ece_post)
    print(f"\n{pop}: pre ECE={ece_pre:.3f}  post ECE={ece_post:.3f}  (target <= 0.05)  -> {cls}\n")

    return {"pop": pop, "ece_pre": ece_pre, "ece_post": ece_post, "classification": cls}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=POPULATIONS,
        default=None,
        help="Audit a single population (default: both).",
    )
    args = parser.parse_args(argv)

    pops = (args.population,) if args.population else POPULATIONS
    results: list[dict[str, float | str]] = []
    for pop in pops:
        res = _audit_population(pop)
        if res is None:
            return 2
        results.append(res)

    print("=== Summary ===")
    for r in results:
        print(f"  {r['pop']}: pre ECE={r['ece_pre']:.3f}  post ECE={r['ece_post']:.3f}  -> {r['classification']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
