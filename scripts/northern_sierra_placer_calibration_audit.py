"""Per-population calibration audit + validation for the northern-Sierra placer raster.

Two analyses run per population:

1. Anchor-set audit (carried over from v2 Phase I). The seven anchor districts
   are snapped to their nearest grid cell and treated as the positive set;
   the pre-isotonic stack (`p_stack`) and the post-isotonic calibrated score
   (`p_cal`) are binned into 10 equal-width buckets; ECE is computed against
   the anchor positives. Outputs: `calibration_audit_<pop>.csv`,
   `calibration_audit_anchors_<pop>.png`. Classified into
   clean/marginal/disclaim per:

     ECE <= 0.05:                 clean
     0.05 < ECE <= 0.10:          marginal
     ECE > 0.10:                  disclaim

2. v3 Phase E.1 calibration validation. Uses the full `is_placer_<pop>`
   label set (158 Tertiary pit-centroid positives, 573 Quaternary MRDS
   positives) joined from `features_northern_sierra_placer_250m.parquet`.
   Bins `p_cal` into 10 equal-frequency buckets (`pd.qcut` with
   `duplicates='drop'` because the v2 calibrated distribution is bimodal
   and does not have 10 distinct quantiles). Outputs:
   `calibration_validation_<pop>.csv` and `calibration_reliability_<pop>.png`;
   pre-isotonic counterparts use the `_precal` suffix on both files.

Inputs (under data/derived/northern_sierra_placer/):
  pop_predictions_<pop>_250m.parquet     (columns: row, col, x, y, p_stack, ...)
  pop_calibrated_<pop>_250m.parquet      (columns: row, col, x, y, p_cal)

Additional input (under data/derived/):
  features_northern_sierra_placer_250m.parquet  (columns: row, col, ..., is_placer_<pop>)

Outputs (under data/derived/northern_sierra_placer/):
  calibration_audit_<pop>.csv              # anchor-set audit, both pre + post
  calibration_audit_anchors_<pop>.png      # anchor-set reliability diagram
  calibration_validation_<pop>.csv         # E.1 post-isotonic equal-freq table
  calibration_reliability_<pop>.png        # E.1 post-isotonic reliability diagram
  calibration_validation_<pop>_precal.csv  # E.1 pre-isotonic equal-freq table
  calibration_reliability_<pop>_precal.png # E.1 pre-isotonic reliability diagram

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
FEATURES_PARQUET = DATA_DERIVED / f"features_{REGION.data_prefix}_250m.parquet"

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
    out_png = OUT_DIR / f"calibration_audit_anchors_{pop}.png"

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


# ---------------------------------------------------------------------------
# v3 Phase E.1: per-population calibration validation against real labels
# ---------------------------------------------------------------------------


def _equal_freq_reliability_table(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Equal-frequency reliability table; collapses duplicate quantile edges.

    Bimodal v2 distributions can have <n_bins distinct quantiles, so
    `pd.qcut(..., duplicates='drop')` is used and the resulting number of
    bins may be smaller than `n_bins`. Returns columns
    `bin, count, mean_pred, observed_rate, abs_diff`.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    if scores.shape != y_true.shape:
        raise ValueError(
            f"scores and y_true shape mismatch: {scores.shape} vs {y_true.shape}"
        )
    finite = np.isfinite(scores)
    scores = scores[finite]
    y_true = y_true[finite]
    if scores.size == 0:
        return pd.DataFrame(
            columns=["bin", "count", "mean_pred", "observed_rate", "abs_diff"]
        )

    bins = pd.qcut(scores, q=n_bins, duplicates="drop", labels=False)
    df = pd.DataFrame({"bin": bins, "score": scores, "y": y_true})
    grouped = df.groupby("bin", sort=True).agg(
        count=("score", "size"),
        mean_pred=("score", "mean"),
        observed_rate=("y", "mean"),
    ).reset_index()
    grouped["abs_diff"] = (grouped["observed_rate"] - grouped["mean_pred"]).abs()
    # Re-number bins 0..k-1 so the CSV always has a contiguous index.
    grouped["bin"] = np.arange(len(grouped), dtype=int)
    return grouped[["bin", "count", "mean_pred", "observed_rate", "abs_diff"]]


def _ece_from_table(table: pd.DataFrame) -> float:
    """ECE = sum_b (n_b / N) * |observed - mean_pred|, given an equal-freq table."""
    if table.empty:
        return float("nan")
    n_total = float(table["count"].sum())
    if n_total <= 0:
        return float("nan")
    weights = table["count"].to_numpy(dtype=float) / n_total
    return float((weights * table["abs_diff"].to_numpy(dtype=float)).sum())


def _plot_validation_reliability(
    *,
    out_path: Path,
    table: pd.DataFrame,
    ece: float,
    pop: str,
    label: str,
) -> None:
    """Reliability diagram with binomial-SE error bars and diagonal reference."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="perfect")

    if not table.empty:
        mean_pred = table["mean_pred"].to_numpy(dtype=float)
        obs = table["observed_rate"].to_numpy(dtype=float)
        n = table["count"].to_numpy(dtype=float)
        # Wald binomial SE; clipped at zero count guards already applied.
        se = np.sqrt(np.clip(obs * (1.0 - obs), 0.0, None) / np.where(n > 0, n, 1))
        ax.errorbar(
            mean_pred, obs, yerr=se,
            fmt="o", color="C0", ecolor="C0", capsize=3, linewidth=2,
            label=f"{label}: ECE={ece:.4f} ({len(table)} bins)",
        )
        for x, y, c in zip(mean_pred, obs, table["count"].to_numpy(dtype=int)):
            if np.isfinite(y):
                ax.annotate(
                    f"n={c}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="C0",
                )

    ax.set_xlabel("Mean predicted P (equal-frequency bin)")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration validation — {pop} ({label})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _validate_population_calibration(pop: str) -> dict[str, float] | None:
    """v3 E.1: bin p_cal + p_stack against true is_placer_<pop> labels.

    Returns {'pop', 'ece_post', 'ece_pre'} on success, None if any required
    parquet is missing (warning emitted in that case so the caller can skip).
    """
    label_col = f"is_{pop}"
    cal_path = OUT_DIR / f"pop_calibrated_{pop}_250m.parquet"
    pred_path = OUT_DIR / f"pop_predictions_{pop}_250m.parquet"
    features_path = FEATURES_PARQUET

    missing = [p for p in (cal_path, pred_path, features_path) if not p.exists()]
    if missing:
        print(
            f"WARNING: E.1 validation skipped for {pop}; missing inputs:\n"
            + "\n".join(f"  {m}" for m in missing),
            file=sys.stderr,
        )
        return None

    features = pd.read_parquet(features_path, columns=["row", "col", label_col])
    if label_col not in features.columns:
        print(
            f"WARNING: E.1 validation skipped for {pop}; features parquet missing "
            f"label column {label_col!r}.",
            file=sys.stderr,
        )
        return None

    cal = pd.read_parquet(cal_path, columns=["row", "col", "p_cal"])
    preds = pd.read_parquet(pred_path, columns=["row", "col", "p_stack"])

    merged = features.merge(cal, on=["row", "col"], how="inner").merge(
        preds, on=["row", "col"], how="inner"
    )
    if merged.empty:
        print(f"WARNING: E.1 validation skipped for {pop}; merged frame empty.", file=sys.stderr)
        return None

    y = merged[label_col].to_numpy(dtype=float)
    p_cal = merged["p_cal"].to_numpy(dtype=float)
    p_stack = merged["p_stack"].to_numpy(dtype=float)
    n_pos = int(y.sum())

    table_post = _equal_freq_reliability_table(p_cal, y, n_bins=N_BINS)
    table_pre = _equal_freq_reliability_table(p_stack, y, n_bins=N_BINS)
    ece_post = _ece_from_table(table_post)
    ece_pre = _ece_from_table(table_pre)

    out_csv_post = OUT_DIR / f"calibration_validation_{pop}.csv"
    out_csv_pre = OUT_DIR / f"calibration_validation_{pop}_precal.csv"
    out_png_post = OUT_DIR / f"calibration_reliability_{pop}.png"
    out_png_pre = OUT_DIR / f"calibration_reliability_{pop}_precal.png"

    out_csv_post.parent.mkdir(parents=True, exist_ok=True)
    table_post.to_csv(out_csv_post, index=False)
    table_pre.to_csv(out_csv_pre, index=False)
    print(f"    wrote {out_csv_post}")
    print(f"    wrote {out_csv_pre}")

    _plot_validation_reliability(
        out_path=out_png_post, table=table_post,
        ece=ece_post, pop=pop, label="post-isotonic p_cal",
    )
    _plot_validation_reliability(
        out_path=out_png_pre, table=table_pre,
        ece=ece_pre, pop=pop, label="pre-isotonic p_stack",
    )
    print(f"    wrote {out_png_post}")
    print(f"    wrote {out_png_pre}")

    print(
        f"\nE.1 validation {pop}: n_cells={len(merged):,d}  n_positive={n_pos:d}  "
        f"pre ECE={ece_pre:.4f}  post ECE={ece_post:.4f}\n"
    )

    return {"pop": pop, "ece_pre": ece_pre, "ece_post": ece_post, "n_pos": float(n_pos)}


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
    anchor_results: list[dict[str, float | str]] = []
    e1_results: list[dict[str, float]] = []
    for pop in pops:
        pred_path = OUT_DIR / f"pop_predictions_{pop}_250m.parquet"
        cal_path = OUT_DIR / f"pop_calibrated_{pop}_250m.parquet"
        if not pred_path.exists() or not cal_path.exists():
            print(
                f"WARNING: skipping {pop}; missing prediction/calibration parquet "
                f"(predictions exists={pred_path.exists()}, calibrated exists={cal_path.exists()}).",
                file=sys.stderr,
            )
            continue

        res = _audit_population(pop)
        if res is not None:
            anchor_results.append(res)

        e1 = _validate_population_calibration(pop)
        if e1 is not None:
            e1_results.append(e1)

    print("=== Anchor-set audit summary ===")
    for r in anchor_results:
        print(
            f"  {r['pop']}: pre ECE={r['ece_pre']:.3f}  post ECE={r['ece_post']:.3f}  "
            f"-> {r['classification']}"
        )
    if not anchor_results:
        print("  (no populations processed)")

    print("=== E.1 calibration validation summary ===")
    for r in e1_results:
        print(
            f"  {r['pop']}: n_pos={int(r['n_pos']):d}  "
            f"pre ECE={r['ece_pre']:.4f}  post ECE={r['ece_post']:.4f}"
        )
    if not e1_results:
        print("  (no populations processed)")

    return 0 if (anchor_results or e1_results) else 2


if __name__ == "__main__":
    sys.exit(main())
