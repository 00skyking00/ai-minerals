"""Phase E.3: Platt vs isotonic calibration ablation for the placer stack.

For each population (placer_tertiary, placer_quaternary), load the OOF
stacking probability written by
`scripts/northern_sierra_placer_train_predict_250m.py`
(`pop_predictions_<pop>_250m.parquet`, column `p_stack`), train two
`CalibratedClassifierCV` instances against the population's binary label
column from `features_<region>_250m.parquet` (`is_<pop>`), and compare
ECE, Brier score, and top-decile precision/recall/F1 on out-of-fold
predictions.

The two calibrators are:

  * isotonic (the v2/v3 default; non-parametric step function)
  * sigmoid  (Platt scaling; one-parameter logistic; the heuristic
              fallback when positives < 30)

The ablation forces isotonic even when positives are sparse, but warns.
This is the answer to "decide per-population which calibrator ships,
based on the comparison, not a heuristic" from
`~/.claude/plans/hazy-humming-lynx.md` Phase E.3.

Inputs (under data/derived/):
  features_<region>_250m.parquet            (columns: row, col, is_<pop>, ...)
  northern_sierra_placer/
    pop_predictions_<pop>_250m.parquet      (columns: row, col, p_stack)

Outputs (under data/derived/northern_sierra_placer/):
  calibration_ablation_<pop>.csv
      one row per method (isotonic, sigmoid) with:
      method, ece, brier, top_decile_precision,
      top_decile_recall, top_decile_f1

Usage:
    .venv/bin/python scripts/northern_sierra_placer_calibration_ablation.py
    .venv/bin/python scripts/northern_sierra_placer_calibration_ablation.py \
        --population placer_tertiary
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ai_minerals.metrics.calibration import expected_calibration_error
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
POPULATIONS = ("placer_tertiary", "placer_quaternary")
CALIBRATION_CV = 5
ISOTONIC_MIN_POSITIVES = 30
N_ECE_BINS = 10
TOP_DECILE_FRACTION = 0.10

# ECE tie-break threshold: if two methods' ECEs are within this distance,
# break the tie on top-decile F1 instead.
ECE_TIE_TOLERANCE = 0.005

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
FEATURES_PATH = DATA_DERIVED / f"features_{REGION.data_prefix}_250m.parquet"
OUT_DIR = DATA_DERIVED / REGION.data_prefix


def _equal_frequency_ece(
    p: np.ndarray, y: np.ndarray, *, n_bins: int = N_ECE_BINS
) -> float:
    """ECE using equal-frequency bins (quantile cuts on the score distribution).

    The `expected_calibration_error` helper in ai_minerals.metrics.calibration
    uses equal-width bins on [0, 1], which is fine for well-spread predictions
    but understates miscalibration when the score distribution piles up at one
    end (a known issue for sparse-positive prospectivity scores). The plan asks
    for equal-frequency bins; we keep both available so the audit script stays
    consistent and this script answers the plan's request directly.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = p.shape[0]
    if n == 0:
        return 0.0
    # quantile-based bin edges; np.quantile handles ties by interpolation
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    # Force strict monotonicity by nudging duplicates; otherwise digitize
    # collapses adjacent bins. We accept the small numerical adjustment in
    # exchange for stable per-bin counts on heavy-tailed score distributions.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(p[mask].mean())
        pos_rate = float(y[mask].mean())
        ece += (count / n) * abs(pos_rate - mean_pred)
    return float(ece)


def _top_decile_metrics(
    p: np.ndarray, y: np.ndarray, *, fraction: float = TOP_DECILE_FRACTION
) -> tuple[float, float, float]:
    """Top-k% precision, recall, F1.

    Threshold by score quantile; treat all cells with score >= cutoff as
    positive predictions, then score against true labels.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    n = p.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    cutoff = float(np.quantile(p, 1.0 - fraction))
    pred = p >= cutoff
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _calibrated_oof(
    p_stack: np.ndarray, y: np.ndarray, *, method: str, n_splits: int, seed: int = 42
) -> np.ndarray:
    """Out-of-fold calibrated probabilities via sklearn cross_val_predict.

    Wraps a `CalibratedClassifierCV` whose base estimator is a 1-D logistic
    regression on `p_stack`. cross_val_predict uses the outer split for the
    held-out fold; the calibrator's inner CV is `n_splits` as well, matching
    the v2/v3 train_predict pattern.
    """
    n_pos = int(y.sum())
    cv_inner = min(n_splits, max(2, n_pos))
    base = LogisticRegression(max_iter=1000)
    cal = CalibratedClassifierCV(base, method=method, cv=cv_inner)
    outer = StratifiedKFold(
        n_splits=min(n_splits, max(2, n_pos)),
        shuffle=True,
        random_state=seed,
    )
    X = p_stack.reshape(-1, 1)
    proba = cross_val_predict(cal, X, y, cv=outer, method="predict_proba")
    return proba[:, 1]


def _recommend(
    isotonic_metrics: dict[str, float], sigmoid_metrics: dict[str, float]
) -> str:
    """Recommend the winning method.

    Lower ECE wins; if the two ECEs are within `ECE_TIE_TOLERANCE`, break the
    tie on top-decile F1.
    """
    ece_iso = isotonic_metrics["ece"]
    ece_sig = sigmoid_metrics["ece"]
    if abs(ece_iso - ece_sig) <= ECE_TIE_TOLERANCE:
        f1_iso = isotonic_metrics["top_decile_f1"]
        f1_sig = sigmoid_metrics["top_decile_f1"]
        return "isotonic" if f1_iso >= f1_sig else "sigmoid"
    return "isotonic" if ece_iso < ece_sig else "sigmoid"


def _ablate_population(pop: str) -> dict | None:
    pred_path = OUT_DIR / f"pop_predictions_{pop}_250m.parquet"
    if not pred_path.exists():
        print(
            f"[{pop}] predictions parquet missing: {pred_path}; skipping.",
            file=sys.stderr,
        )
        return None
    if not FEATURES_PATH.exists():
        print(
            f"[{pop}] features parquet missing: {FEATURES_PATH}; cannot recover labels.",
            file=sys.stderr,
        )
        return None

    label_col = f"is_{pop}"
    preds = pd.read_parquet(pred_path, columns=["row", "col", "p_stack"])
    feats = pd.read_parquet(FEATURES_PATH, columns=["row", "col", label_col])

    merged = preds.merge(feats, on=["row", "col"], how="inner")
    if merged.empty:
        print(f"[{pop}] joined frame empty; check row/col alignment.", file=sys.stderr)
        return None

    p = merged["p_stack"].to_numpy(dtype=np.float64)
    y = merged[label_col].to_numpy(dtype=np.int64)
    finite = np.isfinite(p)
    p = p[finite]
    y = y[finite]
    n_pos = int(y.sum())
    n = int(y.shape[0])
    print(
        f"[{pop}] {n:,} cells; {n_pos:,} positives "
        f"({100.0 * n_pos / max(n, 1):.4f}% base rate)",
        flush=True,
    )

    if n_pos < 2:
        print(
            f"[{pop}] fewer than 2 positives; skipping ablation.",
            file=sys.stderr,
        )
        return None

    if n_pos < ISOTONIC_MIN_POSITIVES:
        warnings.warn(
            f"[{pop}] {n_pos} positives < {ISOTONIC_MIN_POSITIVES}; "
            f"isotonic results will be unstable but reported anyway "
            f"(per the Phase E.3 ablation brief)."
        )

    rows: list[dict[str, float | str]] = []
    method_metrics: dict[str, dict[str, float]] = {}
    for method in ("isotonic", "sigmoid"):
        print(f"[{pop}] fitting calibrator method={method}...", flush=True)
        p_cal = _calibrated_oof(
            p, y, method=method, n_splits=CALIBRATION_CV
        )
        ece = _equal_frequency_ece(p_cal, y, n_bins=N_ECE_BINS)
        brier = float(brier_score_loss(y, p_cal))
        precision, recall, f1 = _top_decile_metrics(
            p_cal, y, fraction=TOP_DECILE_FRACTION
        )
        metrics = {
            "ece": ece,
            "brier": brier,
            "top_decile_precision": precision,
            "top_decile_recall": recall,
            "top_decile_f1": f1,
        }
        method_metrics[method] = metrics
        rows.append({"method": method, **metrics})
        print(
            f"[{pop}]   {method}: ece={ece:.4f}  brier={brier:.6f}  "
            f"top-decile P/R/F1={precision:.4f}/{recall:.4f}/{f1:.4f}",
            flush=True,
        )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "method",
            "ece",
            "brier",
            "top_decile_precision",
            "top_decile_recall",
            "top_decile_f1",
        ],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"calibration_ablation_{pop}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{pop}] wrote {out_csv}", flush=True)

    winner = _recommend(method_metrics["isotonic"], method_metrics["sigmoid"])
    print(
        f"[{pop}] recommendation: {winner} "
        f"(isotonic ECE={method_metrics['isotonic']['ece']:.4f}, "
        f"sigmoid ECE={method_metrics['sigmoid']['ece']:.4f})",
        flush=True,
    )
    return {"pop": pop, "winner": winner, "metrics": method_metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=POPULATIONS,
        default=None,
        help="Ablate a single population (default: both).",
    )
    args = parser.parse_args(argv)

    pops = (args.population,) if args.population else POPULATIONS
    any_ok = False
    results: list[dict] = []
    for pop in pops:
        res = _ablate_population(pop)
        if res is not None:
            any_ok = True
            results.append(res)

    if not any_ok:
        return 2

    print("=== Summary ===")
    for r in results:
        m = r["metrics"]
        print(
            f"  {r['pop']}: winner={r['winner']}  "
            f"isotonic ECE={m['isotonic']['ece']:.4f}, sigmoid ECE={m['sigmoid']['ece']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
