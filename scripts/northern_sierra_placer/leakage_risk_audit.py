"""Phase E.5: leakage-risk audit for v3 suspect features.

For each suspect feature in v3, computes the Pearson and Spearman correlation
against each population label (is_placer_tertiary, is_placer_quaternary)
across all cells in the assembled feature parquet. Features whose
|Pearson r| > 0.10 (or |Spearman r| > 0.10) are flagged as a leakage warning.

Also reports the median feature value at positive vs negative cells and the
gap ratio (|pos_median - neg_median| / (|neg_median| + 1e-9)) so an inspector
can see whether the feature is mechanically inverse-of-label (the v2
hydraulic_pit_proximity_m failure mode).

Suspect features absent from the parquet are skipped and noted in stdout.

Inputs:
  data/derived/features_northern_sierra_placer_250m.parquet

Outputs:
  data/derived/northern_sierra_placer/leakage_risk_audit.csv

Usage:
    .venv/bin/python scripts/northern_sierra_placer/leakage_risk_audit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PARQUET = (
    REPO_ROOT / "data" / "derived" / "features_northern_sierra_placer_250m.parquet"
)
OUT_DIR = REPO_ROOT / "data" / "derived" / REGION.data_prefix
OUT_CSV = OUT_DIR / "leakage_risk_audit.csv"

POPULATIONS = ("is_placer_tertiary", "is_placer_quaternary")

SUSPECT_FEATURES = [
    "hydraulic_pit_proximity_m_buffered",
    "is_quaternary_alluvium",
    "distance_downstream_from_lode_m",
    "catchment_au_hawkes",
    "distance_to_lode_m",
    "motherlode_prob",
]

LEAKAGE_THRESHOLD = 0.10


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (pearson_r, spearman_r) after dropping NaNs, or (nan, nan) if not enough."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan"), float("nan")
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return float(pr), float(sr)


def _gap_ratio(pos_median: float, neg_median: float) -> float:
    if not np.isfinite(pos_median) or not np.isfinite(neg_median):
        return float("nan")
    return float(abs(pos_median - neg_median) / (abs(neg_median) + 1e-9))


def audit_feature(
    df: pd.DataFrame, feature: str, population: str
) -> dict[str, float | str | bool]:
    x = df[feature].to_numpy(dtype=float)
    y = df[population].to_numpy(dtype=float)
    pr, sr = _safe_corr(x, y)
    pos_mask = (y == 1) & np.isfinite(x)
    neg_mask = (y == 0) & np.isfinite(x)
    pos_median = float(np.median(x[pos_mask])) if pos_mask.any() else float("nan")
    neg_median = float(np.median(x[neg_mask])) if neg_mask.any() else float("nan")
    pos_std = float(np.std(x[pos_mask])) if pos_mask.sum() > 1 else float("nan")
    neg_std = float(np.std(x[neg_mask])) if neg_mask.sum() > 1 else float("nan")
    gap = _gap_ratio(pos_median, neg_median)
    flagged = (
        np.isfinite(pr) and abs(pr) > LEAKAGE_THRESHOLD
    ) or (np.isfinite(sr) and abs(sr) > LEAKAGE_THRESHOLD)
    return {
        "feature": feature,
        "population": population,
        "pearson_r": pr,
        "spearman_r": sr,
        "pos_median": pos_median,
        "neg_median": neg_median,
        "pos_std": pos_std,
        "neg_std": neg_std,
        "gap_ratio": gap,
        "flagged": bool(flagged),
    }


def main() -> int:
    if not FEATURE_PARQUET.exists():
        print(f"ERROR: feature parquet not found at {FEATURE_PARQUET}")
        return 1
    df = pd.read_parquet(FEATURE_PARQUET)
    print(f"loaded feature parquet: {df.shape[0]} cells, {df.shape[1]} columns")

    present: list[str] = []
    missing: list[str] = []
    for feat in SUSPECT_FEATURES:
        if feat in df.columns:
            present.append(feat)
        else:
            missing.append(feat)

    if missing:
        print(
            "skipping suspect features absent from parquet: "
            + ", ".join(missing)
        )

    rows: list[dict[str, float | str | bool]] = []
    for feat in present:
        for pop in POPULATIONS:
            if pop not in df.columns:
                print(f"WARN: population column {pop} missing; skipping")
                continue
            rows.append(audit_feature(df, feat, pop))

    audit = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(audit)} rows)")

    flagged = audit[audit["flagged"]]
    print()
    print("=" * 64)
    print(f"leakage warnings (|r| > {LEAKAGE_THRESHOLD}):")
    print("=" * 64)
    if flagged.empty:
        print("none")
    else:
        for _, r in flagged.iterrows():
            print(
                f"  {r['feature']:42s}  {r['population']:22s}  "
                f"pearson={r['pearson_r']:+.3f}  spearman={r['spearman_r']:+.3f}  "
                f"gap_ratio={r['gap_ratio']:.2f}"
            )
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
