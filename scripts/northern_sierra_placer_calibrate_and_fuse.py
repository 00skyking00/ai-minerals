"""Phase F: fuse the per-population calibrated rasters into the deliverable.

Sierra placer divides cleanly into two populations (Tertiary deep-gravel +
Quaternary modern-channel) with different geomorphic signatures. Phase E
trains and calibrates one classifier per population and writes a calibrated
parquet for each. This script fuses them and writes the deliverable GeoTIFFs.

Two fusion modes ship side-by-side:

1. Per-cell `np.maximum` on the calibrated probabilities. Cheap, transparent,
   and the v2 deliverable. Always written.

2. Calibrated logistic stacking (v3 Phase C.3, calibrated logistic stacking
   per the MMDS framing, Marini et al., paper S0169136825003282). A small
   `sklearn.linear_model.LogisticRegression` meta-learner takes the two
   calibrated probabilities plus a couple of context features
   (`distance_downstream_from_lode_m`, `is_quaternary_alluvium`) and produces
   a single combined probability per cell. Training labels are the union
   `is_placer_tertiary | is_placer_quaternary` from the feature parquet.
   Written when both calibrated parquets exist; otherwise warned and skipped.

The np.max fusion ships unchanged so the v3 deliverable remains comparable to
v2. The logistic-fusion raster is a side-by-side artifact for the v3 model
card and downstream A/B comparison; it is not yet the canonical deliverable.

Inputs (from Phase E):
  data/derived/northern_sierra_placer/pop_calibrated_placer_tertiary_250m.parquet
  data/derived/northern_sierra_placer/pop_calibrated_placer_quaternary_250m.parquet
  data/derived/features_northern_sierra_placer_250m.parquet  (for the logistic
                                                              meta-learner)

Outputs (under data/derived/northern_sierra_placer/):
  prospectivity_placer_northern_sierra_250m_calibrated_3310.tif    (np.max)
  prospectivity_placer_northern_sierra_250m_calibrated_4326.tif    (np.max, deliverable)
  prospectivity_placer_combined_250m_calibrated_3310.tif           (logistic stack)
  prospectivity_placer_combined_250m_calibrated_4326.tif           (logistic stack)
  fusion_meta.json   per-cell summary + logistic meta coefficients (when run)

Usage:
    .venv/bin/python scripts/northern_sierra_placer_calibrate_and_fuse.py
    .venv/bin/python scripts/northern_sierra_placer_calibrate_and_fuse.py --no-logistic-fusion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ai_minerals.io.geotiff import write_geotiff_dual_crs
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
RES_M = 250.0

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
OUT_DIR = DATA_DERIVED / REGION.data_prefix

IN_TERTIARY = OUT_DIR / "pop_calibrated_placer_tertiary_250m.parquet"
IN_QUATERNARY = OUT_DIR / "pop_calibrated_placer_quaternary_250m.parquet"
IN_FEATURES = DATA_DERIVED / f"features_{REGION.data_prefix}_250m.parquet"

OUT_TIF_3310 = OUT_DIR / "prospectivity_placer_northern_sierra_250m_calibrated_3310.tif"
OUT_TIF_4326 = OUT_DIR / "prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
OUT_TIF_LOGISTIC_3310 = OUT_DIR / "prospectivity_placer_combined_250m_calibrated_3310.tif"
OUT_TIF_LOGISTIC_4326 = OUT_DIR / "prospectivity_placer_combined_250m_calibrated_4326.tif"
OUT_FUSION_PARQUET = OUT_DIR / "prospectivity_placer_northern_sierra_250m_fused.parquet"
OUT_LOGISTIC_PARQUET = OUT_DIR / "prospectivity_placer_combined_250m_calibrated.parquet"
OUT_META = OUT_DIR / "fusion_meta.json"

# Context features sampled into the meta-learner alongside the two calibrated
# probabilities. Both are known leakage-adjacent (the distance feature has a
# 1500 m radius leakage guard upstream; the alluvium flag is a direct
# Quaternary-defining lithology), but we are stacking calibrated probabilities
# from population-specific models that already saw these features, so the
# meta is restricted to a tiny input set on purpose.
LOGISTIC_CONTEXT_FEATURES = (
    "distance_downstream_from_lode_m",
    "is_quaternary_alluvium",
)


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


def _logistic_fusion(
    fused: pd.DataFrame,
    features_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Train a small LogisticRegression meta on (p_cal_t, p_cal_q, context)
    against the union of positive labels; return per-cell combined probability
    and a meta-summary (coefficients, intercept, positive count).

    Per the MMDS framing (Marini et al., S0169136825003282): calibrated logistic
    stacking on a small input set is more principled than per-cell np.max
    because it learns how to weight the two populations from the observed
    label union rather than assuming the maximum is the right combiner.
    """
    feat = pd.read_parquet(features_path)
    required = {"row", "col", "is_placer_tertiary", "is_placer_quaternary"}
    missing = required - set(feat.columns)
    if missing:
        raise ValueError(f"{features_path} missing columns: {sorted(missing)}")

    # Context features: keep only the ones present.
    context_cols = [c for c in LOGISTIC_CONTEXT_FEATURES if c in feat.columns]
    skipped = [c for c in LOGISTIC_CONTEXT_FEATURES if c not in feat.columns]
    if skipped:
        print(f"    [logistic] context features not in parquet, skipping: {skipped}")

    keep_cols = ["row", "col", "is_placer_tertiary", "is_placer_quaternary"] + context_cols
    feat = feat[keep_cols].copy()
    feat["y_union"] = (
        (feat["is_placer_tertiary"].fillna(0).astype(int) > 0)
        | (feat["is_placer_quaternary"].fillna(0).astype(int) > 0)
    ).astype(int)

    # Join calibrated probabilities by (row, col).
    merged = fused[["row", "col", "x", "y", "p_tertiary", "p_quaternary"]].merge(
        feat[["row", "col", "y_union", *context_cols]],
        on=["row", "col"],
        how="left",
    )

    # Training rows: positives plus the cells where both calibrated probs are
    # finite. Fill missing per-population probs with 0 (the population
    # genuinely scored that cell as "not its kind"). Context-feature NaNs are
    # median-imputed inside the LR pipeline below.
    p_t = merged["p_tertiary"].fillna(0.0).to_numpy()
    p_q = merged["p_quaternary"].fillna(0.0).to_numpy()

    X_full_cols = [p_t, p_q]
    for c in context_cols:
        col = merged[c].to_numpy(dtype=float)
        # Simple median imputation on the in-AOI cells.
        med = float(np.nanmedian(col)) if np.isfinite(np.nanmedian(col)) else 0.0
        col = np.where(np.isfinite(col), col, med)
        X_full_cols.append(col)
    X_full = np.column_stack(X_full_cols)

    y = merged["y_union"].fillna(0).to_numpy(dtype=int)
    n_pos = int(y.sum())
    if n_pos < 10:
        raise ValueError(
            f"logistic fusion: only {n_pos} positive labels in union; "
            f"need at least 10 to fit a meta-learner."
        )

    # Train on cells where at least one calibrated branch was finite (the
    # "scored" mask). Cells outside the AOI for both branches are excluded.
    scored = fused["p_tertiary"].notna().to_numpy() | fused["p_quaternary"].notna().to_numpy()
    X_train = X_full[scored]
    y_train = y[scored]
    n_pos_train = int(y_train.sum())
    print(f"    [logistic] training on {int(scored.sum()):,} scored cells, "
          f"{n_pos_train} positives ({n_pos} positives total in union)")

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train, y_train)

    # Predict on the full scored set; leave unscored cells as NaN.
    p_combined = np.full(len(merged), np.nan, dtype=float)
    p_combined[scored] = lr.predict_proba(X_train)[:, 1]

    out = merged[["row", "col", "x", "y"]].copy()
    out["p_combined"] = p_combined

    feature_names = ["p_cal_tertiary", "p_cal_quaternary", *context_cols]
    meta = {
        "framework": "calibrated logistic stacking per the MMDS framing "
                     "(Marini et al., S0169136825003282)",
        "feature_names": feature_names,
        "coefficients": {name: float(c) for name, c in zip(feature_names, lr.coef_[0])},
        "intercept": float(lr.intercept_[0]),
        "n_train_cells": int(scored.sum()),
        "n_train_positives": n_pos_train,
        "n_positives_in_union": n_pos,
        "context_features_used": list(context_cols),
        "context_features_skipped": list(skipped),
        "p_combined_min": float(np.nanmin(p_combined)),
        "p_combined_mean": float(np.nanmean(p_combined)),
        "p_combined_max": float(np.nanmax(p_combined)),
    }
    return out, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="Calibrated-probability cutoff for the 'above-cutoff' diagnostic count (default 0.5).",
    )
    parser.add_argument(
        "--no-logistic-fusion",
        dest="logistic_fusion",
        action="store_false",
        default=True,
        help="Skip the v3 calibrated logistic stacking fusion step. "
             "Default: enabled when both calibrated parquets exist.",
    )
    args = parser.parse_args(argv)

    t_missing = not IN_TERTIARY.exists()
    q_missing = not IN_QUATERNARY.exists()
    if t_missing and q_missing:
        print(f"==> [error] Both calibrated parquets missing; nothing to fuse.")
        print(f"    Run scripts/northern_sierra_placer_train_predict_250m.py first.")
        return 1

    if t_missing:
        print(f"==> [warn] {IN_TERTIARY} missing; np.max fusion degrades to Quaternary-only.")
        t_df = None
    else:
        print(f"==> Loading Tertiary: {IN_TERTIARY}")
        t_df = _load(IN_TERTIARY)
        print(f"    {len(t_df):,} cells, p_cal finite={int(t_df['p_cal'].notna().sum()):,}")

    if q_missing:
        print(f"==> [warn] {IN_QUATERNARY} missing; np.max fusion degrades to Tertiary-only.")
        print(f"    Run train_predict --population placer_quaternary to add the second branch.")
        q_df = None
    else:
        print(f"==> Loading Quaternary: {IN_QUATERNARY}")
        q_df = _load(IN_QUATERNARY)
        print(f"    {len(q_df):,} cells, p_cal finite={int(q_df['p_cal'].notna().sum()):,}")

    # If only one branch exists, construct an empty sibling so the per-cell
    # max degrades to that branch's score and the v2 deliverable still ships.
    if t_df is None:
        t_df = q_df[["row", "col", "x", "y"]].copy()
        t_df["p_cal"] = np.nan
    if q_df is None:
        q_df = t_df[["row", "col", "x", "y"]].copy()
        q_df["p_cal"] = np.nan

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

    print(f"==> Writing np.max GeoTIFFs ({REGION.working_crs} + EPSG:4326)")
    write_geotiff_dual_crs(
        fused["p_fused"].values,
        fused[["x", "y"]],
        resolution_m=RES_M,
        src_crs=REGION.working_crs,
        out_src=OUT_TIF_3310,
        out_4326=OUT_TIF_4326,
    )
    print(f"    wrote {OUT_TIF_3310}")
    print(f"    wrote {OUT_TIF_4326}  ← np.max deliverable (v2 contract)")

    logistic_meta: dict | None = None
    if args.logistic_fusion:
        if t_missing or q_missing:
            missing_branch = "Tertiary" if t_missing else "Quaternary"
            print(f"==> [warn] Skipping logistic fusion: {missing_branch} calibrated parquet "
                  f"missing; need both branches to fit the meta-learner.")
        elif not IN_FEATURES.exists():
            print(f"==> [warn] Skipping logistic fusion: feature parquet not found at {IN_FEATURES}")
        else:
            try:
                print("==> Calibrated logistic stacking fusion (MMDS framing)")
                combined, logistic_meta = _logistic_fusion(fused, IN_FEATURES)
                combined.to_parquet(OUT_LOGISTIC_PARQUET, index=False)
                print(f"    wrote logistic-fusion parquet: {OUT_LOGISTIC_PARQUET}")
                print(f"==> Writing logistic-fusion GeoTIFFs ({REGION.working_crs} + EPSG:4326)")
                write_geotiff_dual_crs(
                    combined["p_combined"].values,
                    combined[["x", "y"]],
                    resolution_m=RES_M,
                    src_crs=REGION.working_crs,
                    out_src=OUT_TIF_LOGISTIC_3310,
                    out_4326=OUT_TIF_LOGISTIC_4326,
                )
                print(f"    wrote {OUT_TIF_LOGISTIC_3310}")
                print(f"    wrote {OUT_TIF_LOGISTIC_4326}  ← v3 logistic-stacking comparison raster")
                print(f"    meta coefficients: {logistic_meta['coefficients']}")
                print(f"    intercept: {logistic_meta['intercept']:.4f}")
            except (FileNotFoundError, ValueError) as e:
                print(f"==> [warn] Logistic fusion failed, falling back to np.max only: {e}")
                logistic_meta = None
    else:
        print("==> Logistic fusion skipped (--no-logistic-fusion)")

    meta_payload = {
        "region": REGION.slug,
        "resolution_m": RES_M,
        "compute_crs": REGION.working_crs,
        "deliverable_crs": "EPSG:4326",
        "inputs": {
            "tertiary": str(IN_TERTIARY),
            "quaternary": str(IN_QUATERNARY),
            "features": str(IN_FEATURES),
        },
        "outputs": {
            "deliverable_4326_tif": str(OUT_TIF_4326),
            "compute_3310_tif": str(OUT_TIF_3310),
            "fused_parquet": str(OUT_FUSION_PARQUET),
            "logistic_4326_tif": str(OUT_TIF_LOGISTIC_4326) if logistic_meta else None,
            "logistic_3310_tif": str(OUT_TIF_LOGISTIC_3310) if logistic_meta else None,
            "logistic_parquet": str(OUT_LOGISTIC_PARQUET) if logistic_meta else None,
        },
        "fusion": "per-cell np.maximum on calibrated probabilities",
        "logistic_fusion": logistic_meta,
        "summary": summary,
    }
    OUT_META.write_text(json.dumps(meta_payload, indent=2))
    print(f"==> Wrote fusion metadata: {OUT_META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
