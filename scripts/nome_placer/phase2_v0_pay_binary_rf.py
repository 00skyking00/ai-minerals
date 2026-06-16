"""Phase 2 v0 baseline: RandomForest pay/barren classifier with
leave-one-claim-MS-out cross-validation.

Uses ``features.v2_assemble.assemble`` to load bearcub's 104 labelled
drillholes joined with the v1.5 v2 prospectivity raster + bedrock
depth + surface_class. Trains a RandomForestClassifier on pay_binary
with default sklearn parameters (no hyperparameter tuning - this is a
baseline, not a polished result).

Reports:
- Per-fold AUC + train/test counts
- Aggregated AUC across all held-out predictions
- Feature importance ranking
- A CSV of per-hole predictions (for chapter and for goldbug)

Run:
    uv run python -m scripts.nome_placer.phase2_v0_pay_binary_rf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ai_minerals.features.v2_assemble import (
    PROSPECTIVITY_BAND_KEYS,
    assemble,
    claim_ms_kfold_split,
)
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION


DRILLHOLES = Path(NOME_PLACER_REGION.raw_paths["drillholes"])
RASTER = Path(
    "data/derived/nome_placer/prospectivity_v1p5/"
    "nome_placer_prospectivity_v1p5_v2_4326.tif"
)
OUT_DIR = Path("data/derived/nome_placer/phase2_v0")
OUT_PREDICTIONS = OUT_DIR / "phase2_v0_per_hole_predictions.csv"
OUT_FEATURE_IMPORTANCE = OUT_DIR / "phase2_v0_feature_importance.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Assembling feature frame ...")
    ff = assemble(DRILLHOLES, RASTER)
    n = len(ff.holes)
    print(f"  {n} holes; feature columns: {ff.feature_columns}")

    # Filter to rows with pay_binary defined
    has_label = ff.holes["pay_binary"].notna()
    X_all = ff.X()[has_label].fillna(-999.0).to_numpy(dtype=np.float32)
    y_all = ff.holes.loc[has_label, "pay_binary"].astype(int).to_numpy()
    holes_labelled = ff.holes.loc[has_label].reset_index(drop=True).copy()
    print(f"  {len(y_all)} holes have pay_binary defined ({y_all.sum()} pay, {(y_all == 0).sum()} barren)")

    splits = claim_ms_kfold_split(holes_labelled)
    print(f"  {len(splits)} LOO folds by claim_ms")

    fold_records = []
    per_hole_prob = np.full(len(y_all), np.nan, dtype=np.float32)

    for k, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        per_hole_prob[test_idx] = prob

        # Per-fold AUC; skip if only one class in test set
        try:
            if len(np.unique(y_test)) >= 2:
                fold_auc = roc_auc_score(y_test, prob)
            else:
                fold_auc = float("nan")
        except Exception:
            fold_auc = float("nan")
        ms_value = holes_labelled.iloc[test_idx[0]]["claim_ms"]
        fold_records.append({
            "fold": k,
            "test_claim_ms": ms_value,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_test_pay": int(y_test.sum()),
            "fold_auc": float(fold_auc) if not np.isnan(fold_auc) else None,
        })
        print(f"  fold {k:2d}: MS={ms_value}  train {len(y_train):3d}  test {len(y_test):3d}  pay {int(y_test.sum()):2d}  AUC={fold_auc:.3f}" if not np.isnan(fold_auc) else f"  fold {k:2d}: MS={ms_value}  train {len(y_train):3d}  test {len(y_test):3d}  pay {int(y_test.sum()):2d}  AUC=NA (single-class fold)")

    # Aggregate AUC across all held-out predictions
    finite = ~np.isnan(per_hole_prob)
    if len(np.unique(y_all[finite])) >= 2:
        aggregate_auc = roc_auc_score(y_all[finite], per_hole_prob[finite])
    else:
        aggregate_auc = float("nan")
    print(f"\nAggregate leave-one-claim-MS-out AUC: {aggregate_auc:.3f}")
    valid_aucs = [r["fold_auc"] for r in fold_records if r["fold_auc"] is not None]
    if valid_aucs:
        print(f"Mean per-fold AUC (folds with both classes): {np.mean(valid_aucs):.3f} (n={len(valid_aucs)})")

    # Save per-hole predictions
    holes_labelled["pred_pay_prob"] = per_hole_prob
    cols_out = [
        "hole_id", "lon", "lat", "claim_ms", "deposit_class",
        "pay_binary", "pay_zone_average_oz_cuyd",
        "bedrock_depth_ft", "bottom_of_pay_ft", "surface_mineable",
        "pred_pay_prob",
    ]
    cols_out = [c for c in cols_out if c in holes_labelled.columns]
    holes_labelled[cols_out].to_csv(OUT_PREDICTIONS, index=False)
    print(f"\nWrote per-hole predictions: {OUT_PREDICTIONS}")

    # Train one final model on ALL data for feature importance
    final = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    final.fit(X_all, y_all)
    importances = sorted(
        zip(ff.feature_columns, final.feature_importances_),
        key=lambda p: -p[1],
    )
    print(f"\nFeature importance (full-data RF):")
    for feat, imp in importances:
        print(f"  {feat:<28} {imp:.4f}")
    pd.DataFrame(importances, columns=["feature", "importance"]).to_csv(
        OUT_FEATURE_IMPORTANCE, index=False,
    )
    print(f"Wrote feature importance: {OUT_FEATURE_IMPORTANCE}")


if __name__ == "__main__":
    main()
