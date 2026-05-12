"""Phase 2: Mother Lode model training + spatial-block CV + SHAP.

Saves intermediate artifacts to data/derived/motherlode/:
- shap_rf_motherlode.npz   — SHAP values for the trained RF
- cv_metrics_motherlode.json  — fold-by-fold and bootstrap-CI metrics
- model_predictions_motherlode.parquet  — per-cell scores from RF + bagging-PU

These feed into the Phase 3 within-Sierra validation notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from ai_minerals.model import (
    add_lithology_onehot, build_training_set, sample_pseudo_negatives,
    NON_FEATURE_COLUMNS,
)
from ai_minerals.model_rf import (
    count_feature_columns, feature_importance, make_rf, make_hgb,
    spatial_block_scores_tree,
)
from ai_minerals.model_pu import fit_pu_bagging
from ai_minerals.regions.motherlode import MOTHERLODE


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
OUT_DIR = DATA_DERIVED / "motherlode"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    print(f"feature frame: {df.shape}  |  positives: {int(df['is_orogenic_gold'].sum())}")

    # Top-10 lithology classes for one-hot encoding.
    top_classes = (
        df["lithology_class"].value_counts().head(10).index.tolist()
    )
    print(f"top lithology_class codes: {top_classes}")

    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")

    # 1. Build training set: positives + pseudo-negatives, one-hot lith, drop labels from features.
    print("\n=== build_training_set ===")
    X, y = build_training_set(
        df, top_classes,
        n_per_positive=30, random_state=42,
        label_col=label_col, label_cols=label_cols,
    )
    drop_cols = count_feature_columns(list(X.columns))
    X_trim = X.drop(columns=drop_cols)
    print(f"  full-feature training: {X.shape}")
    print(f"  count-free training:   {X_trim.shape}  (dropped {len(drop_cols)} count cols)")

    # 2. Spatial-block CV on full + trimmed (20 km blocks, BCGT pattern).
    print("\n=== spatial-block CV (20 km) ===")
    negs = sample_pseudo_negatives(df, n_per_positive=30, random_state=42, label_col=label_col)
    rows = pd.concat(
        [df[df[label_col] == 1][["row", "col", "x", "y"]], negs[["row", "col", "x", "y"]]],
        ignore_index=True,
    )

    rf_full_cv  = spatial_block_scores_tree(X,      y, rows, model_factory=make_rf,  block_size_m=20_000)
    print(f"  RF full   : {len(rf_full_cv)} valid folds", flush=True)
    rf_trim_cv  = spatial_block_scores_tree(X_trim, y, rows, model_factory=make_rf,  block_size_m=20_000)
    print(f"  RF no-cnt : {len(rf_trim_cv)} valid folds", flush=True)
    hgb_trim_cv = spatial_block_scores_tree(X_trim, y, rows, model_factory=make_hgb, block_size_m=20_000)
    print(f"  HGB no-cnt: {len(hgb_trim_cv)} valid folds", flush=True)

    def _stats(cv_df: pd.DataFrame) -> dict:
        return {
            "auc_mean": float(cv_df["roc_auc"].mean()),
            "auc_std": float(cv_df["roc_auc"].std()),
            "pr_auc_mean": float(cv_df["pr_auc"].mean()),
            "pr_auc_std": float(cv_df["pr_auc"].std()),
            "n_folds": int(len(cv_df)),
            "fold_aucs": cv_df["roc_auc"].tolist(),
            "fold_pr_aucs": cv_df["pr_auc"].tolist(),
        }

    rf_full_stats = _stats(rf_full_cv)
    rf_trim_stats = _stats(rf_trim_cv)
    hgb_trim_stats = _stats(hgb_trim_cv)

    print(f"\n  RF full   :  AUC {rf_full_stats['auc_mean']:.3f} ± {rf_full_stats['auc_std']:.3f}  "
          f"PR-AUC {rf_full_stats['pr_auc_mean']:.3f} ± {rf_full_stats['pr_auc_std']:.3f}  "
          f"folds={rf_full_stats['n_folds']}", flush=True)
    print(f"  RF no-cnt :  AUC {rf_trim_stats['auc_mean']:.3f} ± {rf_trim_stats['auc_std']:.3f}  "
          f"PR-AUC {rf_trim_stats['pr_auc_mean']:.3f} ± {rf_trim_stats['pr_auc_std']:.3f}", flush=True)
    print(f"  HGB no-cnt:  AUC {hgb_trim_stats['auc_mean']:.3f} ± {hgb_trim_stats['auc_std']:.3f}  "
          f"PR-AUC {hgb_trim_stats['pr_auc_mean']:.3f} ± {hgb_trim_stats['pr_auc_std']:.3f}", flush=True)

    # Bootstrap 95% CI on per-fold AUC, RF no-count. Filter NaN folds
    # (single-class test blocks return NaN AUC; including them produces
    # NaN bootstrap means).
    rng = np.random.default_rng(0)
    fold_aucs = np.array(rf_trim_stats["fold_aucs"])
    fold_aucs = fold_aucs[~np.isnan(fold_aucs)]
    rf_trim_stats["n_folds_after_nan_filter"] = int(len(fold_aucs))
    if len(fold_aucs):
        boot = np.array([rng.choice(fold_aucs, size=len(fold_aucs)).mean() for _ in range(2000)])
        rf_trim_stats["auc_bootstrap_ci95"] = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
        rf_trim_stats["auc_mean_filtered"] = float(fold_aucs.mean())
        print(f"  RF no-cnt mean AUC (NaN-filtered, n={len(fold_aucs)} folds): "
              f"{fold_aucs.mean():.3f}", flush=True)
        print(f"  RF no-cnt 95% CI on mean AUC (2000-resample): "
              f"[{rf_trim_stats['auc_bootstrap_ci95'][0]:.3f}, {rf_trim_stats['auc_bootstrap_ci95'][1]:.3f}]", flush=True)

    # 3. Fit final RF on full training set, score every cell, compute SHAP.
    print("\n=== final RF on no-count features ===")
    rf_final = make_rf(random_state=42)
    rf_final.fit(X_trim.fillna(-9999).to_numpy(), y)

    df_oh = add_lithology_onehot(df, top_classes)
    feat_cols = [c for c in X_trim.columns]
    X_all = df_oh[feat_cols].fillna(-9999).to_numpy()
    p_rf = rf_final.predict_proba(X_all)[:, 1]
    print(f"  scored {len(p_rf):,} cells  |  median P {np.median(p_rf):.3f}  |  mean P {p_rf.mean():.3f}")

    print("\n=== SHAP top-15 features ===")
    sv_path = OUT_DIR / "shap_rf_motherlode.npz"
    if sv_path.exists():
        sv_z = np.load(sv_path)
        shap_values = sv_z["shap_values"]
        feat_names = sv_z["feat_names"].tolist()
    else:
        # SHAP on a sample — full grid would be huge; sample 5k cells for stable means
        rng_idx = np.random.default_rng(0)
        sample_idx = rng_idx.choice(len(X_all), size=min(5_000, len(X_all)), replace=False)
        explainer = shap.TreeExplainer(rf_final)
        shap_values = explainer.shap_values(X_all[sample_idx], check_additivity=False)
        feat_names = list(X_trim.columns)
        np.savez_compressed(sv_path, shap_values=shap_values, feat_names=np.array(feat_names))

    # Normalize SHAP output shape to 2D (n_samples, n_features) for the
    # positive class. Newer SHAP versions return a 3D array
    # (n_samples, n_features, n_classes); older versions return a
    # list-of-2D [neg_class_array, pos_class_array]. Handle both.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    print(shap_df.head(15).to_string(index=False))

    # 4. Bagging-PU baseline (weaker positive-only setup; for comparison).
    print("\n=== bagging-PU ensemble ===")
    p_pu, pu_feats = fit_pu_bagging(
        df_oh, top_classes,
        label_col=label_col,
        n_bags=30, random_state=42,
    )
    print(f"  scored {(~np.isnan(p_pu)).sum():,} cells (others have 0 OOB coverage)")

    # 5. Per-cell predictions out for Phase 3 validation.
    pred = df[["row", "col", "x", "y", label_col, "any_mineral_occurrence", "lithology_class"]].copy()
    pred["p_rf_no_count"] = p_rf
    pred["p_pu_bagging"] = p_pu
    pred.to_parquet(OUT_DIR / "model_predictions_motherlode.parquet")
    print(f"\nSaved predictions to {OUT_DIR / 'model_predictions_motherlode.parquet'}")

    # 6. Persist all metrics.
    metrics = {
        "label_col": label_col,
        "n_positives": int(df[label_col].sum()),
        "n_cells": int(len(df)),
        "rf_full":   rf_full_stats,
        "rf_trim":   rf_trim_stats,
        "hgb_trim":  hgb_trim_stats,
        "shap_top15": shap_df.head(15).to_dict(orient="records"),
    }
    (OUT_DIR / "cv_metrics_motherlode.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Saved metrics to {OUT_DIR / 'cv_metrics_motherlode.json'}")


if __name__ == "__main__":
    main()
