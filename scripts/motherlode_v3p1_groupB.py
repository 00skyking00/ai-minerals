"""Group B v3.1 verification: re-run RF spatial-block CV with new features.

Compares v3 (no derivatives, no major1/2/3) vs v3.1 (with derivatives
+ major1/2/3) on the same Mother Lode AOI. Reports AUC delta, capture-
curve delta, and SHAP top-15 feature ranking so we can see whether the
new features are actually being used by the model.

The v3 baseline numbers come from `motherlode/cv_metrics_motherlode.json`
(written by the v3 Phase 2 run); v3.1 numbers come from a fresh CV run
on the rebuilt feature frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from ai_minerals.regions.motherlode import MOTHERLODE
from ai_minerals.model import (
    add_lithology_onehot, build_training_set, sample_pseudo_negatives,
    NON_FEATURE_COLUMNS,
)
from ai_minerals.model_rf import (
    count_feature_columns, feature_importance, make_rf, make_hgb,
    spatial_block_scores_tree,
)

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
V3P1_DIR = ML_DIR / "v3p1"
V3P1_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    print(f"v3.1 feature frame: {df.shape}")
    new_cols = [c for c in df.columns if "magnetic_" in c and c != "magnetic" or "major" in c]
    print(f"v3.1-added cols: {new_cols}")

    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")
    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    print("\n=== build_training_set (with v3.1 cols and major1/2/3 one-hots) ===")
    X, y = build_training_set(
        df, top_classes,
        n_per_positive=30, random_state=42,
        label_col=label_col, label_cols=label_cols,
    )
    drop = count_feature_columns(list(X.columns))
    X_trim = X.drop(columns=drop)
    print(f"  full-feature: {X.shape}")
    print(f"  count-free:   {X_trim.shape}  (dropped {len(drop)} count cols)")

    print("\n=== spatial-block CV (20 km, RF only) ===")
    negs = sample_pseudo_negatives(df, n_per_positive=30, random_state=42, label_col=label_col)
    rows = pd.concat(
        [df[df[label_col] == 1][["row", "col", "x", "y"]], negs[["row", "col", "x", "y"]]],
        ignore_index=True,
    )

    cv = spatial_block_scores_tree(X_trim, y, rows, model_factory=make_rf, block_size_m=20_000)
    aucs = cv["roc_auc"].dropna().to_numpy()
    pr_aucs = cv["pr_auc"].dropna().to_numpy()
    print(f"  RF v3.1: AUC {aucs.mean():.3f} ± {aucs.std():.3f}  "
          f"PR-AUC {pr_aucs.mean():.3f} ± {pr_aucs.std():.3f}  folds={len(aucs)}")

    # Bootstrap CI
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(aucs, len(aucs)).mean() for _ in range(2000)])
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    print(f"  Bootstrap 95% CI on mean AUC: [{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\n=== Final RF + SHAP top-15 ===")
    rf_final = make_rf(random_state=42)
    rf_final.fit(X_trim.fillna(-9999).to_numpy(), y)

    df_oh = add_lithology_onehot(df, top_classes,
                                 extra_class_columns=_extra(df, top_n=10))
    feat_cols = list(X_trim.columns)
    # Align: any one-hot column the training set has that the full-data
    # one-hot is missing (because the major1/2/3 top-10 were drawn from a
    # different sampling) gets a zero-filled column.
    for c in feat_cols:
        if c not in df_oh.columns:
            df_oh[c] = 0
    X_all = df_oh[feat_cols].fillna(-9999).to_numpy()

    sample_idx = rng.choice(len(X_all), size=min(5_000, len(X_all)), replace=False)
    expl = shap.TreeExplainer(rf_final)
    shap_values = expl.shap_values(X_all[sample_idx], check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).head(15).reset_index(drop=True)
    print(shap_df.to_string(index=False))

    np.savez_compressed(V3P1_DIR / "shap_rf_motherlode_v3p1.npz",
                        shap_values=shap_values,
                        feat_names=np.array(feat_cols))

    # Compare to v3
    v3 = json.loads((ML_DIR / "cv_metrics_motherlode.json").read_text())
    v3_auc = v3["rf_trim"]["auc_mean"]
    v3_auc_std = v3["rf_trim"]["auc_std"]
    v3_pr = v3["rf_trim"]["pr_auc_mean"]

    delta_auc = aucs.mean() - v3_auc
    print(f"\n=== v3 vs v3.1 RF (no count features) ===")
    print(f"  v3:    AUC {v3_auc:.3f} ± {v3_auc_std:.3f}  PR-AUC {v3_pr:.3f}")
    print(f"  v3.1:  AUC {aucs.mean():.3f} ± {aucs.std():.3f}  PR-AUC {pr_aucs.mean():.3f}  CI95={ci}")
    print(f"  delta AUC: {delta_auc:+.3f}")

    out = {
        "v3_auc_mean": float(v3_auc),
        "v3_auc_std": float(v3_auc_std),
        "v3p1_auc_mean": float(aucs.mean()),
        "v3p1_auc_std": float(aucs.std()),
        "v3p1_auc_ci95": list(ci),
        "v3p1_pr_auc_mean": float(pr_aucs.mean()),
        "n_folds": int(len(aucs)),
        "delta_auc": float(delta_auc),
        "shap_top15": shap_df.to_dict(orient="records"),
        "v3p1_added_features": new_cols,
        "shap_top15_includes_derivative": bool(any("magnetic_" in r["feature"] and r["feature"] != "magnetic"
                                                  for r in shap_df.to_dict(orient="records"))),
        "shap_top15_includes_major": bool(any("major" in r["feature"]
                                              for r in shap_df.to_dict(orient="records"))),
    }
    (V3P1_DIR / "group_b_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {V3P1_DIR / 'group_b_metrics.json'}")


def _extra(df: pd.DataFrame, top_n: int = 10) -> dict[str, list[int]]:
    out = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df.columns:
            out[col] = df[col][df[col] >= 0].value_counts().head(top_n).index.tolist()
    return out


if __name__ == "__main__":
    main()
