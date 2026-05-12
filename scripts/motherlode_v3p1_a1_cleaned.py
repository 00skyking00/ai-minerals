"""Re-run A1 bootstrap capture-CI on Cox-Singer-cleaned labels.

The original A1 (group_a_metrics.json) was computed against the v3
predictions parquet, which was generated before the D1 Cox-Singer
cleanup dropped 7,713 -> 6,149 positives. This script regenerates the
RF predictions against the cleaned feature frame, then recomputes
bootstrap CIs on capture-at-top-k%.

Output: data/derived/motherlode/v3p1/a1_cleaned_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.model import (
    add_lithology_onehot, build_training_set, NON_FEATURE_COLUMNS,
)
from ai_minerals.model_rf import count_feature_columns, make_rf

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
V3P1_DIR = ML_DIR / "v3p1"


def main() -> None:
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    n_pos = int(df["is_orogenic_gold"].sum())
    print(f"feature frame: {df.shape}  cleaned positives: {n_pos:,}")

    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")
    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    X, y = build_training_set(
        df, top_classes, n_per_positive=30, random_state=42,
        label_col=label_col, label_cols=label_cols,
    )
    drop = count_feature_columns(list(X.columns))
    X_trim = X.drop(columns=drop)
    print(f"  training: {X_trim.shape}")

    rf = make_rf(random_state=42)
    rf.fit(X_trim.fillna(-9999).to_numpy(), y)

    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df.columns:
            extra[col] = df[col][df[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df, top_classes, extra_class_columns=extra or None)
    feat_cols = list(X_trim.columns)
    for c in feat_cols:
        if c not in df_oh.columns:
            df_oh[c] = 0.0
    X_all = df_oh[feat_cols].fillna(-9999).to_numpy()
    p_rf = rf.predict_proba(X_all)[:, 1]

    pos = (df[label_col] == 1).to_numpy()
    n = len(p_rf)
    pos_idx = np.where(pos)[0]
    rng = np.random.default_rng(0)

    out = {}
    print("\n=== A1 bootstrap capture CIs (cleaned labels) ===")
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        top_k_mask = np.zeros(n, dtype=bool)
        top_k_mask[np.argsort(-p_rf)[:k]] = True
        boot = []
        for _ in range(2000):
            sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            captured = top_k_mask[sample].sum()
            boot.append(captured / len(sample))
        boot = np.array(boot)
        rate = top_k_mask[pos].sum() / pos.sum()
        out[f"top_{p}_pct"] = {
            "rate": float(rate),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
        }
        print(f"  top {p:>3}%: rate={rate*100:>5.1f}%  CI95=[{out[f'top_{p}_pct']['ci95_low']*100:>5.1f}%, "
              f"{out[f'top_{p}_pct']['ci95_high']*100:>5.1f}%]")

    pred_path = ML_DIR / "model_predictions_motherlode_cleaned.parquet"
    pred = df[["row", "col", "x", "y", label_col, "any_mineral_occurrence", "lithology_class"]].copy()
    pred["p_rf_no_count"] = p_rf
    pred.to_parquet(pred_path)
    print(f"\nSaved {pred_path}")

    metrics = {
        "n_positives_cleaned": int(n_pos),
        "n_features": int(X_trim.shape[1]),
        "A1_bootstrap_capture_ci_cleaned": out,
    }
    json_path = V3P1_DIR / "a1_cleaned_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
