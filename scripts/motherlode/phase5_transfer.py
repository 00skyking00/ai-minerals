"""Phase 5: Sierra → Klamath cross-region transfer.

The marquee validation. Train a Random Forest on Sierra Mother Lode
features, score Klamath cells WITHOUT retraining, report whether known
Klamath orogenic-Au districts get ranked high.

Why this is the strongest validation we have:
- Klamath is a different geological province (terrane-quilted oceanic
  accretion vs Sierra's coherent foothills belt).
- Klamath positives were never in the training set.
- Both regions host orogenic Au of the same Cox-Singer 36a class, so
  asking the model to generalize across them makes geological sense
  (vs e.g. porphyry to orogenic, which would be the wrong test).
- A clear positive result (Klamath districts in top-10%) confirms the
  model picked up rock-type-level signal, not location-specific clusters.

Caveats called out plainly in the writeup:
- This is geological-generalization, not discovery-validity.
- Klamath has more ultramafic exposure than Sierra; magnetic/gravity
  signatures partially differ.
- We hold the same SGMC GENERALIZED_LITH controlled vocabulary across
  both regions, so lithology-one-hot encoding is comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.regions.motherlode import MOTHERLODE
from ai_minerals.regions.klamath import KLAMATH
from ai_minerals.features.assemble import build_feature_frame
from ai_minerals.model import (
    add_lithology_onehot, build_training_set, NON_FEATURE_COLUMNS,
)
from ai_minerals.model_rf import (
    count_feature_columns, make_rf,
)

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
OUT_DIR = DATA_DERIVED / "motherlode" / "transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # 1. Build (or load) the Mother Lode and Klamath feature frames.
    ml_path = DATA_DERIVED / "features_motherlode_500m.parquet"
    kl_path = DATA_DERIVED / "features_klamath_500m.parquet"

    print("[Sierra] loading feature frame...")
    ml = pd.read_parquet(ml_path)
    print(f"  ML: {ml.shape}, positives: {int(ml['is_orogenic_gold'].sum())}")

    if kl_path.exists():
        print("[Klamath] loading cached feature frame...")
        kl = pd.read_parquet(kl_path)
    else:
        print("[Klamath] building feature frame (first time)...")
        kl = build_feature_frame(KLAMATH, resolution_m=500)
        kl.to_parquet(kl_path)
    print(f"  KL: {kl.shape}, positives: {int(kl['is_orogenic_gold'].sum())}")

    # 2. Determine top lithology classes from Sierra (not Klamath) so the
    # one-hot encoding uses the same Sierra-trained schema for both.
    top_classes = ml["lithology_class"].value_counts().head(10).index.tolist()
    print(f"\nSierra top lithology classes: {top_classes}")

    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")

    # 3. Train RF on Sierra (no count features, BCGT lesson).
    print("\n=== Training RF on Sierra ===")
    X_train, y_train = build_training_set(
        ml, top_classes,
        n_per_positive=30, random_state=42,
        label_col=label_col, label_cols=label_cols,
    )
    drop_cols = count_feature_columns(list(X_train.columns))
    X_train = X_train.drop(columns=drop_cols)
    feat_cols = list(X_train.columns)
    print(f"Sierra training: {X_train.shape}")

    rf = make_rf(random_state=42)
    rf.fit(X_train.fillna(-9999).to_numpy(), y_train)

    # 4. Score Klamath cells with the same feature schema.
    print("\n=== Scoring Klamath ===")
    kl_oh = add_lithology_onehot(kl, top_classes)
    # Some lithology classes from Sierra may be absent in Klamath; the
    # one-hot encoder will produce missing columns → fill with zeros.
    for c in feat_cols:
        if c not in kl_oh.columns:
            kl_oh[c] = 0.0
    X_kl = kl_oh[feat_cols].fillna(-9999).to_numpy()
    p_kl = rf.predict_proba(X_kl)[:, 1]
    print(f"Klamath scores: range [{p_kl.min():.3f}, {p_kl.max():.3f}]  "
          f"median {np.median(p_kl):.3f}  mean {p_kl.mean():.3f}")

    # 5. Capture curves on Klamath positives (the orogenic-Au-bearing
    # cells per MRDS commodity filter).
    pos = (kl["is_orogenic_gold"] == 1).to_numpy()
    n_pos = pos.sum()
    n = len(p_kl)
    if n_pos == 0:
        print("WARNING: no orogenic_gold positives in Klamath frame; "
              "can't compute transfer-test capture curves")
        return

    order = np.argsort(-p_kl)
    sorted_pos = pos[order]
    capture = {}
    print(f"\nCross-region capture (Sierra-trained RF on Klamath):")
    print(f"  positives in Klamath: {n_pos} / {n} cells "
          f"({100*n_pos/n:.2f}% base rate)")
    rng = np.random.default_rng(0)
    rand = rng.random(n)
    rand_order = np.argsort(-rand)
    rand_sorted_pos = pos[rand_order]
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        captured_rf = sorted_pos[:k].sum()
        captured_rand = rand_sorted_pos[:k].sum()
        rate_rf = float(captured_rf / n_pos)
        rate_rand = float(captured_rand / n_pos)
        capture[f"top_{p}_pct"] = {
            "rf_capture": rate_rf,
            "random_capture": rate_rand,
            "lift": rate_rf / max(rate_rand, 1e-6),
        }
        print(f"  top {p:>3}% (k={k:>5}):  random {rate_rand*100:>5.1f}%  "
              f"RF {rate_rf*100:>5.1f}%  lift {rate_rf/max(rate_rand,1e-6):.2f}x")

    # 6. Save artifact for Phase 6 cover-page integration.
    pred_kl = kl[["row", "col", "x", "y", "is_orogenic_gold"]].copy()
    pred_kl["p_rf_sierra"] = p_kl
    pred_kl.to_parquet(OUT_DIR / "klamath_predictions_sierra_rf.parquet")
    metrics = {
        "n_klamath_cells": int(n),
        "n_klamath_positives": int(n_pos),
        "klamath_score_stats": {
            "min": float(p_kl.min()),
            "max": float(p_kl.max()),
            "mean": float(p_kl.mean()),
            "median": float(np.median(p_kl)),
        },
        "capture_curves": capture,
        "sierra_top_lithology_classes": [int(c) for c in top_classes],
        "sierra_n_train_pos": int((y_train == 1).sum()),
        "sierra_n_train_neg": int((y_train == 0).sum()),
    }
    (OUT_DIR / "transfer_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"\nSaved {OUT_DIR / 'transfer_metrics.json'}")


if __name__ == "__main__":
    main()
