"""Phase 4b: apply DevNet (DEEP-SEAM port) to Mother Lode.

Trains DevNet on the Mother Lode feature frame using the same
positive-vs-unlabeled framing as the bagging-PU baseline (positives are
cells with `is_orogenic_gold == 1`; unlabeled are everything else).
Scores all 192,968 cells, saves to the same predictions parquet.

Compared to bagging-PU, DevNet:
- doesn't sample fake negatives; it treats unlabeled as ambiguous and
  uses the deviation loss to push positives above a Gaussian-prior
  reference distribution.
- gives a single ranking instead of an ensemble average.
- is fast on CPU (small MLP).

This script extends Phase 2's predictions parquet with a `p_devnet`
column for direct comparison in the writeup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.model import add_lithology_onehot, NON_FEATURE_COLUMNS
from ai_minerals.model_rf import count_feature_columns
from ai_minerals.model_devnet import fit_devnet, DevNetConfig

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
OUT_DIR = DATA_DERIVED / "motherlode"


def main() -> None:
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    print(f"feature frame: {df.shape}  positives: {int(df['is_orogenic_gold'].sum())}")

    # Drop count features (per BCGT lesson) and prepare canonical feature cols.
    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df, top_classes)

    label_cols = ("is_orogenic_gold", "is_low_sulfidation")
    drop_count = count_feature_columns(list(df_oh.columns))
    feat_cols = [
        c for c in df_oh.columns
        if c not in NON_FEATURE_COLUMNS
        and c not in label_cols
        and c not in drop_count
        and c != "lithology_class"
    ]
    print(f"feature columns: {len(feat_cols)}")

    # Train DevNet on (df_oh, is_orogenic_gold). DEEP-SEAM hyperparameters.
    cfg = DevNetConfig(
        hidden=(24, 12),
        learning_rate=0.005,
        batch_size=128,
        n_epochs=500,
        n_ref=5000,
        confidence_margin=5.0,
        seed=42,
    )

    print("\nTraining DevNet on Mother Lode...")
    scores, cfg_used, model = fit_devnet(
        df_oh, feat_cols=feat_cols, label_col="is_orogenic_gold", config=cfg,
    )
    print(f"\nMother Lode DevNet scores: range [{scores.min():.3f}, {scores.max():.3f}]")

    # Headline capture-curve numbers (training-only — no holdout split here;
    # this is "model can recall its training labels" sanity, not a true
    # out-of-sample test. The Phase 5 cross-region transfer is the
    # external test.)
    pos = (df["is_orogenic_gold"] == 1).to_numpy()
    n_pos = pos.sum()
    n = len(scores)
    order = np.argsort(-scores)
    sorted_pos = pos[order]
    capture_pcts = {}
    print("\nDevNet capture curve on Mother Lode:")
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        captured = sorted_pos[:k].sum()
        rate = float(captured / n_pos)
        capture_pcts[f"top_{p}_pct"] = rate
        print(f"  top {p:>3}% (k={k:>6}): {captured}/{n_pos} = {rate*100:.1f}%")

    # Extend the Phase 2 predictions parquet with a p_devnet column.
    pred_path = OUT_DIR / "model_predictions_motherlode.parquet"
    pred = pd.read_parquet(pred_path)
    # Align: scores are produced in the order of df_oh rows, which matches
    # df rows since add_lithology_onehot preserves order.
    pred["p_devnet"] = scores
    pred.to_parquet(pred_path)
    print(f"\nUpdated {pred_path} with p_devnet column "
          f"({pred.shape[0]} cells × {pred.shape[1]} cols)")

    metrics = {
        "n_positives": int(n_pos),
        "n_cells": int(n),
        "n_features_used": len(feat_cols),
        "config": {
            "hidden": list(cfg_used.hidden),
            "lr": cfg_used.learning_rate,
            "batch_size": cfg_used.batch_size,
            "n_epochs": cfg_used.n_epochs,
            "confidence_margin": cfg_used.confidence_margin,
        },
        "score_stats": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
        },
        "training_capture_pcts": capture_pcts,
    }
    out_json = OUT_DIR / "devnet_metrics_motherlode.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
