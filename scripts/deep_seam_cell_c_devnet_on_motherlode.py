"""DEEP-SEAM Cell C: DevNet on Mother Lode (our data + their methodology).

Trains DevNet (the model from the DEEP-SEAM paper) on the v3.1
cleaned-label Mother Lode feature frame, then scores every cell. Uses
single-pass full-data scoring (the same measurement DEEP-SEAM uses for
their headline) so the comparison is methodology-faithful.

Important: this is full-data scoring, not OOF. The DEEP-SEAM paper
itself reports full-data-style metrics; the question Cell C answers is
"does DevNet do meaningfully better than our RF on the same Mother Lode
data when measured the same way they measure," not "is DevNet's OOF
generalization better."

Output: data/derived/deep_seam_cell_c_motherlode.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai_minerals.model import (
    add_lithology_onehot, sample_pseudo_negatives, non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns
from ai_minerals.model_devnet import fit_devnet, DevNetConfig

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
FEATURE_PARQUET = DATA_DERIVED / "features_motherlode_500m.parquet"
OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/deep_seam_cell_c_motherlode.json")

LABEL_COL = "is_orogenic_gold"
LABEL_COLS = ("is_orogenic_gold", "is_low_sulfidation")


def main() -> None:
    df = pd.read_parquet(FEATURE_PARQUET)
    n_pos = int((df[LABEL_COL] == 1).sum())
    print(f"feature frame: {df.shape}, cleaned positives: {n_pos:,}", flush=True)

    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    pos_mask = (df[LABEL_COL] == 1).to_numpy()
    pos_indices = np.where(pos_mask)[0]
    neg_df = sample_pseudo_negatives(df, n_per_positive=30, random_state=42, label_col=LABEL_COL)
    df_keys = list(zip(df["row"].to_numpy().tolist(), df["col"].to_numpy().tolist()))
    df_key_to_idx = {k: i for i, k in enumerate(df_keys)}
    neg_indices = np.array(
        [df_key_to_idx[(int(r), int(c))] for r, c in zip(neg_df["row"], neg_df["col"])]
    )
    train_indices = np.concatenate([pos_indices, neg_indices])
    y_train = np.concatenate(
        [np.ones(len(pos_indices), dtype=np.int64),
         np.zeros(len(neg_indices), dtype=np.int64)]
    )
    print(f"training set: {len(train_indices):,} ({len(pos_indices):,} pos, {len(neg_indices):,} neg)", flush=True)

    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df.columns:
            extra[col] = df[col][df[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df, top_classes, extra_class_columns=extra or None)

    non_feat = non_feature_columns(label_cols=LABEL_COLS)
    feat_cols = [c for c in df_oh.columns if c not in non_feat]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    leak_check = [c for c in feat_cols if c in LABEL_COLS]
    assert not leak_check, f"label leak: {leak_check}"
    print(f"feature columns: {len(feat_cols)} (leak-checked)", flush=True)

    # Match DEEP-SEAM's preprocessing: median-impute NaN, then min-max scale to [0, 1].
    X_raw = df_oh[feat_cols].to_numpy(dtype=np.float32)
    col_medians = np.nanmedian(X_raw, axis=0)
    X_imputed = np.where(np.isnan(X_raw), col_medians, X_raw)
    col_min = X_imputed.min(axis=0)
    col_max = X_imputed.max(axis=0)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)
    X_all = ((X_imputed - col_min) / col_range).astype(np.float32)
    print(f"feature preprocessing: median-impute + min-max scaled to [0, 1]", flush=True)
    print(f"  any NaN remaining? {np.isnan(X_all).any()}", flush=True)
    print(f"  range after scale: [{X_all.min():.3f}, {X_all.max():.3f}]", flush=True)

    train_df = pd.DataFrame(X_all[train_indices], columns=feat_cols)
    train_df["y"] = y_train
    train_df = train_df.copy()

    cfg = DevNetConfig(
        hidden=(24, 12),
        learning_rate=0.005,
        batch_size=128,
        n_epochs=500,
        n_ref=5000,
        confidence_margin=5.0,
        seed=42,
    )

    print("\nfitting DevNet on Mother Lode (this is full-data, not OOF)...", flush=True)
    t0 = time.time()
    train_scores, cfg_used, model = fit_devnet(
        train_df, feat_cols=feat_cols, label_col="y", config=cfg,
    )
    elapsed = time.time() - t0
    print(f"trained in {elapsed:.0f}s", flush=True)

    # Score the FULL feature frame (192,968 cells). Features already min-max
    # scaled; fit_devnet does an internal z-score using train mean/std, so we
    # need to apply the same z-score with the training-set stats.
    print("scoring full grid...", flush=True)
    Xtf = X_all[train_indices]
    mu = Xtf.mean(axis=0)
    sd = Xtf.std(axis=0) + 1e-8
    Xn = (X_all - mu) / sd
    model.eval()
    with torch.no_grad():
        scores_all = model(torch.from_numpy(Xn.astype(np.float32))).squeeze(-1).numpy()
    print(f"  scores: range [{scores_all.min():.3f}, {scores_all.max():.3f}], "
          f"median {np.median(scores_all):.3f}", flush=True)

    # Capture-at-top-k% (DEEP-SEAM-style: full-data scoring; comparable to
    # their published headline measurement).
    n = len(scores_all)
    print("\n=== Cell C: capture-at-top-k% (DevNet on Mother Lode, full-data scoring) ===", flush=True)
    capture = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        top_k_idx = np.argsort(-scores_all)[:k]
        captured = int(pos_mask[top_k_idx].sum())
        rate = captured / max(n_pos, 1)
        lift = rate / (p / 100)
        capture[f"top_{p}_pct"] = {
            "rate": float(rate), "lift": float(lift),
            "captured": captured, "n_top": int(k),
        }
        print(f"  top {p:>3}%: rate={rate*100:>5.1f}%  lift={lift:.2f}x  "
              f"captured={captured}/{n_pos}", flush=True)

    out = {
        "methodology": "Cell C: our data (Mother Lode v3.1 cleaned) + DevNet from paper",
        "n_cells": int(n),
        "n_positives": int(n_pos),
        "n_features": len(feat_cols),
        "training_set_size": len(train_indices),
        "score_stats": {
            "min": float(scores_all.min()),
            "max": float(scores_all.max()),
            "mean": float(scores_all.mean()),
            "median": float(np.median(scores_all)),
        },
        "capture_at_top_k": capture,
        "config": {
            "hidden": list(cfg.hidden),
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "n_epochs": cfg.n_epochs,
            "confidence_margin": cfg.confidence_margin,
            "seed": cfg.seed,
        },
        "elapsed_seconds": float(elapsed),
        "comparison_baseline_v3p1_RF_full_data": {
            "top_5_pct": "37.0% (the inflated number; full-data scoring of RF)",
            "note": "RF OOF top 5% on same data was 19.5%, lift 3.91x (held-out, leak-free)",
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
