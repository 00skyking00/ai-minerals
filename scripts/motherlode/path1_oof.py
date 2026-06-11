"""Path 1: out-of-fold spatial-block CV scoring of the entire Mother Lode AOI.

For each 20-km spatial block:
  1. Take training rows (positives + pseudo-negatives) NOT in this block.
  2. Fit RF on those rows.
  3. Predict on every cell whose spatial block is the held-out one.
  4. Save predictions.

After all folds, every cell has exactly one prediction from a model that did
not see its block during training. Then recompute capture-at-top-k% on these
honest held-out predictions and compare to the inflation-prone full-data
scoring number (37% top 5%).

Output:
  - data/derived/motherlode/model_predictions_motherlode_oof.parquet
  - data/derived/motherlode/v3p1/path1_oof_metrics.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.model import (
    add_lithology_onehot, sample_pseudo_negatives, non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns, make_rf

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
V3P1_DIR = ML_DIR / "v3p1"

BLOCK_SIZE_M = 20_000.0
LABEL_COL = "is_orogenic_gold"


def main() -> None:
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
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
    y_train_full = np.concatenate(
        [np.ones(len(pos_indices), dtype=np.int64),
         np.zeros(len(neg_indices), dtype=np.int64)]
    )
    print(f"training set: {len(train_indices):,} rows ({len(pos_indices):,} pos, {len(neg_indices):,} neg)", flush=True)

    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df.columns:
            extra[col] = df[col][df[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df, top_classes, extra_class_columns=extra or None)

    non_feat = non_feature_columns(label_cols=("is_orogenic_gold", "is_low_sulfidation"))
    feat_cols = [c for c in df_oh.columns if c not in non_feat]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    leak_check = [c for c in feat_cols if c in ("is_orogenic_gold", "is_low_sulfidation")]
    assert not leak_check, f"label leak still present: {leak_check}"
    print(f"feature columns: {len(feat_cols)} (dropped {len(drop)} count cols, leak-checked)", flush=True)

    X_all = df_oh[feat_cols].fillna(-9999).to_numpy()

    x_all = df["x"].to_numpy()
    y_coord = df["y"].to_numpy()
    bx = (x_all // BLOCK_SIZE_M).astype(int)
    by = (y_coord // BLOCK_SIZE_M).astype(int)
    block_ids = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
    unique_blocks = np.unique(block_ids)
    print(f"spatial blocks: {len(unique_blocks)}", flush=True)

    train_block_ids = block_ids[train_indices]

    oof = np.full(len(df), np.nan, dtype=np.float32)

    t0 = time.time()
    n_done = 0
    for held_block in unique_blocks:
        train_mask = train_block_ids != held_block
        n_train_pos = int(y_train_full[train_mask].sum())
        if train_mask.sum() < 100 or n_train_pos < 5:
            continue
        test_mask = block_ids == held_block
        if test_mask.sum() == 0:
            continue

        X_train = X_all[train_indices[train_mask]]
        y_train = y_train_full[train_mask]

        rf = make_rf(random_state=42)
        rf.fit(X_train, y_train)

        test_idx = np.where(test_mask)[0]
        proba = rf.predict_proba(X_all[test_idx])[:, 1]
        oof[test_idx] = proba

        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            n_scored = int((~np.isnan(oof)).sum())
            print(f"  fold {n_done}/{len(unique_blocks)}  scored {n_scored:,} cells  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    n_scored = int((~np.isnan(oof)).sum())
    print(f"\ncomplete: {n_done} folds in {elapsed/60:.1f} min", flush=True)
    print(f"cells scored: {n_scored:,} / {len(oof):,}", flush=True)

    valid = ~np.isnan(oof)
    valid_preds = oof[valid]
    valid_pos = pos_mask[valid]
    n_valid = int(valid.sum())
    n_valid_pos = int(valid_pos.sum())

    print(f"\nout-of-fold capture-at-top-k%:", flush=True)
    out = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n_valid * p / 100))
        top_k_idx = np.argsort(-valid_preds)[:k]
        captured = int(valid_pos[top_k_idx].sum())
        rate = captured / max(n_valid_pos, 1)
        lift = rate / (p / 100)
        out[f"top_{p}_pct"] = {
            "rate": float(rate),
            "lift": float(lift),
            "captured": captured,
            "n_top": int(k),
        }
        print(f"  top {p:>3}%: rate={rate*100:>5.1f}%  lift={lift:.2f}x  "
              f"captured={captured}/{n_valid_pos}", flush=True)

    out_pred_path = ML_DIR / "model_predictions_motherlode_oof.parquet"
    pred_out = df[["row", "col", "x", "y", LABEL_COL]].copy()
    pred_out["p_rf_oof"] = oof
    pred_out.to_parquet(out_pred_path)
    print(f"\nsaved {out_pred_path}", flush=True)

    metrics_path = V3P1_DIR / "path1_oof_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "block_size_m": BLOCK_SIZE_M,
        "n_folds_used": int(n_done),
        "n_blocks_total": int(len(unique_blocks)),
        "n_cells_scored": n_valid,
        "n_positives_scored": n_valid_pos,
        "elapsed_minutes": elapsed / 60,
        "capture_at_top_k": out,
    }, indent=2))
    print(f"saved {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
