"""Train RF on 250m Mother Lode feature frame and predict on all cells.

Replicates the v3 RF training recipe (positives + 30x pseudo-negatives,
lithology stratified, 5-km exclusion buffer) but at 250m, and writes
per-cell predictions for the entire AOI for rasterization to GeoTIFF.

Output:
    data/derived/motherlode/model_predictions_motherlode_250m.parquet
        columns: row, col, x, y, p_rf
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.model import (
    add_lithology_onehot,
    sample_pseudo_negatives,
    non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns, make_rf


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
LABEL_COL = "is_orogenic_gold"
LABEL_COLS = ("is_orogenic_gold", "is_low_sulfidation")
IN_PARQUET = DATA_DERIVED / "features_motherlode_250m.parquet"
OUT_PARQUET = DATA_DERIVED / "motherlode" / "model_predictions_motherlode_250m.parquet"


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(IN_PARQUET)
    n_pos = int((df[LABEL_COL] == 1).sum())
    print(f"feature frame: {df.shape}, positives ({LABEL_COL}): {n_pos:,}", flush=True)

    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df.columns:
            extra[col] = df[col][df[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df, top_classes, extra_class_columns=extra or None)

    non_feat = non_feature_columns(label_cols=LABEL_COLS)
    feat_cols = [c for c in df_oh.columns if c not in non_feat]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    print(f"feature columns: {len(feat_cols)}", flush=True)

    pos_mask = (df[LABEL_COL] == 1).to_numpy()
    pos_indices = np.where(pos_mask)[0]
    neg_df = sample_pseudo_negatives(df, n_per_positive=30, random_state=42, label_col=LABEL_COL)
    df_keys = list(zip(df["row"].to_numpy().tolist(), df["col"].to_numpy().tolist()))
    df_key_to_idx = {k: i for i, k in enumerate(df_keys)}
    neg_indices = np.array(
        [df_key_to_idx[(int(r), int(c))]
         for r, c in zip(neg_df["row"], neg_df["col"])
         if (int(r), int(c)) in df_key_to_idx]
    )
    print(f"  positives: {len(pos_indices):,}; negatives: {len(neg_indices):,}", flush=True)

    train_indices = np.concatenate([pos_indices, neg_indices])
    y_train = np.concatenate(
        [np.ones(len(pos_indices), dtype=np.int64),
         np.zeros(len(neg_indices), dtype=np.int64)]
    )

    X_all = df_oh[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)

    print(f"fitting RF on {len(train_indices):,} samples...", flush=True)
    rf = make_rf(random_state=42)
    rf.fit(X_all[train_indices], y_train)
    print(f"  fit done in {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"predicting on all {len(df):,} cells...", flush=True)
    t1 = time.time()
    preds = rf.predict_proba(X_all)[:, 1]
    print(f"  predict done in {(time.time()-t1)/60:.1f} min", flush=True)

    out = df[["row", "col", "x", "y"]].copy()
    out["p_rf"] = preds.astype(np.float32)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET} ({len(out):,} rows)", flush=True)
    print(f"  P min/p50/p99/max: "
          f"{preds.min():.4f} / {np.median(preds):.4f} / "
          f"{np.quantile(preds, 0.99):.4f} / {preds.max():.4f}", flush=True)
    print(f"total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
