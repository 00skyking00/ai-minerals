"""Path 3 Stage D: AZ 4-cell OOF experiment.

  RF on raw features    (v3.2-style baseline)
  RF on decomposed      (DEEP-SEAM-style features on AZ)
  DevNet on raw         (DEEP-SEAM architecture, raw features)
  DevNet on decomposed  (DEEP-SEAM architecture + DEEP-SEAM-style features)

Same OOF spatial-block CV harness as Tanacross Path 5.

Output:
  data/derived/arizona/path3_decomposed_metrics.json
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
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.model_devnet import fit_devnet, DevNetConfig

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
AZ_DIR = DATA_DERIVED / "arizona"
AZ_DIR.mkdir(parents=True, exist_ok=True)

RAW_PARQUET = DATA_DERIVED / "features_arizona_500m.parquet"
DECOMPOSED_PARQUET = DATA_DERIVED / "features_arizona_500m_decomposed.parquet"

LABEL_COL = "is_porphyry_cu"
LABEL_COLS = ("is_porphyry_cu", "is_porphyry_mo", "is_skarn_cu")
BLOCK_SIZE_M = 20_000.0
N_UNLABELED_SAMPLE = 5000  # for DevNet


def setup(parquet_path):
    df = pd.read_parquet(parquet_path)
    n_pos = int((df[LABEL_COL] == 1).sum())
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
    leak = [c for c in feat_cols if c in LABEL_COLS]
    assert not leak, f"label leak: {leak}"
    return df, df_oh, feat_cols, n_pos


def setup_blocks(df):
    bx = (df["x"].to_numpy() // BLOCK_SIZE_M).astype(int)
    by = (df["y"].to_numpy() // BLOCK_SIZE_M).astype(int)
    block_ids = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
    return block_ids, np.unique(block_ids)


def run_rf(df, df_oh, feat_cols):
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
    X_all = df_oh[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    block_ids, unique_blocks = setup_blocks(df)
    train_block_ids = block_ids[train_indices]
    oof = np.full(len(df), np.nan, dtype=np.float32)
    n_done = 0
    t0 = time.time()
    for held_block in unique_blocks:
        train_mask = train_block_ids != held_block
        if train_mask.sum() < 50 or y_train_full[train_mask].sum() < 3:
            continue
        test_idx = np.where(block_ids == held_block)[0]
        if len(test_idx) == 0:
            continue
        rf = make_rf(random_state=42)
        rf.fit(X_all[train_indices[train_mask]], y_train_full[train_mask])
        oof[test_idx] = rf.predict_proba(X_all[test_idx])[:, 1]
        n_done += 1
    print(f"  RF folds done: {n_done}, {(time.time()-t0)/60:.1f} min", flush=True)
    return oof, pos_mask


def run_devnet(df, df_oh, feat_cols):
    pos_mask = (df[LABEL_COL] == 1).to_numpy()
    pos_indices = np.where(pos_mask)[0]
    unl_indices = np.where(~pos_mask)[0]

    X_raw = df_oh[feat_cols].to_numpy(dtype=np.float32)
    col_medians = np.nanmedian(X_raw, axis=0)
    X_imputed = np.where(np.isnan(X_raw), col_medians, X_raw)
    col_min = X_imputed.min(axis=0)
    col_max = X_imputed.max(axis=0)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)
    X_all = ((X_imputed - col_min) / col_range).astype(np.float32)

    block_ids, unique_blocks = setup_blocks(df)
    pos_block_ids = block_ids[pos_indices]
    unl_block_ids = block_ids[unl_indices]
    rng = np.random.default_rng(42)
    cfg = DevNetConfig(hidden=(24, 12), learning_rate=0.005, batch_size=128,
                       n_epochs=500, n_ref=5000, confidence_margin=5.0, seed=42)

    oof = np.full(len(df), np.nan, dtype=np.float32)
    n_done = 0
    t0 = time.time()
    for held_block in unique_blocks:
        train_pos = pos_indices[pos_block_ids != held_block]
        if len(train_pos) < 3:
            continue
        unl_pool = unl_indices[unl_block_ids != held_block]
        if len(unl_pool) < 100:
            continue
        n_sample = min(N_UNLABELED_SAMPLE, len(unl_pool))
        train_unl = rng.choice(unl_pool, size=n_sample, replace=False)
        train_idx = np.concatenate([train_pos, train_unl])
        y_train = np.concatenate([np.ones(len(train_pos), dtype=np.int64),
                                   np.zeros(len(train_unl), dtype=np.int64)])
        train_df = pd.DataFrame(X_all[train_idx], columns=feat_cols)
        train_df["y"] = y_train
        _, _, model = fit_devnet(train_df, feat_cols=feat_cols, label_col="y", config=cfg)
        Xtf = X_all[train_idx]
        mu = Xtf.mean(axis=0); sd = Xtf.std(axis=0) + 1e-8
        test_idx = np.where(block_ids == held_block)[0]
        Xn = ((X_all[test_idx] - mu) / sd).astype(np.float32)
        model.eval()
        with torch.no_grad():
            proba = model(torch.from_numpy(Xn)).squeeze(-1).numpy()
        oof[test_idx] = proba
        n_done += 1
    print(f"  DevNet folds done: {n_done}, {(time.time()-t0)/60:.1f} min", flush=True)
    return oof, pos_mask


def report(name, oof, pos_mask):
    valid = ~np.isnan(oof)
    valid_preds = oof[valid]
    valid_pos = pos_mask[valid]
    n_valid = int(valid.sum())
    n_valid_pos = int(valid_pos.sum())
    print(f"\n--- {name} OOF capture ---", flush=True)
    capture = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n_valid * p / 100))
        top_k_idx = np.argsort(-valid_preds)[:k]
        captured = int(valid_pos[top_k_idx].sum())
        rate = captured / max(n_valid_pos, 1)
        lift = rate / (p / 100)
        capture[f"top_{p}_pct"] = {"rate": float(rate), "lift": float(lift),
                                    "captured": captured, "n_top": int(k)}
        print(f"  top {p:>3}%: rate={rate*100:>5.1f}%  lift={lift:.2f}x  captured={captured}/{n_valid_pos}", flush=True)
    return capture


def main() -> None:
    print("=" * 60); print("AZ Cell A: RF on raw features"); print("=" * 60)
    df_raw, df_raw_oh, feat_raw, n_pos = setup(RAW_PARQUET)
    print(f"raw: {df_raw.shape}, {len(feat_raw)} features, {n_pos} positives", flush=True)
    rf_raw_oof, _ = run_rf(df_raw, df_raw_oh, feat_raw)
    rf_raw_capture = report("RF raw", rf_raw_oof, (df_raw[LABEL_COL] == 1).to_numpy())

    print("\n" + "=" * 60); print("AZ Cell B: DevNet on raw features"); print("=" * 60)
    devnet_raw_oof, _ = run_devnet(df_raw, df_raw_oh, feat_raw)
    devnet_raw_capture = report("DevNet raw", devnet_raw_oof, (df_raw[LABEL_COL] == 1).to_numpy())

    print("\n" + "=" * 60); print("AZ Cell C: RF on decomposed features"); print("=" * 60)
    df_dec, df_dec_oh, feat_dec, _ = setup(DECOMPOSED_PARQUET)
    print(f"decomposed: {df_dec.shape}, {len(feat_dec)} features", flush=True)
    rf_dec_oof, _ = run_rf(df_dec, df_dec_oh, feat_dec)
    rf_dec_capture = report("RF decomposed", rf_dec_oof, (df_dec[LABEL_COL] == 1).to_numpy())

    print("\n" + "=" * 60); print("AZ Cell D: DevNet on decomposed features"); print("=" * 60)
    devnet_dec_oof, _ = run_devnet(df_dec, df_dec_oh, feat_dec)
    devnet_dec_capture = report("DevNet decomposed", devnet_dec_oof, (df_dec[LABEL_COL] == 1).to_numpy())

    metrics = {
        "stage": "Path 3 Stage D: AZ 4-cell OOF",
        "label_col": LABEL_COL,
        "n_positives": n_pos,
        "n_features_raw": len(feat_raw),
        "n_features_decomposed": len(feat_dec),
        "n_unlabeled_per_fold_devnet": N_UNLABELED_SAMPLE,
        "block_size_m": BLOCK_SIZE_M,
        "RF_raw": {"capture_at_top_k": rf_raw_capture},
        "DevNet_raw": {"capture_at_top_k": devnet_raw_capture},
        "RF_decomposed": {"capture_at_top_k": rf_dec_capture},
        "DevNet_decomposed": {"capture_at_top_k": devnet_dec_capture},
        "tanacross_baselines": {
            "RF_raw_top_1": "22.2%, lift 22.2x",
            "RF_decomposed_top_1": "15.6%, lift 15.6x",
            "DevNet_raw_top_1": "13.3%, lift 13.3x",
            "DevNet_decomposed_top_1": "13.3%, lift 13.3x",
        },
    }
    out_path = AZ_DIR / "path3_decomposed_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
