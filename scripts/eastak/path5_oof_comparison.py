"""Path 5 Stage B: DevNet + RF on Tanacross v3.3 (DEEP-SEAM-style features).

Runs both methods on the same v3.3 feature frame (ILR-PCA geochem +
GLCM geophysics textures) for apples-to-apples comparison vs the v3.2
baselines:
  - v3.2 RF: top-1% capture 22.2% lift 22.2x; Kenorland top 41.4% of map
  - v3.2 DevNet: top-1% capture 13.3% lift 13.3x; Kenorland top 60.3% of map

Question: does DEEP-SEAM-style preprocessing change the result?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point

from ai_minerals.model import (
    add_lithology_onehot, sample_pseudo_negatives, non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.model_devnet import fit_devnet, DevNetConfig

DATA_RAW = Path("/home/sky/src/learning/ai-minerals/data/raw")
DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
EASTAK_DIR = DATA_DERIVED / "eastak"

FEATURE_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p3.parquet"
KENORLAND_CSV = DATA_RAW / "kenorland" / "kenorland_tanacross_collars.csv"

LABEL_COL = "is_porphyry_clean"
LABEL_COLS = ("is_porphyry", "is_porphyry_strict", "is_porphyry_clean")
BLOCK_SIZE_M = 20_000.0
WORKING_CRS = "EPSG:6393"
N_UNLABELED_SAMPLE = 5000


def setup_features():
    df = pd.read_parquet(FEATURE_PARQUET)
    n_pos = int((df[LABEL_COL] == 1).sum())
    print(f"v3.3 frame: {df.shape}, positives: {n_pos}", flush=True)
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
    assert not leak
    print(f"feature columns: {len(feat_cols)} (leak-checked)", flush=True)
    return df, df_oh, feat_cols


def setup_blocks(df):
    x_all = df["x"].to_numpy()
    y_coord = df["y"].to_numpy()
    bx = (x_all // BLOCK_SIZE_M).astype(int)
    by = (y_coord // BLOCK_SIZE_M).astype(int)
    block_ids = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
    return block_ids, np.unique(block_ids)


def kenorland_query(df, oof, valid_preds, n_valid):
    kenorland_df = pd.read_csv(KENORLAND_CSV)
    target = kenorland_df[kenorland_df["hole_id"] == "23ETD062"].iloc[0]
    pt = gpd.GeoSeries([Point(target["lon"], target["lat"])], crs="EPSG:4326").to_crs(WORKING_CRS).iloc[0]
    from scipy.spatial import cKDTree
    tree = cKDTree(df[["x", "y"]].to_numpy())
    _, idx = tree.query([(pt.x, pt.y)])
    target_idx = int(idx[0])
    target_pred = float(oof[target_idx]) if not np.isnan(oof[target_idx]) else None
    if target_pred is None:
        return {"note": "target not in valid OOF fold"}
    rank = int((valid_preds > target_pred).sum())
    return {
        "oof_prediction": target_pred,
        "n_cells_with_higher_pred": rank,
        "percentile_rank": 100.0 * (n_valid - rank) / n_valid,
        "top_pct_position": (rank / n_valid) * 100,
    }


def run_devnet(df, df_oh, feat_cols):
    print("\n========== DevNet on v3.3 ==========", flush=True)
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
    t0 = time.time()
    n_done = 0
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
        if n_done % 30 == 0:
            elapsed = time.time() - t0
            print(f"  fold {n_done}  scored {(~np.isnan(oof)).sum():,}  elapsed={elapsed:.0f}s", flush=True)
    elapsed = time.time() - t0
    print(f"DevNet OOF complete: {n_done} folds, {elapsed/60:.1f} min", flush=True)
    return oof, pos_mask


def run_rf(df, df_oh, feat_cols):
    print("\n========== RF on v3.3 ==========", flush=True)
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
    t0 = time.time()
    n_done = 0
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
        if n_done % 30 == 0:
            elapsed = time.time() - t0
            print(f"  fold {n_done}  scored {(~np.isnan(oof)).sum():,}  elapsed={elapsed:.0f}s", flush=True)
    elapsed = time.time() - t0
    print(f"RF OOF complete: {n_done} folds, {elapsed/60:.1f} min", flush=True)
    return oof, pos_mask


def report(name, oof, pos_mask, df):
    valid = ~np.isnan(oof)
    valid_preds = oof[valid]
    valid_pos = pos_mask[valid]
    n_valid = int(valid.sum())
    n_valid_pos = int(valid_pos.sum())
    print(f"\n--- {name} capture-at-top-k% ---", flush=True)
    capture = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n_valid * p / 100))
        top_k_idx = np.argsort(-valid_preds)[:k]
        captured = int(valid_pos[top_k_idx].sum())
        rate = captured / max(n_valid_pos, 1)
        lift = rate / (p / 100)
        capture[f"top_{p}_pct"] = {
            "rate": float(rate), "lift": float(lift),
            "captured": captured, "n_top": int(k),
        }
        print(f"  top {p:>3}%: rate={rate*100:>5.1f}%  lift={lift:.2f}x  captured={captured}/{n_valid_pos}", flush=True)
    kenorland = kenorland_query(df, oof, valid_preds, n_valid)
    print(f"  Kenorland 23ETD062 top {kenorland.get('top_pct_position', 'N/A'):.2f}% of map" if 'top_pct_position' in kenorland else f"  Kenorland: {kenorland.get('note')}", flush=True)
    return capture, kenorland


def main():
    df, df_oh, feat_cols = setup_features()

    # Run both methods.
    devnet_oof, pos_mask = run_devnet(df, df_oh, feat_cols)
    devnet_capture, devnet_kenorland = report("DevNet v3.3", devnet_oof, pos_mask, df)

    rf_oof, _ = run_rf(df, df_oh, feat_cols)
    rf_capture, rf_kenorland = report("RF v3.3", rf_oof, pos_mask, df)

    # Save predictions and metrics.
    pred_out = df[["row", "col", "x", "y", LABEL_COL, "is_porphyry"]].copy()
    pred_out["p_devnet_v3p3_oof"] = devnet_oof
    pred_out["p_rf_v3p3_oof"] = rf_oof
    pred_out.to_parquet(EASTAK_DIR / "model_predictions_eastak_v3p3.parquet")

    metrics = {
        "stage": "Path 5: ILR-PCA geochem + GLCM textures + DevNet/RF",
        "feature_frame": str(FEATURE_PARQUET),
        "n_features": len(feat_cols),
        "n_positives": int(pos_mask.sum()),
        "DevNet_v3p3": {
            "capture_at_top_k": devnet_capture,
            "kenorland_blind_test": devnet_kenorland,
        },
        "RF_v3p3": {
            "capture_at_top_k": rf_capture,
            "kenorland_blind_test": rf_kenorland,
        },
        "v3p2_baselines_for_comparison": {
            "RF_top_1_pct_capture": "22.2% lift 22.2x",
            "RF_top_5_pct_capture": "64.4% lift 12.9x",
            "RF_kenorland_top_pct": 41.4,
            "DevNet_top_1_pct_capture": "13.3% lift 13.3x",
            "DevNet_top_5_pct_capture": "28.9% lift 5.8x",
            "DevNet_kenorland_top_pct": 60.3,
        },
    }
    (EASTAK_DIR / "path5_decomposed_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {EASTAK_DIR / 'path5_decomposed_metrics.json'}")


if __name__ == "__main__":
    main()
