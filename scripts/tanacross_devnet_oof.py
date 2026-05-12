"""Tanacross / Eastern Alaska + DEEP-SEAM methodology.

Applies the DevNet architecture (ported from Pang 2019 + Luo et al.
2026 DEEP-SEAM, in `model_devnet.py`) to the Tanacross/Eastak v3.2
feature frame, with leak-free out-of-fold spatial-block CV scoring.
Direct apples-to-apples comparison vs the RF result from Path 2 Stage
1 (top 1% OOF capture: 22.2%, lift 22.2x; Kenorland 23ETD062 at top
41.4% of map).

Implementation notes:
  - Training set per fold: 45 cleaned-label positives in non-held-out
    blocks PLUS a random sample of 5,000 unlabeled cells from
    non-held-out blocks. This matches the DEEP-SEAM pattern (positives
    + unlabeled) rather than our prior RF pseudo-negative pattern.
  - Feature preprocessing: median-impute NaN, min-max scale to [0, 1].
    DEEP-SEAM applies PCA on top of this for geochem, but for an
    initial DevNet-on-our-data test we use raw min-max-scaled features
    so the difference vs RF is purely the model architecture, not the
    feature preprocessing.
  - DevNet config: 24-12-1 ReLU+linear, lr=0.005, batch=128, 500 epochs,
    confidence_margin=5, n_ref=5000 (Pang 2019 / DEEP-SEAM defaults).
  - 20-km spatial blocks, same as Stage 1.

Output:
  data/derived/eastak/model_predictions_eastak_devnet_oof.parquet
  data/derived/eastak/path4_devnet_metrics.json
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
    add_lithology_onehot, non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns
from ai_minerals.model_devnet import fit_devnet, DevNetConfig

DATA_RAW = Path("/home/sky/src/learning/ai-minerals/data/raw")
DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
EASTAK_DIR = DATA_DERIVED / "eastak"

FEATURE_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p2.parquet"
KENORLAND_CSV = DATA_RAW / "kenorland" / "kenorland_tanacross_collars.csv"
OUT_PRED = EASTAK_DIR / "model_predictions_eastak_devnet_oof.parquet"
OUT_METRICS = EASTAK_DIR / "path4_devnet_metrics.json"

LABEL_COL = "is_porphyry_clean"
LABEL_COLS = ("is_porphyry", "is_porphyry_strict", "is_porphyry_clean")
BLOCK_SIZE_M = 20_000.0
WORKING_CRS = "EPSG:6393"
N_UNLABELED_SAMPLE = 5000  # per fold


def main() -> None:
    df = pd.read_parquet(FEATURE_PARQUET)
    n_pos = int((df[LABEL_COL] == 1).sum())
    print(f"feature frame v3.2: {df.shape}, positives: {n_pos}", flush=True)

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
    leak_check = [c for c in feat_cols if c in LABEL_COLS]
    assert not leak_check, f"label leak: {leak_check}"
    print(f"feature columns: {len(feat_cols)} (leak-checked)", flush=True)

    # Feature preprocessing: median-impute + min-max scale to [0, 1]. Uses
    # full-frame stats; per-fold scaling adds noise without changing the
    # broad picture for DevNet (Pang 2019 reference distribution is robust
    # to absolute scaling once features are bounded).
    X_raw = df_oh[feat_cols].to_numpy(dtype=np.float32)
    col_medians = np.nanmedian(X_raw, axis=0)
    X_imputed = np.where(np.isnan(X_raw), col_medians, X_raw)
    col_min = X_imputed.min(axis=0)
    col_max = X_imputed.max(axis=0)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)
    X_all = ((X_imputed - col_min) / col_range).astype(np.float32)
    print(f"preprocessing: median-impute + min-max scaled to [0, 1], range "
          f"[{X_all.min():.3f}, {X_all.max():.3f}]", flush=True)

    pos_mask = (df[LABEL_COL] == 1).to_numpy()
    pos_indices = np.where(pos_mask)[0]
    unl_indices = np.where(~pos_mask)[0]

    # Spatial blocks.
    x_all = df["x"].to_numpy()
    y_coord = df["y"].to_numpy()
    bx = (x_all // BLOCK_SIZE_M).astype(int)
    by = (y_coord // BLOCK_SIZE_M).astype(int)
    block_ids = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
    unique_blocks = np.unique(block_ids)
    print(f"spatial blocks: {len(unique_blocks)}", flush=True)

    pos_block_ids = block_ids[pos_indices]
    unl_block_ids = block_ids[unl_indices]
    rng = np.random.default_rng(42)

    cfg = DevNetConfig(
        hidden=(24, 12),
        learning_rate=0.005,
        batch_size=128,
        n_epochs=500,
        n_ref=5000,
        confidence_margin=5.0,
        seed=42,
    )

    oof = np.full(len(df), np.nan, dtype=np.float32)
    t0 = time.time()
    n_done = 0

    for held_block in unique_blocks:
        # Training rows: positives in non-held-out blocks + random sample
        # of N_UNLABELED_SAMPLE unlabeled cells from non-held-out blocks.
        train_pos = pos_indices[pos_block_ids != held_block]
        if len(train_pos) < 3:
            continue
        unl_pool = unl_indices[unl_block_ids != held_block]
        if len(unl_pool) < 100:
            continue
        n_sample = min(N_UNLABELED_SAMPLE, len(unl_pool))
        train_unl = rng.choice(unl_pool, size=n_sample, replace=False)
        train_idx = np.concatenate([train_pos, train_unl])
        y_train = np.concatenate([
            np.ones(len(train_pos), dtype=np.int64),
            np.zeros(len(train_unl), dtype=np.int64),
        ])

        # Fit DevNet on this fold.
        train_df = pd.DataFrame(X_all[train_idx], columns=feat_cols)
        train_df["y"] = y_train
        scores_train, cfg_used, model = fit_devnet(
            train_df, feat_cols=feat_cols, label_col="y", config=cfg,
        )

        # Score every cell in held-out block. fit_devnet z-score-normalizes
        # internally using train-fold mean/std; we apply the same here.
        Xtf = X_all[train_idx]
        mu = Xtf.mean(axis=0)
        sd = Xtf.std(axis=0) + 1e-8
        test_mask = block_ids == held_block
        test_idx = np.where(test_mask)[0]
        Xn = ((X_all[test_idx] - mu) / sd).astype(np.float32)
        model.eval()
        with torch.no_grad():
            proba = model(torch.from_numpy(Xn)).squeeze(-1).numpy()
        oof[test_idx] = proba

        n_done += 1
        if n_done % 20 == 0:
            elapsed = time.time() - t0
            n_scored = int((~np.isnan(oof)).sum())
            print(f"  fold {n_done}  scored {n_scored:,}  elapsed={elapsed:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    n_scored = int((~np.isnan(oof)).sum())
    print(f"\nOOF complete: {n_done} folds, {elapsed/60:.1f} min", flush=True)
    print(f"cells scored: {n_scored:,} / {len(oof):,}", flush=True)

    # Score distribution diagnostics.
    valid = ~np.isnan(oof)
    valid_preds = oof[valid]
    valid_pos = pos_mask[valid]
    n_valid = int(valid.sum())
    n_valid_pos = int(valid_pos.sum())
    print(f"\nscore distribution:", flush=True)
    print(f"  positives (n={n_valid_pos}): "
          f"min={valid_preds[valid_pos].min():.3f}  "
          f"median={np.median(valid_preds[valid_pos]):.3f}  "
          f"max={valid_preds[valid_pos].max():.3f}", flush=True)
    print(f"  non-positives:                "
          f"min={valid_preds[~valid_pos].min():.3f}  "
          f"median={np.median(valid_preds[~valid_pos]):.3f}  "
          f"max={valid_preds[~valid_pos].max():.3f}", flush=True)

    # Capture-at-top-k%.
    print(f"\nout-of-fold capture-at-top-k% (DevNet on Tanacross):", flush=True)
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
        print(f"  top {p:>3}%: rate={rate*100:>5.1f}%  lift={lift:.2f}x  "
              f"captured={captured}/{n_valid_pos}", flush=True)

    # Save predictions.
    pred_out = df[["row", "col", "x", "y", LABEL_COL, "is_porphyry"]].copy()
    pred_out["p_devnet_oof"] = oof
    pred_out.to_parquet(OUT_PRED)
    print(f"\nsaved {OUT_PRED}", flush=True)

    # Kenorland blind test.
    print("\n=== Kenorland 23ETD062 blind test (DevNet) ===", flush=True)
    kenorland_df = pd.read_csv(KENORLAND_CSV)
    target = kenorland_df[kenorland_df["hole_id"] == "23ETD062"].iloc[0]
    kenorland_pt = gpd.GeoSeries(
        [Point(target["lon"], target["lat"])], crs="EPSG:4326"
    ).to_crs(WORKING_CRS).iloc[0]
    target_x, target_y = kenorland_pt.x, kenorland_pt.y

    from scipy.spatial import cKDTree
    grid_xy = df[["x", "y"]].to_numpy()
    tree = cKDTree(grid_xy)
    dist, idx = tree.query([(target_x, target_y)])
    target_cell_idx = int(idx[0])
    target_pred = float(oof[target_cell_idx]) if not np.isnan(oof[target_cell_idx]) else None
    if target_pred is not None:
        rank = int((valid_preds > target_pred).sum())
        percentile = 100.0 * (n_valid - rank) / n_valid
        position_pct = (rank / n_valid) * 100
        print(f"  OOF prediction: {target_pred:.4f}", flush=True)
        print(f"  cells with higher pred: {rank:,} / {n_valid:,}", flush=True)
        print(f"  percentile rank: {percentile:.1f}", flush=True)
        print(f"  position: top {position_pct:.2f}% of map", flush=True)
        kenorland_metrics = {
            "hole_id": str(target["hole_id"]),
            "lat": float(target["lat"]),
            "lon": float(target["lon"]),
            "oof_prediction": target_pred,
            "n_cells_with_higher_pred": rank,
            "percentile_rank": percentile,
            "top_pct_position": position_pct,
        }
    else:
        kenorland_metrics = {"note": "target cell not in valid OOF fold"}

    metrics = {
        "stage": "Path 4: DEEP-SEAM (DevNet) on Tanacross v3.2",
        "feature_frame": str(FEATURE_PARQUET),
        "label_col": LABEL_COL,
        "n_positives": n_pos,
        "n_features": len(feat_cols),
        "n_unlabeled_per_fold": N_UNLABELED_SAMPLE,
        "n_folds_used": int(n_done),
        "n_cells_scored": n_valid,
        "elapsed_minutes": elapsed / 60,
        "capture_at_top_k": capture,
        "kenorland_blind_test": kenorland_metrics,
        "rf_baseline_path2_stage1": {
            "top_1_pct_capture": "22.2% (10/45) lift 22.2x",
            "top_5_pct_capture": "64.4% (29/45) lift 12.9x",
            "kenorland_top_pct": 41.4,
        },
        "devnet_config": {
            "hidden": list(cfg.hidden),
            "lr": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "n_epochs": cfg.n_epochs,
            "confidence_margin": cfg.confidence_margin,
            "seed": cfg.seed,
        },
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"saved {OUT_METRICS}", flush=True)


if __name__ == "__main__":
    main()
