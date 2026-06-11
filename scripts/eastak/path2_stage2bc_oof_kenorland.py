"""Path 2 Stage 2B+2C: OOF spatial-block CV + Kenorland blind test on Eastak v3.2 feature frame.

Same logic as Stage 1B+1C but reads the v3.2 frame which adds:
  - gravity_isostatic_hp (high-pass-filtered Bouguer gravity)
  - major1/2/3_class (Alaska fine-grained lithology codes)

The major1/2/3 codes get expanded into one-hots via add_lithology_onehot.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ai_minerals.model import (
    add_lithology_onehot, sample_pseudo_negatives, non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns, make_rf

DATA_RAW = Path("/home/sky/src/learning/ai-minerals/data/raw")
DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
EASTAK_DIR = DATA_DERIVED / "eastak"

FEATURE_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p2.parquet"
KENORLAND_CSV = DATA_RAW / "kenorland" / "kenorland_tanacross_collars.csv"
OUT_PRED = EASTAK_DIR / "model_predictions_eastak_oof_v3p2.parquet"
OUT_METRICS = EASTAK_DIR / "path2_stage2_metrics.json"

LABEL_COL = "is_porphyry_clean"
LABEL_COLS = ("is_porphyry", "is_porphyry_strict", "is_porphyry_clean")
BLOCK_SIZE_M = 20_000.0
WORKING_CRS = "EPSG:6393"


def main() -> None:
    df = pd.read_parquet(FEATURE_PARQUET)
    n_pos = int((df[LABEL_COL] == 1).sum())
    print(f"feature frame v3.2: {df.shape}, positives: {n_pos}", flush=True)

    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    # Build training set tracking df row indices.
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
    print(f"training set: {len(train_indices):,} ({len(pos_indices)} pos, {len(neg_indices):,} neg)", flush=True)

    # v3.2: enable major1/2/3 one-hots.
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
        if train_mask.sum() < 50 or y_train_full[train_mask].sum() < 3:
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
        if n_done % 20 == 0:
            elapsed = time.time() - t0
            n_scored = int((~np.isnan(oof)).sum())
            print(f"  fold {n_done}/{len(unique_blocks)}  scored {n_scored:,} cells  elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    n_scored = int((~np.isnan(oof)).sum())
    print(f"\ncomplete: {n_done} folds in {elapsed/60:.1f} min", flush=True)
    print(f"cells scored: {n_scored:,} / {len(oof):,}", flush=True)

    valid = ~np.isnan(oof)
    valid_preds = oof[valid]
    valid_pos = pos_mask[valid]
    n_valid = int(valid.sum())
    n_valid_pos = int(valid_pos.sum())

    print(f"\nout-of-fold capture-at-top-k% (v3.2):", flush=True)
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

    pred_out = df[["row", "col", "x", "y", LABEL_COL, "is_porphyry"]].copy()
    pred_out["p_rf_oof"] = oof
    pred_out.to_parquet(OUT_PRED)
    print(f"\nsaved {OUT_PRED}", flush=True)

    # Kenorland blind test.
    print("\n=== Kenorland 23ETD062 blind test (v3.2) ===", flush=True)
    kenorland_df = pd.read_csv(KENORLAND_CSV)
    target = kenorland_df[kenorland_df["hole_id"] == "23ETD062"].iloc[0]
    print(f"  target: {target['hole_id']} ({target['lat']}, {target['lon']}) Cu={target['cu_pct']}%", flush=True)
    kenorland_pt = gpd.GeoSeries([Point(target["lon"], target["lat"])], crs="EPSG:4326").to_crs(WORKING_CRS).iloc[0]
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
            "target_cell_idx": target_cell_idx,
            "oof_prediction": target_pred,
            "n_cells_with_higher_pred": rank,
            "percentile_rank": percentile,
            "top_pct_position": position_pct,
        }
    else:
        kenorland_metrics = {"note": "target cell not in valid OOF fold"}

    metrics = {
        "stage": "2 (mag derivatives + cleaned label + isostatic-hp gravity + Alaska fine-grained lithology)",
        "feature_frame": str(FEATURE_PARQUET),
        "label_col": LABEL_COL,
        "n_positives": n_pos,
        "n_features": len(feat_cols),
        "n_folds_used": int(n_done),
        "n_cells_scored": n_valid,
        "elapsed_minutes": elapsed / 60,
        "capture_at_top_k": capture,
        "kenorland_blind_test": kenorland_metrics,
        "v1_baseline_kenorland_percentile": 62.0,
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {OUT_METRICS}", flush=True)


if __name__ == "__main__":
    main()
