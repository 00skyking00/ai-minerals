"""v2 ML prospectivity for gldbg integration.

Differences from v1 (motherlode_train_predict_250m.py):
- Label is **MRDS producer-only** (dev_stat in Producer/Past Producer, commod
  AU). Cleaner economic signal than the v1 "any MRDS Au occurrence" label.
- Restricted to **within-belt cells** (within 5 km of any AU MRDS site).
  v1's full-AOI scope was essentially a belt/no-belt classifier; v2 forces
  the model to discriminate within the belt where the user actually stakes.
- Tighter **pseudo-negative exclusion** (1 km vs 5 km). Lets the model see
  fine-scale negative context near positives.

Outputs:
    data/derived/motherlode/v2_predictions_motherlode_250m.parquet
        columns: row, col, x, y, p_rf_v2 (within-belt only; others NaN)
    data/derived/motherlode/v2_within_belt_mask.parquet
        columns: row, col, within_belt
    data/derived/motherlode/v2_rf_state.npz
        feature_cols + RF predictions on training set (for SHAP later)
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ai_minerals.features.labels import assign_cells
from ai_minerals.grid import build_grid
from ai_minerals.model import (
    add_lithology_onehot,
    sample_pseudo_negatives,
    non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.regions.motherlode import MOTHERLODE


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
IN_PARQUET = DATA_DERIVED / "features_motherlode_250m.parquet"
MRDS_GPKG = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds_motherlode.gpkg")

OUT_PREDS = ML_DIR / "v2_predictions_motherlode_250m.parquet"
OUT_MASK = ML_DIR / "v2_within_belt_mask.parquet"
OUT_STATE = ML_DIR / "v2_rf_state.npz"

BELT_BUFFER_M = 5_000.0     # within-belt definition
NEG_EXCLUSION_M = 1_000.0    # pseudo-negative exclusion (was 5000 in v1)
N_NEG_PER_POS = 30


def producer_filter(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only Au-commodity producer / past producer sites."""
    commod = gdf["commod1"].astype(str).str.upper()
    has_au = commod.str.contains("AU|GOLD", regex=True, na=False)
    dev = gdf["dev_stat"].astype(str)
    is_producer = dev.isin(["Producer", "Past Producer"])
    return gdf[has_au & is_producer].copy()


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(IN_PARQUET)
    print(f"feature frame: {df.shape}", flush=True)

    # === Build the new label: is_producer ===
    mrds = gpd.read_file(MRDS_GPKG)
    producers = producer_filter(mrds)
    print(f"MRDS producer-only Au sites: {len(producers):,}", flush=True)

    # Map producers to grid cells
    grid = build_grid(MOTHERLODE.aoi, resolution_m=250, working_crs=MOTHERLODE.working_crs)
    cell_assign = assign_cells(producers, grid)
    producer_cells = set(zip(cell_assign["row"].astype(int), cell_assign["col"].astype(int)))
    df["is_producer"] = df.apply(
        lambda r: 1 if (int(r["row"]), int(r["col"])) in producer_cells else 0,
        axis=1,
    ).astype(np.int8)
    n_pos = int(df["is_producer"].sum())
    print(f"is_producer positives at 250m: {n_pos:,}", flush=True)

    # === Within-belt mask: within 5km of any AU MRDS site (all dev_stat) ===
    au_mrds = mrds[mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)].copy()
    au_mrds = au_mrds.to_crs(MOTHERLODE.working_crs)
    au_xy = np.column_stack([au_mrds.geometry.x.values, au_mrds.geometry.y.values])
    cell_xy = np.column_stack([df["x"].values, df["y"].values])
    tree = cKDTree(au_xy)
    dists, _ = tree.query(cell_xy, k=1)
    df["within_belt"] = (dists <= BELT_BUFFER_M).astype(np.int8)
    n_belt = int(df["within_belt"].sum())
    print(f"within-belt cells (≤{BELT_BUFFER_M:.0f}m of any Au MRDS): {n_belt:,} of {len(df):,} "
          f"({100*n_belt/len(df):.1f}%)", flush=True)

    # Save the mask as a separate artifact so the rasterizer can use it
    df[["row", "col", "within_belt"]].to_parquet(OUT_MASK, index=False)

    # === Subset to within-belt for training + prediction ===
    df_belt = df[df["within_belt"] == 1].reset_index(drop=True)
    n_pos_belt = int(df_belt["is_producer"].sum())
    print(f"within-belt positives: {n_pos_belt:,}", flush=True)

    # === Feature encoding ===
    top_classes = df_belt["lithology_class"].value_counts().head(10).index.tolist()
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df_belt.columns:
            extra[col] = df_belt[col][df_belt[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df_belt, top_classes, extra_class_columns=extra or None)

    # Exclude all label columns from features
    non_feat = non_feature_columns(label_cols=("is_orogenic_gold", "is_low_sulfidation", "is_producer", "within_belt"))
    feat_cols = [c for c in df_oh.columns if c not in non_feat]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    print(f"feature columns: {len(feat_cols)}", flush=True)

    # === Pseudo-negatives with tighter exclusion ===
    pos_mask = (df_belt["is_producer"] == 1).to_numpy()
    pos_indices = np.where(pos_mask)[0]

    # Custom pseudo-negative sampling: exclude cells within NEG_EXCLUSION_M of ANY MRDS Au site
    # (not just producers — any occurrence). This is more conservative than 5km exclusion only
    # for producers because it also keeps non-producer trace occurrences out of the negative set.
    cell_xy_belt = np.column_stack([df_belt["x"].values, df_belt["y"].values])
    excl_dists, _ = tree.query(cell_xy_belt, k=1)
    eligible_neg = (excl_dists > NEG_EXCLUSION_M) & ~pos_mask
    n_eligible = int(eligible_neg.sum())
    print(f"eligible negatives (>{NEG_EXCLUSION_M:.0f}m from any Au MRDS): {n_eligible:,}", flush=True)

    # Lithology-stratified sample
    n_neg_target = N_NEG_PER_POS * n_pos_belt
    print(f"sampling {n_neg_target:,} negatives across lithology classes...", flush=True)
    rng = np.random.default_rng(42)
    neg_indices_list = []
    classes = df_belt.loc[eligible_neg, "lithology_class"].value_counts()
    per_class = max(1, n_neg_target // len(classes))
    for lc, _ in classes.items():
        candidates = np.where(eligible_neg & (df_belt["lithology_class"].values == lc))[0]
        if len(candidates) == 0:
            continue
        take = min(len(candidates), per_class)
        chosen = rng.choice(candidates, size=take, replace=False)
        neg_indices_list.append(chosen)
    neg_indices = np.concatenate(neg_indices_list)
    if len(neg_indices) > n_neg_target:
        neg_indices = rng.choice(neg_indices, size=n_neg_target, replace=False)
    print(f"  drew {len(neg_indices):,} negatives across {len(classes)} lithology classes", flush=True)

    train_indices = np.concatenate([pos_indices, neg_indices])
    y_train = np.concatenate(
        [np.ones(len(pos_indices), dtype=np.int64),
         np.zeros(len(neg_indices), dtype=np.int64)]
    )

    X_all = df_oh[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)

    print(f"fitting RF on {len(train_indices):,} samples ({y_train.sum()} pos / {(y_train==0).sum()} neg)...", flush=True)
    rf = make_rf(random_state=42)
    rf.fit(X_all[train_indices], y_train)
    print(f"  fit done in {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"predicting on all {len(df_belt):,} within-belt cells...", flush=True)
    t1 = time.time()
    preds = rf.predict_proba(X_all)[:, 1]
    print(f"  predict done in {(time.time()-t1)/60:.1f} min", flush=True)

    out = df_belt[["row", "col", "x", "y"]].copy()
    out["p_rf_v2"] = preds.astype(np.float32)
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PREDS, index=False)
    print(f"wrote {OUT_PREDS} ({len(out):,} within-belt cells)", flush=True)
    print(f"  P min/p50/p99/max: {preds.min():.4f} / {np.median(preds):.4f} / "
          f"{np.quantile(preds, 0.99):.4f} / {preds.max():.4f}", flush=True)

    # === Save model state for SHAP + lookalike-distance computation ===
    np.savez_compressed(
        OUT_STATE,
        feature_cols=np.array(feat_cols, dtype=object),
        train_indices=train_indices,
        pos_indices=pos_indices,
        x_all_shape=np.array(X_all.shape),
    )
    print(f"wrote {OUT_STATE}", flush=True)
    print(f"total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
