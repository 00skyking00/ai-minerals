"""Compute per-cell SHAP top-3 features for the top-K most-prospective
within-belt cells, in checkpointed chunks. Avoids the OOM/timeout failure
mode of the original 20k single-pass attempt.

Strategy:
- Top 5,000 cells by P_v2 (covers all the cells gldbg users care about).
- Process in 500-cell chunks; checkpoint progress to disk after each chunk.
- If a chunk fails, skip it; the rest still get done.
- Final step merges the SHAP results into the existing rationale parquet.

Output (in place):
    data/derived/motherlode/v2_rationale_motherlode_250m.parquet
        column updated: top_shap_features (string per cell, "" if no SHAP)
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shap
from scipy.spatial import cKDTree

from ai_minerals.grid import build_grid
from ai_minerals.model import add_lithology_onehot, non_feature_columns
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.regions.motherlode import MOTHERLODE


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
IN_FEATURES = DATA_DERIVED / "features_motherlode_250m.parquet"
IN_MASK = ML_DIR / "v2_within_belt_mask.parquet"
IN_RATIONALE = ML_DIR / "v2_rationale_motherlode_250m.parquet"
MRDS_GPKG = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds_motherlode.gpkg")
CHECKPOINT = ML_DIR / "v2_shap_checkpoint.parquet"

N_TOP = 25_000
CHUNK = 500


def main() -> None:
    t0 = time.time()
    print("Loading inputs...", flush=True)
    feats = pd.read_parquet(IN_FEATURES)
    mask = pd.read_parquet(IN_MASK)
    rationale = pd.read_parquet(IN_RATIONALE)
    mrds = gpd.read_file(MRDS_GPKG).to_crs(MOTHERLODE.working_crs)

    df = feats.merge(mask, on=["row", "col"], how="left")
    df_belt = df[df["within_belt"] == 1].reset_index(drop=True)

    # Rebuild the producer label exactly as the v2 training script does
    producers = mrds[
        mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)
        & mrds["dev_stat"].astype(str).isin(["Producer", "Past Producer"])
    ].copy()
    grid = build_grid(MOTHERLODE.aoi, resolution_m=250, working_crs=MOTHERLODE.working_crs)
    r_res = 250.0
    grid_x0 = grid.xs[0] - r_res / 2
    grid_y0 = grid.ys[0] - r_res / 2
    grid_x1 = grid.xs[-1] + r_res / 2
    grid_y1 = grid.ys[-1] + r_res / 2
    px = producers.geometry.x.to_numpy()
    py = producers.geometry.y.to_numpy()
    p_inside = (px >= grid_x0) & (px < grid_x1) & (py >= grid_y0) & (py < grid_y1)
    producers = producers[p_inside].copy()
    producers["row"] = np.floor((producers.geometry.y.to_numpy() - grid_y0) / r_res).astype(int)
    producers["col"] = np.floor((producers.geometry.x.to_numpy() - grid_x0) / r_res).astype(int)
    producer_set = set(zip(producers["row"].astype(int), producers["col"].astype(int)))
    df_belt["is_producer_cell"] = df_belt.apply(
        lambda r: (int(r["row"]), int(r["col"])) in producer_set, axis=1,
    )
    print(f"within-belt: {len(df_belt):,}; producer cells: {int(df_belt['is_producer_cell'].sum()):,}",
          flush=True)

    # Same feature encoding as v2 training
    top_classes = df_belt["lithology_class"].value_counts().head(10).index.tolist()
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df_belt.columns:
            extra[col] = df_belt[col][df_belt[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df_belt, top_classes, extra_class_columns=extra or None)
    non_feat = non_feature_columns(label_cols=("is_orogenic_gold", "is_low_sulfidation", "is_producer", "within_belt"))
    feat_cols = [c for c in df_oh.columns
                 if c not in non_feat
                 and c not in ("is_producer_cell",)]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    print(f"feature columns: {len(feat_cols)}", flush=True)
    X = df_oh[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)

    # Pseudo-negatives at 1-km exclusion (matching v2 training)
    pos_idx = np.where(df_belt["is_producer_cell"].values)[0]
    au_mrds = mrds[
        mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)
    ].copy()
    au_xy = np.column_stack([au_mrds.geometry.x.values, au_mrds.geometry.y.values])
    kd = cKDTree(au_xy)
    cell_xy = np.column_stack([df_belt["x"].values, df_belt["y"].values])
    excl, _ = kd.query(cell_xy, k=1)
    eligible = (excl > 1_000.0) & ~df_belt["is_producer_cell"].values
    rng = np.random.default_rng(42)
    classes = df_belt.loc[eligible, "lithology_class"].value_counts()
    per_class = max(1, (30 * len(pos_idx)) // len(classes))
    neg_idx_list = []
    for lc in classes.index:
        cands = np.where(eligible & (df_belt["lithology_class"].values == lc))[0]
        if len(cands) == 0:
            continue
        neg_idx_list.append(rng.choice(cands, size=min(len(cands), per_class), replace=False))
    neg_idx = np.concatenate(neg_idx_list)
    train_idx = np.concatenate([pos_idx, neg_idx])
    y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

    print(f"refitting v2 RF on {len(train_idx):,} samples...", flush=True)
    rf = make_rf(random_state=42)
    rf.fit(X[train_idx], y_train)
    print(f"  fit done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Predict on all within-belt cells to identify top-K
    print("predicting + picking top-K...", flush=True)
    preds = rf.predict_proba(X)[:, 1]
    df_belt["p_v2_recomputed"] = preds
    topk_idx_full = np.argsort(-preds)[:N_TOP]

    # Resume from checkpoint if present
    if CHECKPOINT.exists():
        ckpt = pd.read_parquet(CHECKPOINT)
        done_keys = set(zip(ckpt["row"].astype(int), ckpt["col"].astype(int)))
        print(f"  checkpoint: {len(done_keys):,} cells already done", flush=True)
    else:
        ckpt = pd.DataFrame(columns=["row", "col", "top_shap_features"])
        done_keys = set()

    # Identify which top-K cells still need SHAP
    todo_idx = []
    for i in topk_idx_full:
        rc = (int(df_belt.iloc[i]["row"]), int(df_belt.iloc[i]["col"]))
        if rc not in done_keys:
            todo_idx.append(i)
    todo_idx = np.array(todo_idx, dtype=int)
    print(f"  cells to process: {len(todo_idx):,}", flush=True)

    # SHAP in chunks with checkpointing
    explainer = shap.TreeExplainer(rf)
    results = [ckpt]
    n_chunks = (len(todo_idx) + CHUNK - 1) // CHUNK
    for k in range(n_chunks):
        chunk = todo_idx[k * CHUNK : (k + 1) * CHUNK]
        if len(chunk) == 0:
            break
        t_chunk = time.time()
        try:
            sv = explainer.shap_values(X[chunk])
        except Exception as exc:
            print(f"  chunk {k+1}/{n_chunks} FAILED: {exc}", flush=True)
            continue
        # sv shape: (N, F, 2) for binary or (N, F)
        if isinstance(sv, list):
            sv_pos = sv[1]
        elif sv.ndim == 3:
            sv_pos = sv[:, :, 1]
        else:
            sv_pos = sv
        top3 = np.argsort(-np.abs(sv_pos), axis=1)[:, :3]
        chunk_rows = []
        for j, ci in enumerate(chunk):
            note = "; ".join(
                f"{feat_cols[fi]}({sv_pos[j, fi]:+.3f})"
                for fi in top3[j]
            )
            chunk_rows.append({
                "row": int(df_belt.iloc[ci]["row"]),
                "col": int(df_belt.iloc[ci]["col"]),
                "top_shap_features": note,
            })
        chunk_df = pd.DataFrame(chunk_rows)
        results.append(chunk_df)
        # Checkpoint
        merged = pd.concat(results, ignore_index=True).drop_duplicates(subset=["row", "col"], keep="last")
        merged.to_parquet(CHECKPOINT, index=False)
        elapsed_chunk = time.time() - t_chunk
        elapsed_total = (time.time() - t0) / 60
        print(f"  chunk {k+1}/{n_chunks} ({len(chunk)} cells, {elapsed_chunk:.1f}s) "
              f"total {len(merged):,}/{N_TOP:,}  [{elapsed_total:.1f} min elapsed]", flush=True)

    # Merge SHAP results into the existing rationale parquet
    print("\nMerging SHAP into rationale parquet...", flush=True)
    shap_df = pd.read_parquet(CHECKPOINT)
    # Drop the empty top_shap_features column from existing rationale; replace with SHAP
    rationale = rationale.drop(columns=["top_shap_features"], errors="ignore")
    out = rationale.merge(shap_df, on=["row", "col"], how="left")
    out["top_shap_features"] = out["top_shap_features"].fillna("")
    out.to_parquet(IN_RATIONALE, index=False)
    print(f"wrote {IN_RATIONALE} with SHAP populated for {(out['top_shap_features'] != '').sum():,} cells",
          flush=True)
    print(f"total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
