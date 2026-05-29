"""Validity tests for the v2 ML prospectivity model.

Five tests:
1. Spatial-blocked CV capture rate (the rigorous "does it generalize?" test)
2. Random hold-out capture rate (baseline; biased upward by spatial autocorrelation)
3. Famous-producer sanity check (does the model know the famous deposits?)
4. Calibration plot data (binned P vs observed-positive fraction)
5. Comparison: v2 vs v1 vs random-rank baseline at top-K capture

All metrics are on the within-belt subset (the v2 model's training scope).
The "positives" are the same producer-status Au sites the v2 model targets.

Output:
    data/derived/motherlode/v2_validity.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ai_minerals.grid import build_grid
from ai_minerals.model import add_lithology_onehot, non_feature_columns
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.regions.motherlode import MOTHERLODE


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
IN_FEATURES = DATA_DERIVED / "features_motherlode_250m.parquet"
IN_MASK = ML_DIR / "v2_within_belt_mask.parquet"
IN_V1_PREDS = ML_DIR / "model_predictions_motherlode_250m.parquet"
IN_V2_PREDS = ML_DIR / "v2_predictions_motherlode_250m.parquet"
MRDS_GPKG = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds_motherlode.gpkg")
OUT_JSON = ML_DIR / "v2_validity.json"

BLOCK_SIZE_M = 20_000.0
TOPK_PCTS = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

# Famous Mother Lode Au producers (approximate centroids; lat, lon)
FAMOUS_PRODUCERS = [
    ("Empire-Star (Grass Valley)",  39.21,  -121.06),
    ("Idaho-Maryland (Grass Valley)", 39.227, -121.025),
    ("North Star (Grass Valley)",   39.190, -121.085),
    ("Argonaut (Jackson)",          38.347, -120.769),
    ("Kennedy (Jackson)",           38.353, -120.769),
    ("Sutter Creek",                38.392, -120.802),
    ("Carson Hill (Melones)",       38.024, -120.547),
    ("Plumas-Eureka (Johnsville)",  39.755, -120.700),
    ("Original Sixteen-to-One (Alleghany)", 39.473, -120.840),
    ("Sheep Ranch",                 38.211, -120.476),
    ("Mother Lode (Sonora)",        37.984, -120.382),
    ("Angels Camp - Utica",         38.078, -120.539),
]


def setup() -> dict:
    feats = pd.read_parquet(IN_FEATURES)
    mask = pd.read_parquet(IN_MASK)
    v1 = pd.read_parquet(IN_V1_PREDS)
    v2 = pd.read_parquet(IN_V2_PREDS)
    mrds = gpd.read_file(MRDS_GPKG).to_crs(MOTHERLODE.working_crs)

    df = feats.merge(mask, on=["row", "col"], how="left")
    df_belt = df[df["within_belt"] == 1].reset_index(drop=True)

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
    df_belt = df_belt.merge(v1[["row", "col", "p_rf"]].rename(columns={"p_rf": "p_v1"}),
                            on=["row", "col"], how="left")
    df_belt = df_belt.merge(v2[["row", "col", "p_rf_v2"]],
                            on=["row", "col"], how="left")

    # Feature encoding (matches v2 training)
    top_classes = df_belt["lithology_class"].value_counts().head(10).index.tolist()
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df_belt.columns:
            extra[col] = df_belt[col][df_belt[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df_belt, top_classes, extra_class_columns=extra or None)
    non_feat = non_feature_columns(label_cols=("is_orogenic_gold", "is_low_sulfidation", "is_producer", "within_belt"))
    feat_cols = [c for c in df_oh.columns
                 if c not in non_feat
                 and c not in ("p_v1", "p_rf_v2", "is_producer_cell")]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    X = df_oh[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)

    au_mrds = mrds[mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)].to_crs(MOTHERLODE.working_crs)
    au_xy = np.column_stack([au_mrds.geometry.x.values, au_mrds.geometry.y.values])

    return {
        "df_belt": df_belt, "X": X, "feat_cols": feat_cols,
        "au_xy": au_xy, "grid_meta": (grid_x0, grid_y0, r_res),
    }


def sample_negatives(df_belt: pd.DataFrame, au_xy: np.ndarray,
                     pos_indices: np.ndarray, seed: int = 42) -> np.ndarray:
    """Same pseudo-negative recipe as v2 training: 1km exclusion + lithology-stratified."""
    cell_xy = np.column_stack([df_belt["x"].values, df_belt["y"].values])
    kd = cKDTree(au_xy)
    excl, _ = kd.query(cell_xy, k=1)
    pos_mask = df_belt["is_producer_cell"].values
    eligible = (excl > 1_000.0) & ~pos_mask
    rng = np.random.default_rng(seed)
    classes = df_belt.loc[eligible, "lithology_class"].value_counts()
    per_class = max(1, (30 * len(pos_indices)) // len(classes))
    neg_idx_list = []
    for lc in classes.index:
        cands = np.where(eligible & (df_belt["lithology_class"].values == lc))[0]
        if len(cands) == 0:
            continue
        neg_idx_list.append(rng.choice(cands, size=min(len(cands), per_class), replace=False))
    return np.concatenate(neg_idx_list)


def capture_at_topk(scores: np.ndarray, pos_mask: np.ndarray) -> dict[str, dict]:
    """Compute capture-at-top-k% for a vector of scores against a positive mask.
    Only scores cells with finite values."""
    valid = np.isfinite(scores)
    s = scores[valid]
    p = pos_mask[valid]
    n = len(s)
    n_pos = int(p.sum())
    out = {}
    order = np.argsort(-s)
    p_sorted = p[order]
    for k in TOPK_PCTS:
        n_top = max(1, int(np.ceil(n * k / 100.0)))
        captured = int(p_sorted[:n_top].sum())
        rate = captured / max(n_pos, 1)
        out[f"top_{k}pct"] = {
            "rate": float(rate),
            "captured": captured,
            "total_positives": n_pos,
            "lift_vs_random": float(rate / (k / 100.0)),
        }
    return out


def spatial_blocked_cv(df_belt: pd.DataFrame, X: np.ndarray,
                       au_xy: np.ndarray) -> dict:
    """20-km spatial-block CV. For each block: train on the rest, predict on this block.
    Stitch all OOF predictions and compute capture-at-K."""
    print("\n=== Spatial-blocked CV (20-km blocks) ===", flush=True)
    bx = (df_belt["x"].to_numpy() // BLOCK_SIZE_M).astype(int)
    by = (df_belt["y"].to_numpy() // BLOCK_SIZE_M).astype(int)
    block_ids = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
    unique_blocks = np.unique(block_ids)
    print(f"  blocks: {len(unique_blocks)}", flush=True)

    pos_mask = df_belt["is_producer_cell"].values
    pos_indices = np.where(pos_mask)[0]
    neg_indices = sample_negatives(df_belt, au_xy, pos_indices, seed=42)
    train_pool = np.concatenate([pos_indices, neg_indices])
    y_pool = np.concatenate([np.ones(len(pos_indices)), np.zeros(len(neg_indices))])
    pool_block_ids = block_ids[train_pool]

    oof = np.full(len(df_belt), np.nan, dtype=np.float32)
    n_done = 0
    t0 = time.time()
    for block in unique_blocks:
        train_keep = pool_block_ids != block
        if train_keep.sum() < 100 or y_pool[train_keep].sum() < 5:
            continue
        test_idx = np.where(block_ids == block)[0]
        if len(test_idx) == 0:
            continue
        rf = make_rf(random_state=42)
        rf.fit(X[train_pool[train_keep]], y_pool[train_keep])
        oof[test_idx] = rf.predict_proba(X[test_idx])[:, 1]
        n_done += 1
    print(f"  folds completed: {n_done}; {(time.time()-t0)/60:.1f} min", flush=True)
    return {
        "n_folds": int(n_done),
        "capture": capture_at_topk(oof, pos_mask),
    }


def famous_producer_check(df_belt: pd.DataFrame, grid_meta: tuple) -> list[dict]:
    """For each famous mine, find its grid cell and report v2 P + lookalike-y context."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", MOTHERLODE.working_crs, always_xy=True)
    grid_x0, grid_y0, r_res = grid_meta
    results = []
    df_indexed = df_belt.set_index(["row", "col"])
    for name, lat, lon in FAMOUS_PRODUCERS:
        x, y = transformer.transform(lon, lat)
        row = int((y - grid_y0) / r_res)
        col = int((x - grid_x0) / r_res)
        try:
            r = df_indexed.loc[(row, col)]
            results.append({
                "name": name, "lat": lat, "lon": lon,
                "row": row, "col": col,
                "within_belt": bool(r["within_belt"] == 1),
                "is_producer_cell": bool(r["is_producer_cell"]),
                "p_v1": float(r["p_v1"]) if pd.notna(r["p_v1"]) else None,
                "p_v2": float(r["p_rf_v2"]) if pd.notna(r["p_rf_v2"]) else None,
            })
        except KeyError:
            results.append({
                "name": name, "lat": lat, "lon": lon,
                "row": row, "col": col,
                "within_belt": False, "is_producer_cell": False,
                "p_v1": None, "p_v2": None, "note": "cell not in within-belt frame",
            })
    return results


def calibration(df_belt: pd.DataFrame) -> dict:
    """Bin v2 P into 10 deciles; report fraction-positive per bin."""
    p = df_belt["p_rf_v2"].values
    pos = df_belt["is_producer_cell"].values
    edges = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, 9)
    out = []
    for b in range(10):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        out.append({
            "p_lo": float(edges[b]), "p_hi": float(edges[b+1]),
            "n_cells": int(m.sum()),
            "n_positive": int(pos[m].sum()),
            "positive_rate": float(pos[m].mean()),
        })
    return {"bins": out}


def compare_methods(df_belt: pd.DataFrame) -> dict:
    """Capture-at-K for: v2 (in-sample), v1 (in-sample), MRDS-density-only,
    random baseline. All on within-belt cells."""
    pos = df_belt["is_producer_cell"].values
    # MRDS-density-only baseline: count Au MRDS sites within 1km of each cell
    cell_xy = np.column_stack([df_belt["x"].values, df_belt["y"].values])
    rng = np.random.default_rng(123)
    methods = {
        "v2_in_sample": df_belt["p_rf_v2"].values,
        "v1_in_sample_within_belt": df_belt["p_v1"].values,
        "random_baseline": rng.uniform(size=len(df_belt)),
    }
    out = {}
    for name, scores in methods.items():
        out[name] = capture_at_topk(scores, pos)
    return out


def main() -> None:
    t0 = time.time()
    print("Loading + setting up...", flush=True)
    ctx = setup()
    df_belt = ctx["df_belt"]
    print(f"within-belt cells: {len(df_belt):,}; producer cells: {int(df_belt['is_producer_cell'].sum()):,}", flush=True)

    results: dict = {}

    print("\n[1/5] Spatial-blocked CV (the rigorous test)...", flush=True)
    results["spatial_blocked_cv"] = spatial_blocked_cv(df_belt, ctx["X"], ctx["au_xy"])

    print("\n[2/5] Famous-producer sanity check...", flush=True)
    results["famous_producers"] = famous_producer_check(df_belt, ctx["grid_meta"])
    for r in results["famous_producers"]:
        if r.get("p_v2") is not None:
            print(f"  {r['name']:36s} v2_P={r['p_v2']:.3f}  in_belt={r['within_belt']}  is_producer={r['is_producer_cell']}",
                  flush=True)

    print("\n[3/5] Calibration plot data...", flush=True)
    results["calibration"] = calibration(df_belt)
    for b in results["calibration"]["bins"]:
        print(f"  P=[{b['p_lo']:.1f},{b['p_hi']:.1f}): n={b['n_cells']:7d}  pos_rate={b['positive_rate']:.4f}",
              flush=True)

    print("\n[4/5] Comparison: v2 vs v1 vs random...", flush=True)
    results["method_comparison"] = compare_methods(df_belt)
    for method, caps in results["method_comparison"].items():
        cap5 = caps["top_5.0pct"]
        cap1 = caps["top_1.0pct"]
        print(f"  {method:30s} top-1%={cap1['rate']*100:5.1f}%  top-5%={cap5['rate']*100:5.1f}%  "
              f"lift@5%={cap5['lift_vs_random']:5.1f}x",
              flush=True)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_JSON}  (total: {(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
