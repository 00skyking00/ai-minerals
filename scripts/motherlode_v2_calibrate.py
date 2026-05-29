"""Calibrate v2 RF predictions with isotonic regression.

The raw v2 RF is monotonic in ranking but over-confident: cells with raw P=0.9-1.0
have an actual ~16% positive rate, not 90%. For pure ranking (gldbg's main use)
this is fine, but a calibrated probability is more interpretable.

Uses sklearn's CalibratedClassifierCV with method='isotonic' and cv=5 (random
folds). Calibration doesn't need spatial CV — it's a 1D mapping fit on
out-of-fold predictions, and random folds give enough data per bin.

Output:
    data/derived/motherlode/prospectivity_motherlode_v2_250m_calibrated_3310.tif
    data/derived/motherlode/prospectivity_motherlode_v2_250m_calibrated_4326.tif
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from scipy.spatial import cKDTree
from sklearn.calibration import CalibratedClassifierCV

from ai_minerals.grid import build_grid
from ai_minerals.model import add_lithology_onehot, non_feature_columns
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.regions.motherlode import MOTHERLODE


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
IN_FEATURES = DATA_DERIVED / "features_motherlode_250m.parquet"
IN_MASK = ML_DIR / "v2_within_belt_mask.parquet"
IN_V1_PREDS = ML_DIR / "model_predictions_motherlode_250m.parquet"
MRDS_GPKG = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds_motherlode.gpkg")
OUT_3310 = ML_DIR / "prospectivity_motherlode_v2_250m_calibrated_3310.tif"
OUT_4326 = ML_DIR / "prospectivity_motherlode_v2_250m_calibrated_4326.tif"
RES_M = 250.0


def write_geotiff(values: np.ndarray, df_xy: pd.DataFrame) -> None:
    x_min = df_xy["x"].min() - RES_M / 2
    x_max = df_xy["x"].max() + RES_M / 2
    y_min = df_xy["y"].min() - RES_M / 2
    y_max = df_xy["y"].max() + RES_M / 2
    width = int(round((x_max - x_min) / RES_M))
    height = int(round((y_max - y_min) / RES_M))
    transform = Affine(RES_M, 0.0, x_min, 0.0, -RES_M, y_max)
    arr = np.full((height, width), np.nan, dtype=np.float32)
    cols_idx = ((df_xy["x"].values - x_min) / RES_M).astype(int)
    rows_idx = ((y_max - df_xy["y"].values) / RES_M).astype(int)
    cols_idx = np.clip(cols_idx, 0, width - 1)
    rows_idx = np.clip(rows_idx, 0, height - 1)
    arr[rows_idx, cols_idx] = values.astype(np.float32)

    with rasterio.open(
        OUT_3310, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=CRS.from_string("EPSG:3310"),
        transform=transform, nodata=float("nan"), compress="deflate",
    ) as dst:
        dst.write(arr, 1)

    with rasterio.open(OUT_3310) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": CRS.from_string("EPSG:4326"),
                       "transform": dst_transform, "width": dst_width,
                       "height": dst_height, "compress": "deflate"})
        with rasterio.open(OUT_4326, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=CRS.from_string("EPSG:4326"),
                resampling=Resampling.bilinear,
            )


def main() -> None:
    t0 = time.time()
    print("Loading inputs...", flush=True)
    feats = pd.read_parquet(IN_FEATURES)
    mask = pd.read_parquet(IN_MASK)
    mrds = gpd.read_file(MRDS_GPKG).to_crs(MOTHERLODE.working_crs)

    df = feats.merge(mask, on=["row", "col"], how="left")
    df_belt = df[df["within_belt"] == 1].reset_index(drop=True)

    # Producer label (correct south-to-north row formula)
    producers = mrds[
        mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)
        & mrds["dev_stat"].astype(str).isin(["Producer", "Past Producer"])
    ].copy()
    grid = build_grid(MOTHERLODE.aoi, resolution_m=250, working_crs=MOTHERLODE.working_crs)
    grid_x0 = grid.xs[0] - RES_M / 2
    grid_y0 = grid.ys[0] - RES_M / 2
    grid_x1 = grid.xs[-1] + RES_M / 2
    grid_y1 = grid.ys[-1] + RES_M / 2
    px = producers.geometry.x.to_numpy()
    py = producers.geometry.y.to_numpy()
    inside = (px >= grid_x0) & (px < grid_x1) & (py >= grid_y0) & (py < grid_y1)
    producers = producers[inside].copy()
    producers["row"] = np.floor((producers.geometry.y.to_numpy() - grid_y0) / RES_M).astype(int)
    producers["col"] = np.floor((producers.geometry.x.to_numpy() - grid_x0) / RES_M).astype(int)
    producer_set = set(zip(producers["row"].astype(int), producers["col"].astype(int)))
    df_belt["is_producer_cell"] = df_belt.apply(
        lambda r: (int(r["row"]), int(r["col"])) in producer_set, axis=1,
    )
    n_pos = int(df_belt["is_producer_cell"].sum())
    print(f"within-belt: {len(df_belt):,}; producer cells: {n_pos:,}", flush=True)

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
    X = df_oh[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)

    # Build training set: producer positives + 30:1 stratified negatives, 1-km exclusion
    pos_idx = np.where(df_belt["is_producer_cell"].values)[0]
    au_mrds = mrds[mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)].copy()
    au_xy = np.column_stack([au_mrds.geometry.x.values, au_mrds.geometry.y.values])
    cell_xy = np.column_stack([df_belt["x"].values, df_belt["y"].values])
    excl, _ = cKDTree(au_xy).query(cell_xy, k=1)
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
    print(f"training set: {len(train_idx):,} ({len(pos_idx):,} pos / {len(neg_idx):,} neg)", flush=True)

    # CalibratedClassifierCV: trains 5 base RF models on 5-fold splits, fits
    # isotonic regression on out-of-fold predictions
    print("fitting CalibratedClassifierCV (isotonic, 5-fold)...", flush=True)
    base_rf = make_rf(random_state=42)
    cal = CalibratedClassifierCV(base_rf, method="isotonic", cv=5, n_jobs=1)
    cal.fit(X[train_idx], y_train)
    print(f"  fit done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Predict on all within-belt cells with the calibrated ensemble
    print(f"predicting on all {len(df_belt):,} within-belt cells...", flush=True)
    t1 = time.time()
    preds = cal.predict_proba(X)[:, 1]
    print(f"  predict done in {(time.time()-t1)/60:.1f} min", flush=True)
    print(f"  P_calibrated min/p50/p99/max: {preds.min():.4f} / {np.median(preds):.4f} / "
          f"{np.quantile(preds, 0.99):.4f} / {preds.max():.4f}", flush=True)

    # Calibration check: bin by predicted P and report observed positive rate
    pos = df_belt["is_producer_cell"].values
    edges = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(preds, edges, right=False) - 1, 0, 9)
    print()
    print("=== Calibrated raster vs observed positive rate ===")
    for b in range(10):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        rate = pos[m].mean()
        print(f"  P=[{edges[b]:.1f},{edges[b+1]:.1f}): n={m.sum():7d}  pos_rate={rate:.4f}")

    # For outside-belt cells: use the v1 raster value * 0.5 (same as combined raster)
    v1 = pd.read_parquet(IN_V1_PREDS)
    combined = feats.merge(mask, on=["row", "col"], how="left")
    combined = combined.merge(v1[["row", "col", "p_rf"]].rename(columns={"p_rf": "p_v1"}),
                              on=["row", "col"], how="left")
    # Build the calibrated full-AOI grid
    out_full = np.full(len(combined), 0.0, dtype=np.float32)
    belt_mask_full = combined["within_belt"].values == 1
    # Place calibrated predictions back into the full-AOI ordering
    belt_to_orig = combined.index[belt_mask_full]
    out_full[belt_to_orig] = preds.astype(np.float32)
    # Outside belt: v1 * 0.5 (clamped at 0.5)
    out_belt_no = ~belt_mask_full
    v1_outside = combined.loc[out_belt_no, "p_v1"].fillna(0.0).values * 0.5
    out_full[out_belt_no] = v1_outside.astype(np.float32)

    write_geotiff(out_full, combined)
    print(f"\nwrote {OUT_4326}", flush=True)
    print(f"total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
