"""v2 post-training:
1. Combined raster (within-belt = v2 P; outside-belt = v1 P, demoted) → GeoTIFF
   in EPSG:4326 for gldbg
2. Lookalike-distance: nearest producer in feature space + the producer's name
   + geographic distance → sidecar parquet for gldbg rationale
3. SHAP rationale per cell (top-20k most-prospective only): top-3 features
   contributing to each cell's P → sidecar parquet

Inputs:
    data/derived/features_motherlode_250m.parquet
    data/derived/motherlode/v2_predictions_motherlode_250m.parquet
    data/derived/motherlode/v2_within_belt_mask.parquet
    data/derived/motherlode/model_predictions_motherlode_250m.parquet (v1)
    data/raw/mrds/mrds_motherlode.gpkg

Outputs:
    data/derived/motherlode/prospectivity_motherlode_v2_250m_4326.tif
    data/derived/motherlode/lookalike_motherlode_v2_250m_4326.tif
    data/derived/motherlode/v2_rationale_motherlode_250m.parquet
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
from sklearn.preprocessing import StandardScaler

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
IN_FEATURES = DATA_DERIVED / "features_motherlode_250m.parquet"
IN_V2_PREDS = ML_DIR / "v2_predictions_motherlode_250m.parquet"
IN_MASK = ML_DIR / "v2_within_belt_mask.parquet"
IN_V1_PREDS = ML_DIR / "model_predictions_motherlode_250m.parquet"
MRDS_GPKG = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds_motherlode.gpkg")

OUT_RASTER = ML_DIR / "prospectivity_motherlode_v2_250m_4326.tif"
OUT_LOOKALIKE_RASTER = ML_DIR / "lookalike_motherlode_v2_250m_4326.tif"
OUT_RATIONALE = ML_DIR / "v2_rationale_motherlode_250m.parquet"
RES_M = 250.0
SRC_CRS = "EPSG:3310"
DST_CRS = "EPSG:4326"

N_SHAP_TOP = 20_000  # compute SHAP only for the top-K cells by P


def write_geotiff(values: np.ndarray, df_xy: pd.DataFrame, out_3310: Path, out_4326: Path) -> None:
    """Build a (row, col)-indexed raster in EPSG:3310, reproject to EPSG:4326, write both."""
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
        out_3310, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_string(SRC_CRS),
        transform=transform,
        nodata=float("nan"),
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)

    with rasterio.open(out_3310) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": CRS.from_string(DST_CRS),
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "compress": "deflate",
        })
        with rasterio.open(out_4326, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=CRS.from_string(DST_CRS),
                resampling=Resampling.bilinear,
            )


def main() -> None:
    t0 = time.time()
    print("Loading inputs...", flush=True)
    feats = pd.read_parquet(IN_FEATURES)
    v2 = pd.read_parquet(IN_V2_PREDS)
    v1 = pd.read_parquet(IN_V1_PREDS)
    mask = pd.read_parquet(IN_MASK)
    mrds = gpd.read_file(MRDS_GPKG).to_crs(MOTHERLODE.working_crs)

    # === Combined raster ===
    print("\nBuilding combined raster...", flush=True)
    combined = feats.merge(mask, on=["row", "col"], how="left")
    combined = combined.merge(v1[["row", "col", "p_rf"]].rename(columns={"p_rf": "p_v1"}),
                              on=["row", "col"], how="left")
    combined = combined.merge(v2[["row", "col", "p_rf_v2"]],
                              on=["row", "col"], how="left")

    # Within-belt: use v2. Outside-belt: use v1 demoted by 0.5 (so a 1.0 v1 caps at 0.5,
    # ensuring belt cells always rank above non-belt by default).
    in_belt = (combined["within_belt"] == 1)
    p_combined = np.where(
        in_belt.values,
        combined["p_rf_v2"].values,
        combined["p_v1"].values * 0.5,
    ).astype(np.float32)
    p_combined = np.nan_to_num(p_combined, nan=0.0)

    out_3310 = ML_DIR / "prospectivity_motherlode_v2_250m_3310.tif"
    write_geotiff(p_combined, combined, out_3310, OUT_RASTER)
    print(f"  wrote {OUT_RASTER} ({OUT_RASTER.stat().st_size/1e6:.1f} MB)", flush=True)

    # === Lookalike: nearest-producer in feature space, with geo-distance ===
    print("\nBuilding lookalike-distance score...", flush=True)
    # Producer cells in MRDS
    producers = mrds[
        mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)
        & mrds["dev_stat"].astype(str).isin(["Producer", "Past Producer"])
    ].copy()
    grid = build_grid(MOTHERLODE.aoi, resolution_m=250, working_crs=MOTHERLODE.working_crs)
    cell_assign = assign_cells(producers, grid)
    producer_cells = set(zip(cell_assign["row"].astype(int), cell_assign["col"].astype(int)))

    # Within-belt feature frame (matches v2 training scope)
    belt_idx = combined["within_belt"] == 1
    df_belt = combined[belt_idx].reset_index(drop=True)
    df_belt["is_producer_cell"] = df_belt.apply(
        lambda r: (int(r["row"]), int(r["col"])) in producer_cells, axis=1,
    )

    # Same one-hot + feature filtering as the training script
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

    # Standardize so cosine similarity is meaningful
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    producer_mask = df_belt["is_producer_cell"].values
    producer_X = X_std[producer_mask]
    n_producers = producer_mask.sum()
    print(f"  producer cells in within-belt set: {n_producers:,}", flush=True)

    # Nearest-producer in feature space (KDTree on standardized features)
    tree = cKDTree(producer_X)
    t1 = time.time()
    feat_dist, feat_idx = tree.query(X_std, k=1)
    print(f"  feature-space NN: {(time.time()-t1):.1f}s", flush=True)

    # Geographic distance from each cell to the producer it matches in feature space
    producer_xy = np.column_stack([df_belt.loc[producer_mask, "x"].values,
                                    df_belt.loc[producer_mask, "y"].values])
    cell_xy = np.column_stack([df_belt["x"].values, df_belt["y"].values])
    matched_producer_xy = producer_xy[feat_idx]
    geo_dist = np.linalg.norm(cell_xy - matched_producer_xy, axis=1)

    # Lookalike score: high when geological similarity is high (feat_dist low)
    #   AND geographic novelty is high (geo_dist large).
    # Normalize each to [0,1] over the within-belt set.
    feat_sim = 1.0 / (1.0 + feat_dist)                  # 1 = identical, 0 = very different
    geo_norm = np.minimum(geo_dist / 20_000.0, 1.0)     # 20 km cap; 1 = at-or-beyond cap
    lookalike = feat_sim * geo_norm
    print(f"  lookalike: min/p50/p99/max = "
          f"{lookalike.min():.4f} / {np.median(lookalike):.4f} / "
          f"{np.quantile(lookalike, 0.99):.4f} / {lookalike.max():.4f}", flush=True)

    # Rasterize lookalike. Outside-belt cells get 0.
    lookalike_full = np.full(len(combined), 0.0, dtype=np.float32)
    lookalike_full[belt_idx.values] = lookalike
    out_3310 = ML_DIR / "lookalike_motherlode_v2_250m_3310.tif"
    write_geotiff(lookalike_full, combined, out_3310, OUT_LOOKALIKE_RASTER)
    print(f"  wrote {OUT_LOOKALIKE_RASTER}", flush=True)

    # Lookup nearest-producer name for the rationale.
    # Compute (row, col) for each producer point directly (bypassing assign_cells)
    # so we can keep the producers GeoDataFrame's columns alongside.
    producers_xy = producers.copy()
    r_res = 250.0
    grid_x0 = grid.xs[0] - r_res / 2
    grid_y0 = grid.ys[0] - r_res / 2
    grid_x1 = grid.xs[-1] + r_res / 2
    grid_y1 = grid.ys[-1] + r_res / 2
    px = producers_xy.geometry.x.to_numpy()
    py = producers_xy.geometry.y.to_numpy()
    p_inside = (px >= grid_x0) & (px < grid_x1) & (py >= grid_y0) & (py < grid_y1)
    producers_xy = producers_xy[p_inside].copy()
    px = producers_xy.geometry.x.to_numpy()
    py = producers_xy.geometry.y.to_numpy()
    producers_xy["row"] = np.floor((py - grid_y0) / r_res).astype(int)
    producers_xy["col"] = np.floor((px - grid_x0) / r_res).astype(int)
    # First site_name per (row, col) cell
    cell_to_site = (
        producers_xy[["row", "col", "site_name"]]
        .drop_duplicates(subset=["row", "col"], keep="first")
    )
    name_by_rc = {
        (int(r), int(c)): (str(n) if pd.notna(n) else "unnamed")
        for r, c, n in zip(cell_to_site["row"], cell_to_site["col"], cell_to_site["site_name"])
    }

    producer_names_belt = df_belt.loc[producer_mask].reset_index(drop=True)
    producer_rowcol = list(zip(
        producer_names_belt["row"].astype(int).tolist(),
        producer_names_belt["col"].astype(int).tolist(),
    ))
    nearest_names = [name_by_rc.get(rc, "unnamed") for rc in producer_rowcol]
    matched_names = [nearest_names[i] for i in feat_idx]

    # === Rationale sidecar (top-K cells by combined score for now; SHAP TBD) ===
    print("\nBuilding rationale sidecar (top-K cells)...", flush=True)
    df_belt_out = df_belt[["row", "col", "x", "y"]].copy()
    df_belt_out["p_rf_v2"] = df_belt["p_rf_v2"].astype(np.float32)
    df_belt_out["lookalike_score"] = lookalike.astype(np.float32)
    df_belt_out["nearest_producer_name"] = matched_names
    df_belt_out["nearest_producer_geo_dist_m"] = geo_dist.astype(np.float32)
    df_belt_out["nearest_producer_feat_dist"] = feat_dist.astype(np.float32)

    # SHAP top-3 features per cell — disabled by default. The trained RF
    # has no max_depth so trees go very deep on the real geological data
    # (50+ levels), which makes shap.TreeExplainer prohibitively slow
    # (~1+ hour for 20k cells). The nearest-producer name + lookalike
    # score in the sidecar are the load-bearing rationale signals; SHAP
    # is nice-to-have. Re-enable when we have a depth-capped RF or run
    # this offline as a one-shot.
    do_shap = False

    if do_shap:
        print(f"  computing SHAP for top {N_SHAP_TOP:,} cells by P...", flush=True)
        # Retrain RF (we didn't pickle in v2 training) — fast
        rf = make_rf(random_state=42)
        # Same training set as v2 script: producer positives + tight-excl negatives
        pos_idx = np.where(df_belt["is_producer_cell"].values)[0]
        # eligible negatives: cells > 1km from any AU MRDS site
        au_mrds = mrds[
            mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)
        ].to_crs(MOTHERLODE.working_crs)
        au_xy = np.column_stack([au_mrds.geometry.x.values, au_mrds.geometry.y.values])
        kd_au = cKDTree(au_xy)
        excl_dists, _ = kd_au.query(cell_xy, k=1)
        eligible = (excl_dists > 1_000.0) & ~df_belt["is_producer_cell"].values
        rng = np.random.default_rng(42)
        classes = df_belt.loc[eligible, "lithology_class"].value_counts()
        per_class = max(1, (30 * len(pos_idx)) // len(classes))
        neg_idx_list = []
        for lc in classes.index:
            cands = np.where(eligible & (df_belt["lithology_class"].values == lc))[0]
            if len(cands) == 0:
                continue
            take = min(len(cands), per_class)
            neg_idx_list.append(rng.choice(cands, size=take, replace=False))
        neg_idx = np.concatenate(neg_idx_list)
        train_idx = np.concatenate([pos_idx, neg_idx])
        y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

        rf.fit(X[train_idx], y_train)
        print(f"  rf refit done", flush=True)

        # Top-K cells by P
        topk_idx = np.argsort(-df_belt["p_rf_v2"].values)[:N_SHAP_TOP]
        t_shap = time.time()
        import shap as _shap
        explainer = _shap.TreeExplainer(rf)
        sv = explainer.shap_values(X[topk_idx])
        print(f"  shap on {N_SHAP_TOP:,} cells: {(time.time()-t_shap)/60:.1f} min", flush=True)
        # sv shape: (N, F, 2) for binary classification — take class 1
        if isinstance(sv, list):
            sv_pos = sv[1]
        elif sv.ndim == 3:
            sv_pos = sv[:, :, 1]
        else:
            sv_pos = sv

        top3_idx = np.argsort(-np.abs(sv_pos), axis=1)[:, :3]
        top_features = [
            "; ".join(f"{feat_cols[fi]}({sv_pos[i, fi]:+.3f})" for fi in top3_idx[i])
            for i in range(N_SHAP_TOP)
        ]
        shap_df = pd.DataFrame({
            "row": df_belt.loc[topk_idx, "row"].values,
            "col": df_belt.loc[topk_idx, "col"].values,
            "top_shap_features": top_features,
        })
        df_belt_out = df_belt_out.merge(shap_df, on=["row", "col"], how="left")
    else:
        df_belt_out["top_shap_features"] = ""

    df_belt_out.to_parquet(OUT_RATIONALE, index=False)
    print(f"  wrote {OUT_RATIONALE} ({len(df_belt_out):,} rows)", flush=True)
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
