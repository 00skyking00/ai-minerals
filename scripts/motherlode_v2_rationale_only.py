"""Build the v2 rationale sidecar parquet WITHOUT SHAP.

The full postprocess crashed on SHAP (likely OOM on 20k cells × 75 features
× deep RF trees). This script does just the nearest-producer-name + lookalike
lookup, which is the higher-value piece anyway. SHAP rationale is left as a
follow-up; the parquet just has an empty top_shap_features column.

Output:
    data/derived/motherlode/v2_rationale_motherlode_250m.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from ai_minerals.grid import build_grid
from ai_minerals.model import add_lithology_onehot, non_feature_columns
from ai_minerals.model_rf import count_feature_columns
from ai_minerals.regions.motherlode import MOTHERLODE


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
IN_FEATURES = DATA_DERIVED / "features_motherlode_250m.parquet"
IN_V2_PREDS = ML_DIR / "v2_predictions_motherlode_250m.parquet"
IN_MASK = ML_DIR / "v2_within_belt_mask.parquet"
MRDS_GPKG = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds_motherlode.gpkg")
OUT_RATIONALE = ML_DIR / "v2_rationale_motherlode_250m.parquet"


def main() -> None:
    t0 = time.time()
    print("Loading inputs...", flush=True)
    feats = pd.read_parquet(IN_FEATURES)
    v2 = pd.read_parquet(IN_V2_PREDS)
    mask = pd.read_parquet(IN_MASK)
    mrds = gpd.read_file(MRDS_GPKG).to_crs(MOTHERLODE.working_crs)

    combined = feats.merge(mask, on=["row", "col"], how="left")
    combined = combined.merge(v2[["row", "col", "p_rf_v2"]], on=["row", "col"], how="left")
    df_belt = combined[combined["within_belt"] == 1].reset_index(drop=True)
    print(f"within-belt cells: {len(df_belt):,}", flush=True)

    # Producer subset
    producers = mrds[
        mrds["commod1"].astype(str).str.upper().str.contains("AU|GOLD", regex=True, na=False)
        & mrds["dev_stat"].astype(str).isin(["Producer", "Past Producer"])
    ].copy()
    print(f"producers (Au + producer/past producer): {len(producers):,}", flush=True)

    # Compute (row, col) for each producer point against the 250m grid
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
    px = producers.geometry.x.to_numpy()
    py = producers.geometry.y.to_numpy()
    producers["row"] = np.floor((py - grid_y0) / r_res).astype(int)
    producers["col"] = np.floor((px - grid_x0) / r_res).astype(int)

    cell_to_site = (
        producers[["row", "col", "site_name"]]
        .drop_duplicates(subset=["row", "col"], keep="first")
    )
    name_by_rc = {
        (int(r), int(c)): (str(n) if pd.notna(n) else "unnamed")
        for r, c, n in zip(cell_to_site["row"], cell_to_site["col"], cell_to_site["site_name"])
    }

    # Identify producer cells in df_belt
    producer_set = set(name_by_rc.keys())
    df_belt["is_producer_cell"] = df_belt.apply(
        lambda r: (int(r["row"]), int(r["col"])) in producer_set, axis=1,
    )

    # Same feature encoding as the v2 training script
    top_classes = df_belt["lithology_class"].value_counts().head(10).index.tolist()
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df_belt.columns:
            extra[col] = df_belt[col][df_belt[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df_belt, top_classes, extra_class_columns=extra or None)
    non_feat = non_feature_columns(label_cols=("is_orogenic_gold", "is_low_sulfidation", "is_producer", "within_belt"))
    feat_cols = [c for c in df_oh.columns
                 if c not in non_feat
                 and c not in ("p_rf_v2", "is_producer_cell")]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    print(f"feature columns: {len(feat_cols)}", flush=True)

    X = df_oh[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    X_std = StandardScaler().fit_transform(X)

    producer_mask = df_belt["is_producer_cell"].values
    producer_X = X_std[producer_mask]
    tree = cKDTree(producer_X)
    print(f"  feature-space NN over {producer_mask.sum():,} producers...", flush=True)
    t1 = time.time()
    feat_dist, feat_idx = tree.query(X_std, k=1)
    print(f"  done in {(time.time()-t1):.1f}s", flush=True)

    producer_xy = np.column_stack([df_belt.loc[producer_mask, "x"].values,
                                    df_belt.loc[producer_mask, "y"].values])
    cell_xy = np.column_stack([df_belt["x"].values, df_belt["y"].values])
    matched_xy = producer_xy[feat_idx]
    geo_dist = np.linalg.norm(cell_xy - matched_xy, axis=1)

    feat_sim = 1.0 / (1.0 + feat_dist)
    geo_norm = np.minimum(geo_dist / 20_000.0, 1.0)
    lookalike = feat_sim * geo_norm

    producer_names_belt = df_belt.loc[producer_mask].reset_index(drop=True)
    producer_rowcol = list(zip(
        producer_names_belt["row"].astype(int).tolist(),
        producer_names_belt["col"].astype(int).tolist(),
    ))
    nearest_names = [name_by_rc.get(rc, "unnamed") for rc in producer_rowcol]
    matched_names = [nearest_names[i] for i in feat_idx]

    out = df_belt[["row", "col", "x", "y"]].copy()
    out["p_rf_v2"] = df_belt["p_rf_v2"].astype(np.float32)
    out["lookalike_score"] = lookalike.astype(np.float32)
    out["nearest_producer_name"] = matched_names
    out["nearest_producer_geo_dist_m"] = geo_dist.astype(np.float32)
    out["nearest_producer_feat_dist"] = feat_dist.astype(np.float32)
    out["top_shap_features"] = ""  # SHAP deferred; column kept so loader schema is stable

    OUT_RATIONALE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_RATIONALE, index=False)
    print(f"wrote {OUT_RATIONALE} ({len(out):,} rows) in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
