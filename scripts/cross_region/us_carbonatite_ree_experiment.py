"""US Western Carbonatite REE — DEEP-SEAM transfer-within-niche experiment.

Tests whether DEEP-SEAM's published Curnamona advantage transfers to
a different REE province (US Western Carbonatite Belt — Bear Lodge,
Bearpaw, Elk Creek, Gem Park, Iron Hill, North Fork, Ravalli,
Wet Mountains; 8 carbonatite positives across MT-WY-CO-NE-ID).

Methodological note: this experiment reuses the Lawley 2022 H3 cube
as the feature source rather than building a new region-specific
feature frame. The Lawley features are MVT-tuned (proximity-to-
black-shale, proximity-to-passive-margin, sedimentary dictionaries)
rather than REE-specific (which would emphasize alkaline-intrusive
proximity, Th/U anomalies, F/CO2 alteration). The trade-off:
faster experiment (uses existing data) at the cost of feature
relevance. The experiment can still answer the core question — does
DevNet beat tree-based methods on a small-positive REE label set —
even if the absolute capture numbers don't match Curnamona's.

Comparison target: Curnamona (7 carbonatite-IOCG REE positives, heavy
PCA + GLCM preprocessing) showed DevNet at 86% top-2% capture vs
RF at near-zero. Same direction of advantage here would be a
within-niche transfer; opposite direction would tighten the
"Curnamona is dataset-specific" finding.

Outputs:
  data/derived/us_carbonatite_ree/metrics.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

REPO_ROOT = Path("/home/sky/src/learning/ai-minerals")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sri-ta3-baselines"))

from ai_minerals.model_devnet import fit_devnet, DevNetConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "derived" / "us_carbonatite_ree"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAWLEY_PARQUET = REPO_ROOT / "data" / "raw" / "lawley2022" / "datacube.parquet"
REE_SHAPEFILE = REPO_ROOT / "data" / "raw" / "usgs_ree" / "ree" / "ree.shp"

# Western US Carbonatite Belt AOI (37-49N, 116-96W).
AOI_MIN_LAT, AOI_MAX_LAT = 37.0, 49.0
AOI_MIN_LON, AOI_MAX_LON = -116.0, -96.0

# Match-radius for spatial join of REE point to H3 hexagon centroid.
# H3 hexes are ~5 km² each (effective ~1.3 km radius). At 1.5 km, each
# REE deposit point matches exactly the H3 hexagon containing it, with
# occasional fall-throughs into the adjacent neighbor. This keeps the
# positive count at the deposit count (small-positive regime matching
# Curnamona's 7-deposit setup) instead of dilating each deposit into
# 15+ cells.
MATCH_RADIUS_KM = 1.5

# Lawley feature columns relevant to REE prospectivity. Drop the
# MVT-specific ones (BlackShale, PassiveMargin, Sedimentary_Dictionary).
REE_RELEVANT_COLS = [
    "Latitude_EPSG4326", "Longitude_EPSG4326",
    "Geology_Lithology_Majority", "Geology_Lithology_Minority",
    "Geology_Period_Maximum_Majority", "Geology_Period_Minimum_Majority",
    "Geology_Dictionary_Alkalic", "Geology_Dictionary_Felsic",
    "Geology_Dictionary_Intermediate", "Geology_Dictionary_UltramaficMafic",
    "Geology_Dictionary_Pegmatitic",  # carbonatites can resemble pegmatites
    "Seismic_LAB_Priestley", "Seismic_Moho",
    "Gravity_GOCE_ShapeIndex",
    "Geology_Paleolatitude_Period_Maximum",
    "Terrane_Proximity", "Geology_Fault_Proximity",
    "Gravity_Bouguer", "Gravity_Bouguer_HGM", "Gravity_Bouguer_UpCont30km_HGM",
    "Gravity_Bouguer_HGM_Worms_Proximity",
    "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity",
    "Magnetic_HGM", "Magnetic_LongWavelength_HGM",
    "Magnetic_HGM_Worms_Proximity", "Magnetic_LongWavelength_HGM_Worms_Proximity",
]


def load_ree_positives() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(REE_SHAPEFILE)
    carb = gdf[(gdf["DEPTYPE"] == "carb")
               & (gdf.geometry.x >= AOI_MIN_LON)
               & (gdf.geometry.x <= AOI_MAX_LON)
               & (gdf.geometry.y >= AOI_MIN_LAT)
               & (gdf.geometry.y <= AOI_MAX_LAT)].copy()
    print(f"Carbonatite REE positives in AOI: {len(carb)}", flush=True)
    for _, row in carb.iterrows():
        print(f"  REC_ID {row['REC_ID']:>4} {row['DEPNAME']:40s} "
              f"({row['STPROV']}) at ({row.geometry.y:.3f}, {row.geometry.x:.3f})",
              flush=True)
    return carb


def load_lawley_aoi() -> pd.DataFrame:
    print(f"\nReading Lawley parquet AOI subset ...", flush=True)
    t0 = time.time()
    cols = REE_RELEVANT_COLS + ["Continent_Majority"]
    df = pd.read_parquet(LAWLEY_PARQUET, columns=cols)
    df = df[df["Continent_Majority"] == "North America"]
    df = df[(df["Latitude_EPSG4326"] >= AOI_MIN_LAT)
            & (df["Latitude_EPSG4326"] <= AOI_MAX_LAT)
            & (df["Longitude_EPSG4326"] >= AOI_MIN_LON)
            & (df["Longitude_EPSG4326"] <= AOI_MAX_LON)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"  AOI cells: {len(df):,}  ({time.time()-t0:.1f}s)", flush=True)
    return df


def tag_positives(df: pd.DataFrame, ree_pos: gpd.GeoDataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Mark each AOI cell as positive if any REE deposit centroid is
    within MATCH_RADIUS_KM of the cell's lat/lon. Returns the labeled
    frame plus a per-positive list of matched cell counts.

    Uses haversine-flat approximation: dlat * 111 km, dlon * 111 km *
    cos(lat). Accurate enough at this latitude band for a 5 km match
    radius.
    """
    deg_to_km = 111.0
    is_pos = np.zeros(len(df), dtype=np.int8)
    per_dep_counts = []
    df_lat = df["Latitude_EPSG4326"].to_numpy()
    df_lon = df["Longitude_EPSG4326"].to_numpy()
    for _, ree in ree_pos.iterrows():
        dlat = df_lat - ree.geometry.y
        dlon = (df_lon - ree.geometry.x) * np.cos(np.radians(ree.geometry.y))
        dist_km = np.hypot(dlat * deg_to_km, dlon * deg_to_km)
        match = dist_km < MATCH_RADIUS_KM
        n_match = int(match.sum())
        per_dep_counts.append((ree["DEPNAME"], n_match))
        is_pos[match] = 1
    df["target"] = is_pos
    print(f"\nSpatial join (radius {MATCH_RADIUS_KM} km):", flush=True)
    for name, count in per_dep_counts:
        print(f"  {name:40s}: {count} cells", flush=True)
    print(f"  total positive cells: {int(is_pos.sum()):,}", flush=True)
    return df, np.array(per_dep_counts, dtype=object)


def factorize_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Factorize the four string categorical features so the model can
    consume them as integer codes (matching Phase 1b)."""
    for col in ("Geology_Lithology_Majority", "Geology_Lithology_Minority",
                "Geology_Period_Maximum_Majority", "Geology_Period_Minimum_Majority"):
        if col in df.columns:
            codes, _ = pd.factorize(df[col])
            df[col] = codes.astype("uint8")
    return df


def evaluate_top_k(y_true: np.ndarray, scores: np.ndarray, ks_pct: list[int]) -> dict:
    n = len(y_true)
    out = {}
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    for k_pct in ks_pct:
        k = max(1, int(np.ceil(n * k_pct / 100)))
        captured = int(sorted_y[:k].sum())
        total_pos = max(int(y_true.sum()), 1)
        rate = captured / total_pos
        lift = rate / (k_pct / 100)
        out[f"top_{k_pct}_pct"] = {
            "rate": float(rate), "lift": float(lift),
            "captured": captured, "n_top": int(k),
        }
    return out


def train_rf(X_tr: np.ndarray, y_tr: np.ndarray, X_all: np.ndarray, seed: int = 42) -> np.ndarray:
    clf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_all)[:, 1]


def train_devnet(X_tr: np.ndarray, y_tr: np.ndarray, X_all: np.ndarray,
                 feat_cols: list[str], n_epochs: int = 500) -> np.ndarray:
    cfg = DevNetConfig(hidden=(24, 12), learning_rate=0.005, batch_size=128,
                       n_epochs=n_epochs, n_ref=5000, confidence_margin=5.0,
                       seed=42)
    tr_df = pd.DataFrame(X_tr, columns=feat_cols)
    tr_df["y"] = y_tr.astype(np.int8)
    _, _, model = fit_devnet(tr_df, feat_cols=feat_cols, label_col="y", config=cfg)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    Xn = ((X_all - mu) / sd).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(Xn)).squeeze(-1).numpy()


def leave_one_out_eval(df: pd.DataFrame, feat_cols: list[str],
                        model_name: str) -> dict:
    """For each positive cell, drop it from training and score it from
    the resulting model. Returns top-k capture across all dropped
    positives' scores against unlabeled cells.
    """
    X_all = df[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    y_all = df["target"].to_numpy(dtype=np.int8)
    pos_idx = np.where(y_all == 1)[0]
    pos_scores = np.zeros(len(pos_idx), dtype=np.float32)

    print(f"\n[{model_name}] LOO over {len(pos_idx)} positives ...", flush=True)
    for i, p in enumerate(pos_idx):
        keep = np.ones(len(df), dtype=bool)
        keep[p] = False
        if model_name == "RF":
            scores = train_rf(X_all[keep], y_all[keep], X_all)
        elif model_name == "DevNet":
            scores = train_devnet(X_all[keep], y_all[keep], X_all, feat_cols,
                                   n_epochs=300)
        else:
            raise ValueError(model_name)
        pos_scores[i] = float(scores[p])
        # rank percentile of this held-out positive among all unlabeled cells
        n_higher = int((scores > scores[p]).sum())
        pct = n_higher / len(scores) * 100
        print(f"  pos {i+1}/{len(pos_idx)} (idx={p}): score={scores[p]:.4f}, "
              f"rank pct={pct:.2f}", flush=True)

    # Build "OOF score" array: for the cells that are positive, use the LOO
    # held-out score; for unlabeled cells, use a single full-fit score.
    print(f"\n  full-data fit for unlabeled-cell scores ...", flush=True)
    if model_name == "RF":
        all_scores = train_rf(X_all, y_all, X_all)
    else:
        all_scores = train_devnet(X_all, y_all, X_all, feat_cols, n_epochs=500)
    all_scores[pos_idx] = pos_scores

    capture = evaluate_top_k(y_all, all_scores, ks_pct=[1, 2, 5, 10, 30])
    return {
        "n_positives": int(len(pos_idx)),
        "loo_pos_scores": pos_scores.tolist(),
        "capture_at_top_k": capture,
    }


def main() -> None:
    print("=== US Western Carbonatite REE — DEEP-SEAM transfer test ===", flush=True)
    print(f"AOI: {AOI_MIN_LAT}-{AOI_MAX_LAT}°N, {AOI_MIN_LON}-{AOI_MAX_LON}°W",
          flush=True)
    print(f"Features: Lawley 2022 H3 cube (REE-relevant subset of {len(REE_RELEVANT_COLS)} cols)",
          flush=True)

    ree_pos = load_ree_positives()
    df = load_lawley_aoi()
    df, dep_counts = tag_positives(df, ree_pos)
    df = factorize_cats(df)

    feat_cols = [c for c in df.columns
                  if c not in ("target", "Latitude_EPSG4326", "Longitude_EPSG4326",
                               "Continent_Majority")]
    print(f"\nFeatures used: {len(feat_cols)}", flush=True)

    results = {
        "stage": "US Western Carbonatite REE LOO comparison",
        "aoi": [AOI_MIN_LAT, AOI_MAX_LAT, AOI_MIN_LON, AOI_MAX_LON],
        "match_radius_km": MATCH_RADIUS_KM,
        "n_cells": int(len(df)),
        "n_positives_total": int(df["target"].sum()),
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "ree_deposits": [
            {"name": str(row["DEPNAME"]), "state": str(row["STPROV"]),
             "lat": float(row.geometry.y), "lon": float(row.geometry.x)}
            for _, row in ree_pos.iterrows()
        ],
    }

    rf_result = leave_one_out_eval(df, feat_cols, "RF")
    devnet_result = leave_one_out_eval(df, feat_cols, "DevNet")
    results["RF"] = rf_result
    results["DevNet"] = devnet_result

    print(f"\n=== SUMMARY ===", flush=True)
    for k in (1, 2, 5, 10, 30):
        rf_cap = rf_result["capture_at_top_k"][f"top_{k}_pct"]
        dn_cap = devnet_result["capture_at_top_k"][f"top_{k}_pct"]
        print(f"  top {k:>3}%: RF={rf_cap['rate']*100:>5.1f}% "
              f"(lift {rf_cap['lift']:.1f}x)  vs  "
              f"DevNet={dn_cap['rate']*100:>5.1f}% (lift {dn_cap['lift']:.1f}x)",
              flush=True)
    print(f"\nCurnamona reference target: DevNet 86% top-2%, RF near 0%.",
          flush=True)

    out_path = OUT_DIR / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
