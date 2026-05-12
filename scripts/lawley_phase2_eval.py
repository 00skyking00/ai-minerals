"""Phase 2: tightened evaluation on top of the Phase 1b leak-corrected baseline.

Phase 1 reproduced Lawley 2022's 0.983 AUC. Phase 1b removed a label
leak (`Training_MVT_Occurrence` aggregates into the target),
producing a corrected 0.972. The remaining gap between 0.972 and a
truly-held-out generalization number is what Phase 2 quantifies.

Three adds on top of Phase 1b:

1. **2-D spatial-block CV.** Lawley's 1-D latitude-band scheme keeps
   geographically adjacent (across longitude) cells in train + test
   in the same fold. Replace with 12×12 = 144 lat × lon quantile
   blocks → 6 folds, one held out.

2. **Bootstrap CI.** 1,000 resamples of test cells (with replacement);
   recompute AUC and top-k% capture rate per resample; report 95% CI.

3. **Held-out cross-continent transfer.** Train on US/Canada
   (n_pos = 167 deposits → ~1,500 after neighbor expansion), predict
   on Australia (n_pos = 46 deposits → ~500 expanded). Reverse
   direction too. Reports per-continent AUC + top-k%.

Output:
  data/derived/lawley/path2_tightened_eval_metrics.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/home/sky/src/learning/ai-minerals")
SRI_REPO = REPO_ROOT / "third_party" / "sri-ta3-baselines"
sys.path.insert(0, str(SRI_REPO))
import utilities as utils  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lawley_phase1_gbm import (  # noqa: E402
    PREFERRED_MVT_COLS, filter_continent, neighbor_deposits_fast,
)

DATACUBE_PARQUET = REPO_ROOT / "data" / "raw" / "lawley2022" / "datacube.parquet"
OUT_DIR = REPO_ROOT / "data" / "derived" / "lawley"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 2 extends the Phase 1 column list with longitude (needed for
# 2-D blocking).
PHASE2_COLS = PREFERRED_MVT_COLS + ["Longitude_EPSG4326"]

DROP_BEFORE_FIT = (
    "target", "Latitude_EPSG4326", "Longitude_EPSG4326", "group",
    "H3_Geometry", "Continent_Majority",
    "Training_MVT_Deposit", "Training_MVT_Occurrence",
    "lat_bin", "lon_bin", "cell_2d", "fold_2d", "fold_1d",
)

TOP_K_PCT = [1, 2, 5, 10, 30]
N_BOOT = 1000
RNG_SEED = 1234


# ----------------------------------------------------------------------
# Feature pipeline (mirrors Phase 1b, plus longitude)
# ----------------------------------------------------------------------

def build_feature_frame() -> pd.DataFrame:
    t0 = time.time()
    data = pd.read_parquet(DATACUBE_PARQUET, columns=PHASE2_COLS)
    print(f"loaded {data.shape[0]:,} rows × {data.shape[1]} cols  "
          f"({time.time()-t0:.1f}s, "
          f"{data.memory_usage(deep=True).sum()/1e6:.0f} MB in RAM)",
          flush=True)

    aus = filter_continent(data, "Oceania")
    uscan = filter_continent(data, "North America")
    del data; gc.collect()

    aus = neighbor_deposits_fast(aus, deptype="MVT")
    uscan = neighbor_deposits_fast(uscan, deptype="MVT")

    cols_dict = utils.load_features_dict(deptype="MVT", baseline="preferred")
    cols_aus, _ = utils.extract_cols(aus, cols_dict)
    cols_uscan, _ = utils.extract_cols(uscan, cols_dict)

    for src, dst in [(aus, cols_aus), (uscan, cols_uscan)]:
        dst["target"] = src["MVT_Deposit"].values
        dst["Continent_Majority"] = src["Continent_Majority"].values
        dst["Latitude_EPSG4326"] = src["Latitude_EPSG4326"].values
        dst["Longitude_EPSG4326"] = src["Longitude_EPSG4326"].values
        dst["Training_MVT_Deposit"] = src["Training_MVT_Deposit"].values
    del aus, uscan; gc.collect()

    df = pd.concat([cols_aus, cols_uscan], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    del cols_aus, cols_uscan; gc.collect()

    for col in ("Geology_Lithology_Majority", "Geology_Lithology_Minority",
                "Geology_Period_Maximum_Majority",
                "Geology_Period_Minimum_Majority"):
        if col in df.columns:
            codes, _ = pd.factorize(df[col])
            df[col] = codes.astype("uint8")
    if "H3_Geometry" in df.columns:
        df = df.drop(columns=["H3_Geometry"])

    print(f"feature frame: {df.shape}  positives: {int(df['target'].sum())}",
          flush=True)
    return df


# ----------------------------------------------------------------------
# Spatial-CV variants
# ----------------------------------------------------------------------

def assign_1d_lat_bands(df: pd.DataFrame, nbins: int = 72, k: int = 5) -> pd.DataFrame:
    """Same scheme as SRI's get_spatial_cross_val_idx, broken out so we
    can use it identically and add a 2-D analogue alongside."""
    target_df = df.loc[df["Training_MVT_Deposit"], "Latitude_EPSG4326"]
    _, bins = pd.qcut(target_df, nbins, retbins=True, duplicates="drop")
    bins[0] = -np.inf
    bins[-1] = np.inf
    df["_lat_bin"] = pd.cut(df["Latitude_EPSG4326"], bins, labels=False)
    bin_to_fold = (np.arange(len(bins) - 1) % (k + 1))
    df["fold_1d"] = df["_lat_bin"].map(lambda i: bin_to_fold[int(i)] if pd.notna(i) else -1)
    df = df.drop(columns=["_lat_bin"])
    return df


def assign_2d_blocks(df: pd.DataFrame, nbins: int = 12, k: int = 5) -> pd.DataFrame:
    """12 × 12 = 144 quantile-based lat × lon blocks, assigned into
    k+1 folds via cell_id % (k+1)."""
    pos_lat = df.loc[df["Training_MVT_Deposit"], "Latitude_EPSG4326"]
    pos_lon = df.loc[df["Training_MVT_Deposit"], "Longitude_EPSG4326"]
    _, lat_bins = pd.qcut(pos_lat, nbins, retbins=True, duplicates="drop")
    _, lon_bins = pd.qcut(pos_lon, nbins, retbins=True, duplicates="drop")
    lat_bins[0], lat_bins[-1] = -np.inf, np.inf
    lon_bins[0], lon_bins[-1] = -np.inf, np.inf
    df["lat_bin"] = pd.cut(df["Latitude_EPSG4326"], lat_bins, labels=False)
    df["lon_bin"] = pd.cut(df["Longitude_EPSG4326"], lon_bins, labels=False)
    n_lon = len(lon_bins) - 1
    df["cell_2d"] = df["lat_bin"] * n_lon + df["lon_bin"]
    df["fold_2d"] = df["cell_2d"] % (k + 1)
    return df


# ----------------------------------------------------------------------
# Eval helpers
# ----------------------------------------------------------------------

def top_k_capture(y_true: np.ndarray, scores: np.ndarray, k_pct: int) -> tuple[float, float, int]:
    """Return (capture rate, lift, n_captured)."""
    n = len(y_true)
    n_top = int(np.ceil(n * k_pct / 100))
    order = np.argsort(-scores)
    top_idx = order[:n_top]
    captured = int(y_true[top_idx].sum())
    total_pos = max(int(y_true.sum()), 1)
    rate = captured / total_pos
    lift = rate / (k_pct / 100)
    return rate, lift, captured


def bootstrap_metrics(
    y_true: np.ndarray, scores: np.ndarray, *,
    n_boot: int = N_BOOT, seed: int = RNG_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_samples = np.empty(n_boot, dtype=np.float32)
    capture_samples = {k: np.empty(n_boot, dtype=np.float32) for k in TOP_K_PCT}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = scores[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            auc_samples[b] = np.nan
        else:
            auc_samples[b] = roc_auc_score(y_b, s_b)
        for k in TOP_K_PCT:
            rate, _, _ = top_k_capture(y_b, s_b, k)
            capture_samples[k][b] = rate
    out = {
        "auc": {
            "mean": float(np.nanmean(auc_samples)),
            "ci95": [float(np.nanpercentile(auc_samples, 2.5)),
                     float(np.nanpercentile(auc_samples, 97.5))],
        },
        "capture": {},
    }
    for k in TOP_K_PCT:
        rate, lift, captured = top_k_capture(y_true, scores, k)
        out["capture"][f"top_{k}_pct"] = {
            "rate": float(rate),
            "lift": float(lift),
            "captured": int(captured),
            "rate_ci95": [float(np.nanpercentile(capture_samples[k], 2.5)),
                          float(np.nanpercentile(capture_samples[k], 97.5))],
        }
    return out


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def make_clf(cat_mask: np.ndarray) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=110, max_depth=7,
        min_samples_leaf=48, max_leaf_nodes=64, verbose=0,
        l2_regularization=0, class_weight={0: 1, 1: 400},
        validation_fraction=0.1, random_state=1234,
        categorical_features=cat_mask,
    )


def fit_and_predict(
    tr_df: pd.DataFrame, te_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    cat_mask = np.asarray(
        [tr_df[c].dtype.kind in ("u", "b", "i") for c in feature_cols]
    ).astype(bool)
    clf = make_clf(cat_mask)
    t1 = time.time()
    clf.fit(tr_df[feature_cols], tr_df["target"])
    print(f"    fit done in {time.time()-t1:.1f}s "
          f"(train {len(tr_df):,}, test {len(te_df):,})",
          flush=True)
    proba_test = clf.predict_proba(te_df[feature_cols])[:, 1]
    proba_train = clf.predict_proba(tr_df[feature_cols])[:, 1]
    return proba_train, proba_test


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    print(f"=== Lawley 2022 Phase 2: tightened evaluation ===", flush=True)
    df = build_feature_frame()
    df = assign_1d_lat_bands(df)
    df = assign_2d_blocks(df)
    print(f"  1-D folds (fold_1d) value counts: "
          f"{dict(df['fold_1d'].value_counts().sort_index())}", flush=True)
    print(f"  2-D folds (fold_2d) value counts: "
          f"{dict(df['fold_2d'].value_counts().sort_index())}", flush=True)

    feature_cols = [c for c in df.columns if c not in DROP_BEFORE_FIT]
    print(f"  feature_cols ({len(feature_cols)}): {feature_cols}", flush=True)

    results: dict = {
        "stage": "Phase 2: tightened evaluation on Phase 1b leak-corrected baseline",
        "n_total": int(len(df)),
        "n_positives_total": int(df["target"].sum()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_boot": N_BOOT,
        "top_k_pct": TOP_K_PCT,
    }

    # --- 1-D latitude-band CV (matches Phase 1b) ---
    print(f"\n[1/4] 1-D latitude-band CV (fold_1d == 0 held out)", flush=True)
    tr_mask = df["fold_1d"] != 0
    te_mask = df["fold_1d"] == 0
    _, proba_te_1d = fit_and_predict(df[tr_mask], df[te_mask], feature_cols)
    y_te_1d = df.loc[te_mask, "target"].to_numpy().astype(np.int8)
    print(f"    bootstrap CI on test (n={len(y_te_1d):,}, "
          f"pos={int(y_te_1d.sum())})...", flush=True)
    results["cv_1d_lat_band"] = bootstrap_metrics(y_te_1d, proba_te_1d)

    # --- 2-D quantile-block CV ---
    print(f"\n[2/4] 2-D quantile-block CV (fold_2d == 0 held out)", flush=True)
    tr_mask = df["fold_2d"] != 0
    te_mask = df["fold_2d"] == 0
    _, proba_te_2d = fit_and_predict(df[tr_mask], df[te_mask], feature_cols)
    y_te_2d = df.loc[te_mask, "target"].to_numpy().astype(np.int8)
    print(f"    bootstrap CI on test (n={len(y_te_2d):,}, "
          f"pos={int(y_te_2d.sum())})...", flush=True)
    results["cv_2d_blocks"] = bootstrap_metrics(y_te_2d, proba_te_2d)

    # --- Cross-continent: train USCAN → predict Aus ---
    print(f"\n[3/4] Cross-continent transfer: USCAN → Australia", flush=True)
    tr_mask = df["Continent_Majority"] == "North America"
    te_mask = df["Continent_Majority"] == "Oceania"
    _, proba_te_aus = fit_and_predict(df[tr_mask], df[te_mask], feature_cols)
    y_te_aus = df.loc[te_mask, "target"].to_numpy().astype(np.int8)
    print(f"    bootstrap CI on test (n={len(y_te_aus):,}, "
          f"pos={int(y_te_aus.sum())})...", flush=True)
    results["transfer_uscan_to_aus"] = bootstrap_metrics(y_te_aus, proba_te_aus)

    # --- Cross-continent: train Aus → predict USCAN ---
    print(f"\n[4/4] Cross-continent transfer: Australia → USCAN", flush=True)
    tr_mask = df["Continent_Majority"] == "Oceania"
    te_mask = df["Continent_Majority"] == "North America"
    _, proba_te_us = fit_and_predict(df[tr_mask], df[te_mask], feature_cols)
    y_te_us = df.loc[te_mask, "target"].to_numpy().astype(np.int8)
    print(f"    bootstrap CI on test (n={len(y_te_us):,}, "
          f"pos={int(y_te_us.sum())})...", flush=True)
    results["transfer_aus_to_uscan"] = bootstrap_metrics(y_te_us, proba_te_us)

    # --- Summary ---
    print(f"\n=== SUMMARY ===", flush=True)
    for name in ("cv_1d_lat_band", "cv_2d_blocks",
                 "transfer_uscan_to_aus", "transfer_aus_to_uscan"):
        r = results[name]
        auc = r["auc"]
        top1 = r["capture"]["top_1_pct"]
        top5 = r["capture"]["top_5_pct"]
        print(f"  {name:30s}  AUC = {auc['mean']:.4f} "
              f"[{auc['ci95'][0]:.3f}, {auc['ci95'][1]:.3f}]  "
              f"top1 = {top1['rate']*100:.1f}% (lift {top1['lift']:.1f}x)  "
              f"top5 = {top5['rate']*100:.1f}% (lift {top5['lift']:.1f}x)",
              flush=True)

    out_path = OUT_DIR / "path2_tightened_eval_metrics.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
