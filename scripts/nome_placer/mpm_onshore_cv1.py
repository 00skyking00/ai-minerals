#!/usr/bin/env python
"""Onshore-placer MPM: first spatial-CV fold, honest comparison vs the live v3.1 map.

Assembles the onshore-placer covariate stack at the 104 bearcub drillholes and
runs leave-one-spatial-block-out CV for a RandomForest pay/barren classifier.
Reports held-out AUC for three things on the SAME held-out holes:

  baseline_v31   the live v3.1 prospectivity composite score, used directly
                 (no training) -- this is the map ADR-013 says stays live until
                 an MPM beats it.
  mpm_full       RF on the full stack (v3.1 sub-bands + composite + geochem IDW
                 + IFSAR terrain + per-hole bedrock depth).
  mpm_new_only   RF on ONLY the covariates v3.1 does not already use (geochem
                 + IFSAR terrain + bedrock depth) -- does the newly-pulled data
                 carry independent pay/barren signal at the hole scale?

Covariate sources (all EPSG:3338, 25 m, sampled by easting/northing_3338):
  - v3.1 prospectivity raster, 8 bands (BL/AP/TB/SS/BC/QM/buried-BL geomorphic
    sub-scores + composite). These re-expose the Tuck stand surfaces, beach
    lines and beach-stream confluence -- the "reuse from my tree" layers.
  - AGDB4 stream-sediment geochem IDW, 6 bands (log10 Au/As/Sb/Bi/Hg/W).
  - IFSAR 5 m terrain: DEM, slope, TPI.
  - Per-hole measured bedrock_depth_ft (a real datum at each hole).

Labels: bearcub 104-hole pay/barren (89 with pay_binary defined: 69 pay, 20
barren). ARDF placer occurrences are NOT folded into this supervised pay/barren
CV -- the 104 holes already sit on the ARDF placer creeks, so adding them as
positives is near-circular. A separate descriptive ARDF check is reported.

Spatial CV: holes are assigned to square blocks (default 750 m); leave-one-
block-out so the dense MS-1178/485 hole cluster cannot train-test leak. A
leave-one-claim_ms-out scheme is reported alongside for comparability with the
phase2_v0 baseline. Analysis only -- emits nothing to goldbug, cuts no raster.

Run:
    .venv/bin/python -m scripts.nome_placer.mpm_onshore_cv1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DRILLHOLES = Path("data/raw/bearcub_nome/drillholes_nome_placer.parquet")
V3P1 = Path("data/derived/nome_placer/prospectivity_v1p5/"
            "nome_placer_prospectivity_v1p5_v3p1_3338.tif")
GEOCHEM = Path("data/derived/nome_placer/covariates_mpm/geochem_ss_idw_log_3338.tif")
IFSAR = {
    "ifsar_dem": Path("data/raw/nome_mpm/ifsar_dem_3338.tif"),
    "ifsar_slope": Path("data/raw/nome_mpm/ifsar_slope_3338.tif"),
    "ifsar_tpi": Path("data/raw/nome_mpm/ifsar_tpi_3338.tif"),
}
ARDF = Path("data/raw/nome_mpm/ardf_nome.geojson")

OUT_DIR = Path("data/derived/nome_placer/mpm_onshore")
OUT_JSON = OUT_DIR / "mpm_onshore_cv1_metrics.json"
OUT_PRED = OUT_DIR / "mpm_onshore_cv1_per_hole.csv"

V3P1_BANDS = ["bl", "ap", "tb", "ss", "bc", "qm", "buried_bl", "composite"]
GEOCHEM_BANDS = ["log10_Au", "log10_As", "log10_Sb", "log10_Bi", "log10_Hg", "log10_W"]
BLOCK_M = 750.0
SEED = 42


def sample_raster(path: Path, pts: list[tuple[float, float]], cols: list[str]) -> pd.DataFrame:
    """Sample a multi-band raster at (easting, northing); nodata -> NaN."""
    with rasterio.open(path) as src:
        nodata = src.nodata
        arr = np.array(list(src.sample(pts)), dtype="float64")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return pd.DataFrame(arr, columns=cols)


def leave_one_group_out(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for g in pd.unique(groups):
        test = np.where(groups == g)[0]
        train = np.where(groups != g)[0]
        splits.append((train, test))
    return splits


def cv_auc(X: np.ndarray, y: np.ndarray, splits) -> tuple[float, np.ndarray]:
    """Leave-one-group-out RF; return aggregate held-out AUC + per-hole prob."""
    prob = np.full(len(y), np.nan, dtype="float64")
    for train_idx, test_idx in splits:
        ytr = y[train_idx]
        if len(np.unique(ytr)) < 2:
            continue  # degenerate fold: training set is single-class
        m = RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1,
        )
        m.fit(X[train_idx], ytr)
        prob[test_idx] = m.predict_proba(X[test_idx])[:, 1]
    fin = ~np.isnan(prob)
    auc = roc_auc_score(y[fin], prob[fin]) if len(np.unique(y[fin])) >= 2 else float("nan")
    return auc, prob


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dh = pd.read_parquet(DRILLHOLES)
    pts = list(zip(dh.easting_3338.astype(float), dh.northing_3338.astype(float)))

    feats = pd.concat(
        [
            sample_raster(V3P1, pts, [f"v31_{b}" for b in V3P1_BANDS]),
            sample_raster(GEOCHEM, pts, GEOCHEM_BANDS),
            *[sample_raster(p, pts, [name]) for name, p in IFSAR.items()],
        ],
        axis=1,
    )
    feats["bedrock_depth_ft"] = dh.bedrock_depth_ft.astype(float).values
    df = pd.concat([dh.reset_index(drop=True), feats], axis=1)

    # Label set: holes with pay_binary defined AND all raster covariates valid
    cov_cols = list(feats.columns)
    valid = df["pay_binary"].notna() & df[[c for c in cov_cols if c != "bedrock_depth_ft"]].notna().all(axis=1)
    work = df[valid].reset_index(drop=True).copy()
    y = work["pay_binary"].astype(int).to_numpy()
    print(f"holes: {len(df)} total; {int(df['pay_binary'].notna().sum())} labelled; "
          f"{len(work)} labelled AND fully-covered ({y.sum()} pay, {(y == 0).sum()} barren)")

    # Spatial blocks (declustering): square BLOCK_M grid on 3338 coords
    bx = np.floor(work.easting_3338.astype(float) / BLOCK_M).astype(int)
    by = np.floor(work.northing_3338.astype(float) / BLOCK_M).astype(int)
    block = bx.astype(str) + "_" + by.astype(str)
    work["spatial_block"] = block.values
    n_blocks = work["spatial_block"].nunique()
    claim = work["claim_ms"].fillna("none").astype(str).to_numpy()
    n_claims = len(pd.unique(claim))
    print(f"spatial blocks @ {BLOCK_M:.0f} m: {n_blocks};  claim_ms groups: {n_claims}")

    # Feature matrices
    new_only = GEOCHEM_BANDS + list(IFSAR) + ["bedrock_depth_ft"]
    full = [f"v31_{b}" for b in V3P1_BANDS] + new_only
    X_full = work[full].fillna(-999.0).to_numpy("float32")
    X_new = work[new_only].fillna(-999.0).to_numpy("float32")

    results = {}
    for scheme, groups in [("spatial_block", work["spatial_block"].to_numpy()),
                           ("claim_ms", claim)]:
        splits = leave_one_group_out(groups)
        # baseline: v3.1 composite score directly (no training)
        base = work["v31_composite"].to_numpy("float64")
        base_auc = roc_auc_score(y, base) if len(np.unique(y)) >= 2 else float("nan")
        full_auc, p_full = cv_auc(X_full, y, splits)
        new_auc, p_new = cv_auc(X_new, y, splits)
        results[scheme] = {
            "n_folds": len(splits),
            "baseline_v31_composite_auc": round(float(base_auc), 4),
            "mpm_full_auc": round(float(full_auc), 4),
            "mpm_new_only_auc": round(float(new_auc), 4),
        }
        print(f"\n[{scheme}] folds={len(splits)}  "
              f"baseline_v31={base_auc:.3f}  mpm_full={full_auc:.3f}  mpm_new_only={new_auc:.3f}")
        if scheme == "spatial_block":
            work["pred_mpm_full"] = p_full
            work["pred_mpm_new_only"] = p_new

    # Feature importance (full-data RF, full stack) -- descriptive only
    fm = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                random_state=SEED, n_jobs=-1).fit(X_full, y)
    imp = sorted(zip(full, fm.feature_importances_), key=lambda p: -p[1])
    print("\nfeature importance (full-data RF):")
    for f, v in imp[:12]:
        print(f"  {f:<20} {v:.4f}")

    # Descriptive ARDF check (NOT in the CV): placer occurrences vs other ARDF
    import geopandas as gpd
    ardf = gpd.read_file(ARDF).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str)).str.lower()
    is_placer = mc.str.startswith("39") | txt.str.contains("placer")
    apts = [(g.x, g.y) for g in ardf.geometry]
    comp = sample_raster(V3P1, apts, [f"v31_{b}" for b in V3P1_BANDS])["v31_composite"].to_numpy()
    au = sample_raster(GEOCHEM, apts, GEOCHEM_BANDS)["log10_Au"].to_numpy()
    ardf_check = {
        "n_placer": int(is_placer.sum()),
        "n_other": int((~is_placer).sum()),
        "v31_composite_median_placer": round(float(np.nanmedian(comp[is_placer.values])), 4),
        "v31_composite_median_other": round(float(np.nanmedian(comp[~is_placer.values])), 4),
        "geochem_log10Au_median_placer": round(float(np.nanmedian(au[is_placer.values])), 4),
        "geochem_log10Au_median_other": round(float(np.nanmedian(au[~is_placer.values])), 4),
    }
    print("\nARDF descriptive check (not in CV):", json.dumps(ardf_check))

    summary = {
        "n_holes_total": int(len(df)),
        "n_labelled": int(df["pay_binary"].notna().sum()),
        "n_labelled_covered": int(len(work)),
        "n_pay": int(y.sum()),
        "n_barren": int((y == 0).sum()),
        "block_size_m": BLOCK_M,
        "n_spatial_blocks": int(n_blocks),
        "n_claim_groups": int(n_claims),
        "feature_stack_full": full,
        "feature_stack_new_only": new_only,
        "cv": results,
        "feature_importance_full": [{"feature": f, "importance": round(float(v), 4)} for f, v in imp],
        "ardf_descriptive_check": ardf_check,
        "note": "Analysis only. v3.1 stays the live map (ADR-013). ARDF placer "
                "occurrences excluded from the supervised CV to avoid circularity "
                "with the on-creek drillholes.",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    cols = ["hole_id", "lon", "lat", "claim_ms", "spatial_block", "deposit_class",
            "pay_binary", "v31_composite", "pred_mpm_full", "pred_mpm_new_only"]
    work[[c for c in cols if c in work.columns]].to_csv(OUT_PRED, index=False)
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_PRED}")


if __name__ == "__main__":
    main()
