"""Phase 2 v1: onshore-placer MPM under honest spatial CV vs the v3.1 baseline.

Differences from the v0 baseline (phase2_v0_pay_binary_rf.py):

1. Covariates are MAPPABLE-ONLY (every-cell), so the model is a true MPM:
   7 v3.1 population surfaces (BL/AP/TB/SS/BC/QM/buried_BL) + 6 AGDB4
   stream-sediment geochem IDW surfaces (Au/As/Sb/Bi/Hg/W) + 3 IFSAR
   terrain surfaces (DEM/slope/TPI). The v0 model's top feature was
   ``bedrock_depth_ft`` -- a per-hole measurement that cannot be rasterised,
   so it is excluded here and reported only as an ablation.

2. Evaluation is spatial-cluster leave-block-out CV (KMeans on collar
   coordinates), which breaks the spatial autocorrelation that LOO/claim-MS
   CV leaks across the tight 4.5x2.5 km hole cluster. claim-MS group CV is
   also reported for continuity with v0.

3. The honest comparison is against the live v3.1 composite sampled at the
   same held-out holes.

Run: uv run python -m scripts.nome_placer.phase2_v1_spatial_cv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ai_minerals.features.v2_assemble import claim_ms_kfold_split
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION

DRILL = Path(NOME_PLACER_REGION.raw_paths["drillholes"])
DD = Path("data/derived/nome_placer")
V3P1 = DD / "prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif"
GEOCHEM = DD / "covariates_mpm/geochem_ss_idw_log_3338.tif"
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_3338.tif")
OUT_DIR = DD / "phase2_v1"

POP_NAMES = ["bl", "ap", "tb", "ss", "bc", "qm", "buried_bl"]      # v3p1 bands 1-7
GEO_NAMES = ["au", "as", "sb", "bi", "hg", "w"]                    # geochem bands 1-6
N_SPATIAL_CLUSTERS = 8
RNG = 42


def sample_bands(path: Path, ex: np.ndarray, ny: np.ndarray,
                 bands: list[int], names: list[str]) -> pd.DataFrame:
    with rasterio.open(path) as ds:
        samp = np.asarray(list(ds.sample(list(zip(ex, ny)))), dtype=float)
        nod = ds.nodata
    out = {}
    for bi, nm in zip(bands, names):
        col = samp[:, bi - 1]
        if nod is not None:
            col = np.where(col == nod, np.nan, col)
        out[nm] = col
    return pd.DataFrame(out)


def spatial_cv_auc(X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray) -> tuple[float, list[float]]:
    """Leave-one-group-out; aggregate AUC over pooled held-out predictions."""
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RNG, n_jobs=-1)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) >= 2:
            fold_aucs.append(roc_auc_score(y[te], oof[te]))
    ok = ~np.isnan(oof)
    agg = roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) >= 2 else float("nan")
    return agg, fold_aucs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = pd.read_parquet(DRILL)
    h = h[h["pay_binary"].notna() & h["easting_3338"].notna()
          & h["northing_3338"].notna()].reset_index(drop=True)
    ex = h["easting_3338"].to_numpy(float)
    ny = h["northing_3338"].to_numpy(float)
    y = h["pay_binary"].astype(int).to_numpy()
    print(f"{len(y)} labelled holes ({y.sum()} pay, {(y==0).sum()} barren)")

    pop = sample_bands(V3P1, ex, ny, list(range(1, 8)), POP_NAMES)
    comp = sample_bands(V3P1, ex, ny, [8], ["composite"])["composite"]
    geo = sample_bands(GEOCHEM, ex, ny, list(range(1, 7)), GEO_NAMES)
    dem = sample_bands(DEM, ex, ny, [1], ["dem"])["dem"]
    slope = sample_bands(SLOPE, ex, ny, [1], ["slope"])["slope"]
    tpi = sample_bands(TPI, ex, ny, [1], ["tpi"])["tpi"]

    feat = pd.concat([pop, geo,
                      dem.rename("dem"), slope.rename("slope"), tpi.rename("tpi")],
                     axis=1)
    mappable_cols = list(feat.columns)
    X_map = feat.fillna(-999.0).to_numpy(np.float32)
    X_abl = (feat.assign(bedrock_depth_ft=h["bedrock_depth_ft"].to_numpy())
             .fillna(-999.0).to_numpy(np.float32))

    # spatial-cluster groups + claim_ms groups
    sp = KMeans(n_clusters=N_SPATIAL_CLUSTERS, random_state=RNG, n_init=10
                ).fit_predict(np.column_stack([ex, ny]))
    ms = h["claim_ms"].fillna("__unk__").astype(str).to_numpy()
    print(f"spatial clusters: {len(np.unique(sp))} | claim_ms groups: {len(np.unique(ms))}")

    # baseline: live v3.1 composite at the same holes
    auc_v31 = roc_auc_score(y, comp.fillna(comp.median()).to_numpy())

    results = {"n_holes": int(len(y)), "n_pay": int(y.sum()),
               "n_barren": int((y == 0).sum()),
               "covariates_mappable": mappable_cols,
               "baseline_v31_composite_auc": round(float(auc_v31), 3)}

    for tag, X in [("mappable", X_map), ("mappable+bedrock_depth(ablation)", X_abl)]:
        sp_auc, sp_folds = spatial_cv_auc(X, y, sp)
        ms_auc, ms_folds = spatial_cv_auc(X, y, ms)
        results[tag] = {
            "spatial_cluster_cv_auc": round(float(sp_auc), 3),
            "spatial_cluster_fold_aucs_n": len(sp_folds),
            "claim_ms_cv_auc": round(float(ms_auc), 3),
        }
        print(f"\n[{tag}]")
        print(f"  spatial-cluster CV AUC = {sp_auc:.3f} ({len(sp_folds)} dual-class folds)")
        print(f"  claim_ms group CV AUC  = {ms_auc:.3f}")

    # feature importance, full mappable model
    fm = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RNG, n_jobs=-1).fit(X_map, y)
    imp = sorted(zip(mappable_cols, fm.feature_importances_), key=lambda p: -p[1])
    results["feature_importance_mappable"] = [[k, round(float(v), 4)] for k, v in imp]

    print(f"\nBaseline live v3.1 composite AUC @ holes = {auc_v31:.3f}")
    print("Top mappable features:", ", ".join(f"{k}={v:.3f}" for k, v in imp[:6]))
    (OUT_DIR / "phase2_v1_spatial_cv_report.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR/'phase2_v1_spatial_cv_report.json'}")


if __name__ == "__main__":
    main()
