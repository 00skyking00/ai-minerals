"""Phase 2 v1b: regional MPM evaluated as presence/background on ARDF placer
occurrences (the spatially-spread label set), under spatial-block CV.

The drill-hole pay/barren task (phase2_v1_spatial_cv.py) cannot test a regional
map: the 79 holes are packed into one small high-prospectivity cluster where pay
and barren interleave at sub-100 m scale. ARDF placer occurrences are spread
across the district, so presence-vs-background actually tests whether the map
ranks known gold above background.

Caveats reported alongside the numbers:
- v3.1 was partly tuned against ARDF, so v3.1-vs-ARDF is optimistic.
- Stream-sediment Au geochem is effectively a gold detector, so an MPM that
  includes it predicts placer occurrence near-tautologically. The honest
  regional test is the geomorphology+terrain model WITHOUT geochem.

Run: uv run python -m scripts.nome_placer.phase2_v1_ardf_presence_cv
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DD = Path("data/derived/nome_placer")
V3P1 = DD / "prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif"
GEOCHEM = DD / "covariates_mpm/geochem_ss_idw_log_3338.tif"
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_3338.tif")
ARDF = Path("data/raw/nome_mpm/ardf_nome.geojson")
OUT_DIR = DD / "phase2_v1"
N_BG = 2000
RNG = 42

POP = ["bl", "ap", "tb", "ss", "bc", "qm", "buried_bl"]
GEO = ["au", "as", "sb", "bi", "hg", "w"]
GEOMORPH = POP + ["dem", "slope", "tpi"]            # honest regional test (no geochem)
FULL = POP + GEO + ["dem", "slope", "tpi"]


def samp(path: Path, ex, ny, bands, names) -> pd.DataFrame:
    with rasterio.open(path) as ds:
        a = np.asarray(list(ds.sample(list(zip(ex, ny)))), float)
        nod = ds.nodata
    return pd.DataFrame(
        {nm: (np.where(a[:, b - 1] == nod, np.nan, a[:, b - 1]) if nod is not None
              else a[:, b - 1]) for b, nm in zip(bands, names)})


def features(ex, ny) -> pd.DataFrame:
    return pd.concat([
        samp(V3P1, ex, ny, list(range(1, 8)), POP),
        samp(GEOCHEM, ex, ny, list(range(1, 7)), GEO),
        samp(DEM, ex, ny, [1], ["dem"]),
        samp(SLOPE, ex, ny, [1], ["slope"]),
        samp(TPI, ex, ny, [1], ["tpi"]),
    ], axis=1)


def spatial_cv_auc(X, y, groups) -> float:
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        if len(np.unique(y[~te])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RNG, n_jobs=-1).fit(X[~te], y[~te])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return float(roc_auc_score(y[ok], oof[ok]))


def main() -> None:
    # ARDF placer positives inside the AOI grid
    with rasterio.open(V3P1) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
        H, W, T = ds.height, ds.width, ds.transform
    ardf = gpd.read_file(ARDF).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str)).str.lower()
    placer = ardf[mc.str.startswith("39") | txt.str.contains("placer")].copy()
    placer = placer.cx[bl:br, bb:bt]
    px = placer.geometry.x.to_numpy(); py = placer.geometry.y.to_numpy()
    print(f"ARDF placer positives in AOI: {len(px)}")

    # background: random valid (on-land) cells
    with rasterio.open(DEM) as ds:
        dem = ds.read(1); nod = ds.nodata
    valid = np.argwhere(dem != nod)
    rng = np.random.default_rng(RNG)
    pick = valid[rng.choice(len(valid), size=N_BG, replace=False)]
    bx, by = rasterio.transform.xy(T, pick[:, 0], pick[:, 1])
    bx = np.asarray(bx); by = np.asarray(by)

    ex = np.concatenate([px, bx]); ny = np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    feat = features(ex, ny).fillna(-999.0)
    comp = samp(V3P1, ex, ny, [8], ["composite"])["composite"].fillna(0.0).to_numpy()
    groups = KMeans(n_clusters=10, random_state=RNG, n_init=10
                    ).fit_predict(np.column_stack([ex, ny]))

    res = {"n_placer": int(len(px)), "n_background": int(len(bx)),
           "baseline_v31_composite_auc": round(roc_auc_score(y, comp), 3),
           "mpm_geomorph_terrain_spatialCV_auc":
               round(spatial_cv_auc(feat[GEOMORPH].to_numpy(np.float32), y, groups), 3),
           "mpm_full_with_geochem_spatialCV_auc":
               round(spatial_cv_auc(feat[FULL].to_numpy(np.float32), y, groups), 3),
           "note": ("v3.1 partly tuned to ARDF (optimistic); geochem Au is a gold "
                    "detector so full model is near-circular; geomorph+terrain is the "
                    "honest regional test")}
    print(json.dumps(res, indent=2))
    (OUT_DIR / "phase2_v1_ardf_presence_report.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT_DIR/'phase2_v1_ardf_presence_report.json'}")


if __name__ == "__main__":
    main()
