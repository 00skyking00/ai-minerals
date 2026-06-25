"""Lode-gold MPM v0: presence/background CV on ARDF lode occurrences (ADR-013).

Covariates (all every-cell, EPSG:3338 / 25 m, on the placer AOI grid):
  - bedrock geology (SIM 3131 / OFR 2009-1254), one-hot map unit
  - akmag aeromagnetic anomaly (1 km composite)
  - distance-to-fault (SIM 3131 fault arcs)
  - IFSAR terrain (DEM/slope/TPI)
  - AGDB4 stream-sed geochem (Au + pathfinders As/Sb/Bi/Hg/W) -- added only in the
    "full" model, because stream-sed pathfinders co-locate with mineralisation and
    are near-circular with the lode labels (same caveat as Au for the placer model).

Labels: ARDF lode occurrences (Cox-Singer model 36/27/22 or lode/vein/skarn/greisen)
inside the AOI grid. NOTE the scope limit: this AOI is the coastal placer core, so it
holds 33 of the district's ~104 lode sites and EXCLUDES the Big Hurrah anchor (-164.25,
~30 km east) and the northern/eastern hinterland lodes. A district-wide grid is the
proper lode scope (next build); this is a fast v0 on covariates already in hand.

The v3.1 placer composite is reported as a reference baseline (does the placer map
predict lode? expected near chance -- different deposit style).

Run: uv run python -m scripts.nome_placer.mpm_lode_presence_cv
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DD = Path("data/derived/nome_placer")
CM = DD / "covariates_mpm"
V3P1 = DD / "prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif"
AKMAG = CM / "akmag_nome_3338.tif"
GEOL = CM / "geology_unit_code_nome_3338.tif"
DFAULT = CM / "dist_to_fault_nome_3338.tif"
GEOCHEM = CM / "geochem_ss_idw_log_3338.tif"
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_3338.tif")
ARDF = Path("data/raw/nome_mpm/ardf_nome.geojson")
OUT_DIR = DD / "mpm_lode"
GEO_CODES = {1: "DOx", 2: "Dcs", 3: "Ocs", 4: "Qs", 5: "water"}
GEO_OH = [f"geol_{v}" for v in GEO_CODES.values()]
GEOCHEM_NAMES = ["au", "as", "sb", "bi", "hg", "w"]
N_BG = 2000
RNG = 42


def samp(path: Path, ex, ny, bands, names) -> pd.DataFrame:
    with rasterio.open(path) as ds:
        a = np.asarray(list(ds.sample(list(zip(ex, ny)))), float)
        nod = ds.nodata
    return pd.DataFrame(
        {nm: (np.where(a[:, b - 1] == nod, np.nan, a[:, b - 1]) if nod is not None
              else a[:, b - 1]) for b, nm in zip(bands, names)})


def spatial_cv_auc(X, y, groups, coords=None, buffer_m=0.0) -> float:
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if buffer_m > 0 and coords is not None:
            d, _ = cKDTree(coords[te]).query(coords, k=1)
            tr = tr & (d >= buffer_m)
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RNG, n_jobs=-1).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return float(roc_auc_score(y[ok], oof[ok]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(V3P1) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
        T = ds.transform

    ardf = gpd.read_file(ARDF).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str)).str.lower()
    lode = ardf[mc.str.startswith(("36", "27", "22"))
                | txt.str.contains("lode|vein|skarn|greisen")].cx[bl:br, bb:bt]
    px, py = lode.geometry.x.to_numpy(), lode.geometry.y.to_numpy()
    print(f"ARDF lode positives in AOI: {len(px)}")

    with rasterio.open(DEM) as ds:
        dem = ds.read(1); nod = ds.nodata
    valid = np.argwhere(dem != nod)
    rng = np.random.default_rng(RNG)
    pick = valid[rng.choice(len(valid), size=N_BG, replace=False)]
    bx, by = rasterio.transform.xy(T, pick[:, 0], pick[:, 1])
    bx, by = np.asarray(bx), np.asarray(by)

    ex = np.concatenate([px, bx]); ny = np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    coords = np.column_stack([ex, ny])

    geol_code = samp(GEOL, ex, ny, [1], ["g"])["g"].fillna(0).astype(int)
    geol_oh = pd.DataFrame({f"geol_{v}": (geol_code == k).astype(float)
                            for k, v in GEO_CODES.items()})
    base = pd.concat([
        geol_oh,
        samp(AKMAG, ex, ny, [1], ["akmag"]),
        samp(DFAULT, ex, ny, [1], ["dist_fault"]),
        samp(DEM, ex, ny, [1], ["dem"]),
        samp(SLOPE, ex, ny, [1], ["slope"]),
        samp(TPI, ex, ny, [1], ["tpi"]),
    ], axis=1)
    geochem = samp(GEOCHEM, ex, ny, list(range(1, 7)), GEOCHEM_NAMES)
    full = pd.concat([base, geochem], axis=1).fillna(-999.0)
    base = base.fillna(-999.0)
    comp = samp(V3P1, ex, ny, [8], ["c"])["c"].fillna(0.0).to_numpy()

    groups = KMeans(n_clusters=10, random_state=RNG, n_init=10).fit_predict(coords)
    Xi = base.to_numpy(np.float32)
    res = {
        "aoi": "placer core grid (excludes Big Hurrah + N/E hinterland lodes)",
        "n_lode": int(len(px)), "n_background": int(len(bx)),
        "ref_v31_placer_composite_auc": round(roc_auc_score(y, comp), 3),
        "lode_indep_geol_akmag_struct_terrain_spatialCV_auc":
            round(spatial_cv_auc(Xi, y, groups), 3),
        "lode_indep_buffered300m_auc":
            round(spatial_cv_auc(Xi, y, groups, coords, 300.0), 3),
        "lode_full_with_geochem_spatialCV_auc":
            round(spatial_cv_auc(full.to_numpy(np.float32), y, groups), 3),
        "pathfinders_only_AsSbBiHgW_spatialCV_auc":
            round(spatial_cv_auc(geochem[["as", "sb", "bi", "hg", "w"]].fillna(-999.0)
                                 .to_numpy(np.float32), y, groups), 3),
        "note": ("33 of ~104 district lode sites; geochem pathfinders are stream-sed so "
                 "near-circular with lode labels (indep model is the honest structural test); "
                 "v3.1 is a placer map, low lode skill expected; widen to district grid next"),
    }
    fm = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RNG, n_jobs=-1).fit(Xi, y)
    res["feature_importance_indep"] = [[k, round(float(v), 4)] for k, v in
                                       sorted(zip(base.columns, fm.feature_importances_),
                                              key=lambda p: -p[1])]
    print(json.dumps(res, indent=2))
    (OUT_DIR / "mpm_lode_presence_report.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT_DIR/'mpm_lode_presence_report.json'}")


if __name__ == "__main__":
    main()
