"""Lode-gold MPM: presence/background spatial-CV on the DISTRICT-WIDE grid (ADR-013).

Mirrors mpm_lode_presence_cv.py exactly, re-scoped from the coastal placer-core AOI
(33 of ~104 district lode sites, no Big Hurrah) to the full Nome lode district. Every
covariate is the `_district` build at 100 m / EPSG:3338 (see mpm_build_district_grid.py
and mpm_fetch_district_terrain.py).

Labels: ARDF lode occurrences (Cox-Singer model 36/27/22 OR lode/vein/skarn/greisen
keyword) inside the district grid -- now including the Big Hurrah anchor (-164.25,
64.65) and the eastern/northern hinterland lodes that the placer-core AOI excluded.

Background: 3000 random on-land cells (DEM valid).

Models (presence/background, KMeans 10-cluster spatial-block CV, RandomForest):
  - structural        : geology one-hot + akmag + dist_fault + dem/slope/tpi
  - structural buf300 : same, with a 300 m positive buffer dropped from each fold's train set
  - full              : structural + AGDB4 stream-sed geochem (Au + As/Sb/Bi/Hg/W)
  - pathfinders_only  : As/Sb/Bi/Hg/W only

Geochem is stream-sed so it co-locates with mineralisation and is near-circular with
the lode labels; the structural model is the clean test of terrain/structure skill.

Run: uv run python -m scripts.nome_placer.mpm_lode_district_cv
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
TEMPLATE = CM / "_district_grid_template_3338.tif"
AKMAG = CM / "akmag_district_3338.tif"
GEOL = CM / "geology_unit_code_district_3338.tif"
GEOL_LEGEND = CM / "geology_unit_code_district_legend.json"
DFAULT = CM / "dist_to_fault_district_3338.tif"
GEOCHEM = CM / "geochem_ss_idw_log_district_3338.tif"
DEM = Path("data/raw/nome_mpm/ifsar_dem_district_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_district_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_district_3338.tif")
ARDF = Path("data/raw/nome_mpm/ardf_nome.geojson")
OUT_DIR = DD / "mpm_lode"

GEOCHEM_NAMES = ["au", "as", "sb", "bi", "hg", "w"]
N_BG = 3000
RNG = 42


def load_geo_codes() -> dict[int, str]:
    """Map int unit code -> label, from the district legend (0 = unmapped, dropped)."""
    legend = json.loads(GEOL_LEGEND.read_text())
    return {int(k): v for k, v in legend.items() if int(k) != 0}


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
    geo_codes = load_geo_codes()
    print(f"geology units ({len(geo_codes)}): {list(geo_codes.values())}")

    with rasterio.open(TEMPLATE) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
        T = ds.transform

    # ---- positives: ARDF lode in the district grid ----------------------
    ardf = gpd.read_file(ARDF).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str)).str.lower()
    lode = ardf[mc.str.startswith(("36", "27", "22"))
                | txt.str.contains("lode|vein|skarn|greisen")].cx[bl:br, bb:bt]
    px, py = lode.geometry.x.to_numpy(), lode.geometry.y.to_numpy()
    site_l = lode.get("site", pd.Series([""] * len(lode))).astype(str)
    big_hurrah_in = bool(site_l.str.contains("Hurrah", case=False).any())
    print(f"ARDF lode positives in district: {len(px)} (Big Hurrah included: {big_hurrah_in})")

    # ---- background: 3000 random on-land (DEM valid) cells ---------------
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

    # ---- assemble feature blocks ----------------------------------------
    geol_code = samp(GEOL, ex, ny, [1], ["g"])["g"].fillna(0).astype(int)
    geol_oh = pd.DataFrame({f"geol_{v}": (geol_code == k).astype(float)
                            for k, v in geo_codes.items()})
    structural = pd.concat([
        geol_oh,
        samp(AKMAG, ex, ny, [1], ["akmag"]),
        samp(DFAULT, ex, ny, [1], ["dist_fault"]),
        samp(DEM, ex, ny, [1], ["dem"]),
        samp(SLOPE, ex, ny, [1], ["slope"]),
        samp(TPI, ex, ny, [1], ["tpi"]),
    ], axis=1)
    geochem = samp(GEOCHEM, ex, ny, list(range(1, 7)), GEOCHEM_NAMES)
    full = pd.concat([structural, geochem], axis=1).fillna(-999.0)
    structural = structural.fillna(-999.0)

    groups = KMeans(n_clusters=10, random_state=RNG, n_init=10).fit_predict(coords)
    Xs = structural.to_numpy(np.float32)

    res = {
        "scope": "district-wide grid (Nome+Solomon quads; includes Big Hurrah + hinterland)",
        "grid": {"width": 1038, "height": 709, "res_m": 100, "crs": "EPSG:3338",
                 "wgs84_bbox": {"lon": [-166.0, -164.0], "lat": [64.3, 64.8]}},
        "n_lode": int(len(px)),
        "big_hurrah_included": big_hurrah_in,
        "n_background": int(len(bx)),
        "n_geology_units": len(geo_codes),
        "structural_geol_akmag_fault_terrain_spatialCV_auc":
            round(spatial_cv_auc(Xs, y, groups), 3),
        "structural_buffered300m_auc":
            round(spatial_cv_auc(Xs, y, groups, coords, 300.0), 3),
        "full_with_geochem_spatialCV_auc":
            round(spatial_cv_auc(full.to_numpy(np.float32), y, groups), 3),
        "pathfinders_only_AsSbBiHgW_spatialCV_auc":
            round(spatial_cv_auc(geochem[["as", "sb", "bi", "hg", "w"]].fillna(-999.0)
                                 .to_numpy(np.float32), y, groups), 3),
        "note": ("stream-sed geochem is near-circular with lode labels (structural model is "
                 "the clean structure/terrain test); v0 placer-core was 0.802 / 0.791 buffered "
                 "on 33 sites without Big Hurrah"),
    }
    fm = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RNG, n_jobs=-1).fit(Xs, y)
    res["feature_importance_structural"] = [[k, round(float(v), 4)] for k, v in
                                            sorted(zip(structural.columns, fm.feature_importances_),
                                                   key=lambda p: -p[1])]
    print(json.dumps(res, indent=2))
    out = OUT_DIR / "mpm_lode_district_report.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
