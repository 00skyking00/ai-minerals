"""F1: leak-guarded spatial-CV re-baseline of the Nome placer & lode models.

Re-grades three existing presence/background datasets under the leak-guarded
scheme in ``ai_minerals.spatial_cv`` (residual-variogram block sizing + a 1 km
dead-zone that exceeds our 400-800 m proximity buffers), and reports each AUC
next to its old KMeans-10 / 300 m number so the move is attributable to the CV
scheme, not to data drift:

  1. placer onshore  -- geomorph + terrain   (old: 0.733 / 0.741 buffered)
  2. lode  in-box    -- structural, old feats (old: 0.802 / 0.791 buffered)
  3. lode  district  -- structural, old feats (old: 0.620 / 0.582 buffered)

The lode in-box vs district pair is the acceptance check: the harness must still
show in-box >> district (restriction of range) on the existing structural
features. The district run additionally reports AUC on the placer-core in-box
subset as a single-fit alarm.

Run: uv run python -m scripts.nome_placer.f1_leak_guarded_rebaseline
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from ai_minerals.data.adapters.occurrences import kg as kg_occ
from ai_minerals.spatial_cv import default_rf_factory, leak_guarded_evaluate

DD = Path("data/derived/nome_placer")
CM = DD / "covariates_mpm"
V3P1 = DD / "prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif"
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_3338.tif")
KG_EXPORT = Path("data/raw/fossick_kg/kg_nome.jsonld")
ARDF = Path("data/raw/nome_mpm/ardf_nome.geojson")
OUT_DIR = DD / "f1_rebaseline"

RNG = 42
DEAD_ZONE_M = 1000.0          # > our 400-800 m proximity buffers (Airola 2018)
N_FOLDS = 10
POP = ["bl", "ap", "tb", "ss", "bc", "qm", "buried_bl"]
GEOMORPH = POP + ["dem", "slope", "tpi"]
GEO_CODES_INBOX = {1: "DOx", 2: "Dcs", 3: "Ocs", 4: "Qs", 5: "water"}


def samp(path: Path, ex, ny, bands, names) -> pd.DataFrame:
    with rasterio.open(path) as ds:
        a = np.asarray(list(ds.sample(list(zip(ex, ny)))), float)
        nod = ds.nodata
    return pd.DataFrame(
        {nm: (np.where(a[:, b - 1] == nod, np.nan, a[:, b - 1]) if nod is not None
              else a[:, b - 1]) for b, nm in zip(bands, names)})


def _background(dem_path: Path, transform, n_bg: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1); nod = ds.nodata
    valid = np.argwhere(dem != nod)
    pick = valid[np.random.default_rng(RNG).choice(len(valid), size=n_bg, replace=False)]
    bx, by = rasterio.transform.xy(transform, pick[:, 0], pick[:, 1])
    return np.asarray(bx), np.asarray(by)


def old_style_auc(X, y, coords, *, buffer_m: float) -> float:
    """The pre-F1 scheme: KMeans-10 spatial groups, optional positive buffer.
    Computed here on the identical rows so the delta is purely the CV change."""
    groups = KMeans(n_clusters=10, random_state=RNG, n_init=10).fit_predict(coords)
    oof = np.full(len(y), np.nan)
    factory = default_rf_factory(seed=RNG)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if buffer_m > 0:
            d, _ = cKDTree(coords[te]).query(coords, k=1)
            tr = tr & (d >= buffer_m)
        if len(np.unique(y[tr])) < 2:
            continue
        oof[te] = factory().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return round(float(roc_auc_score(y[ok], oof[ok])), 3)


def feature_importance(X, y, names) -> list[list]:
    fm = default_rf_factory(seed=RNG)().fit(X, y)
    return [[n, round(float(v), 4)] for n, v in
            sorted(zip(names, fm.feature_importances_), key=lambda p: -p[1])][:6]


# --------------------------------------------------------------------------- #
# dataset loaders (faithful to the existing presence/background scripts)
# --------------------------------------------------------------------------- #
def load_placer():
    with rasterio.open(V3P1) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
        T = ds.transform
    occ = kg_occ.load(KG_EXPORT).to_crs("EPSG:3338")
    comm = occ["commodity"].astype(str).str.lower()
    code39 = occ["deposit_codes"].apply(
        lambda t: any(str(c).split(":")[-1].startswith("39") for c in t))
    placer = occ[comm.str.contains("placer") | code39].cx[bl:br, bb:bt]
    px, py = placer.geometry.x.to_numpy(), placer.geometry.y.to_numpy()
    order = np.lexsort((py, px))  # canonical order (ADR-017 row-order guard)
    px, py = px[order], py[order]
    bx, by = _background(DEM, T, 2000)
    ex, ny = np.concatenate([px, bx]), np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    feat = pd.concat([
        samp(V3P1, ex, ny, list(range(1, 8)), POP),
        samp(DEM, ex, ny, [1], ["dem"]),
        samp(SLOPE, ex, ny, [1], ["slope"]),
        samp(TPI, ex, ny, [1], ["tpi"]),
    ], axis=1).fillna(-999.0)
    return feat[GEOMORPH], y, np.column_stack([ex, ny]), GEOMORPH, None


def _lode_positives(bounds) -> tuple[np.ndarray, np.ndarray]:
    bl, bb, br, bt = bounds
    ardf = gpd.read_file(ARDF).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str)).str.lower()
    lode = ardf[mc.str.startswith(("36", "27", "22"))
                | txt.str.contains("lode|vein|skarn|greisen")].cx[bl:br, bb:bt]
    return lode.geometry.x.to_numpy(), lode.geometry.y.to_numpy()


def _lode_features(ex, ny, geol_path, geo_codes, akmag, dfault, dem, slope, tpi):
    geol_code = samp(geol_path, ex, ny, [1], ["g"])["g"].fillna(0).astype(int)
    geol_oh = pd.DataFrame({f"geol_{v}": (geol_code == k).astype(float)
                            for k, v in geo_codes.items()})
    base = pd.concat([
        geol_oh,
        samp(akmag, ex, ny, [1], ["akmag"]),
        samp(dfault, ex, ny, [1], ["dist_fault"]),
        samp(dem, ex, ny, [1], ["dem"]),
        samp(slope, ex, ny, [1], ["slope"]),
        samp(tpi, ex, ny, [1], ["tpi"]),
    ], axis=1).fillna(-999.0)
    return base


def load_lode_inbox():
    with rasterio.open(V3P1) as ds:
        bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
        T = ds.transform
    px, py = _lode_positives(bounds)
    bx, by = _background(DEM, T, 2000)
    ex, ny = np.concatenate([px, bx]), np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    base = _lode_features(ex, ny, CM / "geology_unit_code_nome_3338.tif", GEO_CODES_INBOX,
                          CM / "akmag_nome_3338.tif", CM / "dist_to_fault_nome_3338.tif",
                          DEM, SLOPE, TPI)
    return base, y, np.column_stack([ex, ny]), list(base.columns), None


def load_lode_district():
    template = CM / "_district_grid_template_3338.tif"
    dem_d = Path("data/raw/nome_mpm/ifsar_dem_district_3338.tif")
    legend = json.loads((CM / "geology_unit_code_district_legend.json").read_text())
    geo_codes = {int(k): v for k, v in legend.items() if int(k) != 0}
    with rasterio.open(template) as ds:
        bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
        T = ds.transform
    px, py = _lode_positives(bounds)
    bx, by = _background(dem_d, T, 3000)
    ex, ny = np.concatenate([px, bx]), np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    base = _lode_features(ex, ny, CM / "geology_unit_code_district_3338.tif", geo_codes,
                          CM / "akmag_district_3338.tif", CM / "dist_to_fault_district_3338.tif",
                          dem_d, Path("data/raw/nome_mpm/ifsar_slope_district_3338.tif"),
                          Path("data/raw/nome_mpm/ifsar_tpi_district_3338.tif"))
    # in-box subset = test points inside the placer-core grid bounds
    with rasterio.open(V3P1) as ds:
        pbl, pbb, pbr, pbt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    in_box = (ex >= pbl) & (ex <= pbr) & (ny >= pbb) & (ny <= pbt)
    return base, y, np.column_stack([ex, ny]), list(base.columns), in_box


def run_one(name: str, scope: str, loader, *, old_ref: dict) -> dict:
    X_df, y, coords, names, in_box = loader()
    X = X_df.to_numpy(np.float32)
    print(f"[{name}] n={len(y)} pos={int(y.sum())} feats={len(names)}")
    # Headline = contiguous folds (the stricter, less optimistic test that
    # exposes restriction of range). balanced_scatter is reported alongside as
    # the coordinator's literal "even positives across folds" instruction -- and
    # its optimism vs the contiguous number is itself the finding.
    res = leak_guarded_evaluate(
        X, y, coords, in_box_mask=in_box, model_factory=default_rf_factory(seed=RNG),
        n_folds=N_FOLDS, dead_zone_m=DEAD_ZONE_M, fold_strategy="contiguous", seed=RNG,
    )
    res_scatter = leak_guarded_evaluate(
        X, y, coords, in_box_mask=in_box, model_factory=default_rf_factory(seed=RNG),
        n_folds=N_FOLDS, dead_zone_m=DEAD_ZONE_M, fold_strategy="balanced_scatter", seed=RNG,
    )
    row = asdict(res)
    row.update({
        "name": name, "scope": scope, "features": names,
        "auc_district_scatter": res_scatter.auc_district,
        "auc_in_box_scatter": res_scatter.auc_in_box,
        "old_scheme_auc_nobuffer": old_style_auc(X, y, coords, buffer_m=0.0),
        "old_scheme_auc_buffer300m": old_style_auc(X, y, coords, buffer_m=300.0),
        "old_report_reference": old_ref,
        "feature_importance_top6": feature_importance(X, y, names),
    })
    print(json.dumps({k: row[k] for k in
                      ("auc_district", "auc_in_box", "auc_district_scatter",
                       "variogram_range_m", "block_size_m", "n_pos_scored",
                       "old_scheme_auc_nobuffer", "old_scheme_auc_buffer300m")}, indent=2))
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["name", "scope", "n", "n_pos", "n_pos_in_box", "n_pos_scored",
            "variogram_range_m", "variogram_method", "block_size_m",
            "block_size_multiplier", "dead_zone_m", "n_folds", "n_blocks", "n_scored",
            "auc_district", "auc_in_box", "auc_district_scatter", "auc_in_box_scatter",
            "old_scheme_auc_nobuffer", "old_scheme_auc_buffer300m"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        run_one("placer_onshore", "placer-core grid (geomorph+terrain)", load_placer,
                old_ref={"spatialCV": 0.733, "buffered300m": 0.741}),
        run_one("lode_inbox", "placer-core grid (structural, OLD feats)", load_lode_inbox,
                old_ref={"spatialCV": 0.802, "buffered300m": 0.791}),
        run_one("lode_district", "district grid (structural, OLD feats)", load_lode_district,
                old_ref={"spatialCV": 0.620, "buffered300m": 0.582}),
    ]
    out_json = OUT_DIR / "f1_leak_guarded_rebaseline.json"
    out_json.write_text(json.dumps({
        "scheme": {"dead_zone_m": DEAD_ZONE_M, "n_folds": N_FOLDS,
                   "block_size_multiplier": 2.0, "block_sizing": "2x residual-variogram range",
                   "estimator": "RandomForest(300, balanced, seed=42)",
                   "headline_fold_strategy": "contiguous",
                   "note": ("auc_district / auc_in_box are the contiguous-fold (stricter) "
                            "numbers; auc_*_scatter are the positive-balanced scattered-fold "
                            "numbers reported for comparison")},
        "runs": rows,
    }, indent=2))
    write_csv(rows, OUT_DIR / "f1_leak_guarded_rebaseline.csv")
    print(f"\nwrote {out_json}")
    print(f"wrote {OUT_DIR / 'f1_leak_guarded_rebaseline.csv'}")
    print("report: docs/f1_leak_guarded_cv_rebaseline.md (narrative, written by hand)")


if __name__ == "__main__":
    main()
