"""Round 5 steps 5-6: H1 occurrence CV and H2 coarseness ordinal test.

H1 asks whether distance-to-lode (down-channel, confined valleys) and
distance-to-schist-limestone-contact predict strictly-alluvial placer
occurrence better than a spatial null, under the leak-guarded spatial CV
(KMeans blocks + buffer, reused from mpm_onshore_presence_cv.py), with a
bootstrap CI on every AUC. Two designs:

  H1a full-AOI:        51 alluvial positives vs 2000 random on-land background.
  H1b confined-matched: alluvial-in-confined vs random background ALSO drawn
                        from confined valleys, which removes the valley-
                        membership confound and isolates the source gradient.

Single distance features are scored by their rank AUC (a monotone feature
needs no training, so spatial CV cannot leak through it); multi-feature models
use the RF spatial-block CV. The spatial null is the AUC of distance to N
RANDOM points (N = lode count), repeated, so a real feature must beat what any
clustered point set would score under spatial autocorrelation. The negative
control reruns the features on the marine-beach set.

H2 tests whether gold-coarseness bins (fine=1 / coarse=2 / rough-nuggety=3,
mined from ARDF narratives) differ in median distance (Kruskal-Wallis) and
whether coarseness falls with distance (Spearman). Down-channel n is reported
explicitly because the mapped-lode network is sparse.

Run: uv run python -m scripts.nome_placer.inland_local_source.run_h1_h2
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DD = Path("data/derived/nome_placer")
ILS = DD / "inland_local_source"
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_3338.tif")
CONTACT = DD / "bedrock_contact/dist_to_contact.tif"
DOWNCH = ILS / "down_channel_dist_to_lode.tif"
STRAIGHT = ILS / "straight_line_dist_to_lode.tif"
LODE = DD / "ardf_nome_lode_au_sources.geojson"
RNG = 42
N_BG = 2000
N_BG_CONFINED = 1000
B_BOOT = 2000
K_NULL = 300


def sample_raster(path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as ds:
        vals = np.array([v[0] for v in ds.sample(zip(xs, ys))], dtype=float)
        nod = ds.nodata
    if nod is not None:
        vals = np.where(vals == nod, np.nan, vals)
    return vals


def boot_auc_ci(y: np.ndarray, score: np.ndarray, b: int = B_BOOT) -> tuple[float, float, float]:
    """Stratified-bootstrap percentile CI for AUC of `score` vs `y`."""
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    point = roc_auc_score(y, score)
    rng = np.random.default_rng(RNG)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(b):
        bi = np.concatenate([rng.choice(ip, len(ip), replace=True),
                             rng.choice(ineg, len(ineg), replace=True)])
        aucs.append(roc_auc_score(y[bi], score[bi]))
    return round(point, 3), round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)


def rank_auc(y: np.ndarray, dist: np.ndarray) -> tuple[float, float, float]:
    """AUC treating SHORTER distance as higher placer score (closer to lode)."""
    return boot_auc_ci(y, -dist)


def spatial_cv_oof(X: np.ndarray, y: np.ndarray, coords: np.ndarray,
                   groups: np.ndarray, buffer_m: float = 300.0) -> np.ndarray:
    """Leave-one-block-out RF; buffer drops train points near held-out points.
    Returns pooled out-of-fold P(placer) for every row."""
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if buffer_m > 0:
            d, _ = cKDTree(coords[te]).query(coords, k=1)
            tr = tr & (d >= buffer_m)
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RNG, n_jobs=1).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def model_auc(feat: pd.DataFrame, cols: list[str], y: np.ndarray, coords: np.ndarray,
              groups: np.ndarray) -> dict:
    X = feat[cols].fillna(-999.0).to_numpy(np.float32)
    oof = spatial_cv_oof(X, y, coords, groups)
    pt, lo, hi = boot_auc_ci(y, oof)
    return {"features": cols, "auc": pt, "ci95": [lo, hi]}


def land_mask_cells() -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    with rasterio.open(DEM) as ds:
        dem = ds.read(1); nod = ds.nodata; T = ds.transform
    rows, cols = np.where(dem != nod)
    return rows, cols, T


def snap_sample(path: Path, xs: np.ndarray, ys: np.ndarray, tol_m: float) -> np.ndarray:
    """Value of the nearest FINITE raster cell within tol_m; NaN beyond.

    For the sparse down-channel feature: placers sit beside, not exactly on,
    the confined stream cell, so an exact sample misses them. Snapping to the
    nearest cell that actually carries a value gives the feature its fair
    coverage without inventing data (still NaN if none within tol_m).
    """
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(float); nod = ds.nodata; T = ds.transform
        cellsize = abs(T.a)
    a = np.where(arr == nod, np.nan, arr)
    fy, fx = np.where(np.isfinite(a))
    if len(fy) == 0:
        return np.full(len(xs), np.nan)
    tree = cKDTree(np.column_stack([fx, fy]).astype(float))
    rr, cc = rasterio.transform.rowcol(T, xs, ys)
    q = np.column_stack([np.asarray(cc, float), np.asarray(rr, float)])
    d, idx = tree.query(q, k=1)
    vals = a[fy[idx], fx[idx]]
    vals[d > tol_m / cellsize] = np.nan
    return vals


def main() -> None:
    rng = np.random.default_rng(RNG)
    typed = gpd.read_file(ILS / "placers_typed.geojson")
    confined = rasterio.open(ILS / "confined_valley.tif").read(1) == 1
    zone = rasterio.open(ILS / "zone.tif").read(1)

    def feats(xs, ys):
        return pd.DataFrame({
            "contact": sample_raster(CONTACT, xs, ys),
            "straight": sample_raster(STRAIGHT, xs, ys),
            "downch": sample_raster(DOWNCH, xs, ys),
            "dem": sample_raster(DEM, xs, ys),
            "slope": sample_raster(SLOPE, xs, ys),
            "tpi": sample_raster(TPI, xs, ys),
        })

    # ---- background: random on-land cells (full AOI) ----
    rows, cols, T = land_mask_cells()
    pick = rng.choice(len(rows), size=N_BG, replace=False)
    bx, by = rasterio.transform.xy(T, rows[pick], cols[pick])
    bx, by = np.asarray(bx), np.asarray(by)

    results: dict = {"params": {"n_bg": N_BG, "n_bg_upland": N_BG_CONFINED,
                               "boot": B_BOOT, "k_null": K_NULL, "buffer_m": 300}}

    # =====================================================================
    # H1a: full-AOI presence/background
    # =====================================================================
    al = typed[typed.geol_type == "alluvial-stream"]
    px, py = al.geometry.x.to_numpy(), al.geometry.y.to_numpy()
    ex = np.concatenate([px, bx]); ey = np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    feat = feats(ex, ey)
    coords = np.column_stack([ex, ey])
    groups = KMeans(n_clusters=10, random_state=RNG, n_init=10).fit_predict(coords)

    h1a = {"n_pos": int(len(px)), "n_bg": int(len(bx))}
    # single dense features (rank AUC + bootstrap CI)
    h1a["single"] = {}
    for f in ("contact", "straight"):
        d = feat[f].fillna(feat[f].max()).to_numpy()
        pt, lo, hi = rank_auc(y, d)
        h1a["single"][f] = {"auc": pt, "ci95": [lo, hi]}
    # multi-feature RF spatial CV
    h1a["models"] = {
        "contact_only": model_auc(feat, ["contact"], y, coords, groups),
        "contact+straight": model_auc(feat, ["contact", "straight"], y, coords, groups),
        "contact+downch": model_auc(feat, ["contact", "downch"], y, coords, groups),
        "contact+straight+downch": model_auc(feat, ["contact", "straight", "downch"], y, coords, groups),
        "all_dense+terrain": model_auc(feat, ["contact", "straight", "downch", "dem", "slope", "tpi"], y, coords, groups),
    }
    # spatial null: distance to N random points (N = lode seed count)
    nlode = json.loads((ILS / "distance_meta.json").read_text())["n_lode_in_grid"]
    null_aucs = []
    for k in range(K_NULL):
        rk = np.random.default_rng(1000 + k)
        idx = rk.choice(len(rows), size=nlode, replace=False)
        rxx, ryy = rasterio.transform.xy(T, rows[idx], cols[idx])
        tree = cKDTree(np.column_stack([rxx, ryy]))
        dd, _ = tree.query(coords, k=1)
        null_aucs.append(roc_auc_score(y, -dd))
    h1a["spatial_null_straightline_to_random"] = {
        "mean_auc": round(float(np.mean(null_aucs)), 3),
        "p95_auc": round(float(np.percentile(null_aucs, 95)), 3),
        "note": "distance to N random points; real lode features must beat this",
    }
    results["H1a_full_aoi"] = h1a

    # =====================================================================
    # H1b: upland-matched (background drawn from the SAME upland terrain as
    # the inland positives, removing the broad upland-vs-coastal confound).
    # =====================================================================
    def in_zone3(xs, ys):
        rr, ccc = rasterio.transform.rowcol(T, xs, ys)
        rr = np.clip(rr, 0, zone.shape[0]-1); ccc = np.clip(ccc, 0, zone.shape[1]-1)
        return zone[rr, ccc] == 3
    al_c = al[in_zone3(px, py)]
    pcx, pcy = al_c.geometry.x.to_numpy(), al_c.geometry.y.to_numpy()
    ur, uc = np.where(zone == 3)
    pu = rng.choice(len(ur), size=min(N_BG_CONFINED, len(ur)), replace=False)
    bux, buy = rasterio.transform.xy(T, ur[pu], uc[pu]); bux, buy = np.asarray(bux), np.asarray(buy)
    ecx = np.concatenate([pcx, bux]); ecy = np.concatenate([pcy, buy])
    yc = np.concatenate([np.ones(len(pcx)), np.zeros(len(bux))]).astype(int)
    featc = feats(ecx, ecy)
    coordsc = np.column_stack([ecx, ecy])
    nblk = max(4, min(8, len(pcx) // 3))
    groupsc = KMeans(n_clusters=nblk, random_state=RNG, n_init=10).fit_predict(coordsc)

    h1b = {"n_pos": int(len(pcx)), "n_bg": int(len(bux)), "n_blocks": int(nblk),
           "design": "alluvial-in-upland vs random-upland background"}
    h1b["single"] = {}
    for f in ("contact", "straight"):
        d = featc[f].fillna(featc[f].max()).to_numpy()
        pt, lo, hi = rank_auc(yc, d)
        h1b["single"][f] = {"auc": pt, "ci95": [lo, hi]}
    # down-channel: snap-sampled (nearest defined cell within 150 m), report coverage
    dch_pos = snap_sample(DOWNCH, pcx, pcy, 150.0)
    dch_bg = snap_sample(DOWNCH, bux, buy, 150.0)
    dvalid = np.isfinite(np.concatenate([dch_pos, dch_bg]))
    h1b["downch_coverage"] = {"n_pos_with_downch": int(np.isfinite(dch_pos).sum()),
                              "n_bg_with_downch": int(np.isfinite(dch_bg).sum())}
    if np.isfinite(dch_pos).sum() >= 5 and np.isfinite(dch_bg).sum() >= 5:
        dd = np.concatenate([dch_pos, dch_bg])[dvalid]
        pt, lo, hi = rank_auc(yc[dvalid], dd)
        h1b["single"]["downch_where_defined"] = {"auc": pt, "ci95": [lo, hi], "n": int(dvalid.sum())}
    else:
        h1b["single"]["downch_where_defined"] = {"auc": None,
            "note": "too few points with a defined down-channel distance to test "
                    f"(pos={int(np.isfinite(dch_pos).sum())}, bg={int(np.isfinite(dch_bg).sum())})"}
    h1b["models"] = {
        "contact_only": model_auc(featc, ["contact"], yc, coordsc, groupsc),
        "contact+straight": model_auc(featc, ["contact", "straight"], yc, coordsc, groupsc),
    }
    results["H1b_upland_matched"] = h1b

    # =====================================================================
    # Negative control: marine-beach set vs background (full AOI)
    # =====================================================================
    mb = typed[typed.geol_type == "marine-beach"]
    if len(mb) >= 3:
        mx, my = mb.geometry.x.to_numpy(), mb.geometry.y.to_numpy()
        emx = np.concatenate([mx, bx]); emy = np.concatenate([my, by])
        ym = np.concatenate([np.ones(len(mx)), np.zeros(len(bx))]).astype(int)
        fm = feats(emx, emy)
        nc = {"n_pos": int(len(mx)), "single": {}}
        for f in ("contact", "straight"):
            d = fm[f].fillna(fm[f].max()).to_numpy()
            pt, lo, hi = rank_auc(ym, d)
            nc["single"][f] = {"auc": pt, "ci95": [lo, hi]}
        results["negative_control_marine"] = nc
    results["negative_control_glacial"] = {
        "n_pos": int((typed.geol_type == "glacial").sum()),
        "note": "no occurrence in the AOI types as genetically glacial-drift placer",
    }

    # =====================================================================
    # H2: coarseness ordinal vs distance
    # =====================================================================
    alc = al.copy()
    axs, ays = alc.geometry.x.to_numpy(), alc.geometry.y.to_numpy()
    alc["downch"] = snap_sample(DOWNCH, axs, ays, 150.0)   # sparse -> snapped
    alc["straight"] = sample_raster(STRAIGHT, axs, ays)
    alc["contact"] = sample_raster(CONTACT, axs, ays)
    h2 = {"n_alluvial_with_coarseness": int(alc["coarseness_rank"].notna().sum()),
          "bin_counts": {int(k): int(v) for k, v in alc["coarseness_rank"].value_counts().items()}}
    for feat_name in ("downch", "straight", "contact"):
        d = alc.dropna(subset=["coarseness_rank", feat_name])
        n = len(d)
        bins = [d[d.coarseness_rank == r][feat_name].to_numpy() for r in (1, 2, 3)]
        bins = [b for b in bins if len(b) > 0]
        out = {"n": int(n), "n_bins": len(bins),
               "median_by_bin": {int(r): round(float(d[d.coarseness_rank == r][feat_name].median()), 1)
                                 for r in (1, 2, 3) if (d.coarseness_rank == r).any()}}
        if len(bins) >= 2 and n >= 6:
            kw = kruskal(*bins)
            sp = spearmanr(d["coarseness_rank"], d[feat_name])
            out["kruskal_p"] = round(float(kw.pvalue), 4)
            out["spearman_rho"] = round(float(sp.correlation), 3)
            out["spearman_p"] = round(float(sp.pvalue), 4)
            out["hypothesis"] = "coarser gold (rank up) should have SHORTER distance (rho<0)"
        else:
            out["result"] = "too few points/bins to test"
        h2[feat_name] = out
    results["H2_coarseness"] = h2

    # persist
    (ILS / "h1_h2_results.json").write_text(json.dumps(results, indent=2))
    # per-point feature table for transparency
    al_table = al.copy()
    axs2, ays2 = al_table.geometry.x.to_numpy(), al_table.geometry.y.to_numpy()
    al_table["contact"] = sample_raster(CONTACT, axs2, ays2)
    al_table["straight"] = sample_raster(STRAIGHT, axs2, ays2)
    al_table["downch_snapped"] = snap_sample(DOWNCH, axs2, ays2, 150.0)
    al_table.drop(columns="geometry").to_csv(ILS / "alluvial_points_features.csv", index=False)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {ILS/'h1_h2_results.json'} and alluvial_points_features.csv")


if __name__ == "__main__":
    main()
