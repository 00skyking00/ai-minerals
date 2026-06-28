"""Round-5 follow-up: re-run the H1 occurrence test with the DISPERSED areal
lode-source feature and the PROPER schist-carbonate contact, after resolving the
DOx unit against the published legend.

DOx RESOLUTION (USGS SIM 3131, Till and others 2011, which uses the exact label
DOx): DOx = "Mixed marble, graphitic metasiliceous rock, and schist (Devonian to
Ordovician)" -- interlayered pure/impure marble, graphitic metasiliceous rock,
and pelitic/calc/mafic schist, "dominated locally by one or the other,"
structurally below the Casadepaga Schist (unit Ocs), part of the Nome Complex.
It is NOT a discrete carbonate platform. The two clean carbonate units in the
Nome quad (Oim impure chlorite marble, Pzmm Paleozoic marble) lie 11-33 km
OUTSIDE the Nome district DEM AOI. So within the district the only
carbonate-bearing unit is the mixed DOx, and the geologically-clean
schist-carbonate contact (10.3 km) is a regional feature off the test AOI. We
therefore test the contact BOTH ways and report it honestly:

  contact_dox    distance to the schist-carbonate contact WITH DOx counted as
                 carbonate (154.8 km, within-AOI) -- the "proper derived contact"
                 from PR #49; after the legend check this is honestly a
                 distance-to-mixed-DOx-unit-boundary feature, not a clean
                 schist/limestone lithologic contact.
  contact_clean  distance to the contact with ONLY the clean marbles Oim+Pzmm
                 (10.3 km, off-AOI) -- lithologically strict but a regional ramp
                 inside the district because the marbles are 11-33 km away.

The dispersed-source feature is the main new instrument:

  areal          distance (m) to the nearest Nome Group schist polygon
                 (PzZh/Ocs/Dcs/Zn/Zo), 0 inside, Euclidean out -- the
                 hydrology.distance_to_lode_areal_m semantics realized on the
                 25 m DEM grid. Per Tuck 1942 the gold is "disseminated over a
                 wide area," so the areal schist host, not the sparse mapped
                 36a veins, is the right source geometry.

For apples-to-apples we also re-score round-5's discrete straight-line distance
and prebuilt contact in the same run. Same strictly-alluvial labels, same
leak-guarded spatial CV (KMeans blocks + 300 m buffer, bootstrap CI). The
random-point spatial null (distance to N random points, N = lode count) is the
bar for the line/point distance features; the areal feature gets a
blob-preserving torus-shift null (the real schist mask shifted to a random
position, area and shape preserved). A feature SURVIVES spatial CV if its
single-feature RF spatial-block-CV AUC has a 95% CI strictly above 0.5.

Run: uv run python -m scripts.nome_placer.inland_local_source.run_areal_contact_followup
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely import line_merge
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from scripts.nome_placer.inland_local_source.run_h1_h2 import (
    DEM, SLOPE, TPI, ILS, CONTACT, STRAIGHT,
    sample_raster, boot_auc_ci, rank_auc, spatial_cv_oof, model_auc,
    land_mask_cells, RNG, N_BG, N_BG_CONFINED, K_NULL,
)

GEOL = Path("data/raw/nome_mpm/geol/nmgeol_dd/nm_ddp.shp")
AREAL = ILS / "dist_to_schist_areal_3338.tif"
CONTACT_DOX = Path(
    "data/derived/nome_placer/peninsula_phase2/dist_to_schist_carbonate_contact_3338.tif")
CONTACT_CLEAN = ILS / "dist_to_schist_carbonate_contact_clean_3338.tif"

SCHIST = {"PzZh", "Ocs", "Dcs", "Zn", "Zo"}
CARB_CLEAN = {"Oim", "Pzmm"}        # the unambiguous marbles (DOx excluded)


def _dem_grid():
    with rasterio.open(DEM) as ds:
        dem = ds.read(1); nod = ds.nodata; T = ds.transform
        H, W = ds.height, ds.width
        prof = ds.profile.copy()
    return dem, nod, T, H, W, prof, abs(T.a)


def build_areal_raster() -> None:
    """distance_to_lode_areal_m semantics: 0 inside any Nome Group schist
    polygon, Euclidean distance (m) to the nearest schist cell outside."""
    dem, nod, T, H, W, prof, cs = _dem_grid()
    polys = gpd.read_file(GEOL).to_crs("EPSG:3338")
    schist = polys[polys.LABEL.isin(SCHIST)]
    mask = rasterize(((g, 1) for g in schist.geometry), out_shape=(H, W),
                     transform=T, fill=0, dtype="uint8").astype(bool)
    dist = (distance_transform_edt(~mask) * cs).astype(np.float32)   # 0 inside schist
    dist = np.where(dem == nod, -1.0, dist)
    prof.update(dtype="float32", count=1, nodata=-1.0, compress="lzw")
    with rasterio.open(AREAL, "w", **prof) as dst:
        dst.write(dist, 1)
    print(f"areal: schist {int(mask.sum())} cells ({100*mask.sum()/(dem != nod).sum():.1f}% of land); "
          f"median dist-to-schist {np.median(dist[(dem != nod) & (dist > 0)]):.0f} m")


def _densify(geom, step_m: float = 50.0) -> np.ndarray:
    pts, lines = [], []
    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        lines = [g for g in geom.geoms if g.geom_type == "LineString"]
    for ln in lines:
        n = max(2, int(ln.length // step_m) + 1)
        for i in range(n + 1):
            p = ln.interpolate(min(i * step_m, ln.length))
            pts.append((p.x, p.y))
    return np.array(pts) if pts else np.empty((0, 2))


def build_clean_contact_raster() -> float:
    """schist-vs-(Oim+Pzmm) contact distance raster. Returns the contact length
    (km). The clean marbles are off-AOI so this is a regional ramp here."""
    dem, nod, T, H, W, prof, cs = _dem_grid()
    polys = gpd.read_file(GEOL).to_crs("EPSG:3338")
    su = polys[polys.LABEL.isin(SCHIST)].union_all()
    cu = polys[polys.LABEL.isin(CARB_CLEAN)].union_all()
    contact = su.boundary.intersection(cu.boundary)
    length_km = 0.0 if contact.is_empty else contact.length / 1000.0
    if not contact.is_empty and contact.geom_type != "LineString":
        contact = line_merge(contact)
    pts = _densify(contact, 50.0)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = rasterio.transform.xy(T, rows.ravel(), cols.ravel())
    cell_xy = np.column_stack([np.asarray(cx), np.asarray(cy)])
    if len(pts):
        d, _ = cKDTree(pts).query(cell_xy, k=1)
        dist = d.reshape(H, W).astype(np.float32)
    else:
        dist = np.full((H, W), -1.0, dtype=np.float32)
    dist = np.where(dem == nod, -1.0, dist)
    prof.update(dtype="float32", count=1, nodata=-1.0, compress="lzw")
    with rasterio.open(CONTACT_CLEAN, "w", **prof) as dst:
        dst.write(dist, 1)
    print(f"clean contact (Oim+Pzmm): {length_km:.1f} km; "
          f"median AOI dist {np.median(dist[(dem != nod) & (dist >= 0)]):.0f} m")
    return round(length_km, 1)


def areal_torus_null(y: np.ndarray, coords: np.ndarray, k: int = 200) -> dict:
    """Blob-preserving null for the areal feature: roll the schist mask by a
    random (dy,dx) torus shift (area + shape preserved, position randomized),
    sample distance at coords, take the AUC. Real schist must beat this."""
    dem, nod, T, H, W, prof, cs = _dem_grid()
    polys = gpd.read_file(GEOL).to_crs("EPSG:3338")
    schist = polys[polys.LABEL.isin(SCHIST)]
    mask = rasterize(((g, 1) for g in schist.geometry), out_shape=(H, W),
                     transform=T, fill=0, dtype="uint8").astype(bool)
    rr, cc = rasterio.transform.rowcol(T, coords[:, 0], coords[:, 1])
    rr = np.clip(np.asarray(rr), 0, H - 1); cc = np.clip(np.asarray(cc), 0, W - 1)
    aucs = []
    for i in range(k):
        rk = np.random.default_rng(7000 + i)
        dy, dx = int(rk.integers(0, H)), int(rk.integers(0, W))
        shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        d = distance_transform_edt(~shifted) * cs
        aucs.append(roc_auc_score(y, -d[rr, cc]))
    return {"mean_auc": round(float(np.mean(aucs)), 3),
            "p95_auc": round(float(np.percentile(aucs, 95)), 3),
            "note": "distance to the schist mask shifted to a random position (area/shape preserved)"}


# single-feature columns and their hypothesis (all: shorter distance -> placer)
FEATURES = ["areal", "contact_dox", "contact_clean", "straight", "contact_pre"]


def feats(xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "areal": sample_raster(AREAL, xs, ys),
        "contact_dox": sample_raster(CONTACT_DOX, xs, ys),
        "contact_clean": sample_raster(CONTACT_CLEAN, xs, ys),
        "straight": sample_raster(STRAIGHT, xs, ys),
        "contact_pre": sample_raster(CONTACT, xs, ys),
        "dem": sample_raster(DEM, xs, ys),
        "slope": sample_raster(SLOPE, xs, ys),
        "tpi": sample_raster(TPI, xs, ys),
    })


def single_block(feat: pd.DataFrame, y: np.ndarray, coords: np.ndarray,
                 groups: np.ndarray) -> dict:
    """Rank AUC (+CI) and single-feature RF spatial-CV AUC (+CI) per feature.
    survives = RF spatial-CV CI strictly above 0.5."""
    out = {}
    for f in FEATURES:
        d = feat[f].fillna(feat[f].max()).to_numpy()
        rpt, rlo, rhi = rank_auc(y, d)
        m = model_auc(feat, [f], y, coords, groups)
        out[f] = {
            "rank_auc": rpt, "rank_ci95": [rlo, rhi],
            "rf_spatial_cv_auc": m["auc"], "rf_spatial_cv_ci95": m["ci95"],
            "survives_spatial_cv": bool(m["ci95"][0] > 0.5),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(RNG)
    if not AREAL.exists():
        build_areal_raster()
    clean_km = build_clean_contact_raster()   # always rebuild (cheap, records length)

    typed = gpd.read_file(ILS / "placers_typed.geojson")
    al = typed[typed.geol_type == "alluvial-stream"]
    zone = rasterio.open(ILS / "zone.tif").read(1)
    nlode = json.loads((ILS / "distance_meta.json").read_text())["n_lode_in_grid"]
    rows, cols, T = land_mask_cells()

    results: dict = {
        "dox_resolution": (
            "USGS SIM 3131 (Till 2011) DOx = 'Mixed marble, graphitic metasiliceous "
            "rock, and schist (Devonian-Ordovician)', NOT a clean carbonate platform; "
            "clean marbles Oim+Pzmm are 11-33 km outside the district DEM AOI, so the "
            "clean schist-carbonate contact is regional (Phase-2), and within the "
            "district the only carbonate-bearing unit is the mixed DOx."),
        "contact_lengths_km": {"with_DOx": 154.8, "clean_Oim_Pzmm": clean_km},
        "params": {"n_bg": N_BG, "n_bg_upland": N_BG_CONFINED, "buffer_m": 300,
                   "k_null": K_NULL},
    }

    # ---- background: random on-land cells (full AOI) ----
    pick = rng.choice(len(rows), size=N_BG, replace=False)
    bx, by = rasterio.transform.xy(T, rows[pick], cols[pick])
    bx, by = np.asarray(bx), np.asarray(by)

    # =====================================================================
    # H1a full-AOI: alluvial positives vs random land background
    # =====================================================================
    px, py = al.geometry.x.to_numpy(), al.geometry.y.to_numpy()
    ex = np.concatenate([px, bx]); ey = np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    feat = feats(ex, ey)
    coords = np.column_stack([ex, ey])
    groups = KMeans(n_clusters=10, random_state=RNG, n_init=10).fit_predict(coords)

    h1a = {"n_pos": int(len(px)), "n_bg": int(len(bx))}
    h1a["features"] = single_block(feat, y, coords, groups)
    # multi-feature models (the dispersed-source feature stack)
    h1a["models"] = {
        "areal_only": model_auc(feat, ["areal"], y, coords, groups),
        "areal+contact_dox": model_auc(feat, ["areal", "contact_dox"], y, coords, groups),
        "areal+terrain": model_auc(feat, ["areal", "dem", "slope", "tpi"], y, coords, groups),
    }
    # random-point null (line/point features must beat this)
    null_aucs = []
    for k in range(K_NULL):
        rk = np.random.default_rng(1000 + k)
        idx = rk.choice(len(rows), size=nlode, replace=False)
        rxx, ryy = rasterio.transform.xy(T, rows[idx], cols[idx])
        dd, _ = cKDTree(np.column_stack([rxx, ryy])).query(coords, k=1)
        null_aucs.append(roc_auc_score(y, -dd))
    h1a["spatial_null_random_points"] = {
        "mean_auc": round(float(np.mean(null_aucs)), 3),
        "p95_auc": round(float(np.percentile(null_aucs, 95)), 3),
        "note": "distance to N random points (N = lode count); line/point features must beat this",
    }
    h1a["spatial_null_areal_torus"] = areal_torus_null(y, coords, k=200)
    results["H1a_full_aoi"] = h1a

    # =====================================================================
    # H1b upland-matched: alluvial-in-upland vs random-upland background
    # =====================================================================
    def in_zone3(xs, ys):
        rr, ccc = rasterio.transform.rowcol(T, xs, ys)
        rr = np.clip(rr, 0, zone.shape[0] - 1); ccc = np.clip(ccc, 0, zone.shape[1] - 1)
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
    h1b["features"] = single_block(featc, yc, coordsc, groupsc)
    results["H1b_upland_matched"] = h1b

    # =====================================================================
    # negative control: marine-beach set vs full-AOI background
    # =====================================================================
    mb = typed[typed.geol_type == "marine-beach"]
    if len(mb) >= 3:
        mx, my = mb.geometry.x.to_numpy(), mb.geometry.y.to_numpy()
        emx = np.concatenate([mx, bx]); emy = np.concatenate([my, by])
        ym = np.concatenate([np.ones(len(mx)), np.zeros(len(bx))]).astype(int)
        fm = feats(emx, emy)
        nc = {"n_pos": int(len(mx)), "features": {}}
        for f in ("areal", "contact_dox"):
            d = fm[f].fillna(fm[f].max()).to_numpy()
            pt, lo, hi = rank_auc(ym, d)
            nc["features"][f] = {"rank_auc": pt, "rank_ci95": [lo, hi]}
        results["negative_control_marine"] = nc

    # =====================================================================
    # verdict: which features survive spatial CV
    # =====================================================================
    verdict = {}
    for f in FEATURES:
        a = h1a["features"][f]; b = h1b["features"][f]
        verdict[f] = {
            "full_aoi_survives": a["survives_spatial_cv"],
            "upland_matched_survives": b["survives_spatial_cv"],
            "full_aoi_rank_auc": a["rank_auc"],
            "full_aoi_rf_cv_auc": a["rf_spatial_cv_auc"],
            "upland_rf_cv_auc": b["rf_spatial_cv_auc"],
        }
    results["verdict_survives_spatial_cv"] = verdict

    out = ILS / "areal_contact_followup_results.json"
    out.write_text(json.dumps(results, indent=2))

    # per-point feature table for transparency
    tbl = al.copy()
    axs, ays = tbl.geometry.x.to_numpy(), tbl.geometry.y.to_numpy()
    ft = feats(axs, ays)
    for c in FEATURES:
        tbl[c] = ft[c].to_numpy()
    tbl.drop(columns="geometry").to_csv(ILS / "areal_contact_followup_points.csv", index=False)

    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
