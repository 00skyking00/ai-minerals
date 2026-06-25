"""Production scoring of the onshore placer presence/background MPM.

Companion to `mpm_onshore_presence_cv.py`, which reports the *honest performance
estimate* of this pipeline under spatial-block CV (0.733 spatial-CV, 0.741 with a
300 m buffer). That script never persists a model: it refits a fresh RF for each
held-out block and reports pooled out-of-fold AUC.

This script produces the *served surface*: the SAME GEOMORPH feature set and the
SAME RandomForest hyper-parameters, refit once on all labels, then scored over
every on-land cell of the district grid. The CV script answers "how well does
this pipeline rank held-out ground"; this script answers "what does the pipeline
say about each cell", which is what goldbug serves. The two are deliberately the
same covariates and the same estimator so the served raster reflects the 0.733
model rather than a different dump.

Output (this repo's tree only; no goldbug delivery here, that is a separate,
coordinated handoff):
  data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_3338.tif
  data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_4326.tif
  data/derived/nome_placer/mpm_onshore/mpm_onshore_score_bands.json
  data/derived/nome_placer/mpm_onshore/mpm_onshore_score_report.json

Run: uv run python -m scripts.nome_placer.mpm_onshore_score_district
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.ensemble import RandomForestClassifier

from ai_minerals.data.adapters.occurrences import kg as kg_occ

DD = Path("data/derived/nome_placer")
V3P1 = DD / "prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif"
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_3338.tif")
KG_EXPORT = Path("data/raw/fossick_kg/kg_nome.jsonld")  # ADR-017 placer labels
OUT_DIR = DD / "mpm_onshore"
N_BG = 2000
RNG = 42

POP = ["bl", "ap", "tb", "ss", "bc", "qm", "buried_bl"]
GEOMORPH = POP + ["dem", "slope", "tpi"]  # honest regional test (no geochem)
FILL = -999.0


def _samp_points(path: Path, ex, ny, bands, names) -> dict[str, np.ndarray]:
    """Sample raster at point coords, mapping nodata -> nan (matches the CV
    script's `samp` so the trained model sees identical feature construction)."""
    with rasterio.open(path) as ds:
        a = np.asarray(list(ds.sample(list(zip(ex, ny)))), float)
        nod = ds.nodata
    return {
        nm: (np.where(a[:, b - 1] == nod, np.nan, a[:, b - 1]) if nod is not None
             else a[:, b - 1])
        for b, nm in zip(bands, names)
    }


def _train_features(ex, ny) -> np.ndarray:
    cols: dict[str, np.ndarray] = {}
    cols.update(_samp_points(V3P1, ex, ny, list(range(1, 8)), POP))
    cols.update(_samp_points(DEM, ex, ny, [1], ["dem"]))
    cols.update(_samp_points(SLOPE, ex, ny, [1], ["slope"]))
    cols.update(_samp_points(TPI, ex, ny, [1], ["tpi"]))
    X = np.column_stack([cols[c] for c in GEOMORPH]).astype(np.float32)
    return np.where(np.isnan(X), FILL, X)


def _grid_features() -> tuple[np.ndarray, np.ndarray, rasterio.Affine, int, int]:
    """Full-grid feature matrix (n_cells, n_feat) plus an on-land validity mask.
    All four rasters are co-registered (697x846, EPSG:3338, 25 m, identical
    bounds), so a plain per-band read aligns without resampling."""
    with rasterio.open(V3P1) as ds:
        pop = ds.read(list(range(1, 8))).astype(np.float32)  # (7, H, W), already 0-filled
        transform, H, W = ds.transform, ds.height, ds.width
    bands = [pop[i] for i in range(7)]
    for path in (DEM, SLOPE, TPI):
        with rasterio.open(path) as ds:
            a = ds.read(1).astype(np.float32)
            nod = ds.nodata
        if nod is not None:
            a = np.where(a == nod, np.nan, a)
        bands.append(a)
    with rasterio.open(DEM) as ds:
        valid = ds.read(1) != ds.nodata  # on-land mask
    stack = np.stack(bands, axis=0)  # (10, H, W) in GEOMORPH order
    X = stack.reshape(stack.shape[0], -1).T  # (n_cells, 10)
    X = np.where(np.isnan(X), FILL, X)
    return X, valid.reshape(-1), transform, H, W


def _placer_positives():
    with rasterio.open(V3P1) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    occ = kg_occ.load(KG_EXPORT).to_crs("EPSG:3338")
    comm = occ["commodity"].astype(str).str.lower()
    code39 = occ["deposit_codes"].apply(
        lambda t: any(str(c).split(":")[-1].startswith("39") for c in t))
    placer = occ[comm.str.contains("placer") | code39].copy()
    placer = placer.cx[bl:br, bb:bt]
    px = placer.geometry.x.to_numpy()
    py = placer.geometry.y.to_numpy()
    order = np.lexsort((py, px))  # canonical order (source-row-order independence)
    return px[order], py[order]


def _write_geotiff(path: Path, arr: np.ndarray, transform, nodata: float) -> None:
    H, W = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs=CRS.from_epsg(3338), transform=transform, nodata=nodata,
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.set_band_description(1, "Placer presence MPM probability (GEOMORPH RF)")


def _reproject_4326(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        dst_transform, dw, dh = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update(crs="EPSG:4326", transform=dst_transform, width=dw, height=dh)
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs="EPSG:4326",
                src_nodata=src.nodata, dst_nodata=src.nodata,
                resampling=Resampling.bilinear,
            )
            dst.set_band_description(1, "Placer presence MPM probability (GEOMORPH RF)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Labels: 65 placer positives (fossick KG, ADR-017) + N_BG on-land background.
    px, py = _placer_positives()
    print(f"KG placer positives in AOI: {len(px)}")
    with rasterio.open(DEM) as ds:
        dem = ds.read(1)
        nod = ds.nodata
        T = ds.transform
    valid = np.argwhere(dem != nod)
    rng = np.random.default_rng(RNG)
    pick = valid[rng.choice(len(valid), size=N_BG, replace=False)]
    bx, by = rasterio.transform.xy(T, pick[:, 0], pick[:, 1])
    bx = np.asarray(bx)
    by = np.asarray(by)

    ex = np.concatenate([px, bx])
    ny = np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    X = _train_features(ex, ny)

    # Refit-on-all: same estimator as the CV script (0.733 / 0.741 estimate).
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RNG, n_jobs=1)
    clf.fit(X, y)

    # Score every on-land cell of the district grid.
    Xg, land_mask, transform, H, W = _grid_features()
    proba = np.full(Xg.shape[0], np.nan, dtype=np.float32)
    idx = np.where(land_mask)[0]
    proba[idx] = clf.predict_proba(Xg[idx])[:, 1].astype(np.float32)
    grid = proba.reshape(H, W)
    OUT_NODATA = -1.0
    grid_out = np.where(np.isnan(grid), OUT_NODATA, grid)

    out_3338 = OUT_DIR / "mpm_onshore_score_district_3338.tif"
    out_4326 = OUT_DIR / "mpm_onshore_score_district_4326.tif"
    _write_geotiff(out_3338, grid_out, transform, OUT_NODATA)
    _reproject_4326(out_3338, out_4326)
    print(f"  EPSG:3338 raster -> {out_3338} ({out_3338.stat().st_size:,} B)")
    print(f"  EPSG:4326 raster -> {out_4326} ({out_4326.stat().st_size:,} B)")

    # Score distribution over scored cells (goldbug needs this to recalibrate the
    # served legend: the RF probability is NOT on the same scale as the v3.1
    # rule-based composite, so the old 0.5/0.3/0.1 absolute buckets do not carry).
    s = grid[~np.isnan(grid)]
    pct = {f"p{q}": round(float(np.percentile(s, q)), 4)
           for q in (5, 10, 25, 50, 75, 90, 95, 99)}
    report = {
        "model": "GEOMORPH RandomForest (300 trees, class_weight=balanced, seed=42)",
        "features": GEOMORPH,
        "n_placer": int(len(px)),
        "n_background": int(len(bx)),
        "performance_estimate": {
            "source": "mpm_onshore_presence_report.json (spatial-block CV)",
            "spatialCV_auc": 0.733,
            "buffered300m_auc": 0.741,
            "note": ("These are the honest performance estimates of THIS pipeline "
                     "under spatial-block CV. The served raster below is the same "
                     "pipeline refit on all labels; AUC is a held-out estimate, not "
                     "a property of the all-data refit."),
        },
        "served_surface": {
            "raster_3338": out_3338.name,
            "raster_4326": out_4326.name,
            "scored_cells": int(s.size),
            "nodata": OUT_NODATA,
            "value_range_scored": [round(float(s.min()), 4), round(float(s.max()), 4)],
            "percentiles": pct,
        },
    }
    (OUT_DIR / "mpm_onshore_score_report.json").write_text(json.dumps(report, indent=2))

    bands_meta = {
        "schema_version": "1.0",
        "model_version": "placer-mpm-geomorph-v1",
        "release": "ai-minerals-2026-06-22",
        "raster_3338": out_3338.name,
        "raster_4326": out_4326.name,
        "resolution_m": 25,
        "working_crs": "EPSG:3338",
        "delivery_crs": "EPSG:4326",
        "nodata": OUT_NODATA,
        "scope": "onshore placer presence/background; lode is NOT modelled here",
        "bands": [{
            "index": 1,
            "key": "mpm_proba",
            "name": "Placer presence MPM probability (GEOMORPH RF)",
            "description": ("RandomForest presence/background probability over "
                            "GEOMORPH features (7 v3.1 population bands + DEM + "
                            "slope + TPI). Single band; not the v3.1 composite."),
            "value_range": [0.0, 1.0],
        }],
        "legend_note": ("Single-band probability. Distribution differs from the "
                        "v3.1 rule-based composite; recalibrate display buckets "
                        "from mpm_onshore_score_report.json percentiles rather "
                        "than reusing the v3.1 0.5/0.3/0.1 absolute thresholds."),
    }
    (OUT_DIR / "mpm_onshore_score_bands.json").write_text(json.dumps(bands_meta, indent=2))
    print(json.dumps(report["served_surface"], indent=2))
    print(f"  report  -> {OUT_DIR/'mpm_onshore_score_report.json'}")
    print(f"  sidecar -> {OUT_DIR/'mpm_onshore_score_bands.json'}")


if __name__ == "__main__":
    main()
