"""Production scoring of the served LODE prospectivity surface (struct_groves).

Companion to ``lode_structure_sharpen_cv`` / ``lode_groves_bootstrap``, which
report the *honest performance estimate* of this pipeline under the F1
leak-guarded spatial-block CV: on the typed 36a labels the ``struct_groves`` arm
(base lode features + the two Groves splay/intersection proximities + fold hinge
+ graphitic host) scores auc_gems 0.712, d=+0.170 over base, 95% CI
[+0.108, +0.234] (paired bootstrap on the 48 GeMS-mapped positives). Those
scripts never persist a model: they refit a fresh RF per held-out block and
report pooled out-of-fold AUC.

This script produces the *served surface*: the SAME feature set and the SAME
RandomForest hyper-parameters as the gate arm, refit once on all 36a labels,
then scored over the district grid. It is the lode analogue of
``mpm_onshore_score_district`` (the served placer MPM). The CV script answers
"how well does this pipeline rank held-out ground"; this script answers "what
does the pipeline say about each cell", which is what goldbug serves.

Extent decision (documented in the report + sidecar): the served surface is the
CENTRAL-DISTRICT struct_groves surface, masked to the GeMS-mapped footprint
(``gems_extent`` > 0.5). The two Groves bands that carry the gate
(dist_fault_intersection, dist_splay) are only defined where the Nome district
GeMS maps structure, so outside that footprint the surface is not meaningful and
is written as nodata. The peninsula-wide generalization (auc 0.82-0.89 on
dispersed 36a labels the central model never trained on, ``lode_peninsula_
generalization``) is validated-but-provisional: it rests on regional structure
maps and a length-quantile splay proxy rather than the named-trunk rule, and the
eastern check has only 9 positives. It is reported as context, not served.

Output (this repo's tree only; no goldbug delivery here, that is a separate,
coordinated handoff driven by the program coordinator):
  data/derived/nome_placer/mpm_lode_served/mpm_lode_served_district_3338.tif
  data/derived/nome_placer/mpm_lode_served/mpm_lode_served_district_4326.tif
  data/derived/nome_placer/mpm_lode_served/mpm_lode_served_bands.json
  data/derived/nome_placer/mpm_lode_served/mpm_lode_served_report.json

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.mpm_lode_score_district
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.ensemble import RandomForestClassifier

from ai_minerals.data import nome_structure as ns
from scripts.nome_placer.f1_leak_guarded_rebaseline import samp
from scripts.nome_placer.lode_structure_sharpen_cv import (
    CM, DEM_D, RNG, SLOPE_D, TEMPLATE, TPI_D,
    ensure_structure_bands, load_base, lode_positives,
)
from scripts.nome_placer.newlayers_geophys_rebaseline import sample_bands

OUT_DIR = Path("data/derived/nome_placer/mpm_lode_served")
GATE = Path("data/derived/nome_placer/lode_structure_sharpen/lode_structure_sharpen.json")
BOOT = Path("data/derived/nome_placer/lode_groves_bootstrap/lode_groves_bootstrap.json")
PENI = Path("data/derived/nome_placer/lode_peninsula_generalization/lode_peninsula_generalization.json")

# The confirmed gate arm, verbatim from lode_structure_sharpen_cv.run_label_set.
STRUCT_GROVES = ns.GROVES_BANDS + ["dist_fold_hinge", "carbonaceous_host"]
FILL = -999.0
OUT_NODATA = -1.0


def _train(struct_paths: dict[str, Path]) -> tuple[RandomForestClassifier, list[str], dict]:
    """Refit the struct_groves RF on all 36a labels (same labels/features as the
    gate arm); return the fitted model, its feature-column order, and label counts."""
    with rasterio.open(TEMPLATE) as ds:
        bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    px, py = lode_positives(bounds, "36a")
    base_df, y, coords, in_box = load_base((px, py))
    ex, ny = coords[:, 0], coords[:, 1]
    add = sample_bands(struct_paths, ex, ny).fillna(FILL)
    mapped = (add["gems_extent"] > 0.5).to_numpy()

    names = list(base_df.columns) + STRUCT_GROVES
    X = np.column_stack([base_df.to_numpy(np.float32),
                         add[STRUCT_GROVES].to_numpy(np.float32)]).astype(np.float32)
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RNG, n_jobs=1).fit(X, y)
    counts = {"n": int(len(y)), "n_pos": int(y.sum()),
              "n_pos_gems_mapped": int(y[mapped].sum()),
              "n_pos_placer_core": int(y[in_box].sum()),
              "n_background": int((y == 0).sum())}
    return clf, names, counts


def _grid_feature_frame(names: list[str], struct_paths: dict[str, Path]):
    """Full-grid feature matrix in the trained column order, plus the served mask
    (GeMS-mapped AND on land) and the grid georeference. Every base / structure
    raster is co-registered to the 100 m district template, so a plain per-band
    read reproduces the point-sampled training features without resampling; the
    nodata -> -999 fill matches f1 ``samp`` + ``.fillna(-999)`` exactly."""
    legend = json.loads((CM / "geology_unit_code_district_legend.json").read_text())
    geo_codes = {int(k): v for k, v in legend.items() if int(k) != 0}

    with rasterio.open(TEMPLATE) as ds:
        transform, H, W = ds.transform, ds.height, ds.width

    def read_fill(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            a = ds.read(1).astype(np.float64)
            nod = ds.nodata
        if nod is not None:
            a = np.where(a == nod, np.nan, a)
        return a.reshape(-1)

    cols: dict[str, np.ndarray] = {}
    # geology one-hot: nodata(0) -> 0 -> all-zero one-hot, matching _lode_features
    with rasterio.open(CM / "geology_unit_code_district_3338.tif") as ds:
        geol = ds.read(1).reshape(-1)
        geol = np.where(geol == ds.nodata, 0, geol).astype(int)
    for k, v in geo_codes.items():
        cols[f"geol_{v}"] = (geol == k).astype(np.float32)
    cols["akmag"] = read_fill(CM / "akmag_district_3338.tif")
    cols["dist_fault"] = read_fill(CM / "dist_to_fault_district_3338.tif")
    cols["dem"] = read_fill(DEM_D)
    cols["slope"] = read_fill(SLOPE_D)
    cols["tpi"] = read_fill(TPI_D)
    for b in STRUCT_GROVES:
        cols[b] = read_fill(struct_paths[b])

    X = np.column_stack([np.nan_to_num(cols[n], nan=FILL).astype(np.float32) for n in names])

    gems = read_fill(struct_paths["gems_extent"])
    dem_valid = ~np.isnan(cols["dem"])
    served = (np.nan_to_num(gems, nan=0.0) > 0.5) & dem_valid
    return X, served, transform, H, W


def _write_geotiff(path: Path, arr: np.ndarray, transform) -> None:
    H, W = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs=CRS.from_epsg(3338), transform=transform, nodata=OUT_NODATA,
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.set_band_description(1, "Lode favorability (struct_groves RF, central district)")


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
            dst.set_band_description(1, "Lode favorability (struct_groves RF, central district)")


def _gate_numbers() -> dict:
    """Pull the leak-guarded gate AUC + bootstrap CI + peninsula context from the
    committed evaluation reports, so the served sidecar cites live numbers."""
    out: dict = {}
    if GATE.exists():
        g = json.loads(GATE.read_text())
        d36 = next(d for d in g["datasets"] if d["name"] == "lode_district_36a")
        sg = d36["arms"]["struct_groves"]
        out["auc_gems"] = round(sg["auc_gems"], 3)
        out["d_auc_gems_over_base"] = sg["d_auc_gems"]
        out["auc_full"] = round(sg["auc_full"], 3)
        out["auc_placer_core"] = round(sg["auc_placer_core"], 3)
    if BOOT.exists():
        b = json.loads(BOOT.read_text())
        sg = b["datasets"]["struct_groves"]["gems"]
        out["bootstrap_ci95_delta"] = sg.get("ci95")
        out["bootstrap_p_gt_0"] = sg.get("p_gt_0")
        out["bootstrap_n_pos"] = sg.get("n_pos")
    if PENI.exists():
        p = json.loads(PENI.read_text())
        v = p.get("verdict", {})
        out["peninsula"] = {
            "wider_auc_mapped": v.get("wider_struct_groves_auc_mapped"),
            "wider_auc_east_heldout": v.get("wider_auc_east"),
            "wider_d_auc_mapped": v.get("wider_d_auc_mapped"),
            "status": "validated-but-provisional (regional splay proxy; 9 eastern positives)",
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building/reusing sharpened structure bands (district 100 m) ...")
    sp = ensure_structure_bands()

    clf, names, counts = _train(sp)
    print(f"refit struct_groves on all 36a labels: {counts}")
    print(f"feature order ({len(names)}): {names}")

    X, served, transform, H, W = _grid_feature_frame(names, sp)
    proba = np.full(X.shape[0], np.nan, dtype=np.float32)
    idx = np.where(served)[0]
    proba[idx] = clf.predict_proba(X[idx])[:, 1].astype(np.float32)
    grid = proba.reshape(H, W)
    grid_out = np.where(np.isnan(grid), OUT_NODATA, grid).astype(np.float32)
    print(f"served cells (GeMS-mapped & on-land): {len(idx):,} / {X.shape[0]:,} "
          f"({len(idx) / X.shape[0]:.1%} of the district grid)")

    out_3338 = OUT_DIR / "mpm_lode_served_district_3338.tif"
    out_4326 = OUT_DIR / "mpm_lode_served_district_4326.tif"
    _write_geotiff(out_3338, grid_out, transform)
    _reproject_4326(out_3338, out_4326)
    print(f"  EPSG:3338 raster -> {out_3338} ({out_3338.stat().st_size:,} B)")
    print(f"  EPSG:4326 raster -> {out_4326} ({out_4326.stat().st_size:,} B)")

    s = grid[~np.isnan(grid)]
    pct = {f"p{q}": round(float(np.percentile(s, q)), 4)
           for q in (5, 10, 25, 50, 75, 90, 95, 99)}
    gate = _gate_numbers()
    report = {
        "model": "struct_groves RandomForest (300 trees, class_weight=balanced, seed=42)",
        "features": names,
        "labels": "ARDF Cox-Singer model_code 36a (low-sulfide Au-quartz vein), typed dispersed set",
        **counts,
        "served_extent": {
            "decision": "central-district (GeMS-mapped footprint)",
            "rationale": ("the gate-carrying Groves bands (dist_fault_intersection, "
                          "dist_splay) are only defined inside the Nome district GeMS; "
                          "outside it the surface is not meaningful, so it is masked to "
                          "gems_extent > 0.5 AND on-land."),
            "peninsula_extension": "validated-but-provisional, reported as context, not served",
        },
        "performance_estimate": {
            "source": ("lode_structure_sharpen.json (gate) + lode_groves_bootstrap.json (CI) "
                       "+ lode_peninsula_generalization.json (generalization)"),
            "note": ("These are the leak-guarded held-out estimates of THIS pipeline. The "
                     "served raster below is the same pipeline refit on all 36a labels; "
                     "AUC is a held-out estimate, not a property of the all-data refit."),
            **gate,
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
    (OUT_DIR / "mpm_lode_served_report.json").write_text(json.dumps(report, indent=2))

    bands_meta = {
        "schema_version": "1.0",
        "model_version": "lode-mpm-struct-groves-v1",
        "release": "ai-minerals-2026-06-27",
        "raster_3338": out_3338.name,
        "raster_4326": out_4326.name,
        "resolution_m": 100,
        "working_crs": "EPSG:3338",
        "delivery_crs": "EPSG:4326",
        "nodata": OUT_NODATA,
        "scope": ("central-district orogenic-lode favorability (struct_groves); GeMS-mapped "
                  "footprint only. Placer is modelled separately (mpm_onshore_score_district)."),
        "served_extent": "central-district (GeMS-mapped); peninsula extension provisional, not served",
        "bands": [{
            "index": 1,
            "key": "lode_proba",
            "name": "Lode favorability (struct_groves RF, central district)",
            "description": ("RandomForest presence/background probability over base lode "
                            "features (geology one-hot + aeromag + distance-to-fault + "
                            "terrain) plus the Groves structure set (NE x NW fault-"
                            "intersection proximity, second-order splay proximity, "
                            "fold-hinge distance, graphitic host). Single band."),
            "value_range": [0.0, 1.0],
        }],
        "legend_note": ("Single-band probability over the GeMS-mapped district. Recalibrate "
                        "display buckets from mpm_lode_served_report.json percentiles; the RF "
                        "probability is not on the same scale as the placer MPM, so do not "
                        "reuse the placer legend thresholds. Render the SAME way goldbug "
                        "renders the placer raster: per-claim zonal aggregate + percentile "
                        "buckets."),
    }
    (OUT_DIR / "mpm_lode_served_bands.json").write_text(json.dumps(bands_meta, indent=2))
    print(json.dumps(report["served_surface"], indent=2))
    print(f"  report  -> {OUT_DIR / 'mpm_lode_served_report.json'}")
    print(f"  sidecar -> {OUT_DIR / 'mpm_lode_served_bands.json'}")


if __name__ == "__main__":
    main()
