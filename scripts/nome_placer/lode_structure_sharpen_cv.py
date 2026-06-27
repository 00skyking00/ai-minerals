"""Round-2 LODE rebuild: do typed (36a) labels + sharpened structure features push
the district lode AUC past ~0.70 under the F1 leak-guarded CV?

This promotes the round-1 +0.106 structure screen (generic NE/NW-fault + graphitic
host, AUC_gems 0.540 -> 0.646) toward a verdict by:

  labels    Typed, dispersed positives -- ARDF Cox-Singer ``model_code`` 36a
            (low-sulfide Au-quartz vein), peninsula-wide within the district grid,
            replacing the round-1 "model 36/27/22 OR lode/vein keyword" set. The
            round-1 set is rerun alongside for the typed-vs-untyped comparison.
  features  The sharpened structure bands (``nome_structure``):
              named   per-named-fault distance (Anvil, Penny River, Aurora Creek,
                      Charlie Creek, Boulder Creek, Rodine) -- the per-named-fault
                      split, with the Rodine NE-floor drop fixed (floor 22.5->15).
              groves  NE x NW structure-intersection proximity + second-order-splay
                      proximity (Groves et al. 2018: deposits sit in the
                      splays/intersections, not on the trunk).
            plus the fold hinge (Twin Mountain anticline) and the graphitic host.

The clean test is ``auc_gems`` -- OOF AUC restricted to the GeMS-mapped footprint,
because the structure bands only exist there and a whole-extent AUC mixes structure
signal with a mapping-coverage proxy (the F3 lesson). The gate is auc_gems > 0.70.
``auc_placer_core`` is the standing restriction-of-range alarm (in-box subset).

Folds are sized ONCE per label set from that set's base-model residual variogram
(F1 2x recipe) and reused across arms, so each arm's delta is the features alone.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_structure_sharpen_cv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from ai_minerals.data import nome_structure as ns
from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof, subset_auc
from scripts.nome_placer.f1_leak_guarded_rebaseline import samp, _background, _lode_features
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, evaluate_arm, sample_bands

RNG = 42
CM = Path("data/derived/nome_placer/covariates_mpm")
TEMPLATE = CM / "_district_grid_template_3338.tif"
DEM_D = Path("data/raw/nome_mpm/ifsar_dem_district_3338.tif")
SLOPE_D = Path("data/raw/nome_mpm/ifsar_slope_district_3338.tif")
TPI_D = Path("data/raw/nome_mpm/ifsar_tpi_district_3338.tif")
ARDF = Path("data/raw/nome_mpm/ardf_nome.geojson")
V3P1 = Path("data/derived/nome_placer/prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif")
STRUCT_DIR = Path("data/derived/nome_geophys/nome_structure_sharp/district_100m")
OUT_DIR = Path("data/derived/nome_placer/lode_structure_sharpen")


def lode_positives(bounds, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """ARDF lode positives in ``bounds`` for a label mode.

    mode="round1": Cox-Singer model 36/27/22 OR lode/vein/skarn/greisen keyword
                   (the round-1 district set).
    mode="36a"   : model_code starts with "36a" (low-sulfide Au-quartz vein,
                   incl. the questioned "36a?") -- the typed dispersed set.
    """
    bl, bb, br, bt = bounds
    ardf = gpd.read_file(ARDF).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    if mode == "36a":
        sel = mc.str.startswith("36a")
    else:
        txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
               + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str)).str.lower()
        sel = mc.str.startswith(("36", "27", "22")) | txt.str.contains("lode|vein|skarn|greisen")
    lode = ardf[sel].cx[bl:br, bb:bt]
    return lode.geometry.x.to_numpy(), lode.geometry.y.to_numpy()


def ensure_structure_bands() -> dict[str, Path]:
    """Build (or reuse) the sharpened structure GeoTIFFs on the district grid."""
    bands = ns.SHARP_BANDS + ns.DIST_BANDS + ["gems_extent"]
    if all((STRUCT_DIR / f"struct_{b}.tif").exists() for b in bands):
        return {b: STRUCT_DIR / f"struct_{b}.tif" for b in bands}
    return ns.build_sharpened_structure_bands(TEMPLATE, STRUCT_DIR)


def load_base(positives) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Base lode features (geology one-hot + akmag + dist_fault + terrain) for
    the given positives plus 3000 random on-land background; + in-box mask."""
    legend = json.loads((CM / "geology_unit_code_district_legend.json").read_text())
    geo_codes = {int(k): v for k, v in legend.items() if int(k) != 0}
    with rasterio.open(TEMPLATE) as ds:
        T = ds.transform
    px, py = positives
    bx, by = _background(DEM_D, T, 3000)
    ex, ny = np.concatenate([px, bx]), np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    base = _lode_features(ex, ny, CM / "geology_unit_code_district_3338.tif", geo_codes,
                          CM / "akmag_district_3338.tif", CM / "dist_to_fault_district_3338.tif",
                          DEM_D, SLOPE_D, TPI_D)
    with rasterio.open(V3P1) as ds:
        pbl, pbb, pbr, pbt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    in_box = (ex >= pbl) & (ex <= pbr) & (ny >= pbb) & (ny <= pbt)
    return base, y, np.column_stack([ex, ny]), in_box


def run_label_set(name: str, mode: str, struct_paths: dict[str, Path]) -> dict:
    with rasterio.open(TEMPLATE) as ds:
        bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    px, py = lode_positives(bounds, mode)
    base_df, y, coords, in_box = load_base((px, py))
    X_base = base_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    st = sample_bands(struct_paths, ex, ny)
    mapped = (st["gems_extent"] > 0.5).to_numpy()
    masks = {"gems": mapped, "placer_core": in_box}
    add = st.fillna(-999.0)

    arms = {
        "base": [],
        "struct_generic": ns.STRUCT_BANDS,                       # round-1 (lowered NE floor)
        "struct_named": ns.NAMED_FAULT_BANDS + ["dist_fold_hinge", "carbonaceous_host"],
        "struct_groves": ns.GROVES_BANDS + ["dist_fold_hinge", "carbonaceous_host"],
        "struct_sharp_all": ns.SHARP_BANDS,
    }

    cv, cvmeta = make_cv(X_base, y, coords)
    n_pos = int(y.sum())
    n_pos_gems = int(y[mapped].sum())
    n_pos_box = int(y[in_box].sum())
    print(f"\n[{name}] mode={mode} n={len(y)} pos={n_pos} "
          f"(gems-mapped={n_pos_gems}, placer-core={n_pos_box})  "
          f"block={cvmeta['block_size_m']:.0f}m vrange={cvmeta['variogram_range_m']:.0f}m "
          f"GeMS-mapped cells={mapped.mean():.0%}")

    def X_for(cols):
        return X_base if not cols else np.column_stack([X_base, add[cols].to_numpy(np.float32)])

    results: dict[str, dict] = {}
    base = evaluate_arm(X_base, y, coords, cv, masks)
    base.update(cvmeta)
    results["base"] = base
    print(f"  {'base':17s} full={base['auc_full']}  gems={base.get('auc_gems')}  "
          f"placer_core={base.get('auc_placer_core')}")
    for arm, cols in arms.items():
        if arm == "base":
            continue
        r = evaluate_arm(X_for(cols), y, coords, cv, masks)
        for k in ("auc_full", "auc_gems", "auc_placer_core"):
            b = base.get(k)
            r[f"d_{k}"] = None if (r.get(k) is None or b is None) else round(r[k] - b, 3)
        results[arm] = r
        dg = r.get("d_auc_gems")
        gate = "" if r.get("auc_gems") is None else ("  >=0.70 PASS" if r["auc_gems"] >= 0.70 else "")
        print(f"  {arm:17s} full={r['auc_full']}  gems={r.get('auc_gems')} "
              f"(d={'n/a' if dg is None else f'{dg:+.3f}'}){gate}")

    # feature importance of the full sharpened arm (what carries)
    fm = default_rf_factory(seed=RNG)().fit(X_for(ns.SHARP_BANDS), y)
    names = list(base_df.columns) + ns.SHARP_BANDS
    imp = sorted(zip(names, fm.feature_importances_), key=lambda p: -p[1])
    return {
        "name": name, "label_mode": mode, "n": int(len(y)), "n_pos": n_pos,
        "n_pos_gems_mapped": n_pos_gems, "n_pos_placer_core": n_pos_box,
        "frac_cells_gems_mapped": round(float(mapped.mean()), 3),
        "arms": results,
        "feature_importance_sharp_all_top12": [[k, round(float(v), 4)] for k, v in imp[:12]],
    }


def write_csv(datasets: list[dict], path: Path) -> None:
    cols = ["dataset", "label_mode", "arm", "n_features",
            "auc_full", "d_auc_full", "auc_gems", "d_auc_gems",
            "auc_placer_core", "d_auc_placer_core"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for d in datasets:
            for arm, r in d["arms"].items():
                w.writerow([d["name"], d["label_mode"], arm] + [r.get(c) for c in cols[3:]])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building/reusing sharpened structure bands (district 100 m) ...")
    sp = ensure_structure_bands()

    datasets = [
        run_label_set("lode_district_36a", "36a", sp),
        run_label_set("lode_district_round1", "round1", sp),
    ]
    gate = None
    for d in datasets:
        if d["name"] == "lode_district_36a":
            best = max((a.get("auc_gems") or 0.0) for a in d["arms"].values())
            gate = {"best_auc_gems_36a": round(best, 3), "passes_0p70": bool(best >= 0.70)}

    out = {
        "question": "Do typed 36a labels + sharpened structure push district lode "
                    "auc_gems past 0.70 under F1 leak-guarded CV?",
        "gate": gate,
        "scheme": {"dead_zone_m": 1000.0, "n_folds": 10,
                   "block_sizing": "2x base-model residual-variogram range (F1), fixed across arms",
                   "estimator": "RandomForest(300, balanced, seed=42)", "fold_strategy": "contiguous"},
        "labels": {"source": "ARDF Cox-Singer model_code (data/raw/nome_mpm/ardf_nome.geojson)",
                   "36a": "low-sulfide Au-quartz vein (incl. questioned 36a?)",
                   "round1": "model 36/27/22 OR lode/vein/skarn/greisen keyword"},
        "structure": {"source": "Nome mining district GeMS, Alaska DGGS PDF 94-39",
                      "named_faults": ns.NAMED_FAULTS,
                      "ne_floor_deg": ns.NE_RANGE[0],
                      "splay_near_trunk_m": ns.SPLAY_NEAR_TRUNK_M,
                      "fold_axis": "Twin Mountain Anticline (only named fold in this GeMS; "
                                   "Mount Distin syncline is regional, not mapped here)",
                      "albion_fault": "deposit-scale, NOT in this GeMS; not digitized this round"},
        "auc_keys": {"auc_full": "whole-extent OOF AUC (mixes structure + coverage proxy)",
                     "auc_gems": "OOF AUC on GeMS-mapped cells (the clean structure test / gate)",
                     "auc_placer_core": "in-box subset (restriction-of-range alarm)"},
        "datasets": datasets,
    }
    (OUT_DIR / "lode_structure_sharpen.json").write_text(json.dumps(out, indent=2, default=str))
    write_csv(datasets, OUT_DIR / "lode_structure_sharpen.csv")
    print(f"\nGATE: {gate}")
    print(f"wrote {OUT_DIR / 'lode_structure_sharpen.json'}")
    print(f"wrote {OUT_DIR / 'lode_structure_sharpen.csv'}")


if __name__ == "__main__":
    main()
