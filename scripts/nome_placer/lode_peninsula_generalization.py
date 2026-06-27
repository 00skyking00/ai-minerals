"""Does the Groves structural control on the lode AUC generalize peninsula-wide?

The round-2 rebuild found auc_gems 0.712 (struct_groves, +0.170 over base) on the
typed 36a labels, but both the labels and the structure features lived inside the
central-Nome district GeMS. The open question: is that a district-specific result,
or does the same NE x NW intersection / splay / fold-hinge / graphitic-host control
predict 36a lode occurrences across the wider Seward Peninsula?

This rebuilds the Groves structure features on a WIDER grid from two bedrock maps
that reach beyond central Nome (SIM 3131 Nome+Solomon quadrangles + RI 2024-7
Casadepaga; ``nome_structure_regional``), disperses the 36a labels truly
peninsula-wide (the full statewide ARDF, not the Nome clip), and re-grades under
the F1 leak-guarded CV.

Because the matched base must build identically on both grids, it is intentionally
terrain-free and aeromag-free: base = SIM 3131 geology one-hot + distance-to-any-
regional-fault. Both are regional, so the central-vs-wider deltas are comparable.
The absolute AUC therefore differs from the 0.712 (which used terrain + aeromag);
what transfers, or does not, is the struct_groves DELTA and whether it clears zero.

Three numbers answer the question:
  * d_wider    struct_groves - base on the wider grid (the generalization delta)
  * auc_east   OOF AUC restricted to cells EAST of the central district (the
               held-out Solomon/Council cluster the central model never saw)
  * d_central  the same arms on the central district grid with the SAME base, so
               wider can be read against central apples-to-apples

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_peninsula_generalization
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from ai_minerals.data import nome_structure_regional as nsr
from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof, subset_auc
from scripts.nome_placer.newlayers_bootstrap import boot_delta
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, sample_bands

RNG = 42
N_BG = 3000
RES = 100.0
CENTRAL_TPL = Path("data/derived/nome_placer/covariates_mpm/_district_grid_template_3338.tif")
CENTRAL_RIGHT_EDGE = -473300.0  # x of the central district grid right edge: "east" is beyond it
ARDF_FULL = Path("data/raw/ardf/ardf/ardf.shp")
STRUCT_DIR = Path("data/derived/nome_geophys/nome_structure_regional")
OUT_DIR = Path("data/derived/nome_placer/lode_peninsula_generalization")

GROVES_ARM = nsr.GROVES_BANDS + ["dist_fold_hinge", "carbonaceous_host"]
GENERIC_ARM = ["dist_ne_fault", "dist_nw_fault", "dist_fold_hinge", "carbonaceous_host"]


def build_wider_template(path: Path) -> Path:
    """Wider grid = bbox(central district grid + RI 2024-7 footprint) padded 12 km,
    clipped to the SIM 3131 nm+so extent, at 100 m, EPSG:3338."""
    if path.exists():
        return path
    with rasterio.open(CENTRAL_TPL) as ds:
        cb = ds.bounds
    ri = nsr._ri_layer("MapUnitPolys").total_bounds
    sim = nsr._sim_polys().total_bounds
    pad = 12000.0
    x0 = max(min(cb.left, ri[0]) - pad, sim[0])
    y0 = max(min(cb.bottom, ri[1]) - pad, sim[1])
    x1 = min(max(cb.right, ri[2]) + pad, sim[2])
    y1 = min(max(cb.top, ri[3]) + pad, sim[3])
    w = int(np.ceil((x1 - x0) / RES))
    h = int(np.ceil((y1 - y0) / RES))
    transform = from_origin(x0, y0 + h * RES, RES, RES)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": w, "height": h,
            "crs": "EPSG:3338", "transform": transform, "nodata": -1.0, "compress": "deflate"}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.zeros((h, w), "float32"), 1)
    print(f"wider grid: [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]  {w}x{h} cells "
          f"({(x1 - x0) / 1000:.0f}x{(y1 - y0) / 1000:.0f} km)")
    return path


def ensure_bands(template: Path, tag: str) -> tuple[dict, dict]:
    sd = STRUCT_DIR / tag
    sbands = nsr.GROVES_BANDS + ["dist_ne_fault", "dist_nw_fault", "dist_fold_hinge",
                                 "carbonaceous_host", "gems_extent"]
    if all((sd / f"struct_{b}.tif").exists() for b in sbands):
        sp = {b: sd / f"struct_{b}.tif" for b in sbands}
    else:
        sp = nsr.build_regional_structure_bands(template, sd)
    bd = STRUCT_DIR / tag / "base"
    if (bd / "geology_unit_code.tif").exists() and (bd / "dist_to_fault.tif").exists():
        bp = {"geology_unit_code": bd / "geology_unit_code.tif", "dist_to_fault": bd / "dist_to_fault.tif"}
    else:
        bp = nsr.build_regional_base_bands(template, bd)
    return sp, bp


def lode_positives(bounds) -> tuple[np.ndarray, np.ndarray]:
    ardf = gpd.read_file(ARDF_FULL).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    lode = ardf[mc.str.startswith("36a")].cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    return lode.geometry.x.to_numpy(), lode.geometry.y.to_numpy()


def background_on_land(template: Path, geo_code_path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(geo_code_path) as ds:
        code = ds.read(1)
        T = ds.transform
    rows, cols = np.where(code > 0)
    pick = np.random.default_rng(RNG).choice(len(rows), size=min(n, len(rows)), replace=False)
    rr, cc = rows[pick], cols[pick]
    xs = T.c + (cc + 0.5) * T.a
    ys = T.f + (rr + 0.5) * T.e
    return xs, ys


def one_hot(code: np.ndarray, legend: dict) -> pd.DataFrame:
    cols = {}
    for k in sorted(legend, key=int):
        if int(k) == 0:
            continue
        cols[f"geo_{legend[k]}"] = (code == int(k)).astype(np.float32)
    return pd.DataFrame(cols)


def build_features(grid_tag: str, template: Path, positives) -> dict:
    sp, bp = ensure_bands(template, grid_tag)
    px, py = positives
    bx, by = background_on_land(template, bp["geology_unit_code"], N_BG)
    ex = np.concatenate([px, bx]); ny = np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    coords = np.column_stack([ex, ny])

    geocode = sample_bands({"geology_unit_code": bp["geology_unit_code"]}, ex, ny)["geology_unit_code"].to_numpy()
    legend = json.loads((bp["geology_unit_code"].parent / "geology_legend.json").read_text())
    base_df = one_hot(np.nan_to_num(geocode, nan=0).astype(int), legend)
    dfault = sample_bands({"dist_to_fault": bp["dist_to_fault"]}, ex, ny)
    base_df["dist_to_fault"] = dfault["dist_to_fault"].fillna(-999.0).to_numpy(np.float32)

    st = sample_bands({b: sp[b] for b in sp if b != "gems_extent"}, ex, ny).fillna(-999.0)
    mapped = (sample_bands({"gems_extent": sp["gems_extent"]}, ex, ny)["gems_extent"] > 0.5).to_numpy()
    east = ex > CENTRAL_RIGHT_EDGE
    return {"y": y, "coords": coords, "base": base_df, "struct": st,
            "mapped": mapped, "east": east, "n_pos": int(y.sum()),
            "n_pos_east": int(y[east].sum()), "n_pos_mapped": int(y[mapped].sum())}


def grade(tag: str, feats: dict) -> dict:
    y, coords = feats["y"], feats["coords"]
    X_base = feats["base"].to_numpy(np.float32)
    add = feats["struct"]
    cv, cvmeta = make_cv(X_base, y, coords)
    masks = {"mapped": feats["mapped"], "east": feats["east"]}

    def oof(cols):
        XX = X_base if not cols else np.column_stack([X_base, add[cols].to_numpy(np.float32)])
        return spatial_cv_oof(XX.astype(np.float32), y, coords, cv, model_factory=default_rf_factory(seed=RNG))

    oof_base = oof([])
    arms = {"base": [], "struct_generic": GENERIC_ARM, "struct_groves": GROVES_ARM}
    res = {}
    for arm, cols in arms.items():
        o = oof(cols)
        row = {"auc_full": subset_auc(y, o), "auc_mapped": subset_auc(y, o, mask=feats["mapped"]),
               "auc_east": subset_auc(y, o, mask=feats["east"]), "n_features": int(X_base.shape[1] + len(cols))}
        if arm != "base":
            b = res["base"]
            for k in ("auc_full", "auc_mapped", "auc_east"):
                row[f"d_{k}"] = None if (row[k] is None or b[k] is None) else round(row[k] - b[k], 3)
            row["bootstrap_mapped"] = boot_delta(y, oof_base, o, feats["mapped"])
            if feats["n_pos_east"] >= 2 and int((~y.astype(bool) & feats["east"]).sum()) >= 2:
                row["bootstrap_east"] = boot_delta(y, oof_base, o, feats["east"])
        res[arm] = row
    print(f"\n[{tag}] n={len(y)} pos={feats['n_pos']} (mapped={feats['n_pos_mapped']}, east={feats['n_pos_east']})  "
          f"block={cvmeta['block_size_m']:.0f}m vrange={cvmeta['variogram_range_m']:.0f}m")
    for arm in arms:
        r = res[arm]
        bs = r.get("bootstrap_mapped") or {}
        be = r.get("bootstrap_east") or {}
        print(f"  {arm:15s} full={r['auc_full']} mapped={r['auc_mapped']} east={r['auc_east']}"
              + ("" if arm == "base" else f"  d_mapped={r.get('d_auc_mapped')} "
                 f"CI={bs.get('ci95')} | d_east={r.get('d_auc_east')} eastCI={be.get('ci95')}"))
    return {"scheme": cvmeta, "n": int(len(y)), **{k: feats[k] for k in ("n_pos", "n_pos_east", "n_pos_mapped")},
            "arms": res}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wider_tpl = build_wider_template(STRUCT_DIR / "_wider_grid_template_3338.tif")

    with rasterio.open(wider_tpl) as ds:
        wb = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    with rasterio.open(CENTRAL_TPL) as ds:
        cb = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

    print("building regional structure + base bands (wider, then central) ...")
    wider = build_features("wider_100m", wider_tpl, lode_positives(wb))
    central = build_features("central_100m", CENTRAL_TPL, lode_positives(cb))

    out = {
        "question": "Does the struct_groves control (auc_gems 0.712 central) generalize "
                    "to dispersed peninsula-wide 36a labels under F1 leak-guarded CV?",
        "design": {
            "base": "SIM 3131 geology one-hot + distance-to-any-regional-fault (terrain-free, "
                    "aeromag-free, so it builds identically on both grids; absolute AUC therefore "
                    "differs from the 0.712 which used terrain+aeromag)",
            "struct_groves": "base + dist_fault_intersection + dist_splay + dist_fold_hinge + carbonaceous_host",
            "structure_sources": "SIM 3131 nm+so (faults ARC_CODE 4/30/60, graphitic LABELs) + "
                                 "RI 2024-7 Casadepaga (faults, 223 fold axes, graphitic units)",
            "splay_caveat": "no named trunks regionally; splay uses a top-quintile fault-length "
                            "trunk proxy (TRUNK_LENGTH_QUANTILE=0.80), not the central named-trunk rule",
            "labels": "ARDF model_code 36a from the FULL statewide ARDF (not the Nome clip)",
            "east_definition": f"cells with x > {CENTRAL_RIGHT_EDGE:.0f} (beyond the central district grid)",
            "cv": "F1 leak_guarded contiguous folds, residual-variogram block sizing, 1 km dead-zone, seed 42",
        },
        "wider": {"bounds_3338": [round(v) for v in wb], **grade("wider", wider)},
        "central_matched": {"bounds_3338": [round(v) for v in cb], **grade("central", central)},
    }
    # verdict helper
    w = out["wider"]["arms"]["struct_groves"]
    out["verdict"] = {
        "wider_struct_groves_auc_mapped": w["auc_mapped"],
        "wider_d_auc_mapped": w.get("d_auc_mapped"),
        "wider_bootstrap_ci": (w.get("bootstrap_mapped") or {}).get("ci95"),
        "wider_p_gt_0": (w.get("bootstrap_mapped") or {}).get("p_gt_0"),
        "wider_auc_east": w["auc_east"],
        "wider_d_auc_east": w.get("d_auc_east"),
        "wider_east_bootstrap_ci": (w.get("bootstrap_east") or {}).get("ci95"),
        "wider_east_p_gt_0": (w.get("bootstrap_east") or {}).get("p_gt_0"),
        "central_matched_d_auc_mapped": out["central_matched"]["arms"]["struct_groves"].get("d_auc_mapped"),
    }
    (OUT_DIR / "lode_peninsula_generalization.json").write_text(json.dumps(out, indent=2, default=str))
    with (OUT_DIR / "lode_peninsula_generalization.csv").open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["grid", "arm", "n_features", "auc_full", "auc_mapped", "d_auc_mapped",
                       "auc_east", "d_auc_east", "ci95_lo", "ci95_hi", "p_gt_0"])
        for grid in ("wider", "central_matched"):
            for arm, r in out[grid]["arms"].items():
                ci = (r.get("bootstrap_mapped") or {}).get("ci95") or [None, None]
                wcsv.writerow([grid, arm, r.get("n_features"), r.get("auc_full"), r.get("auc_mapped"),
                               r.get("d_auc_mapped"), r.get("auc_east"), r.get("d_auc_east"),
                               ci[0], ci[1], (r.get("bootstrap_mapped") or {}).get("p_gt_0")])
    print("\nVERDICT:", json.dumps(out["verdict"], indent=2))
    print(f"wrote {OUT_DIR / 'lode_peninsula_generalization.json'}")


if __name__ == "__main__":
    main()
