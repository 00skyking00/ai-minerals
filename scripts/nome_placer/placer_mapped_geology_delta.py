"""ML step 3 Part A: does independently-mapped geology/structure improve the
leak-guarded PLACER PRESENCE model beyond the pure-geomorph baseline (AUC 0.679)?

This is the F3 presence-CV delta for the mapped physical-geology fields, built to
the binding evaluation contract (portfolio ml_step3_g1_validation_plan_2026-07-12
sections 1 + 2). The fields feed the presence-CV delta ONLY; they never touch the
grade-capture headline model (Tuck supplies the grade labels).

Two independently-mapped sources are tested, at opposite scales:

  OFR 2009-1254   USGS regional bedrock compilation (SIM 3131 Seward Peninsula,
                  1:500,000). Sparse over the ~17x21 km placer core (9 contact
                  arcs, 4 fault arcs). Drawn as a regional compilation, not where
                  anyone dug -> low sampling-effort-confound risk, low power.
                  Fields: distance-to-contact (ARC_CODE 1, NEW here),
                  distance-to-fault (codes 4/30/60, already built), host-rock
                  unit one-hot (already built).
  GeMS PDF 94-39  Alaska DGGS Nome mining district geodatabase. Dense structure
                  network + graphitic-host lithology. It is the mining-district
                  map, so its footprint tracks the occurrences (the confound the
                  review flags). Fields: dist_ne_fault, dist_nw_fault,
                  dist_fold_hinge, carbonaceous_host (already built).

Independence checks run and reported BEFORE the delta is trusted (contract 2 i/ii):
  1. occurrence-density confound  linework / map-footprint density inside vs
                                  outside a 1.5 km placer-occurrence buffer.
  2. fault-field cross-check      mapped distance-to-fault vs the coverage-uniform
                                  aeromag field (akmag) + its horizontal gradient.
  3. variance / distribution      per field over the on-land grid; a near-constant
                                  field is a no-variance null, caught here.

The delta itself reuses the F1 leak-guarded recipe verbatim (load_placer base,
folds sized once from the base residual variogram, 1 km dead zone, contiguous
folds, RandomForest(300, balanced), paired bootstrap AUC delta) so each arm's
marginal is attributable to the added fields, not to a shifted fold geometry.

Run: PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_mapped_geology_delta
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from ai_minerals.data import nome_structure
from ai_minerals.data.adapters.occurrences import kg as kg_occ
from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof, subset_auc
from scripts.nome_placer.f1_leak_guarded_rebaseline import (
    GEO_CODES_INBOX,
    load_lode_inbox,
    load_placer,
    samp,
)
from scripts.nome_placer.newlayers_bootstrap import boot_delta
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, sample_bands

RNG = 42
DD = Path("data/derived/nome_placer")
CM = DD / "covariates_mpm"
V3P1 = DD / "prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif"
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
KG_EXPORT = Path("data/raw/fossick_kg/kg_nome.jsonld")
RAW_GEOL = Path("data/raw/nome_mpm/geol/nmgeol_dd")
STRUCT_DIR = Path("data/derived/nome_geophys/nome_structure/inbox_25m")
OUT_DIR = DD / "mapped_geology_delta"

DIST_CONTACT = CM / "dist_to_contact_nome_3338.tif"
DIST_FAULT = CM / "dist_to_fault_nome_3338.tif"
UNIT_CODE = CM / "geology_unit_code_nome_3338.tif"
AKMAG = CM / "akmag_nome_3338.tif"

CONTACT_CODE = 1               # OFR 2009-1254 ARC_CODE 1 = geologic contact
FAULT_CODES = {4, 30, 60}      # normal / uncertain / concealed (matches builder)
OCC_BUFFER_M = 1500.0          # occurrence-density confound buffer
GEMS_BANDS = nome_structure.STRUCT_BANDS  # ne/nw fault, fold hinge, carbonaceous host
# host-rock domains kept as one-hot columns (drop water + unmapped as reference)
HOST_UNITS = {1: "DOx", 2: "Dcs", 3: "Ocs", 4: "Qs"}


# --------------------------------------------------------------------------- #
# field building
# --------------------------------------------------------------------------- #
def build_dist_to_contact() -> dict:
    """Distance-to-nearest mapped geologic contact (OFR 2009-1254 ARC_CODE 1).

    Distinct from dist_to_fault (fault arcs) and geology_unit_code (polygon
    labels). Padded-halo EDT so a cell's nearest contact may lie just outside the
    grid, matching mpm_build_geology_covariates.
    """
    with rasterio.open(V3P1) as t:
        transform, width, height, crs = t.transform, t.width, t.height, t.crs
    arcs = gpd.read_file(RAW_GEOL / "nm_dda.shp").to_crs("EPSG:3338")
    contacts = arcs[arcs["ARC_CODE"] == CONTACT_CODE]
    px = abs(transform.a)
    pad = int(round(8000.0 / px))
    big = Affine(transform.a, transform.b, transform.c - pad * px,
                 transform.d, transform.e, transform.f + pad * px)
    shapes = [(g, 1) for g in contacts.geometry if g is not None and not g.is_empty]
    mask = rasterize(shapes, out_shape=(height + 2 * pad, width + 2 * pad),
                     transform=big, fill=0, dtype="uint8", all_touched=True)
    dist_big = distance_transform_edt(mask == 0, sampling=px)
    dist = dist_big[pad:pad + height, pad:pad + width].astype("float32")
    DIST_CONTACT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DIST_CONTACT, "w", driver="GTiff", dtype="float32", count=1,
                       width=width, height=height, crs=crs, transform=transform,
                       nodata=np.float32(-1.0), compress="deflate", predictor=2,
                       tiled=True) as o:
        o.write(dist, 1)
        o.update_tags(source="USGS OFR 2009-1254 nm_dda ARC_CODE == 1 (contacts)",
                      method="scipy.ndimage.distance_transform_edt, 25 m sampling")
    return {"n_contact_arcs_total": int(len(contacts)),
            "dist_m_min_mean_max": [round(float(dist.min()), 1),
                                    round(float(dist.mean()), 1),
                                    round(float(dist.max()), 1)]}


def ensure_gems_bands() -> dict[str, Path]:
    paths = {b: STRUCT_DIR / f"struct_{b}.tif" for b in GEMS_BANDS + ["gems_extent"]}
    if not all(p.exists() for p in paths.values()):
        paths = nome_structure.build_structure_bands(V3P1, STRUCT_DIR)
    return paths


# --------------------------------------------------------------------------- #
# occurrences + on-land grid helpers
# --------------------------------------------------------------------------- #
def placer_occurrences() -> gpd.GeoDataFrame:
    """The 65 placer positives inside the AOI (same filter as load_placer)."""
    with rasterio.open(V3P1) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    occ = kg_occ.load(KG_EXPORT).to_crs("EPSG:3338")
    comm = occ["commodity"].astype(str).str.lower()
    code39 = occ["deposit_codes"].apply(
        lambda t: any(str(c).split(":")[-1].startswith("39") for c in t))
    return occ[comm.str.contains("placer") | code39].cx[bl:br, bb:bt].copy()


def onland_grid() -> tuple[np.ndarray, Affine, tuple]:
    """On-land cell mask (DEM valid), the grid transform, and (H, W)."""
    with rasterio.open(V3P1) as t:
        transform, shape = t.transform, (t.height, t.width)
    with rasterio.open(DEM) as ds:
        dem = ds.read(1)
        nod = ds.nodata
    return (dem != nod), transform, shape


def occ_buffer_mask(occ: gpd.GeoDataFrame, transform, shape) -> np.ndarray:
    """Cells within OCC_BUFFER_M of any placer occurrence (raster distance)."""
    occ_cells = rasterize(
        [(g, 1) for g in occ.geometry if g is not None and not g.is_empty],
        out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=True)
    px = abs(transform.a)
    dist = distance_transform_edt(occ_cells == 0, sampling=px)
    return dist <= OCC_BUFFER_M


# --------------------------------------------------------------------------- #
# independence check 1: occurrence-density confound
# --------------------------------------------------------------------------- #
def _line_cell_mask(gdf: gpd.GeoDataFrame, transform, shape) -> np.ndarray:
    shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, bool)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0,
                     dtype="uint8", all_touched=True).astype(bool)


def occurrence_density_confound(occ, land, transform, shape, gems_extent_path) -> dict:
    """Line-cell density inside vs outside the occurrence buffer, on-land.

    density = (line cells that are on-land AND in/out of buffer) / (on-land cells
    in/out of buffer). ratio = density_in / density_out. ratio >> 1 means the
    map's linework is denser where the occurrences are (mapping-where-they-mined).
    A GeMS-footprint arm reports the same ratio for the mapped-extent mask itself,
    the direct coverage-proxy the review warns about.
    """
    buf = occ_buffer_mask(occ, transform, shape)
    in_land = land & buf
    out_land = land & ~buf
    n_in, n_out = int(in_land.sum()), int(out_land.sum())

    arcs = gpd.read_file(RAW_GEOL / "nm_dda.shp").to_crs("EPSG:3338")
    ofr_contacts = arcs[arcs["ARC_CODE"] == CONTACT_CODE]
    ofr_faults = arcs[arcs["ARC_CODE"].isin(FAULT_CODES)]
    oriented = nome_structure.load_oriented_structures()
    gems_ne = gpd.GeoDataFrame(geometry=oriented["dist_ne_fault"], crs="EPSG:3338")
    gems_nw = gpd.GeoDataFrame(geometry=oriented["dist_nw_fault"], crs="EPSG:3338")

    def ratio(mask_cells: np.ndarray) -> dict:
        d_in = float((mask_cells & in_land).sum()) / max(n_in, 1)
        d_out = float((mask_cells & out_land).sum()) / max(n_out, 1)
        r = (d_in / d_out) if d_out > 0 else None
        return {"density_in_buffer": round(d_in, 5), "density_out_buffer": round(d_out, 5),
                "ratio_in_over_out": (round(r, 2) if r is not None else None)}

    out = {
        "buffer_m": OCC_BUFFER_M, "n_occurrences": int(len(occ)),
        "n_onland_in_buffer": n_in, "n_onland_out_buffer": n_out,
        "ofr_contacts": ratio(_line_cell_mask(ofr_contacts, transform, shape)),
        "ofr_faults": ratio(_line_cell_mask(ofr_faults, transform, shape)),
        "gems_ne_faults": ratio(_line_cell_mask(gems_ne, transform, shape)),
        "gems_nw_faults": ratio(_line_cell_mask(gems_nw, transform, shape)),
    }
    with rasterio.open(gems_extent_path) as ds:
        ext = ds.read(1)
        ext_cells = ext > 0.5
    out["gems_mapped_footprint"] = ratio(ext_cells)
    out["note"] = ("ratio >> 1 = linework/footprint denser inside the occurrence "
                   "buffer = sampling-effort confound; ~1 = coverage-uniform.")
    return out


# --------------------------------------------------------------------------- #
# independence check 2: fault field vs coverage-uniform aeromag
# --------------------------------------------------------------------------- #
def fault_vs_akmag(land, transform, shape) -> dict:
    """Do mapped faults align with the aeromag field / its gradient, or diverge?

    A fault with a real magnetic expression sits on an aeromag horizontal
    gradient. If mapped-fault cells show no gradient rise over random on-land
    cells, the mapped fault trace is a surface/exposure line, not a geophysical
    structure (divergence). akmag is the coverage-uniform 1 km composite, so it
    does not care where anyone dug.
    """
    with rasterio.open(AKMAG) as ds:
        ak = ds.read(1).astype(float)
        nod = ds.nodata
    ak = np.where(ak == nod, np.nan, ak)
    with rasterio.open(DIST_FAULT) as ds:
        dfault = ds.read(1).astype(float)
    gy, gx = np.gradient(ak, abs(transform.a))
    grad = np.hypot(gx, gy)  # nT / m

    valid = land & np.isfinite(ak) & np.isfinite(grad)
    # dist-to-fault vs akmag intensity + gradient (spearman over on-land cells)
    sub = valid & (dfault >= 0)
    rho_int = float(spearmanr(dfault[sub], ak[sub]).correlation)
    rho_grad = float(spearmanr(dfault[sub], grad[sub]).correlation)

    px = abs(transform.a)
    fault_cells = valid & (dfault < px * 1.5)          # on / adjacent to a mapped fault
    rng = np.random.default_rng(RNG)
    idx = np.argwhere(valid)
    pick = idx[rng.choice(len(idx), size=min(20000, len(idx)), replace=False)]
    rand_grad = grad[pick[:, 0], pick[:, 1]]
    grad_ratio = (float(np.nanmean(grad[fault_cells])) / float(np.nanmean(rand_grad))
                  if fault_cells.any() else None)
    return {
        "spearman_distfault_vs_akmag_intensity": round(rho_int, 3),
        "spearman_distfault_vs_akmag_gradient": round(rho_grad, 3),
        "akmag_gradient_at_fault_cells_nT_per_m": (round(float(np.nanmean(grad[fault_cells])), 4)
                                                   if fault_cells.any() else None),
        "akmag_gradient_at_random_cells_nT_per_m": round(float(np.nanmean(rand_grad)), 4),
        "gradient_ratio_fault_over_random": (round(grad_ratio, 2) if grad_ratio is not None else None),
        "n_mapped_fault_cells_onland": int(fault_cells.sum()),
        "note": ("ratio ~1 (and near-zero spearman) = mapped faults do NOT track the "
                 "aeromag field: independent lines, but also no geophysical "
                 "corroboration at 1 km resolution over a 17 km AOI."),
    }


# --------------------------------------------------------------------------- #
# independence check 3: variance / distribution per field
# --------------------------------------------------------------------------- #
def field_distributions(land, transform, shape, gems_paths) -> dict:
    def stats_continuous(path: Path, name: str) -> dict:
        with rasterio.open(path) as ds:
            a = ds.read(1).astype(float)
            nod = ds.nodata
        a = np.where(a == nod, np.nan, a)
        v = a[land & np.isfinite(a)]
        if v.size == 0:
            return {"name": name, "n": 0, "near_constant": True}
        q = np.percentile(v, [5, 25, 50, 75, 95])
        std, mean = float(v.std()), float(v.mean())
        cv = std / abs(mean) if mean != 0 else None
        nodata_frac = float((land & ~np.isfinite(a)).sum()) / max(int(land.sum()), 1)
        return {"name": name, "n": int(v.size), "nodata_frac_onland": round(nodata_frac, 3),
                "min": round(float(v.min()), 2), "mean": round(mean, 2),
                "max": round(float(v.max()), 2), "std": round(std, 2),
                "cv_std_over_mean": (round(cv, 3) if cv is not None else None),
                "q05_25_50_75_95": [round(float(x), 1) for x in q],
                "near_constant": bool(std < 1e-6 or (cv is not None and cv < 0.05))}

    def stats_unit(path: Path, name: str, legend: dict) -> dict:
        with rasterio.open(path) as ds:
            a = ds.read(1)
        v = a[land]
        frac = {legend.get(int(k), str(int(k))): round(float((v == k).mean()), 4)
                for k in np.unique(v)}
        n_classes = int((np.array(list(frac.values())) > 0.005).sum())
        return {"name": name, "class_fraction": frac, "n_classes_gt_0.5pct": n_classes,
                "near_constant": bool(max(frac.values()) > 0.98)}

    unit_legend = {1: "DOx", 2: "Dcs", 3: "Ocs", 4: "Qs", 5: "water", 0: "<unmapped>"}
    fields = {
        "dist_to_contact": stats_continuous(DIST_CONTACT, "dist_to_contact"),
        "dist_to_fault": stats_continuous(DIST_FAULT, "dist_to_fault"),
        "akmag": stats_continuous(AKMAG, "akmag"),
        "gems_dist_ne_fault": stats_continuous(gems_paths["dist_ne_fault"], "gems_dist_ne_fault"),
        "gems_dist_nw_fault": stats_continuous(gems_paths["dist_nw_fault"], "gems_dist_nw_fault"),
        "gems_dist_fold_hinge": stats_continuous(gems_paths["dist_fold_hinge"], "gems_dist_fold_hinge"),
        "geology_unit_code": stats_unit(UNIT_CODE, "geology_unit_code", unit_legend),
        "gems_carbonaceous_host": stats_unit(gems_paths["carbonaceous_host"],
                                             "gems_carbonaceous_host", {0: "not_host", 1: "graphitic_host"}),
    }
    return fields


# --------------------------------------------------------------------------- #
# the presence-CV delta
# --------------------------------------------------------------------------- #
def _fill_dist(col: pd.Series) -> np.ndarray:
    v = col.fillna(-999.0).to_numpy(np.float32)
    return np.where(v <= -1.0, -999.0, v)  # -1 nodata -> RF -999 sentinel


def host_onehot(ex, ny) -> pd.DataFrame:
    code = samp(UNIT_CODE, ex, ny, [1], ["u"])["u"].fillna(0).astype(int)
    return pd.DataFrame({f"host_{v}": (code == k).astype(np.float32)
                         for k, v in HOST_UNITS.items()})


def placer_delta(gems_paths) -> dict:
    base_df, y, coords, names, _ = load_placer()
    X_base = base_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    dcontact = _fill_dist(sample_bands({"c": DIST_CONTACT}, ex, ny)["c"])
    dfault = _fill_dist(sample_bands({"f": DIST_FAULT}, ex, ny)["f"])
    host = host_onehot(ex, ny)
    gems = sample_bands({b: gems_paths[b] for b in GEMS_BANDS}, ex, ny).fillna(-999.0)
    gems_mapped = (sample_bands({"e": gems_paths["gems_extent"]}, ex, ny)["e"] > 0.5).to_numpy()

    arms = {
        "dist_to_contact": np.column_stack([X_base, dcontact]),
        "dist_to_fault": np.column_stack([X_base, dfault]),
        "host_rock_onehot": np.column_stack([X_base, host.to_numpy(np.float32)]),
        "gems_structure": np.column_stack([X_base, gems[GEMS_BANDS].to_numpy(np.float32)]),
        "all_ofr_geology": np.column_stack(
            [X_base, dcontact, dfault, host.to_numpy(np.float32)]),
    }
    cv, cvmeta = make_cv(X_base, y, coords)

    def oof(X):
        return spatial_cv_oof(X.astype(np.float32), y, coords, cv,
                              model_factory=default_rf_factory(seed=RNG))
    oof_base = oof(X_base)
    full = np.ones(len(y), bool)
    base_auc = subset_auc(y, oof_base)
    res = {"base_auc_0.679_check": round(float(base_auc), 4),
           "n": int(len(y)), "n_pos": int(y.sum()),
           "gems_mapped_frac_all": round(float(gems_mapped.mean()), 3),
           "gems_mapped_frac_pos": round(float(gems_mapped[y.astype(bool)].mean()), 3),
           "scheme": {**cvmeta, "estimator": "RandomForest(300, balanced, seed=42)",
                      "dead_zone_m": 1000.0, "fold_strategy": "contiguous", "n_boot": 2000},
           "arms": {}}
    for name, X in arms.items():
        o = oof(X)
        res["arms"][name] = {"auc_full": round(float(subset_auc(y, o)), 4),
                             "marginal_full": boot_delta(y, oof_base, o, full)}
        # gems_structure also read on its clean mapped-only subset (the coverage guard)
        if name == "gems_structure":
            res["arms"][name]["marginal_gems_mapped_only"] = boot_delta(
                y, oof_base, o, gems_mapped)
    return res


def lode_delta() -> dict:
    """Cheap lode presence ablation: mapped-geology-in minus terrain-only.

    The lode base already CONTAINS the mapped geology (geol one-hot + dist_fault +
    akmag). Dropping those to dem/slope/tpi isolates what the mapped geology buys
    the lode presence model, the mirror of the placer question.
    """
    Xdf, y, coords, names, _ = load_lode_inbox()
    terrain = [c for c in names if c in ("dem", "slope", "tpi")]
    geol_cols = [c for c in names if c not in terrain]
    X_full = Xdf.to_numpy(np.float32)
    X_terr = Xdf[terrain].to_numpy(np.float32)
    cv, cvmeta = make_cv(X_terr, y, coords)  # folds sized on the smaller (terrain) base

    def oof(X):
        return spatial_cv_oof(X.astype(np.float32), y, coords, cv,
                              model_factory=default_rf_factory(seed=RNG))
    oof_terr, oof_full = oof(X_terr), oof(X_full)
    full = np.ones(len(y), bool)
    return {"n": int(len(y)), "n_pos": int(y.sum()),
            "dropped_geology_cols": geol_cols, "kept_terrain_cols": terrain,
            "auc_terrain_only": round(float(subset_auc(y, oof_terr)), 4),
            "auc_with_mapped_geology": round(float(subset_auc(y, oof_full)), 4),
            "marginal_geology_over_terrain": boot_delta(y, oof_terr, oof_full, full),
            "scheme": {**cvmeta, "note": "arm=with-geology, base=terrain-only; d>0 => geology helps lode"}}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building distance-to-contact (OFR 2009-1254 ARC_CODE 1) ...")
    contact_meta = build_dist_to_contact()
    print(f"  {contact_meta}")
    gems_paths = ensure_gems_bands()

    occ = placer_occurrences()
    land, transform, shape = onland_grid()
    print(f"occurrences={len(occ)}  on-land cells={int(land.sum())}")

    print("independence check 1: occurrence-density confound ...")
    conf = occurrence_density_confound(occ, land, transform, shape, gems_paths["gems_extent"])
    for k in ("ofr_contacts", "ofr_faults", "gems_ne_faults", "gems_nw_faults", "gems_mapped_footprint"):
        print(f"  {k:22s} ratio_in/out = {conf[k]['ratio_in_over_out']}")

    print("independence check 2: fault field vs akmag ...")
    fak = fault_vs_akmag(land, transform, shape)
    print(f"  spearman(distfault,akmag)={fak['spearman_distfault_vs_akmag_intensity']}  "
          f"gradient ratio fault/random={fak['gradient_ratio_fault_over_random']}")

    print("independence check 3: variance / distribution ...")
    dists = field_distributions(land, transform, shape, gems_paths)
    for name, s in dists.items():
        print(f"  {name:24s} near_constant={s.get('near_constant')}")

    print("presence-CV delta: placer ...")
    placer = placer_delta(gems_paths)
    for name, a in placer["arms"].items():
        m = a["marginal_full"]
        print(f"  {name:20s} auc_full={a['auc_full']}  d={m.get('point')} CI={m.get('ci95')} "
              f"P(d>0)={m.get('p_gt_0')}")

    print("presence-CV delta: lode ablation ...")
    lode = lode_delta()
    lm = lode["marginal_geology_over_terrain"]
    print(f"  terrain_only={lode['auc_terrain_only']}  with_geology={lode['auc_with_mapped_geology']}  "
          f"d={lm.get('point')} CI={lm.get('ci95')} P(d>0)={lm.get('p_gt_0')}")

    out = {
        "question": ("Does independently-mapped geology/structure improve the leak-guarded "
                     "PLACER PRESENCE model beyond the pure-geomorph baseline (AUC 0.679)?"),
        "contract": "portfolio ml_step3_g1_validation_plan_2026-07-12 (sections 1 + 2)",
        "sources": {
            "ofr_2009_1254": "USGS OFR 2009-1254 / SIM 3131 Seward Peninsula bedrock (1:500,000); "
                             "regional compilation, sparse over the placer core, low confound risk",
            "gems_pdf_94_39": "Alaska DGGS PDF 94-39 Nome mining district geodatabase; dense structure "
                              "+ graphitic host; mining-district footprint tracks occurrences",
        },
        "fields_built_or_reused": {
            "dist_to_contact": {"path": str(DIST_CONTACT), "status": "NEW", **contact_meta},
            "dist_to_fault": {"path": str(DIST_FAULT), "status": "reused (mpm_build_geology_covariates)"},
            "geology_unit_code": {"path": str(UNIT_CODE), "status": "reused; one-hot host domains"},
            "akmag": {"path": str(AKMAG), "status": "reused (cross-check reference only)"},
            "gems_structure": {"paths": {b: str(gems_paths[b]) for b in GEMS_BANDS},
                               "status": "reused (nome_structure PDF 94-39)"},
        },
        "independence_checks": {
            "occurrence_density_confound": conf,
            "fault_vs_akmag": fak,
            "field_distributions": dists,
        },
        "presence_cv_delta_placer": placer,
        "presence_cv_delta_lode_ablation": lode,
    }
    (OUT_DIR / "placer_mapped_geology_delta.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'placer_mapped_geology_delta.json'}")


if __name__ == "__main__":
    main()
