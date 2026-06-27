"""Bedrock-contact drift-on-beach proxy: does it add over the placer base?

Round 4B reopened the drift-on-beach lever with a precise Nome-town source,
USGS MF-247 (Hummel 1962, Nome C-1 quad, 1:63,360). MF-247 maps the whole
coastal plain as one undifferentiated "Unconsolidated deposits" unit, so there
is no discrete drift-vs-beach edge to trace; the one digitizable boundary is the
bedrock (Nome Group) / Quaternary-cover contact, the inland edge of the
drift-and-beach plain. That contact is a placer covariate: the gold-bearing
drift was shed from the bedrock uplands and reworked seaward, so the raised
strandlines crossing the contact mark where reworked drift gold concentrates.

The contact line here is taken from the digital GeMS SIM 3131 Seward Peninsula
geologic map (1:500,000): the inland (bedrock-facing) boundary of the coastal-
plain Qs polygon. MF-247 (1:63,360) is the precise source and was georeferenced
to EPSG:3338 (scripts.nome_placer.mf247_georeference, RMS ~20 m) for the manual
digitizing precision upgrade and for the cross-check; headless, the contact is
sourced from the clean digital SIM 3131 vector and confirmed to follow the
MF-247-mapped contact near the placer ground. The strandline-crossing feature
self-restricts to the coast: the low raised-beach bands only meet the contact
near the shoreline, so the 1:500k contact's inland convolution never enters the
intersection, and the placer positives sit far seaward of the contact, so the
GeMS-vs-MF-247 precision gap (~100-400 m) cannot move the verdict.

  contact      inland boundary of the SIM 3131 coastal-plain Qs polygon (the
               bedrock / Quaternary-cover contact), clipped to the Nome C-1
               footprint (+1.5 km), saved as a deliverable vector.
  strandline   cells on the IfSAR DEM within +/-2.5 m of a raised stand
               (Second +11.58 m, Third +21.34 m, Fourth +36.58 m NAVD88),
               DEM datum offset +1.4 m. Same build as the round-3 drift-on-beach.
  intersection cells BOTH on a strandline AND on the contact (rasterized at
               5 m). Feature = distance to the nearest such cell, defined only
               inside the C-1 footprint (NODATA elsewhere). Secondary arm tests
               plain distance-to-contact, the proxy the brief named.

A null is the expected, final result and a fine one.

Run: PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_bedrock_contact_cv
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof, subset_auc
from scripts.nome_placer.f1_leak_guarded_rebaseline import load_placer
from scripts.nome_placer.newlayers_bootstrap import N_BOOT, boot_delta
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, sample_bands

RNG = 42
FT_TO_M = 0.3048
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
SIM3131 = Path("data/raw/sim3131_seward/nmgeol/nmgeolp.shp")
STANDS_M = [38 * FT_TO_M, 70 * FT_TO_M, 120 * FT_TO_M]  # Second, Third, Fourth (NAVD88)
DEM_OFFSET_M = 1.4
HALF_M = 2.5
# Nome C-1 quad (NAD27 graticule corners) -> the MF-247 footprint
C1_CORNERS_LL = [(-165.5, 64.75), (-165.0, 64.75), (-165.0, 64.50), (-165.5, 64.50)]
OUT_DIR = Path("data/derived/nome_placer/bedrock_contact")
VEC_PATH = OUT_DIR / "nome_bedrock_quaternary_contact_3338.geojson"
NODATA = -1.0


def _c1_quad() -> Polygon:
    pts = []
    for lon, lat in C1_CORNERS_LL:
        o = subprocess.run(["gdaltransform", "-s_srs", "EPSG:4267", "-t_srs", "EPSG:3338"],
                           input=f"{lon} {lat}\n", capture_output=True, text=True, check=True)
        x, y, *_ = o.stdout.split()
        pts.append((float(x), float(y)))
    return Polygon(pts)


def build_contact_vector() -> tuple[gpd.GeoDataFrame, Polygon]:
    """Inland edge of the SIM 3131 coastal-plain Qs polygon, clipped to C-1 (+1.5 km)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad = _c1_quad()
    g = gpd.read_file(SIM3131).to_crs("EPSG:3338")
    qs = g[g["LABEL"] == "Qs"]
    coastal = qs.loc[qs.geometry.area.idxmax()].geometry        # the 1709 km2 coastal plain
    bedrock = unary_union(g[~g["LABEL"].isin(["Qs", "water"])].geometry.values)
    contact = coastal.boundary.intersection(bedrock.buffer(30))  # exclude the seaward water coast
    contact_c1 = contact.intersection(quad.buffer(1500))
    gdf = gpd.GeoDataFrame({"name": ["nome_bedrock_quaternary_contact"],
                            "source": ["GeMS SIM 3131 (1:500k); precise source MF-247 1:63360"]},
                           geometry=[contact_c1], crs="EPSG:3338")
    gdf.to_file(VEC_PATH, driver="GeoJSON")
    return gdf, quad


def build_feature(contact_geom, quad: Polygon) -> tuple[Path, Path, dict]:
    """dist-to-(strandline x contact) and dist-to-contact on the DEM grid; inside C-1 only."""
    with rasterio.open(DEM) as ds:
        elev = ds.read(1).astype(np.float32)
        nod = ds.nodata
        T, res, profile = ds.transform, abs(ds.transform.a), ds.profile.copy()
        shape = (ds.height, ds.width)
    elev = np.where((elev == nod) | ~np.isfinite(elev), np.nan, elev)

    strand = np.zeros(shape, bool)
    for c in STANDS_M:
        strand |= np.isfinite(elev) & (np.abs(elev - (c + DEM_OFFSET_M)) <= HALF_M)

    contact_cells = rasterize([(contact_geom, 1)], out_shape=shape, transform=T,
                              fill=0, dtype="uint8", all_touched=True).astype(bool)
    c1_mask = rasterize([(quad, 1)], out_shape=shape, transform=T, fill=0,
                        dtype="uint8", all_touched=True).astype(bool)

    inter = strand & contact_cells
    dist_int = (distance_transform_edt(~inter) * res).astype(np.float32) if inter.any() \
        else np.full(shape, NODATA, np.float32)
    dist_con = (distance_transform_edt(~contact_cells) * res).astype(np.float32) if contact_cells.any() \
        else np.full(shape, NODATA, np.float32)
    dist_int = np.where(c1_mask, dist_int, NODATA).astype(np.float32)
    dist_con = np.where(c1_mask, dist_con, NODATA).astype(np.float32)

    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    p_int = OUT_DIR / "dist_strandline_x_contact.tif"
    p_con = OUT_DIR / "dist_to_contact.tif"
    for arr, p in ((dist_int, p_int), (dist_con, p_con)):
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(arr, 1)
    meta = {"n_strandline_cells": int(strand.sum()), "n_contact_cells": int(contact_cells.sum()),
            "n_c1_footprint_cells": int(c1_mask.sum()), "n_intersection_cells": int(inter.sum()),
            "stand_elevs_m_navd88": [round(c, 2) for c in STANDS_M], "dem_offset_m": DEM_OFFSET_M,
            "half_band_m": HALF_M, "contact_length_km_in_c1": round(contact_geom.length / 1000, 1)}
    print(f"strandline cells={meta['n_strandline_cells']}  contact cells={meta['n_contact_cells']}  "
          f"C-1 cells={meta['n_c1_footprint_cells']}  intersection cells={meta['n_intersection_cells']}")
    return p_int, p_con, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf, quad = build_contact_vector()
    contact_geom = gdf.geometry.iloc[0]
    p_int, p_con, build_meta = build_feature(contact_geom, quad)

    base_df, y, coords, names, _ = load_placer()
    X_base = base_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    feats = sample_bands({"dist_int": p_int, "dist_con": p_con}, ex, ny)
    in_c1 = (feats["dist_int"] > NODATA).to_numpy()

    def fill(col):
        v = feats[col].fillna(NODATA).to_numpy(np.float32)
        return np.where(v <= NODATA, -999.0, v)
    f_int, f_con = fill("dist_int"), fill("dist_con")

    cv, cvmeta = make_cv(X_base, y, coords)

    def oof(extra):
        XX = X_base if extra is None else np.column_stack([X_base, extra])
        return spatial_cv_oof(XX.astype(np.float32), y, coords, cv,
                              model_factory=default_rf_factory(seed=RNG))
    oof_base = oof(None)
    oof_int = oof(f_int)
    oof_con = oof(f_con)

    masks = {"full": np.ones(len(y), bool), "in_c1": in_c1}
    res = {}
    for arm, o in (("base", oof_base), ("strandline_x_contact", oof_int), ("dist_to_contact", oof_con)):
        res[arm] = {f"auc_{m}": subset_auc(y, o, mask=mk) for m, mk in masks.items()}
    boots = {
        "strandline_x_contact_full": boot_delta(y, oof_base, oof_int, masks["full"]),
        "strandline_x_contact_in_c1": boot_delta(y, oof_base, oof_int, masks["in_c1"]),
        "dist_to_contact_full": boot_delta(y, oof_base, oof_con, masks["full"]),
        "dist_to_contact_in_c1": boot_delta(y, oof_base, oof_con, masks["in_c1"]),
    }

    n_pos = int(y.sum())
    n_pos_c1 = int(y[in_c1].sum())
    print(f"\nplacer n={len(y)} pos={n_pos}  inside C-1: {int(in_c1.sum())} cells, {n_pos_c1} positives  "
          f"n_boot={N_BOOT}")
    for arm in ("base", "strandline_x_contact", "dist_to_contact"):
        print(f"  {arm:22s} auc_full={res[arm]['auc_full']}  auc_in_c1={res[arm]['auc_in_c1']}")
    for k, b in boots.items():
        print(f"  marginal {k:32s} d={b.get('point')} CI={b.get('ci95')} "
              f"P(d>0)={b.get('p_gt_0')} (n_pos={b.get('n_pos')})")

    out = {
        "question": "Does a bedrock-contact drift-on-beach proxy (strandline x bedrock/Quaternary "
                    "contact, and plain distance-to-contact) add over the placer base under F1 CV?",
        "contact_source": "GeMS SIM 3131 (1:500k) inland edge of coastal-plain Qs polygon; precise "
                          "source MF-247 (Hummel 1962, 1:63,360) georeferenced to EPSG:3338 (RMS ~20 m) "
                          "for cross-check + manual-digitizing precision upgrade.",
        "data_limit": "MF-247 maps the coastal plain as one undifferentiated unit (no drift-vs-beach "
                      "edge); the bedrock/Quaternary contact is the digitizable boundary. The strandline "
                      "crossing self-restricts to the coast, so the contact's inland convolution and the "
                      "GeMS-vs-MF-247 precision gap do not affect the verdict. Feature defined inside the "
                      "Nome C-1 footprint only.",
        "build": build_meta,
        "scheme": {**cvmeta, "estimator": "RandomForest(300, balanced, seed=42)", "n_boot": N_BOOT},
        "n": int(len(y)), "n_pos": n_pos, "n_pos_in_c1": n_pos_c1,
        "arms": res, "marginals": boots,
        "contact_vector": str(VEC_PATH),
    }
    (OUT_DIR / "placer_bedrock_contact.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'placer_bedrock_contact.json'}")
    print(f"wrote {VEC_PATH}")


if __name__ == "__main__":
    main()
