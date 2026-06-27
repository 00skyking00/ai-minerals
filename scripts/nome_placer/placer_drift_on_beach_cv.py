"""Drift-on-beach intersection placer feature: does it add over the placer base?

The Nome raised-beach placer model holds that gold concentrates where a raised
strandline crosses gold-bearing glacial drift: the beach reworks and reconcentrates
the drift's gold. PR #33 flagged this strandline x drift intersection as the one
genuinely-new beach-line lever not already in the served base (which scores
strandline elevation, abrasion platform and buried-beach proximity but not the
drift overlay). This builds it and grades the marginal AUC.

  strandline   cells on the inbox IfSAR DEM (25 m) within +/-2.5 m of a raised
               stand elevation (Second +11.58 m, Third +21.34 m, Fourth +36.58 m
               NAVD88), DEM datum offset +1.4 m. Same definition as the PR #33
               build_strandline_on_dem.
  drift        AOF 125 (Riehle et al. 1981, surficial geology Tolstoi Pt -> Cape
               Nome, DOI 10.14509/42). IMPORTANT DATA LIMIT: AOF 125 maps no
               discrete glacial-drift unit. Qu/Qud "undifferentiated deposits"
               (colluvium, eolian, lagoonal/marine, fluvial; Qud a melt-feature
               physiography) are the closest proxy for the glacially-influenced
               upland cover; the beach (Qba/Qbv), alluvium (Qal), intertidal
               (Qif) and bedrock (pQb) units are excluded. So "drift" here is a
               PROXY, and the narrow coastal strip largely excludes the inland
               drift sheet. Both limits are reported, not papered over.
  intersection cells that are BOTH on a strandline AND on the drift proxy. The
               feature is distance to the nearest such cell, defined only inside
               the AOF 125 footprint (NODATA elsewhere): the coverage limit the
               coordinator asked to handle explicitly.

A null is a fine result and the expected one given the data limit.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.placer_drift_on_beach_cv
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof, subset_auc
from scripts.nome_placer.f1_leak_guarded_rebaseline import load_placer
from scripts.nome_placer.newlayers_bootstrap import N_BOOT, boot_delta
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, sample_bands

RNG = 42
FT_TO_M = 0.3048
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
AOF125 = Path("data/raw/nome_surficial_aof125/shp/"
              "tolstoi_point_cape_nome_surficial-open/GM_MapUnitPolys.shp")
DRIFT_UNITS = {"Qu", "Qud"}          # AOF 125 undifferentiated deposits (drift proxy)
STANDS_M = [38 * FT_TO_M, 70 * FT_TO_M, 120 * FT_TO_M]  # Second, Third, Fourth (NAVD88)
DEM_OFFSET_M = 1.4                    # IfSAR-minus-NAVD88 (PR #33)
HALF_M = 2.5
OUT_DIR = Path("data/derived/nome_placer/drift_on_beach")
NODATA = -1.0


def build_drift_on_beach() -> tuple[Path, dict]:
    """Distance-to-(strandline x drift) on the inbox grid; defined inside AOF 125 only."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DEM) as ds:
        elev = ds.read(1).astype(np.float32)
        nod = ds.nodata
        T, res, profile = ds.transform, abs(ds.transform.a), ds.profile.copy()
        shape = (ds.height, ds.width)
    elev = np.where((elev == nod) | ~np.isfinite(elev), np.nan, elev)

    strand = np.zeros(shape, bool)
    for c in STANDS_M:
        strand |= np.isfinite(elev) & (np.abs(elev - (c + DEM_OFFSET_M)) <= HALF_M)

    mup = gpd.read_file(AOF125).to_crs("EPSG:3338")
    ucol = "MapUnit" if "MapUnit" in mup else [c for c in mup.columns if c.lower() == "mapunit"][0]
    drift = mup[mup[ucol].astype(str).isin(DRIFT_UNITS)]
    drift_mask = rasterize([(g, 1) for g in drift.geometry if g and not g.is_empty],
                           out_shape=shape, transform=T, fill=0, dtype="uint8").astype(bool)
    cover_mask = rasterize([(g, 1) for g in mup.geometry if g and not g.is_empty],
                           out_shape=shape, transform=T, fill=0, dtype="uint8").astype(bool)

    b3 = mup.to_crs("EPSG:3338").total_bounds
    gx0, gx1 = T.c, T.c + shape[1] * T.a
    overlaps_grid = bool(max(b3[0], min(gx0, gx1)) < min(b3[2], max(gx0, gx1)))

    inter = strand & drift_mask
    if inter.any():
        dist = (distance_transform_edt(~inter) * res).astype(np.float32)
    else:
        dist = np.full(shape, NODATA, np.float32)
    dist = np.where(cover_mask, dist, NODATA).astype(np.float32)  # defined inside AOF 125 only

    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    out = OUT_DIR / "dist_drift_on_beach.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(dist, 1)
    meta = {"n_strandline_cells": int(strand.sum()), "n_drift_cells": int(drift_mask.sum()),
            "n_aof125_cover_cells": int(cover_mask.sum()), "n_intersection_cells": int(inter.sum()),
            "drift_units": sorted(DRIFT_UNITS), "stand_elevs_m_navd88": [round(c, 2) for c in STANDS_M],
            "dem_offset_m": DEM_OFFSET_M, "half_band_m": HALF_M,
            "aof125_bounds_3338": [round(v) for v in b3],
            "placer_grid_bounds_3338": [round(gx0), round(T.f + shape[0] * T.e), round(gx1), round(T.f)],
            "aof125_overlaps_placer_grid": overlaps_grid,
            "coverage_finding": ("AOF 125 (Tolstoi Pt -> Cape Nome) lies EAST of the placer in-box; "
                                 "zero overlap with the grid, so the drift-on-beach feature is undefined "
                                 "across the whole placer model" if not overlaps_grid else
                                 "AOF 125 partially overlaps the placer grid")}
    print(f"strandline cells={meta['n_strandline_cells']}  drift(Qu/Qud) cells={meta['n_drift_cells']}  "
          f"AOF125 cover cells={meta['n_aof125_cover_cells']}  intersection cells={meta['n_intersection_cells']}")
    return out, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dob_path, build_meta = build_drift_on_beach()

    base_df, y, coords, names, _ = load_placer()
    X_base = base_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    dob = sample_bands({"dist_drift_on_beach": dob_path}, ex, ny)["dist_drift_on_beach"]
    in_cover = (dob > NODATA).to_numpy()  # within AOF 125 footprint (feature defined)
    dob_filled = dob.fillna(NODATA).to_numpy(np.float32)
    dob_filled = np.where(dob_filled <= NODATA, -999.0, dob_filled)

    cv, cvmeta = make_cv(X_base, y, coords)

    def oof(with_dob):
        XX = np.column_stack([X_base, dob_filled]) if with_dob else X_base
        return spatial_cv_oof(XX.astype(np.float32), y, coords, cv, model_factory=default_rf_factory(seed=RNG))

    oof_base = oof(False)
    oof_dob = oof(True)

    masks = {"full": np.ones(len(y), bool), "in_cover": in_cover}
    res = {}
    for arm, o in (("base", oof_base), ("drift_on_beach", oof_dob)):
        res[arm] = {f"auc_{m}": subset_auc(y, o, mask=mk) for m, mk in masks.items()}
    boot_full = boot_delta(y, oof_base, oof_dob, masks["full"])
    boot_cover = boot_delta(y, oof_base, oof_dob, masks["in_cover"])

    n_pos = int(y.sum()); n_pos_cover = int(y[in_cover].sum())
    print(f"\nplacer n={len(y)} pos={n_pos}  in AOF125 cover={int(in_cover.sum())} cells, "
          f"{n_pos_cover} positives  n_boot={N_BOOT}")
    print(f"  base           auc_full={res['base']['auc_full']:.4f} auc_in_cover={res['base']['auc_in_cover']}")
    print(f"  drift_on_beach auc_full={res['drift_on_beach']['auc_full']:.4f} "
          f"auc_in_cover={res['drift_on_beach']['auc_in_cover']}")
    print(f"  marginal (full):     d={boot_full['point']} CI={boot_full['ci95']} P(d>0)={boot_full['p_gt_0']}")
    print(f"  marginal (in_cover): d={boot_cover.get('point')} CI={boot_cover.get('ci95')} "
          f"P(d>0)={boot_cover.get('p_gt_0')} (n_pos={boot_cover.get('n_pos')})")

    out = {
        "question": "Does a drift-on-beach intersection feature (strandline x AOF 125 drift proxy) "
                    "add over the placer base under F1 leak-guarded CV?",
        "data_limit": "AOF 125 maps no discrete glacial-drift unit; Qu/Qud undifferentiated deposits "
                      "are a proxy, and the coastal strip largely excludes the inland drift sheet. "
                      "The feature is defined only inside the AOF 125 footprint.",
        "build": build_meta,
        "scheme": {**cvmeta, "estimator": "RandomForest(300, balanced, seed=42)", "n_boot": N_BOOT},
        "n": int(len(y)), "n_pos": n_pos, "n_pos_in_aof125_cover": n_pos_cover,
        "arms": res,
        "marginal_full": boot_full,
        "marginal_in_cover": boot_cover,
    }
    (OUT_DIR / "placer_drift_on_beach.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'placer_drift_on_beach.json'}")


if __name__ == "__main__":
    main()
