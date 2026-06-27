"""Round-2 PLACER beach-line REM: do the LiDAR strandline features move the placer
number over the current geomorph base, under the F1 leak-guarded CV?

The documented Nome placer control is strandline / sea-stand geometry, not
distance-to-lode (genesis synthesis 2026-06-26). This grades the beach-line
covariates built from the NOAA/USACE 2021 1 m bare-earth LiDAR
(``ai_minerals.data.nome_placer_rem``) against the served placer base (the
geomorph POP priors + 5 m IfSAR terrain, the 0.733/0.679 model):

  base        the current placer features (load_placer): beach-line / abrasion /
              terrace / buried-beach POP priors + dem/slope/tpi.
  rem         + the detrended-DEM (REM).
  strand      + proximity to the Second (+38 ft), Third (+70 ft) and shallow
              submerged (-36..-45 ft) strandlines.
  abrasion    + modern abrasion-platform proximity.
  ew          + the E-W shore-parallel ridge feature.
  beachline   + the whole beach-line group (rem + strand + abrasion + ew).

The LiDAR is a narrow coastal strip, so its features only carry signal where it
covers ground; a whole-extent AUC mixes that with a coverage proxy (the F3
lesson). The clean test is ``auc_strip`` -- OOF AUC restricted to LiDAR-covered
cells. Gate: does the beachline group move auc_strip over base?

Folds are sized ONCE from the base model (F1 2x recipe) and reused across arms.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.placer_beachline_rem_cv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from ai_minerals.data import nome_placer_rem as rem
from ai_minerals.spatial_cv import default_rf_factory
from scripts.nome_placer.f1_leak_guarded_rebaseline import load_placer
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, evaluate_arm, sample_bands

RNG = 42
REM_DIR = Path("data/derived/nome_placer/beachline_rem")
OUT_DIR = Path("data/derived/nome_placer/placer_beachline_rem")


def ensure_rem_bands() -> dict[str, Path]:
    bands = rem.REM_BANDS + ["strip_cover"]
    if all((REM_DIR / f"{b}.tif").exists() for b in bands):
        return {b: REM_DIR / f"{b}.tif" for b in bands}
    return rem.build_rem_bands(REM_DIR)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building/reusing beach-line REM bands (5 m strip) ...")
    rp = ensure_rem_bands()

    base_df, y, coords, names, _ = load_placer()
    X_base = base_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    feats = sample_bands({b: rp[b] for b in rem.REM_BANDS}, ex, ny)
    cover = sample_bands({"strip_cover": rp["strip_cover"]}, ex, ny)["strip_cover"]
    in_strip = (cover > 0.5).to_numpy()
    add = feats.fillna(rem.NODATA)
    masks = {"strip": in_strip}

    arms = {
        "base": [],
        "rem": ["rem"],
        "strand": ["dist_second", "dist_third", "dist_submerged_shallow"],
        "abrasion": ["dist_abrasion_platform"],
        "ew": ["ew_ridge"],
        "beachline": rem.REM_BANDS,
    }

    cv, cvmeta = make_cv(X_base, y, coords)
    n_pos = int(y.sum())
    n_pos_strip = int(y[in_strip].sum())
    n_bg_strip = int((in_strip & (y == 0)).sum())
    print(f"\n[placer] n={len(y)} pos={n_pos}  in-strip: pos={n_pos_strip} bg={n_bg_strip}  "
          f"block={cvmeta['block_size_m']:.0f}m vrange={cvmeta['variogram_range_m']:.0f}m  "
          f"LiDAR-covered cells={in_strip.mean():.0%}")

    def X_for(cols):
        return X_base if not cols else np.column_stack([X_base, add[cols].to_numpy(np.float32)])

    results: dict[str, dict] = {}
    base = evaluate_arm(X_base, y, coords, cv, masks)
    base.update(cvmeta)
    results["base"] = base
    print(f"  {'base':12s} full={base['auc_full']}  strip={base.get('auc_strip')}")
    for arm, cols in arms.items():
        if arm == "base":
            continue
        r = evaluate_arm(X_for(cols), y, coords, cv, masks)
        for k in ("auc_full", "auc_strip"):
            b = base.get(k)
            r[f"d_{k}"] = None if (r.get(k) is None or b is None) else round(r[k] - b, 3)
        results[arm] = r
        ds = r.get("d_auc_strip")
        print(f"  {arm:12s} full={r['auc_full']}  strip={r.get('auc_strip')} "
              f"(d_strip={'n/a' if ds is None else f'{ds:+.3f}'})")

    # --- powered arm: raised-strandline proximity on the full-coverage IfSAR DEM
    # (all 65 positives, so auc_full is clean -- no LiDAR-coverage proxy). This
    # grades the strandline control with power, where the 1 m strip cannot. ---
    IFSAR_DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
    IFSAR_DIR = Path("data/derived/nome_placer/beachline_rem/ifsar_strand")
    if all((IFSAR_DIR / f"ifsar_{b}.tif").exists() for b in rem.IFSAR_STRAND_BANDS):
        ip = {b: IFSAR_DIR / f"ifsar_{b}.tif" for b in rem.IFSAR_STRAND_BANDS}
    else:
        ip = rem.build_strandline_on_dem(IFSAR_DEM, IFSAR_DIR, offset_m=1.4, half_m=2.5)
    ifeat = sample_bands(ip, ex, ny).fillna(rem.NODATA)
    X_if = np.column_stack([X_base, ifeat[rem.IFSAR_STRAND_BANDS].to_numpy(np.float32)])
    r_if = evaluate_arm(X_if, y, coords, cv, {})
    b = base.get("auc_full")
    r_if["d_auc_full"] = None if (r_if.get("auc_full") is None or b is None) else round(r_if["auc_full"] - b, 3)
    results["strand_ifsar_fullcover"] = r_if
    _dif = r_if["d_auc_full"]
    _difs = "n/a" if _dif is None else f"{_dif:+.3f}"
    print(f"  {'strand_ifsar':12s} full={r_if['auc_full']} (d_full={_difs}) "
          f"[all {n_pos} positives, full IfSAR coverage]")

    fm = default_rf_factory(seed=RNG)().fit(X_for(rem.REM_BANDS), y)
    fnames = names + rem.REM_BANDS
    imp = sorted(zip(fnames, fm.feature_importances_), key=lambda p: -p[1])
    beach_delta = results["beachline"].get("d_auc_strip")
    ifsar_delta = r_if.get("d_auc_full")

    out = {
        "question": "Do the LiDAR beach-line / strandline features move the placer "
                    "auc_strip over the geomorph base, under F1 leak-guarded CV?",
        "gate": {"d_auc_strip_beachline": beach_delta,
                 "auc_strip_base": base.get("auc_strip"),
                 "auc_strip_beachline": results["beachline"].get("auc_strip"),
                 "moved_positive": bool(beach_delta is not None and beach_delta > 0),
                 "UNDERPOWERED": f"only {n_pos_strip} of {n_pos} placer positives fall in "
                                 f"the 1 m LiDAR strip; the strip auc deltas are noise",
                 "powered_strand_ifsar_d_auc_full": ifsar_delta,
                 "powered_note": "raised-strandline proximity on the full-coverage IfSAR DEM "
                                 "(all positives) marginal over the POP base, which already "
                                 "scores BL/AP/buried_BL at the documented stand elevations"},
        "caveat_n_pos_strip": n_pos_strip,
        "scheme": {"dead_zone_m": 1000.0, "n_folds": 10,
                   "block_sizing": "2x base-model residual-variogram range (F1), fixed across arms",
                   "estimator": "RandomForest(300, balanced, seed=42)", "fold_strategy": "contiguous"},
        "layers": {"source": "NOAA/USACE 2021 NCMP 1 m bare-earth LiDAR, dataset 10207",
                   "native_res_m": 1, "feature_res_m": rem.RES_M, "vdatum": "NAVD88 m",
                   "bands": rem.REM_BANDS,
                   "strandline_elev_ft": {"second": 38, "third": 70, "submerged_shallow": "-36..-45"},
                   "not_in_strip": "Fourth +120ft (above +25.6m ceiling); deep submerged "
                                   "-55/-70/-80ft (below -13.7m floor); drift-on-beach "
                                   "intersection (needs surficial AOF125/RI2024-6 vector)"},
        "auc_keys": {"auc_full": "whole-extent OOF AUC (mixes signal + LiDAR-coverage proxy)",
                     "auc_strip": "OOF AUC on LiDAR-covered cells (the clean test / gate)"},
        "n": int(len(y)), "n_pos": n_pos, "n_pos_strip": n_pos_strip, "n_bg_strip": n_bg_strip,
        "frac_cells_lidar_covered": round(float(in_strip.mean()), 3),
        "arms": results,
        "feature_importance_beachline_top12": [[k, round(float(v), 4)] for k, v in imp[:12]],
    }
    (OUT_DIR / "placer_beachline_rem.json").write_text(json.dumps(out, indent=2, default=str))
    with (OUT_DIR / "placer_beachline_rem.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_features", "auc_full", "d_auc_full", "auc_strip", "d_auc_strip"])
        for arm, r in results.items():
            w.writerow([arm, r.get("n_features"), r.get("auc_full"), r.get("d_auc_full"),
                        r.get("auc_strip"), r.get("d_auc_strip")])
    print(f"\nGATE: beachline d_auc_strip = {beach_delta} (n_pos_strip={n_pos_strip})")
    print(f"wrote {OUT_DIR / 'placer_beachline_rem.json'}")
    print(f"wrote {OUT_DIR / 'placer_beachline_rem.csv'}")


if __name__ == "__main__":
    main()
