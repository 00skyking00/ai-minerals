"""New-layers re-baseline: do the 400 m DIGHEM geophysics and the targeted
GeMS structure/lithology features move the Nome models under leak-guarded CV?

Re-grades the placer and lode presence/background models under F1's leak-guarded
spatial CV (``ai_minerals.spatial_cv``), adding two families of acquired layer:

  geophysics  1993 Nome DIGHEM-V, 400 m lines (``ai_minerals.data.dighem``):
              residual magnetics + derivatives, EM apparent resistivity. The
              coordinator's first headline: does the 400 m magnetics give the
              LODE model signal the ~1 km statewide composite (``akmag_*``,
              already in the base features) could not?
  structure   Nome mining district GeMS (``ai_minerals.data.nome_structure``):
              distance to NE- and NW-trending faults/lineaments (kept separate),
              distance to fold hinges, and a graphitic-host lithology flag. The
              addendum's targeted orogenic-gold control (Rock Creek / Albion
              fault; Piekenbrock & Odden 2009). Lode datasets only.

Folds are sized ONCE per dataset from the base model's residual variogram (F1
recipe, 2x range) and reused across every arm, so each arm's marginal AUC delta
is attributable to the added features, not to a shifted fold geometry.

Both new layer families have a footprint that correlates with central Nome where
occurrences cluster, so two AUCs are reported for every masked subset:

  auc_full     pooled OOF AUC over the whole modelled extent. Out-of-footprint
               cells carry the -999 sentinel, so this mixes real geophysical /
               structural signal with a coverage proxy.
  auc_dighem   OOF AUC restricted to cells the DIGHEM survey flew.
  auc_gems     OOF AUC restricted to cells inside the GeMS mapped footprint
               (the clean test for the structure features).
  auc_both     restricted to flown AND mapped cells.

For lode_district a placer_core subset is the F1 restriction-of-range alarm. The
marginal deltas that answer the two headlines are, on lode_district:
  geophys: auc_dighem(geophys) - auc_dighem(base)
  struct : auc_gems(struct)    - auc_gems(base)

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.newlayers_geophys_rebaseline
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ai_minerals.data import dighem, nome_structure
from ai_minerals.spatial_cv import (
    LeakGuardedSpatialCV,
    block_size_from_range,
    default_rf_factory,
    residual_variogram,
    spatial_cv_oof,
    subset_auc,
)
from scripts.nome_placer.f1_leak_guarded_rebaseline import (
    load_lode_district,
    load_lode_inbox,
    load_placer,
)

RNG = 42
N_FOLDS = 10
DEAD_ZONE_M = 1000.0
INBOX_TPL = Path("data/derived/nome_placer/prospectivity_v1p5/"
                 "nome_placer_prospectivity_v1p5_v3p1_3338.tif")
DIST_TPL = Path("data/derived/nome_placer/covariates_mpm/_district_grid_template_3338.tif")
DIGHEM_DIR = Path("data/derived/nome_geophys/dighem1993")
STRUCT_DIR = Path("data/derived/nome_geophys/nome_structure")
OUT_DIR = Path("data/derived/nome_placer/newlayers_rebaseline")


def ensure_dighem() -> dict[str, dict[str, Path]]:
    """Build (or reuse) DIGHEM band GeoTIFFs on the in-box and district grids."""
    grids = {}
    for tag, tmpl in (("inbox_25m", INBOX_TPL), ("district_100m", DIST_TPL)):
        out = DIGHEM_DIR / tag
        if all((out / f"dighem_{b}.tif").exists() for b in dighem.DIGHEM_BANDS):
            grids[tag] = {b: out / f"dighem_{b}.tif" for b in dighem.DIGHEM_BANDS}
        else:
            grids[tag] = dighem.build_dighem_bands(tmpl, out)
    return grids


def ensure_structure() -> dict[str, dict[str, Path]]:
    """Build (or reuse) GeMS structure/lithology GeoTIFFs on both grids."""
    bands = nome_structure.STRUCT_BANDS + ["gems_extent"]
    grids = {}
    for tag, tmpl in (("inbox_25m", INBOX_TPL), ("district_100m", DIST_TPL)):
        out = STRUCT_DIR / tag
        if all((out / f"struct_{b}.tif").exists() for b in bands):
            grids[tag] = {b: out / f"struct_{b}.tif" for b in bands}
        else:
            grids[tag] = nome_structure.build_structure_bands(tmpl, out)
    return grids


def sample_bands(paths: dict[str, Path], ex: np.ndarray, ny: np.ndarray) -> pd.DataFrame:
    """Sample each band at the model coords; the band nodata -> NaN."""
    cols = {}
    pts = list(zip(ex, ny))
    for band, p in paths.items():
        with rasterio.open(p) as ds:
            a = np.asarray(list(ds.sample(pts)), float)[:, 0]
            nod = ds.nodata
        cols[band] = np.where(a == nod, np.nan, a)
    return pd.DataFrame(cols)


def make_cv(X_base: np.ndarray, y: np.ndarray, coords: np.ndarray) -> tuple[LeakGuardedSpatialCV, dict]:
    """Size the leak-guarded folds ONCE from the base model (F1 2x-range recipe)."""
    factory = default_rf_factory(seed=RNG)
    vfit = residual_variogram(X_base, y, coords, model_factory=factory, seed=RNG)
    bs = block_size_from_range(vfit.range_m, multiplier=2.0)
    ext = coords.max(0) - coords.min(0)
    cap = float(np.sqrt(max(ext.prod(), 1.0) / (2.0 * N_FOLDS)))
    if bs > cap:
        bs = float(np.ceil(cap / 100.0) * 100.0)
    cv = LeakGuardedSpatialCV(bs, n_folds=N_FOLDS, dead_zone_m=DEAD_ZONE_M,
                              seed=RNG, strategy="contiguous")
    return cv, {"block_size_m": bs, "variogram_range_m": round(vfit.range_m, 1)}


def evaluate_arm(X: np.ndarray, y: np.ndarray, coords: np.ndarray,
                 cv: LeakGuardedSpatialCV, masks: dict[str, np.ndarray]) -> dict:
    oof = spatial_cv_oof(X.astype(np.float32), y, coords, cv,
                         model_factory=default_rf_factory(seed=RNG))
    out = {"auc_full": subset_auc(y, oof), "n_features": int(X.shape[1])}
    for name, m in masks.items():
        out[f"auc_{name}"] = subset_auc(y, oof, mask=m)
    return out


def run_dataset(name: str, loader, dighem_paths: dict[str, Path],
                struct_paths: dict[str, Path] | None = None, *,
                extra_subsets: dict[str, np.ndarray] | None = None) -> dict:
    X_df, y, coords, names, _ = loader()
    X_base = X_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    dig = sample_bands(dighem_paths, ex, ny)
    covered = (dig["mag_residual"] > -998.0).to_numpy() if "mag_residual" in dig else \
        dig.iloc[:, 0].notna().to_numpy()
    masks: dict[str, np.ndarray] = {"dighem": covered}

    add = dig.copy()
    arms = {"base": [], "mag": dighem.MAG_BANDS, "res": dighem.RES_BANDS,
            "geophys": dighem.DIGHEM_BANDS}
    if struct_paths is not None:
        st = sample_bands(struct_paths, ex, ny)
        mapped = (st["gems_extent"] > 0.5).to_numpy()
        masks["gems"] = mapped
        masks["both"] = covered & mapped
        add = pd.concat([dig, st[nome_structure.STRUCT_BANDS]], axis=1)
        arms["struct"] = nome_structure.STRUCT_BANDS
        arms["geophys_struct"] = dighem.DIGHEM_BANDS + nome_structure.STRUCT_BANDS
    add = add.fillna(-999.0)
    if extra_subsets:
        masks.update(extra_subsets)

    cv, cvmeta = make_cv(X_base, y, coords)
    n_pos = int(y.sum())
    print(f"\n[{name}] n={len(y)} pos={n_pos}  block={cvmeta['block_size_m']:.0f}m "
          f"vrange={cvmeta['variogram_range_m']:.0f}m  "
          f"DIGHEM-covered={covered.mean():.0%}"
          + (f"  GeMS-mapped={masks['gems'].mean():.0%}" if "gems" in masks else ""))

    def X_for(cols: list[str]) -> np.ndarray:
        return X_base if not cols else np.column_stack([X_base, add[cols].to_numpy(np.float32)])

    results: dict[str, dict] = {}
    base = evaluate_arm(X_base, y, coords, cv, masks)
    base.update(cvmeta)
    results["base"] = base
    clean_key = "auc_gems" if "gems" in masks else "auc_dighem"
    print(f"  {'base':15s} full={base['auc_full']}  dighem={base['auc_dighem']}"
          + (f"  gems={base.get('auc_gems')}" if 'gems' in masks else ""))
    for arm, cols in arms.items():
        if arm == "base":
            continue
        r = evaluate_arm(X_for(cols), y, coords, cv, masks)
        for k in ("auc_full", "auc_dighem", "auc_gems", "auc_both"):
            b = base.get(k)
            r[f"d_{k}"] = None if (r.get(k) is None or b is None) else round(r[k] - b, 3)
        results[arm] = r
        d_clean = r.get(f"d_{clean_key}")
        d_str = "n/a" if d_clean is None else f"{d_clean:+.3f}"
        print(f"  {arm:15s} full={r['auc_full']}  {clean_key[4:]}={r.get(clean_key)} (d={d_str})")

    out = {"name": name, "n": int(len(y)), "n_pos": n_pos,
           "frac_cells_dighem_covered": round(float(covered.mean()), 3),
           "arms": results}
    if "gems" in masks:
        out["frac_cells_gems_mapped"] = round(float(masks["gems"].mean()), 3)
        out["n_pos_gems_mapped"] = int(y[masks["gems"]].sum())
    out["n_pos_dighem_covered"] = int(y[covered].sum())
    return out


def write_csv(datasets: list[dict], path: Path) -> None:
    cols = ["dataset", "arm", "n_features", "auc_full", "d_auc_full",
            "auc_dighem", "d_auc_dighem", "auc_gems", "d_auc_gems",
            "auc_both", "d_auc_both", "auc_placer_core"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for d in datasets:
            for arm, r in d["arms"].items():
                w.writerow([d["name"], arm] + [r.get(c) for c in cols[2:]])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building/reusing DIGHEM + GeMS-structure bands (in-box 25 m, district 100 m) ...")
    dg = ensure_dighem()
    sg = ensure_structure()

    _, _, _, _, d_in_box = load_lode_district()
    datasets = [
        run_dataset("placer_onshore", load_placer, dg["inbox_25m"]),
        run_dataset("lode_inbox", load_lode_inbox, dg["inbox_25m"], sg["inbox_25m"]),
        run_dataset("lode_district", load_lode_district, dg["district_100m"], sg["district_100m"],
                    extra_subsets={"placer_core": d_in_box}),
    ]

    out = {
        "scheme": {"dead_zone_m": DEAD_ZONE_M, "n_folds": N_FOLDS,
                   "block_sizing": "2x base-model residual-variogram range (F1 recipe), "
                                   "fixed across arms so deltas isolate the features",
                   "estimator": "RandomForest(300, balanced, seed=42)",
                   "fold_strategy": "contiguous"},
        "layers": {
            "geophysics": {"source": "1993 Nome DIGHEM-V, Alaska DGGS GPR 2019-11 (DOI 10.14509/30189)",
                           "line_spacing_m": 400, "native_grid_m": 25, "src_crs": dighem.SRC_CRS,
                           "mag_bands": dighem.MAG_BANDS, "res_bands": dighem.RES_BANDS},
            "structure": {"source": "Nome mining district GeMS, Alaska DGGS PDF 94-39",
                          "bands": nome_structure.STRUCT_BANDS,
                          "ne_azimuth_deg": nome_structure.NE_RANGE,
                          "nw_azimuth_deg": nome_structure.NW_RANGE,
                          "graphitic_units": sorted(nome_structure.GRAPHITIC_UNITS),
                          "datum": "NAD27 UTM 3N -> EPSG:3338 via NADCON5 us_noaa_alaska grid"},
        },
        "auc_keys": {
            "auc_full": "whole-extent OOF AUC (mixes signal + coverage proxy)",
            "auc_dighem": "OOF AUC restricted to DIGHEM-flown cells",
            "auc_gems": "OOF AUC restricted to GeMS-mapped cells (clean structure test)",
            "auc_both": "OOF AUC restricted to flown AND mapped cells",
            "auc_placer_core": "lode_district only: F1 restriction-of-range alarm"},
        "skytem_2024_excluded": ("Kigluaik/Seward SkyTEM block sits north of the modelled extent: "
                                 "0% of the placer-core in-box, only a northern district strip; "
                                 "no value for these model grids. Held for a northern extension."),
        "datasets": datasets,
    }
    (OUT_DIR / "newlayers_geophys_rebaseline.json").write_text(json.dumps(out, indent=2, default=str))
    write_csv(datasets, OUT_DIR / "newlayers_geophys_rebaseline.csv")
    print(f"\nwrote {OUT_DIR / 'newlayers_geophys_rebaseline.json'}")
    print(f"wrote {OUT_DIR / 'newlayers_geophys_rebaseline.csv'}")


if __name__ == "__main__":
    main()
