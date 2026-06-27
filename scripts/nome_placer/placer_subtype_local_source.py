"""Type the Nome placer positives and test for a masked local-source signal.

Round-4B addendum. The pooled placer-presence model does not respond to
distance-to-lode (round 3, a null). Hudson (2006, after Metcalfe & Tuck 1942)
splits the Nome placers into two physically distinct coastal populations:

  abrasion-platform  Monroeville, Intermediate, Center "beaches" + the Inner/
                     Outer Submarine paystreaks. Coarse gold, pyrite +
                     arsenopyrite, abrupt western limits, sitting on a wave-cut
                     bedrock platform. A LOCAL-SOURCE signature.
  strandline-beach   Second and Third beaches (+ the present/First beach by
                     process). Garnet + black-sand rich, finer gold. The
                     WINNOWED-DRIFT signature.

The question: does the abrasion-platform subset respond to a local-source
covariate that the strandline subset ignores, so that typing reveals a signal
the pooled model averages away?

Typing is from the occurrence ``name`` field (the KG records carry the beach
name: "Monroeville Beach", "Second Beach", ...), cross-checked against position.
Most of the 65 KG positives are neither beach type: they are upland residual /
stream-gulch placers (Hudson's "residual alluvial placer" belt). They are kept
as their own group, not silently dumped into "ambiguous".

Two tests, because only ~8 of 65 positives are cleanly-typed coastal beaches:
  1. Distributional contrast (the powered test). For each local-source covariate,
     compare its value at each typed group against background with a one-sided
     Mann-Whitney U + Cliff's delta + a bootstrap CI on the median difference.
     At n_pos = 5 (abrasion) / 3 (strandline) a rank test against the OTHER
     subtype can never reach significance, so the readable signal is each
     subtype vs background and the direction of the medians.
  2. Per-subset leak-guarded CV (the literal deliverable). Pooled OOF under the
     F1 leak-guarded scheme for base and base+covariate on fixed folds, then the
     marginal AUC delta read on each subtype's positives vs background, with a
     paired bootstrap CI. Reported with its small-n caveat; the per-subtype
     numbers are underpowered by construction.

Local-source covariates (all leak-free, defined at every model cell):
  dist_to_36a_lode    nearest ARDF 36a (low-sulfide Au-quartz vein) occurrence.
  dist_to_contact     nearest bedrock(Nome Group)/Quaternary contact = the
                      abrasion-platform inland edge (SIM 3131 coastal-plain Qs
                      inland boundary; same line the round-4B bedrock-contact
                      run used, here defined district-wide, no C-1 clip).
  dist_to_newton_axis perpendicular distance to the Newton Peak coastal-placer
                      belt axis (Hudson 1979/2006 Fig 2). Axis = principal
                      direction of the 36a lode cloud through Newton Peak
                      (GNIS 1406988); the belt Hudson draws from lode + residual
                      placers, fit here on the lode set only so the placer test
                      is not circular.
  dist_to_newton_pt   straight-line distance to Newton Peak (Hudson: most
                      production within 4 mi).

Run: PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_subtype_local_source
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import kruskal, mannwhitneyu
from shapely.ops import unary_union

from ai_minerals.data.adapters.occurrences import kg as kg_occ
from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof, subset_auc
from scripts.nome_placer.f1_leak_guarded_rebaseline import KG_EXPORT, V3P1, load_placer
from scripts.nome_placer.newlayers_bootstrap import N_BOOT, boot_delta
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv

RNG = 42
ARDF = Path("data/raw/nome_mpm/ardf_nome.geojson")
SIM3131 = Path("data/raw/sim3131_seward/nmgeol/nmgeolp.shp")
OUT_DIR = Path("data/derived/nome_placer/subtype_local_source")
# Newton Peak, GNIS feature 1406988 (Summit, Nome AK), USGS GNIS (carto.nationalmap.gov).
NEWTON_PEAK_LL = (-165.31865641368893, 64.55891231556038)
MILE_M = 1609.344
FILL = -999.0

# --------------------------------------------------------------------------- #
# Typing rules. Each (regex on the occurrence name) -> (type, basis). First
# match wins; order matters. Types: abrasion_platform, strandline_beach,
# upland_residual, broad_ambiguous. Hudson 2006 lines 50-52 name the two coastal
# types explicitly; the handoff adds the Inner/Outer Submarine to abrasion-
# platform; everything that is a named creek/gulch/bench is an upland residual
# placer; district-wide or composite names that do not localise are ambiguous.
# --------------------------------------------------------------------------- #
TYPE_RULES: list[tuple[str, str, str]] = [
    (r"nome placer field|nome coastal plain|nome mining district",
     "broad_ambiguous", "district-wide / unlocatable summary record"),
    (r"center creek and flat",
     "broad_ambiguous", "composite of Center flat + upland creeks; not a single beach unit"),
    (r"fourth beach",
     "broad_ambiguous", "raised beach above Third; not typed by Hudson 2006"),
    (r"offshore|submarine",
     "abrasion_platform", "Inner/Outer Submarine paystreak on the wave-cut bedrock platform"),
    (r"monroeville",
     "abrasion_platform", "Hudson 2006 L50: abrasion platform on bedrock"),
    (r"intermediate beach|clam shell",
     "abrasion_platform", "Hudson 2006 L50: abrasion platform on bedrock"),
    (r"center creek beach|center beach",
     "abrasion_platform", "Hudson 2006 L50: Center 'beach' on the abrasion platform"),
    (r"present .*beach|first.?beach",
     "strandline_beach", "modern marine strandline (winnowed, fine gold + black sand)"),
    (r"second beach",
     "strandline_beach", "Hudson 2006 L51: strandline beach, garnet + black-sand rich"),
    (r"third beach",
     "strandline_beach", "Hudson 2006 L51: strandline beach, garnet + black-sand rich"),
]
COASTAL = {"abrasion_platform", "strandline_beach"}


def classify(name: str) -> tuple[str, str]:
    """Type one occurrence by name; default = upland residual/stream-gulch placer."""
    low = name.lower()
    for pat, typ, basis in TYPE_RULES:
        if re.search(pat, low):
            return typ, basis
    return "upland_residual", "named creek/gulch/bench: residual/alluvial upland placer"


# --------------------------------------------------------------------------- #
# Load typed positives, aligned row-for-row to load_placer()'s positive block
# --------------------------------------------------------------------------- #
def load_typed() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (typing_df for the n_pos positives, X_base, y, coords, base_names).

    The typing rows align to the first n_pos rows of load_placer() (same placer
    filter, same lexsort), which is asserted on the coordinates."""
    with rasterio.open(V3P1) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    occ = kg_occ.load(KG_EXPORT).to_crs("EPSG:3338")
    comm = occ["commodity"].astype(str).str.lower()
    code39 = occ["deposit_codes"].apply(
        lambda t: any(str(c).split(":")[-1].startswith("39") for c in t))
    placer = occ[comm.str.contains("placer") | code39].cx[bl:br, bb:bt]
    px, py = placer.geometry.x.to_numpy(), placer.geometry.y.to_numpy()
    order = np.lexsort((py, px))                       # identical to load_placer
    placer = placer.iloc[order]
    px, py = px[order], py[order]

    X_df, y, coords, names, _ = load_placer()
    n_pos = int(y.sum())
    assert n_pos == len(placer), f"{n_pos} != {len(placer)}"
    assert np.allclose(coords[:n_pos, 0], px) and np.allclose(coords[:n_pos, 1], py), \
        "typing rows do not align to load_placer positive block"

    types, bases = zip(*(classify(n) for n in placer["name"]))
    typ = pd.DataFrame({
        "name": placer["name"].to_numpy(),
        "type": list(types), "basis": list(bases),
        "x": px, "y": py,
        "deposit_codes": [",".join(c.split(":")[-1] for c in t) for t in placer["deposit_codes"]],
    })
    return typ, X_df.to_numpy(np.float32), y, coords, names


# --------------------------------------------------------------------------- #
# Local-source covariates
# --------------------------------------------------------------------------- #
def lode_36a_xy() -> np.ndarray:
    """ARDF 36a (low-sulfide Au-quartz vein) occurrence points, EPSG:3338."""
    a = gpd.read_file(ARDF).to_crs("EPSG:3338")
    mc = a.get("model_code", pd.Series([""] * len(a))).astype(str).str.lower().str.replace(" ", "")
    lode = a[mc.str.contains("36a")]
    return np.column_stack([lode.geometry.x.to_numpy(), lode.geometry.y.to_numpy()])


def contact_vertices(densify_m: float = 50.0) -> np.ndarray:
    """Densified vertices of the bedrock/Quaternary contact (SIM 3131 coastal-plain
    Qs inland edge) = the abrasion-platform inland edge. District-wide, no clip."""
    g = gpd.read_file(SIM3131).to_crs("EPSG:3338")
    qs = g[g["LABEL"] == "Qs"]
    coastal = qs.loc[qs.geometry.area.idxmax()].geometry          # the 1709 km2 coastal plain
    bedrock = unary_union(g[~g["LABEL"].isin(["Qs", "water"])].geometry.values)
    contact = coastal.boundary.intersection(bedrock.buffer(30))   # drop the seaward water coast
    line = unary_union(contact)
    n = max(int(line.length / densify_m), 2)
    pts = [line.interpolate(d, normalized=True) for d in np.linspace(0, 1, n)]
    return np.array([(p.x, p.y) for p in pts])


def newton_axis() -> tuple[np.ndarray, np.ndarray, float]:
    """Newton Peak (m) + unit normal to the belt axis (principal direction of the
    36a lode cloud through Newton Peak) + axis azimuth (deg)."""
    nx, ny = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True).transform(*NEWTON_PEAK_LL)
    lode = lode_36a_xy()
    cov = np.cov((lode - lode.mean(0)).T)
    vec = np.linalg.eigh(cov)[1][:, -1]                           # principal direction
    az = (np.degrees(np.arctan2(vec[0], vec[1]))) % 180.0         # 0=N, clockwise; mod 180 (a line)
    normal = np.array([-vec[1], vec[0]])
    return np.array([nx, ny]), normal, round(float(az), 1)


def build_covariates(coords: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Sample the four local-source covariates at every model coord. NaN where a
    covariate is undefined (none are, by construction, but kept for safety)."""
    pts = coords
    lode = lode_36a_xy()
    d_lode = cKDTree(lode).query(pts, k=1)[0]
    d_contact = cKDTree(contact_vertices()).query(pts, k=1)[0]
    npt, normal, az = newton_axis()
    d_axis = np.abs((pts - npt) @ normal)
    d_npt = np.hypot(pts[:, 0] - npt[0], pts[:, 1] - npt[1])
    cov = {"dist_to_36a_lode": d_lode, "dist_to_contact": d_contact,
           "dist_to_newton_axis": d_axis, "dist_to_newton_pt": d_npt}
    meta = {"n_36a_lode": int(len(lode)), "newton_peak_3338": [round(float(npt[0]), 1),
            round(float(npt[1]), 1)], "belt_axis_azimuth_deg": az,
            "newton_peak_gnis": 1406988}
    return {k: v.astype(np.float32) for k, v in cov.items()}, meta


# --------------------------------------------------------------------------- #
# Test 1: distributional contrast (the powered test)
# --------------------------------------------------------------------------- #
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta of a vs b. Negative => a tends smaller (closer) than b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((a[:, None] > b[None, :]).sum(1))
    lt = sum((a[:, None] < b[None, :]).sum(1))
    return round(float((gt - lt) / (len(a) * len(b))), 3)


def boot_median_diff(a: np.ndarray, b: np.ndarray, n: int = N_BOOT) -> list[float]:
    """95% CI on median(a) - median(b) by paired-independent bootstrap."""
    rng = np.random.default_rng(RNG)
    d = [np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
         for _ in range(n)]
    return [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]


def distributional(typ: pd.DataFrame, cov: dict[str, np.ndarray], y: np.ndarray) -> dict:
    """Per-group medians + one-sided MWU(closer than background) + Cliff's delta +
    bootstrap CI on the median difference, for every covariate."""
    n_pos = len(typ)
    grp_idx = {g: np.where((typ["type"] == g).to_numpy())[0] for g in
               ("abrasion_platform", "strandline_beach", "upland_residual")}
    bg = np.where(y == 0)[0]
    out: dict[str, dict] = {}
    for cname, full in cov.items():
        pos_vals = full[:n_pos]
        bg_vals = full[bg]
        bg_vals = bg_vals[np.isfinite(bg_vals)]
        block: dict[str, object] = {"background": {"n": int(len(bg_vals)),
                                    "median_m": round(float(np.median(bg_vals)), 1)}}
        for g, idx in grp_idx.items():
            v = pos_vals[idx]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                block[g] = {"n": 0}
                continue
            try:
                u = mannwhitneyu(v, bg_vals, alternative="less")
                p_closer = round(float(u.pvalue), 4)
            except ValueError:
                p_closer = None
            block[g] = {"n": int(len(v)), "median_m": round(float(np.median(v)), 1),
                        "iqr_m": [round(float(np.percentile(v, 25)), 1),
                                  round(float(np.percentile(v, 75)), 1)],
                        "p_closer_than_bg": p_closer,
                        "cliffs_delta_vs_bg": cliffs_delta(v, bg_vals),
                        "median_diff_vs_bg_ci95": boot_median_diff(v, bg_vals)}
        ap = pos_vals[grp_idx["abrasion_platform"]]
        sb = pos_vals[grp_idx["strandline_beach"]]
        ap, sb = ap[np.isfinite(ap)], sb[np.isfinite(sb)]
        if len(ap) and len(sb):
            try:
                p_ap_lt_sb = round(float(mannwhitneyu(ap, sb, alternative="less").pvalue), 4)
            except ValueError:
                p_ap_lt_sb = None
            block["abrasion_vs_strandline"] = {
                "cliffs_delta_ap_minus_sb": cliffs_delta(ap, sb),
                "p_ap_closer_than_sb": p_ap_lt_sb,
                "median_diff_ap_minus_sb_ci95": boot_median_diff(ap, sb),
                "note": f"n_ap={len(ap)}, n_sb={len(sb)}: a rank test cannot reach "
                        "p<0.05 at this n; read the medians + Cliff's delta direction"}
        placer_groups = [pos_vals[idx][np.isfinite(pos_vals[idx])] for idx in grp_idx.values()
                         if len(idx) > 1]
        if len(placer_groups) >= 2:
            try:
                block["kruskal_across_placer_groups_p"] = round(
                    float(kruskal(*placer_groups).pvalue), 4)
            except ValueError:
                block["kruskal_across_placer_groups_p"] = None
        out[cname] = block
    return out


# --------------------------------------------------------------------------- #
# Test 2: per-subset leak-guarded CV (the literal deliverable)
# --------------------------------------------------------------------------- #
def subset_cv(typ: pd.DataFrame, X_base: np.ndarray, y: np.ndarray,
              coords: np.ndarray, cov: dict[str, np.ndarray]) -> dict:
    """Pooled OOF for base and base+covariate on fixed F1 leak-guarded folds, then
    the marginal AUC delta read on each subtype's positives vs background."""
    cv, cvmeta = make_cv(X_base, y, coords)
    oof_base = spatial_cv_oof(X_base, y, coords, cv, model_factory=default_rf_factory(seed=RNG))

    n_pos = len(typ)
    is_pos = np.zeros(len(y), bool); is_pos[:n_pos] = True
    bg = y == 0

    def grp_mask(g: str) -> np.ndarray:
        m = np.zeros(len(y), bool)
        m[:n_pos] = (typ["type"] == g).to_numpy()
        return m | bg

    masks = {"pooled": np.ones(len(y), bool),
             "abrasion_platform": grp_mask("abrasion_platform"),
             "strandline_beach": grp_mask("strandline_beach"),
             "upland_residual": grp_mask("upland_residual"),
             "coastal_beach_all": (is_pos & typ_in(typ, COASTAL, len(y))) | bg}

    res: dict[str, dict] = {"scheme": cvmeta,
                            "base_auc": {m: subset_auc(y, oof_base, mask=mk)
                                         for m, mk in masks.items()}}
    for cname, full in cov.items():
        XX = np.column_stack([X_base, np.where(np.isfinite(full), full, FILL).astype(np.float32)])
        oof = spatial_cv_oof(XX, y, coords, cv, model_factory=default_rf_factory(seed=RNG))
        res[cname] = {m: boot_delta(y, oof_base, oof, mk) for m, mk in masks.items()}
    res["n_pos_by_mask"] = {m: int(y[mk].sum()) for m, mk in masks.items()}
    return res


def typ_in(typ: pd.DataFrame, types: set[str], n: int) -> np.ndarray:
    """Boolean over n rows, True for positive rows whose type is in ``types``."""
    m = np.zeros(n, bool)
    m[: len(typ)] = typ["type"].isin(types).to_numpy()
    return m


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    typ, X_base, y, coords, base_names = load_typed()

    counts = typ["type"].value_counts().to_dict()
    n_pos = len(typ)
    n_ambiguous = int(counts.get("upland_residual", 0) + counts.get("broad_ambiguous", 0))
    npt, _, az = newton_axis()
    d_npt_pos = np.hypot(coords[:n_pos, 0] - npt[0], coords[:n_pos, 1] - npt[1])

    print(f"typed {n_pos} placer positives:")
    for g in ("abrasion_platform", "strandline_beach", "upland_residual", "broad_ambiguous"):
        print(f"  {g:18s} {counts.get(g, 0)}")
    print(f"  -> handoff 3-way ambiguous (upland+broad) = {n_ambiguous} "
          f"({n_ambiguous / n_pos:.0%})")
    print(f"  within 4 mi of Newton Peak: {int((d_npt_pos <= 4 * MILE_M).sum())}/{n_pos}")

    cov, cov_meta = build_covariates(coords)
    print(f"covariates: 36a lode points={cov_meta['n_36a_lode']}  "
          f"belt-axis azimuth={cov_meta['belt_axis_azimuth_deg']} deg")

    dist = distributional(typ, cov, y)
    cvres = subset_cv(typ, X_base, y, coords, cov)

    # ---- write the typing table ----
    typ_out = typ.copy()
    typ_out["dist_to_newton_pt_mi"] = (d_npt_pos / MILE_M).round(2)
    for cname, full in cov.items():
        typ_out[cname + "_m"] = full[:n_pos].round(1)
    typ_out.to_csv(OUT_DIR / "placer_typing.csv", index=False)

    # ---- write results JSON ----
    out = {
        "question": "Does typing the placer positives (abrasion-platform vs strandline-beach, "
                    "Hudson 2006) reveal a local-source signal the pooled placer model hides?",
        "typing": {
            "source": "occurrence name field (KG kg_nome.jsonld), cross-checked vs position",
            "scheme": "Hudson 2006 (after Metcalfe & Tuck 1942): abrasion-platform = Monroeville/"
                      "Intermediate/Center + Inner/Outer Submarine; strandline-beach = Second/Third "
                      "(+ present/First by process); upland_residual = named creeks/gulches/benches "
                      "(Hudson's residual-alluvial-placer belt); broad_ambiguous = district-wide or "
                      "composite records.",
            "counts": {g: int(counts.get(g, 0)) for g in
                       ("abrasion_platform", "strandline_beach", "upland_residual", "broad_ambiguous")},
            "n": n_pos,
            "handoff_3way_ambiguous_count": n_ambiguous,
            "handoff_3way_ambiguous_fraction": round(n_ambiguous / n_pos, 3),
            "within_4mi_newton_peak": int((d_npt_pos <= 4 * MILE_M).sum()),
        },
        "covariates": {
            "dist_to_36a_lode": "nearest ARDF 36a (low-sulfide Au-quartz vein) occurrence",
            "dist_to_contact": "nearest bedrock/Quaternary contact (SIM 3131 coastal-plain Qs inland "
                               "edge) = abrasion-platform inland edge, district-wide",
            "dist_to_newton_axis": "perpendicular distance to the Newton Peak belt axis (principal "
                                   "direction of the 36a lode cloud through Newton Peak)",
            "dist_to_newton_pt": "straight-line distance to Newton Peak (GNIS 1406988)",
            **cov_meta,
        },
        "test1_distributional": dist,
        "test2_subset_leak_guarded_cv": cvres,
        "interpretation_keys": {
            "p_closer_than_bg": "one-sided Mann-Whitney U: covariate at the subtype is SMALLER "
                                "(closer to the local source) than at background",
            "cliffs_delta_vs_bg": "negative => subtype sits closer to the source than background",
            "subset_cv_point": "marginal OOF AUC delta of adding the covariate to the base model, "
                               "read on that subtype's positives vs background (small n_pos => wide CI)",
        },
        "scheme": {"estimator": "RandomForest(300, balanced, seed=42)", "n_boot": N_BOOT,
                   "cv": "F1 leak-guarded spatial CV (contiguous folds, 1 km dead zone)",
                   "base_features": base_names},
    }
    (OUT_DIR / "placer_subtype_local_source.json").write_text(json.dumps(out, indent=2, default=str))
    write_summary_csv(dist, cvres, OUT_DIR / "placer_subtype_local_source.csv")

    # ---- console summary ----
    print("\nDistributional (median distance, m; p = closer-than-background):")
    for cname, block in dist.items():
        print(f"  {cname}")
        for g in ("abrasion_platform", "strandline_beach", "upland_residual", "background"):
            b = block.get(g, {})
            if b.get("n"):
                extra = (f"  p_closer_bg={b.get('p_closer_than_bg')}  d={b.get('cliffs_delta_vs_bg')}"
                         if g != "background" else "")
                print(f"    {g:18s} n={b['n']:4d}  median={b.get('median_m')}{extra}")
    print("\nPer-subset CV marginal AUC delta (point [CI] P(d>0), n_pos):")
    for cname in cov:
        print(f"  {cname}")
        for m in ("pooled", "upland_residual", "abrasion_platform", "strandline_beach"):
            r = cvres[cname][m]
            print(f"    {m:18s} d={r.get('point')} CI={r.get('ci95')} "
                  f"P(d>0)={r.get('p_gt_0')} n_pos={r.get('n_pos')}")
    print(f"\nwrote {OUT_DIR / 'placer_subtype_local_source.json'}")
    print(f"wrote {OUT_DIR / 'placer_typing.csv'}")
    print(f"wrote {OUT_DIR / 'placer_subtype_local_source.csv'}")


def write_summary_csv(dist: dict, cvres: dict, path: Path) -> None:
    """Flat one-row-per-(covariate, group) summary across both tests."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["covariate", "group", "n", "median_dist_m", "p_closer_than_bg",
                    "cliffs_delta_vs_bg", "cv_marginal_auc_delta", "cv_ci95", "cv_p_gt_0", "cv_n_pos"])
        for cname, block in dist.items():
            for g in ("abrasion_platform", "strandline_beach", "upland_residual"):
                b = block.get(g, {})
                cvr = cvres.get(cname, {}).get(g, {})
                w.writerow([cname, g, b.get("n"), b.get("median_m"), b.get("p_closer_than_bg"),
                            b.get("cliffs_delta_vs_bg"), cvr.get("point"), cvr.get("ci95"),
                            cvr.get("p_gt_0"), cvr.get("n_pos")])


if __name__ == "__main__":
    main()
