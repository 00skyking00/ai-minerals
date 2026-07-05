"""Corrected drill-gold validation harness: capture-efficiency, not raw correlation.

Supersedes the design in ``validate_mpm_vs_drillgold.py``. The NotebookLM
adversarial design review (coordinator dispatch 2026-07-05) found that
correlating raw point-support drill grade against the smoothed 25 m block-support
MPM, on 47 holes that collapse to ~23 spatially-autocorrelated pixels inside one
1.4 km creek at the 77th prospectivity percentile, is mathematically driven
toward zero by change-of-support / regression dilution, the placer nugget effect,
restricted predictor range (attenuation), near-zero effective independent n, and
a continuous-grade response tested against a presence/background classifier. So
the single-creek Spearman near zero says nothing about model quality.

This harness rebuilds the test on the corrected design:

1. Native CRS only. The MPM is sampled by point-to-cell-centre extraction from the
   native EPSG:3338 raster, never the reprojected 4326 delivery raster.
2. Block-upscale grade to the 25 m support. Drill grades are aggregated to each
   grid cell (spatial mean of the holes in the cell). A support/variance-reduction
   correction from point to block support is documented; at ~1 hole per cell in
   one creek the point-support variogram is not estimable, so the block mean is
   used and that assumption is flagged (design-review instruction).
3. Binary economic-presence response. The block grade is thresholded to
   economic-presence (1) vs sub-economic (0) at a proposed cutoff (cents-gold per
   cubic yard at the $20.67/oz basis). The cutoff is FLAGGED for Sky; a ladder is
   reported so the answer's cutoff-sensitivity is visible.
4. Capture-efficiency / prediction-rate, led over correlation. Do the top
   prospectivity ranks capture a disproportionate share of the economic blocks?
   Reported two ways: within the drilled set (MPM as an economic-presence
   classifier, AUC + top-k capture) and against the district surface (percentile
   capture), with a spatially-independent leave-one-drainage-out structure that
   is degenerate at one drainage and becomes real when more creeks land.
5. Range-restriction handling. Any correlation reported is disattenuated for
   restriction of range in the MPM via Thorndike Case II; the additional
   attenuation from single-assay unreliability is noted, not corrected (no
   duplicate assays exist to estimate reliability).
6. Power by spatially-independent drainages, not pixels or holes. The true n is
   the number of independent drainages. Little Creek is one drainage, so this run
   is a LABELED methods-validation / underpowered baseline and cannot be
   conclusive. The authoritative multi-creek run follows the positioned Tuck
   drill facts (fossick Map B Track 2).

Reads (this repo + one cross-repo READ of the fossick export):
  ~/src/learning/fossick/exports/phase4/drill_gold_points.geojson
  data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_3338.tif
  data/derived/nome_placer/mpm_lode_served/mpm_lode_served_district_3338.tif

Writes (this repo only):
  data/derived/nome_placer/drillgold_capture/drillgold_capture_validation.json
  data/derived/nome_placer/drillgold_capture/drillgold_capture_blocks.csv
  data/derived/nome_placer/drillgold_capture/capture_curve.png

Run: .venv/bin/python -m scripts.nome_placer.drillgold_capture_validation
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

DRILL = Path("/home/sky/src/learning/fossick/exports/phase4/drill_gold_points.geojson")
PLACER_MPM = Path("data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_3338.tif")
LODE_MPM = Path("data/derived/nome_placer/mpm_lode_served/mpm_lode_served_district_3338.tif")
OUT_DIR = Path("data/derived/nome_placer/drillgold_capture")
WORK_CRS = 3338  # Alaska Albers, metres; both MPMs are native here
SEED = 42
N_BOOT = 10000
PRICE_C_PER_OZ = 2067.0  # $20.67/oz gold, pre-1934 basis (matches value_c_cuyd/grade_oz)

# Economic-presence cutoff ladder, cents-gold per cubic yard at the $20.67 basis.
# PROPOSED primary is 10 c/yd (order-of-magnitude workable dredge-era pay). This is
# FLAGGED for Sky, not settled; see the report's cutoff rationale. The ladder shows
# how the answer moves with the threshold so no single cutoff choice carries it.
CUTOFFS_C = [5.0, 10.0, 20.0, 40.0]
PRIMARY_CUTOFF_C = 10.0

# Two holes farther apart than this are treated as separate drainages (single-
# linkage). At Little Creek all holes are one cluster -> n_drainages = 1.
DRAINAGE_GAP_M = 1500.0
TOP_FRACS = [0.10, 0.20, 0.50]          # within-drilled-set capture points
DISTRICT_TOPS = [0.01, 0.05, 0.10, 0.20]  # district-percentile capture points


def assign_drainages(coords: np.ndarray, gap_m: float) -> np.ndarray:
    """Single-linkage spatial clustering of hole coords into drainage ids."""
    n = len(coords)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = cKDTree(coords)
    for a, b in tree.query_pairs(gap_m):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def load_grade_holes(drill_path: Path) -> gpd.GeoDataFrame:
    """Collar-positioned, grade-bearing drill holes, in the working CRS.

    Filters generically so the set extends automatically when more positioned
    drill facts land: any point with a metre-scale collar fix and a real grade.
    Claim-centroid and block-approx points are excluded (not a drill fix).
    """
    gdf = gpd.read_file(drill_path)
    keep = (gdf["extent"] == "collar") & gdf["grade_oz_cuyd"].notna()
    holes = gdf.loc[keep].to_crs(WORK_CRS).reset_index(drop=True)
    coords = np.c_[holes.geometry.x.to_numpy(), holes.geometry.y.to_numpy()]
    holes["drainage_id"] = assign_drainages(coords, DRAINAGE_GAP_M)
    return holes


def sample_native(raster: Path, gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-cell-centre MPM value per hole and a distinct-cell id. nodata->nan."""
    xs, ys = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    with rasterio.open(raster) as ds:
        vals = np.array([v[0] for v in ds.sample(list(zip(xs, ys)))], float)
        vals = np.where(vals == ds.nodata, np.nan, vals)
        rc = np.array([ds.index(x, y) for x, y in zip(xs, ys)])
    cell = np.array([f"{r}_{c}" for r, c in rc])
    return vals, cell


def block_upscale(vals: np.ndarray, cell: np.ndarray, grade_c: np.ndarray,
                  drainage: np.ndarray) -> dict[str, np.ndarray]:
    """Collapse holes to distinct MPM cells at the raster support.

    Block grade = spatial mean of the holes in the cell (the point->block support
    aggregation). The affine variance-reduction correction from point to block
    support (Isaaks & Srivastava 1989, change-of-support) would additionally
    shrink the block-grade distribution toward its mean by sqrt(f), f = block/point
    variance; f needs a point-support variogram, which ~1 hole per cell in one
    creek cannot estimate, so f=1 (block mean only) is used and flagged.
    """
    m = np.isfinite(vals) & np.isfinite(grade_c)
    vals, cell, grade_c, drainage = vals[m], cell[m], grade_c[m], drainage[m]
    uniq = np.unique(cell)
    mpm, bg, nh, dr = [], [], [], []
    for u in uniq:
        sel = cell == u
        vv = vals[sel]
        assert np.allclose(vv, vv[0]), f"cell {u} has non-constant MPM {vv}"
        mpm.append(float(vv[0]))
        bg.append(float(grade_c[sel].mean()))
        nh.append(int(sel.sum()))
        dr.append(int(stats.mode(drainage[sel], keepdims=False).mode))
    return {"mpm": np.array(mpm), "block_grade_c": np.array(bg),
            "n_holes": np.array(nh), "drainage": np.array(dr)}


def _auc(mpm: np.ndarray, y: np.ndarray) -> float | None:
    """AUC of the MPM as a classifier of economic (1) vs sub-economic (0) blocks."""
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, mpm))


def _topk_capture(mpm: np.ndarray, y: np.ndarray, fracs: list[float]) -> dict:
    """Fraction of economic blocks that fall in the top-q MPM-ranked drilled blocks,
    with the lift over the base rate (random ranking captures q of them)."""
    order = np.argsort(-mpm)
    y_sorted = y[order]
    n, npos = len(y), int(y.sum())
    out = {}
    for q in fracs:
        k = max(1, int(round(q * n)))
        captured = int(y_sorted[:k].sum())
        frac_captured = captured / npos if npos else None
        out[f"top_{int(q * 100)}pct"] = {
            "k_blocks": k, "captured": captured, "of_economic": npos,
            "frac_captured": frac_captured,
            "lift_vs_random": (frac_captured / q) if frac_captured is not None else None,
        }
    return out


def _district_percentile(raster: Path, mpm_at_blocks: np.ndarray) -> np.ndarray:
    """Percentile of each block's MPM within the full served surface (0-100)."""
    with rasterio.open(raster) as ds:
        a = ds.read(1)
        s = a[a != ds.nodata]
    return np.array([float((s <= v).mean() * 100.0) for v in mpm_at_blocks])


def _district_capture(pct_econ: np.ndarray, tops: list[float]) -> dict:
    """Of the economic blocks, the fraction sitting in the top-q of the district
    surface by percentile. Confounded by restricted range (all drilled ground is
    one high-prospectivity creek); reported with that caveat."""
    out = {}
    for q in tops:
        thresh = 100.0 * (1.0 - q)
        frac = float((pct_econ >= thresh).mean()) if len(pct_econ) else None
        out[f"district_top_{int(q * 100)}pct"] = {
            "percentile_threshold": thresh, "frac_economic_blocks_here": frac,
            "lift_vs_random": (frac / q) if frac is not None else None,
        }
    return out


def thorndike_case2(r: float, sd_restricted: float, sd_unrestricted: float) -> float:
    """Disattenuate a correlation for restriction of range in the predictor (MPM).

    Thorndike Case II: project the restricted-sample correlation to the
    unrestricted population variance. u = SD_unrestricted / SD_restricted.
    """
    if not np.isfinite(r) or sd_restricted <= 0:
        return float("nan")
    u = sd_unrestricted / sd_restricted
    denom = np.sqrt(1.0 - r * r + r * r * u * u)
    return float(r * u / denom)


def _loo_drainage_auc(blocks: dict, y: np.ndarray) -> dict:
    """Leave-one-drainage-out AUC (spatially-independent). Degenerate at one
    drainage; becomes a real held-out capture measure when more creeks land."""
    dr = blocks["drainage"]
    n_dr = int(len(np.unique(dr)))
    if n_dr < 2:
        return {"n_drainages": n_dr, "auc": None,
                "note": "one drainage: no spatially-independent hold-out is possible"}
    preds = np.full(len(y), np.nan)
    # With the served MPM as a fixed surface there is nothing to refit; the
    # leave-one-drainage-out structure instead pools each held-out drainage's
    # blocks and scores capture across drainages. AUC over pooled held-out blocks.
    for d in np.unique(dr):
        preds[dr == d] = blocks["mpm"][dr == d]
    return {"n_drainages": n_dr, "auc": _auc(preds, y),
            "note": "pooled held-out-drainage AUC of the fixed served surface"}


def _bootstrap_auc(mpm: np.ndarray, y: np.ndarray, drainage: np.ndarray,
                   rng: np.random.Generator) -> dict | None:
    """Cluster bootstrap CI on the economic-presence AUC. Resamples whole
    drainages when >1 exist, else whole blocks (flagged as within-cluster)."""
    if len(np.unique(y)) < 2:
        return None
    n_dr = len(np.unique(drainage))
    boot, k = np.empty(N_BOOT), 0
    for _ in range(N_BOOT):
        if n_dr >= 2:
            drs = rng.choice(np.unique(drainage), n_dr, replace=True)
            idx = np.concatenate([np.where(drainage == d)[0] for d in drs])
        else:
            idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot[k] = roc_auc_score(y[idx], mpm[idx])
        k += 1
    boot = boot[:k]
    return {"ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "median": float(np.median(boot)), "frac_gt_0_5": float((boot > 0.5).mean()),
            "n_valid_resamples": int(k),
            "cluster_unit": "drainage" if n_dr >= 2 else "block (within one drainage)"}


def analyze(name: str, raster: Path, holes: gpd.GeoDataFrame,
            rng: np.random.Generator) -> dict:
    vals, cell = sample_native(raster, holes)
    grade_c = holes["value_c_cuyd"].to_numpy(float)
    drainage = holes["drainage_id"].to_numpy(int)
    blocks = block_upscale(vals, cell, grade_c, drainage)

    with rasterio.open(raster) as ds:
        res_m = int(ds.res[0])
    mpm = blocks["mpm"]
    bg = blocks["block_grade_c"]
    sd_restricted = float(mpm.std(ddof=1)) if len(mpm) > 1 else 0.0
    with rasterio.open(raster) as ds:
        a = ds.read(1)
        sd_unrestricted = float(a[a != ds.nodata].std())

    # correlation (secondary), observed + disattenuated for range restriction
    pear = stats.pearsonr(mpm, bg) if len(mpm) >= 3 else (float("nan"), float("nan"))
    spear = stats.spearmanr(mpm, bg) if len(mpm) >= 3 else (float("nan"), float("nan"))
    u = sd_unrestricted / sd_restricted if sd_restricted > 0 else float("inf")
    # Thorndike Case II is unstable when the restriction is extreme: as u grows it
    # drives any non-zero r toward +-1, so a disattenuated value from a large u is
    # an amplified artifact, not a signal. Flag it rather than let it be cited.
    thorn_unstable = u > 10.0
    corr = {
        "pearson_observed": {"r": float(pear[0]), "p": float(pear[1])},
        "spearman_observed": {"rho": float(spear[0]), "p": float(spear[1])},
        "mpm_sd_restricted": sd_restricted, "mpm_sd_unrestricted": sd_unrestricted,
        "restriction_ratio_u": float(u),
        "pearson_disattenuated_thorndike_case2": thorndike_case2(
            float(pear[0]), sd_restricted, sd_unrestricted),
        "disattenuation_stable": bool(not thorn_unstable),
        "disattenuation_note": (
            "u>10: the MPM range at the drill blocks is a tiny slice of the district "
            "range, so Thorndike amplifies a non-significant r toward +-1; the "
            "disattenuated value is an artifact of extreme restriction, do not cite"
            if thorn_unstable else
            "u<=10: mild restriction, the disattenuated r is interpretable"),
        "reliability_note": ("not further corrected for assay unreliability: single "
                             "churn-drill assays with no duplicates cannot estimate "
                             "reliability, so the disattenuated r is an upper-ish bound "
                             "on the range-restriction correction alone"),
    }

    pct = _district_percentile(raster, mpm)
    # capture-efficiency per cutoff (the primary metric family)
    by_cutoff = {}
    for c in CUTOFFS_C:
        y = (bg >= c).astype(int)
        npos = int(y.sum())
        econ_pct = pct[y == 1]
        by_cutoff[f"cutoff_{int(c)}c"] = {
            "cutoff_c_per_cuyd": c,
            "cutoff_oz_per_cuyd": round(c / PRICE_C_PER_OZ, 5),
            "n_economic_blocks": npos, "n_subeconomic_blocks": int((y == 0).sum()),
            "auc_mpm_as_econ_classifier": _auc(mpm, y),
            "auc_bootstrap": _bootstrap_auc(mpm, y, blocks["drainage"], rng),
            "within_set_capture": _topk_capture(mpm, y, TOP_FRACS),
            "district_percentile_capture": _district_capture(econ_pct, DISTRICT_TOPS),
            "loo_drainage": _loo_drainage_auc(blocks, y),
        }

    return {
        "raster": raster.name, "resolution_m": res_m,
        "n_holes": int(len(holes)), "n_distinct_blocks": int(len(mpm)),
        "n_drainages": int(len(np.unique(blocks["drainage"]))),
        "block_grade_c_range": [float(bg.min()), float(bg.max())],
        "block_mpm_range": [float(mpm.min()), float(mpm.max())],
        "block_mpm_district_percentile": {
            "min": float(pct.min()), "median": float(np.median(pct)), "max": float(pct.max())},
        "support_correction": {
            "applied": "block mean only (point->block aggregation)",
            "affine_variance_shrink": "not applied; f=1",
            "reason": "one hole per cell in one creek; point-support variogram not estimable"},
        "correlation_secondary": corr,
        "capture_by_cutoff": by_cutoff,
        "primary_cutoff_c": PRIMARY_CUTOFF_C,
    }


def _figure(placer: dict, blocks_placer: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mpm = blocks_placer["mpm"]
    bg = blocks_placer["block_grade_c"]
    y = (bg >= PRIMARY_CUTOFF_C).astype(int)
    order = np.argsort(-mpm)
    ys = y[order]
    npos = max(int(y.sum()), 1)
    frac_ground = np.arange(1, len(ys) + 1) / len(ys)
    frac_captured = np.cumsum(ys) / npos

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="random ranking")
    ax.step(np.r_[0, frac_ground], np.r_[0, frac_captured], where="post",
            color="#1f4e79", lw=2, label="served placer MPM")
    ax.set_xlabel("Top fraction of drilled blocks by MPM rank")
    ax.set_ylabel(f"Fraction of economic blocks captured\n(cutoff {int(PRIMARY_CUTOFF_C)} c/yd)")
    auc = placer["capture_by_cutoff"][f"cutoff_{int(PRIMARY_CUTOFF_C)}c"]["auc_mpm_as_econ_classifier"]
    auc_s = "n/a" if auc is None else f"{auc:.2f}"
    ax.set_title(f"Drill-gold capture efficiency, Little Creek (1 drainage, underpowered)\n"
                 f"MPM as economic-presence classifier: AUC={auc_s}, "
                 f"{npos} economic of {len(y)} blocks")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capture_curve.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    holes = load_grade_holes(DRILL)
    n_dr = int(holes["drainage_id"].nunique())

    report = {
        "task": "coordinator 2026-07-05 drillgold-validation-corrected-design",
        "status": ("LABELED methods-validation / underpowered baseline. One drainage "
                   "(Little Creek), so this cannot be conclusive and is NOT a G-result. "
                   "The authoritative multi-creek run follows the positioned Tuck drill "
                   "facts (fossick Map B Track 2)."),
        "design_review": ("supersedes validate_mpm_vs_drillgold.py; the single-creek raw "
                          "Spearman near zero is a change-of-support / range-restriction / "
                          "low-n artifact, not a model verdict"),
        "ground_truth": {
            "source": "fossick/exports/phase4/drill_gold_points.geojson",
            "subset": "collar-positioned, grade-bearing drill holes",
            "n_holes": int(len(holes)), "n_drainages": n_dr,
            "grade_field": "value_c_cuyd (cents-gold per cubic yard, $20.67/oz basis)",
        },
        "economic_cutoff": {
            "primary_proposed_c_per_cuyd": PRIMARY_CUTOFF_C,
            "primary_proposed_oz_per_cuyd": round(PRIMARY_CUTOFF_C / PRICE_C_PER_OZ, 5),
            "ladder_c_per_cuyd": CUTOFFS_C,
            "rationale": ("order-of-magnitude workable dredge-era placer pay at the "
                          "$20.67/oz basis; Little Creek was drift-mined for much richer "
                          "bench gravel, so 10 c/yd separates workable from lean rather "
                          "than pay from barren. Anchor to Moffit 1913 (USGS Bull. 533) "
                          "Nome placer economics; Tuck 1942 normalizes the gold price."),
            "FLAG_FOR_SKY": ("confirm the intended cutoff: dredge workable (~10 c/yd), "
                             "drift-mine pay (dollars/yd), or a relative rich/lean split "
                             "at the sample median. The ladder shows the sensitivity."),
        },
        "barren_domain": {
            "problem": ("historic placer records report successes, so true barren-ground "
                        "negatives are scarce. This run avoids fabricated negatives by "
                        "taking the sub-economic class from real low-grade drill blocks "
                        "(the lean tail of the same holes), so both classes are measured."),
            "for_district_capture": ("the district-percentile capture needs a background; "
                                     "proposed pseudo-absences = low-prospectivity "
                                     "non-drilled cells, but that is circular (drawing "
                                     "negatives from low-MPM ground guarantees the MPM "
                                     "wins), so it is reported only with that caveat"),
            "data_hunt": ("genuine barren negatives would come from drill/pit logs that hit "
                          "no pay: Janin's Dry Creek sub-pay holes and Tuck 1942 barren "
                          "property assessments are candidates; not yet positioned"),
        },
        "data_dependency": ("authoritative run gated on fossick Map B Track 2 positioning "
                            "~45 Tuck drill facts across additional creeks; current export "
                            "has only 10 Tuck claim-centroid points with no grade"),
    }

    for name, raster in [("placer_mpm", PLACER_MPM), ("lode_mpm", LODE_MPM)]:
        report[name] = analyze(name, raster, holes, rng)

    (OUT_DIR / "drillgold_capture_validation.json").write_text(json.dumps(report, indent=2))

    # per-block CSV for audit (placer)
    vals, cell = sample_native(PLACER_MPM, holes)
    blocks = block_upscale(vals, cell, holes["value_c_cuyd"].to_numpy(float),
                           holes["drainage_id"].to_numpy(int))
    import csv
    with (OUT_DIR / "drillgold_capture_blocks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["placer_mpm", "block_grade_c_per_cuyd", "n_holes", "drainage_id",
                    "economic_at_primary_cutoff"])
        for i in range(len(blocks["mpm"])):
            w.writerow([blocks["mpm"][i], blocks["block_grade_c"][i], blocks["n_holes"][i],
                        blocks["drainage"][i], int(blocks["block_grade_c"][i] >= PRIMARY_CUTOFF_C)])

    _figure(report["placer_mpm"], blocks)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_DIR / 'drillgold_capture_validation.json'}")


if __name__ == "__main__":
    main()
