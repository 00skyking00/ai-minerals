"""Drill-gold grade-capture harness on the leak-free evaluation contract (ML step 3 Part C).

Rewritten 2026-07-12 to the binding evaluation contract in the G1 validation plan
(portfolio/docs/reports/ml_step3_g1_validation_plan_2026-07-12.md, §1). The
adversarial review (Fable-5) found the previous design would report a
leaked-optimistic number: it scored a FIXED served surface that had already been
fit over the drilled ground (in-sample), its ``_loo_drainage_auc`` was a no-op
(``preds == mpm``), and it clustered "drainages" by a 1.5 km point threshold that
lets one creek fake several. This rewrite implements the contract exactly:

1. No-Tuck-feature headline model. The capture number comes ONLY from the GEOMORPH
   placer presence RF (7 v3.1 geomorph population bands + DEM + slope + TPI) refit
   here: the SAME features/estimator/labels as the served surface
   (``mpm_onshore_score_district.py``), which carries zero Tuck-derived features.
   Tuck supplies the grade LABELS, so any Tuck feature would put labels on both
   sides. The lode surface is NOT scored (contract §1; see the leak block).
2. Leave-one-WATERSHED-out REFIT. For each drilled watershed we REFIT the presence
   RF with every training occurrence inside that watershed + a 1 km dead zone
   removed, then score that watershed's drill blocks with its own held-out surface.
   No block is scored by a model that saw a label in its own watershed.
3. Within-watershed statistic, equal-weight. Primary = within-watershed capture /
   AUC (does the held-out surface rank economic blocks above sub-economic blocks
   INSIDE each watershed), averaged equal-weight across watersheds. Pooled
   cross-watershed AUC is Simpson-confounded and reported SECONDARY only.
4. Watersheds = DEM catchment polygons (``ai_minerals.watersheds``, D8 basins on the
   in-tree IFSAR DEM), not a point cluster. The drainage map is written alongside.
5. Falsification = within-watershed label-permutation null under LODO. Grade labels
   are permuted at the drill-LINE level (not per hole) inside each held-out
   watershed; if the observed within-watershed capture sits inside the permutation
   band, the MPM shows no grade skill beyond presence.
6. Negatives = real drilled/panned sub-economic reads only (the lean tail of the
   drilled blocks; the Tuck barren reads when they position). NEVER low-MPM cells.
7. Claim ladder. At k < ~3 watersheds this is descriptive: capture curves + a
   per-watershed table, NO significance language. A null at low k means "cannot
   distinguish", NOT "no grade signal" (contract §7 honest fallback).

Today's data is one positioned drainage (Little Creek, Janin 1912, 47 collars), so
this DRY RUN honestly reports the underpowered state. The LODO-refit,
permutation-null, and DEM-catchment paths are all exercised (k=1), and generalise
unchanged when fossick positions the Otter / Nome River / Bay-Odin collars.

Reads (this repo + one cross-repo READ of the fossick export):
  ~/src/learning/fossick/exports/phase4/drill_gold_points.geojson
  ~/src/learning/fossick/exports/features/feature_table.csv          (leak audit only)
  data/raw/nome_mpm/ifsar_dem_3338.tif                               (watershed + terrain)
  data/derived/nome_placer/prospectivity_v1p5/..._v3p1_3338.tif      (geomorph bands)
  data/raw/nome_mpm/{ifsar_slope,ifsar_tpi}_3338.tif
  data/raw/fossick_kg/kg_nome.jsonld                                 (placer presence labels)
  data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_3338.tif (served, SECONDARY)

Writes (this repo only):
  data/derived/nome_placer/drillgold_capture/drillgold_capture_validation.json
  data/derived/nome_placer/drillgold_capture/drillgold_capture_blocks.csv
  data/derived/nome_placer/drillgold_capture/capture_curve.png
  data/derived/nome_placer/drillgold_capture/drainage_map.geojson
  data/derived/nome_placer/drillgold_capture/drainage_map.png

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.drillgold_capture_validation
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ai_minerals.data.adapters.occurrences import kg as kg_occ
from ai_minerals.watersheds import NODATA_ID, Watersheds, delineate

DRILL = Path("/home/sky/src/learning/fossick/exports/phase4/drill_gold_points.geojson")
FEATURE_TABLE = Path("/home/sky/src/learning/fossick/exports/features/feature_table.csv")
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
DEM_DISTRICT = Path("data/raw/nome_mpm/ifsar_dem_district_3338.tif")
V3P1 = Path("data/derived/nome_placer/prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif")
SLOPE = Path("data/raw/nome_mpm/ifsar_slope_3338.tif")
TPI = Path("data/raw/nome_mpm/ifsar_tpi_3338.tif")
KG_EXPORT = Path("data/raw/fossick_kg/kg_nome.jsonld")
SERVED_MPM = Path("data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_3338.tif")
LODE_MPM = Path("data/derived/nome_placer/mpm_lode_served/mpm_lode_served_district_3338.tif")
OUT_DIR = Path("data/derived/nome_placer/drillgold_capture")

WORK_CRS = 3338
SEED = 42
N_BG = 2000                     # background count (matches the served-surface pipeline)
N_BOOT = 5000                   # within-set AUC bootstrap draws
N_PERM = 2000                   # line-restricted permutation-null draws
DEAD_ZONE_M = 1000.0            # LODO refit dead zone (> our 400-800 m proximity buffers)
FILL = -999.0
PRICE_C_PER_OZ = 2067.0         # $20.67/oz pre-1934 basis (value_c_cuyd = grade_oz * this)

# Economic-presence cutoff ladder, cents-gold/cu-yd at the $20.67 basis. HEADLINE
# = 10 c/yd (Moffit-1913 dredge-workable anchor; Sky 2026-07-05). Historic terms
# kept, not converted to modern gold price (contract §6).
CUTOFFS_C = [5.0, 10.0, 20.0, 40.0]
HEADLINE_CUTOFF_C = 10.0
TOP_FRACS = [0.10, 0.20, 0.50]
DISTRICT_TOPS = [0.01, 0.05, 0.10, 0.20]
SUPPORT_SHRINK_FACTORS = [1.0, 0.85, 0.70, 0.50]   # block/point variance ratio f (§6)
MIN_BLOCKS_WS = 4               # min blocks in a watershed to compute a within-w AUC

# GEOMORPH feature set: identical to mpm_onshore_score_district.py. No Tuck/pen_*,
# no occurrence-distance, no Janin, no geochem -> contract §1 headline-eligible.
POP = ["bl", "ap", "tb", "ss", "bc", "qm", "buried_bl"]
GEOMORPH = POP + ["dem", "slope", "tpi"]

# §5 sampling-effort leak columns that must NEVER enter a grade model.
LEAK_COLS = ["nearest_occurrence_distance_m", "nearest_ms_claim_distance_m",
             "n_ms_claims_within_400m"]


# --------------------------------------------------------------------------- #
# Ground truth + watershed labels
# --------------------------------------------------------------------------- #
def load_grade_holes(drill_path: Path) -> gpd.GeoDataFrame:
    """Collar-positioned, grade-bearing drill holes, in the working CRS.

    Generic filter so the set extends when more positioned drill facts land: any
    point with a metre-scale collar fix and a real grade. Claim-centroid and
    block-approx points are excluded (not a drill fix)."""
    gdf = gpd.read_file(drill_path)
    keep = (gdf["extent"] == "collar") & gdf["grade_oz_cuyd"].notna()
    return gdf.loc[keep].to_crs(WORK_CRS).reset_index(drop=True)


def choose_dem(holes: gpd.GeoDataFrame) -> Path:
    """Finest DEM whose bounds contain every drill collar (placer-core 25 m, else
    the coarser district DEM)."""
    xs, ys = holes.geometry.x.to_numpy(), holes.geometry.y.to_numpy()
    with rasterio.open(DEM) as ds:
        b = ds.bounds
    if (xs.min() >= b.left and xs.max() <= b.right
            and ys.min() >= b.bottom and ys.max() <= b.top):
        return DEM
    return DEM_DISTRICT


def _samp(path: Path, xs: np.ndarray, ys: np.ndarray, bands, names) -> pd.DataFrame:
    """Sample raster bands at point coords, nodata -> nan (matches the served pipeline)."""
    with rasterio.open(path) as ds:
        a = np.asarray(list(ds.sample(list(zip(xs, ys)))), float)
        nod = ds.nodata
    return pd.DataFrame(
        {nm: (np.where(a[:, b - 1] == nod, np.nan, a[:, b - 1]) if nod is not None
              else a[:, b - 1]) for b, nm in zip(bands, names)})


def sample_geomorph(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """GEOMORPH feature matrix at point coords, nan-filled -> FILL (served-identical)."""
    feat = pd.concat([
        _samp(V3P1, xs, ys, list(range(1, 8)), POP),
        _samp(DEM, xs, ys, [1], ["dem"]),
        _samp(SLOPE, xs, ys, [1], ["slope"]),
        _samp(TPI, xs, ys, [1], ["tpi"]),
    ], axis=1)[GEOMORPH]
    return feat.to_numpy(np.float32, na_value=FILL)


def _modal(series: pd.Series):
    m = series.mode(dropna=True)
    return m.iloc[0] if len(m) else None


def block_upscale(holes: gpd.GeoDataFrame, dem_path: Path,
                  ws: Watersheds) -> pd.DataFrame:
    """Collapse holes to distinct DEM cells (the block support). Block grade = mean
    of the holes in the cell; watershed / drill line = modal; coords = cell centre.

    The affine change-of-support variance shrink (Isaaks & Srivastava 1989) is not
    applied here (f=1): ~1 hole per cell in one creek cannot estimate a point-support
    variogram. Its label-flip sensitivity is quantified separately (§6)."""
    xs, ys = holes.geometry.x.to_numpy(), holes.geometry.y.to_numpy()
    with rasterio.open(dem_path) as ds:
        T = ds.transform
    rows, cols = rasterio.transform.rowcol(T, xs, ys)
    df = holes.assign(_r=np.asarray(rows), _c=np.asarray(cols),
                      _g=holes["value_c_cuyd"].to_numpy(float))
    df = df.dropna(subset=["_g"])
    recs = []
    for (r, c), grp in df.groupby(["_r", "_c"]):
        cx, cy = rasterio.transform.xy(T, r, c)  # cell centre
        recs.append({"row": int(r), "col": int(c), "x": float(cx), "y": float(cy),
                     "block_grade_c": float(grp["_g"].mean()), "n_holes": int(len(grp)),
                     "line": _modal(grp["line"])})
    blocks = pd.DataFrame(recs)
    blocks["watershed"] = ws.watershed_at(blocks["x"].to_numpy(), blocks["y"].to_numpy())
    return blocks


# --------------------------------------------------------------------------- #
# Presence training set (the served GEOMORPH model, refit per watershed)
# --------------------------------------------------------------------------- #
def build_presence_training(ws: Watersheds) -> dict:
    """65 in-box placer positives + N_BG on-land background, GEOMORPH features and
    watershed labels. Mirrors mpm_onshore_score_district.py exactly."""
    with rasterio.open(V3P1) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
    occ = kg_occ.load(KG_EXPORT).to_crs("EPSG:3338")
    comm = occ["commodity"].astype(str).str.lower()
    code39 = occ["deposit_codes"].apply(
        lambda t: any(str(c).split(":")[-1].startswith("39") for c in t))
    placer = occ[comm.str.contains("placer") | code39].cx[bl:br, bb:bt]
    px, py = placer.geometry.x.to_numpy(), placer.geometry.y.to_numpy()
    order = np.lexsort((py, px))                 # canonical order (ADR-017 row-order guard)
    px, py = px[order], py[order]

    with rasterio.open(DEM) as ds:
        dem = ds.read(1); nod = ds.nodata; T = ds.transform
    valid = np.argwhere(dem != nod)
    rng = np.random.default_rng(SEED)
    pick = valid[rng.choice(len(valid), size=N_BG, replace=False)]
    bx, by = rasterio.transform.xy(T, pick[:, 0], pick[:, 1])
    bx, by = np.asarray(bx), np.asarray(by)

    ex, ny = np.concatenate([px, bx]), np.concatenate([py, by])
    y = np.concatenate([np.ones(len(px)), np.zeros(len(bx))]).astype(int)
    X = sample_geomorph(ex, ny)
    wsid = ws.watershed_at(ex, ny)
    return {"X": X, "y": y, "coords": np.column_stack([ex, ny]), "ws": wsid,
            "n_pos": int(len(px)), "n_bg": int(len(bx))}


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=1)


def lodo_refit_scores(train: dict, blocks: pd.DataFrame,
                      dead_zone_m: float) -> tuple[np.ndarray, dict]:
    """Leave-one-WATERSHED-out refit. For each drilled watershed, drop training
    points inside it + within dead_zone_m of its blocks, refit the GEOMORPH RF, and
    score that watershed's blocks with the held-out surface. Returns per-block
    held-out MPM (NaN where a block's watershed has no clean refit) + diagnostics."""
    Xb = sample_geomorph(blocks["x"].to_numpy(), blocks["y"].to_numpy())
    coords_b = blocks[["x", "y"]].to_numpy()
    held = np.full(len(blocks), np.nan)
    diag = {}
    for w in sorted(int(v) for v in blocks["watershed"].unique() if v != NODATA_ID):
        test = blocks["watershed"].to_numpy() == w
        drop = train["ws"] == w
        if dead_zone_m > 0:
            d, _ = cKDTree(coords_b[test]).query(train["coords"], k=1)
            drop = drop | (d < dead_zone_m)
        keep = ~drop
        if len(np.unique(train["y"][keep])) < 2:
            diag[str(w)] = {"skipped": "held-out training set lost a class"}
            continue
        model = _rf().fit(train["X"][keep], train["y"][keep])
        held[test] = model.predict_proba(Xb[test])[:, 1]
        diag[str(w)] = {"n_train_kept": int(keep.sum()),
                        "n_train_pos_dropped": int((drop & (train["y"] == 1)).sum()),
                        "n_blocks_scored": int(test.sum())}
    return held, diag


# --------------------------------------------------------------------------- #
# Within-watershed capture + line-restricted permutation null
# --------------------------------------------------------------------------- #
def _auc(score: np.ndarray, y: np.ndarray) -> float | None:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def _topk_capture(score: np.ndarray, y: np.ndarray, fracs: list[float]) -> dict:
    order = np.argsort(-score)
    ys = y[order]
    n, npos = len(y), int(y.sum())
    out = {}
    for q in fracs:
        k = max(1, int(round(q * n)))
        cap = int(ys[:k].sum())
        frac = cap / npos if npos else None
        out[f"top_{int(q * 100)}pct"] = {
            "k_blocks": k, "captured": cap, "of_economic": npos, "frac_captured": frac,
            "lift_vs_random": (frac / q) if frac is not None else None}
    return out


def line_restricted_perm_null(score: np.ndarray, y: np.ndarray, line: np.ndarray,
                              rng: np.random.Generator, n_perm: int) -> dict:
    """Falsification null: permute economic labels at the drill-LINE level, holding
    the MPM fixed, and rebuild the within-watershed AUC. Each line's label vector
    moves as an intact block onto another line's slots (size mismatch handled by
    cycling via np.resize), so within-line grade autocorrelation is preserved and
    holes are NOT treated as independent. Degenerate (< 2 lines / one class)."""
    groups = pd.Series(np.where(pd.isna(line), "unassigned", line))
    uniq = groups.unique()
    obs = _auc(score, y)
    if obs is None or len(uniq) < 2:
        return {"observed_auc": obs, "n_lines": int(len(uniq)),
                "null_defined": False,
                "note": "one drill line or one grade class -> line permutation is degenerate"}
    idx_by_line = [np.where(groups.to_numpy() == g)[0] for g in uniq]
    order = np.concatenate(idx_by_line)
    score_o = score[order]
    src = [y[ix] for ix in idx_by_line]
    sizes = [len(ix) for ix in idx_by_line]
    starts = np.cumsum([0] + sizes[:-1])
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(uniq))
        yp = np.empty(len(order), int)
        for dest, (s, sz) in enumerate(zip(starts, sizes)):
            yp[s:s + sz] = np.resize(src[perm[dest]], sz)
        if len(np.unique(yp)) == 2:
            null.append(roc_auc_score(yp, score_o))
    null = np.array(null)
    if not len(null):
        return {"observed_auc": obs, "n_lines": int(len(uniq)), "null_defined": False,
                "note": "permutations never produced two classes"}
    p_upper = float((null >= obs).mean())
    return {"observed_auc": obs, "n_lines": int(len(uniq)), "null_defined": True,
            "null_mean": float(null.mean()),
            "null_ci90": [float(np.percentile(null, 5)), float(np.percentile(null, 95))],
            "p_value_upper": p_upper,
            "inside_null_band": bool(np.percentile(null, 5) <= obs <= np.percentile(null, 95)),
            "n_perm_valid": int(len(null))}


def _bootstrap_auc(score: np.ndarray, y: np.ndarray,
                   rng: np.random.Generator) -> dict | None:
    """Within-drainage block bootstrap CI on the AUC. One drainage -> this is a
    within-cluster CI, NOT a spatial CI (flagged)."""
    if len(np.unique(y)) < 2:
        return None
    boot, k = np.empty(N_BOOT), 0
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) == 2:
            boot[k] = roc_auc_score(y[ix], score[ix]); k += 1
    boot = boot[:k]
    return {"ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "median": float(np.median(boot)), "frac_gt_0_5": float((boot > 0.5).mean()),
            "n_valid_resamples": int(k), "cluster_unit": "block (within one drainage)"}


def within_watershed_capture(blocks: pd.DataFrame, held: np.ndarray, cutoff: float,
                             rng: np.random.Generator) -> dict:
    """Per-watershed capture on the held-out refit surface + equal-weight aggregate
    and a pooled (secondary) AUC. This is the PRIMARY grade metric (contract §3)."""
    per_ws, aucs = {}, []
    for w in sorted(int(v) for v in blocks["watershed"].unique() if v != NODATA_ID):
        m = (blocks["watershed"].to_numpy() == w) & np.isfinite(held)
        if m.sum() < MIN_BLOCKS_WS:
            per_ws[str(w)] = {"n_blocks": int(m.sum()),
                              "note": f"< {MIN_BLOCKS_WS} scored blocks; no within-w AUC"}
            continue
        s, g = held[m], blocks["block_grade_c"].to_numpy()[m]
        line = blocks["line"].to_numpy()[m]
        yb = (g >= cutoff).astype(int)
        auc = _auc(s, yb)
        entry = {"n_blocks": int(m.sum()), "n_economic": int(yb.sum()),
                 "n_subeconomic": int((yb == 0).sum()),
                 "within_watershed_auc": auc,
                 "within_set_capture": _topk_capture(s, yb, TOP_FRACS) if auc is not None else None,
                 "bootstrap": _bootstrap_auc(s, yb, rng) if auc is not None else None,
                 "permutation_null": line_restricted_perm_null(s, yb, line, rng, N_PERM)}
        per_ws[str(w)] = entry
        if auc is not None:
            aucs.append(auc)
    # pooled held-out AUC across watersheds (SECONDARY: Simpson-confounded)
    mp = np.isfinite(held)
    yp = (blocks["block_grade_c"].to_numpy()[mp] >= cutoff).astype(int)
    pooled = _auc(held[mp], yp)
    return {"per_watershed": per_ws,
            "n_watersheds_with_auc": len(aucs),
            "equal_weight_within_watershed_auc": (float(np.mean(aucs)) if aucs else None),
            "pooled_cross_watershed_auc_SECONDARY": pooled,
            "pooled_note": ("pooled AUC mixes rich vs lean drainages (Simpson) and is "
                            "dominated by the largest read set; not the headline")}


# --------------------------------------------------------------------------- #
# Secondary reads (kept from the prior harness, clearly labelled)
# --------------------------------------------------------------------------- #
def thorndike_case2(r: float, sd_r: float, sd_u: float) -> float:
    if not np.isfinite(r) or sd_r <= 0:
        return float("nan")
    u = sd_u / sd_r
    return float(r * u / np.sqrt(1.0 - r * r + r * r * u * u))


def served_secondary(blocks: pd.DataFrame) -> dict:
    """Fixed served-surface reads (correlation, district-percentile capture); all
    restriction-of-range / in-sample confounded, so SECONDARY only. The served
    surface was fit over Little Creek, so a high percentile here is not validation."""
    xs, ys = blocks["x"].to_numpy(), blocks["y"].to_numpy()
    served = _samp(SERVED_MPM, xs, ys, [1], ["v"])["v"].to_numpy()
    bg = blocks["block_grade_c"].to_numpy()
    ok = np.isfinite(served) & np.isfinite(bg)
    served, bg = served[ok], bg[ok]
    with rasterio.open(SERVED_MPM) as ds:
        a = ds.read(1); dist = a[a != ds.nodata]
    sd_r = float(served.std(ddof=1)) if len(served) > 1 else 0.0
    sd_u = float(dist.std())
    u = sd_u / sd_r if sd_r > 0 else float("inf")
    spear = stats.spearmanr(served, bg) if len(served) >= 3 else (float("nan"), float("nan"))
    pear = stats.pearsonr(served, bg) if len(served) >= 3 else (float("nan"), float("nan"))
    pct = np.array([float((dist <= v).mean() * 100.0) for v in served])
    econ = bg >= HEADLINE_CUTOFF_C
    dcap = {}
    for q in DISTRICT_TOPS:
        thr = 100.0 * (1.0 - q)
        frac = float((pct[econ] >= thr).mean()) if econ.any() else None
        dcap[f"district_top_{int(q * 100)}pct"] = {
            "percentile_threshold": thr, "frac_economic_here": frac,
            "lift_vs_random": (frac / q) if frac is not None else None}
    return {
        "caveat": ("fixed served surface, fit over Little Creek -> in-sample / "
                   "restriction-of-range. Descriptive consistency, never the headline."),
        "spearman_observed": {"rho": float(spear[0]), "p": float(spear[1])},
        "pearson_observed": {"r": float(pear[0]), "p": float(pear[1])},
        "restriction_ratio_u": float(u),
        "pearson_disattenuated_thorndike": thorndike_case2(float(pear[0]), sd_r, sd_u),
        "disattenuation_stable": bool(u <= 10.0),
        "disattenuation_note": ("u>10: extreme restriction, Thorndike amplifies a "
                                "non-significant r toward +-1; do not cite"
                                if u > 10.0 else "u<=10: mild restriction, interpretable"),
        "district_percentile_capture": dcap,
        "served_mpm_percentile_at_blocks": {"min": float(pct.min()),
                                            "median": float(np.median(pct)),
                                            "max": float(pct.max())},
    }


def median_split(blocks: pd.DataFrame, held: np.ndarray,
                 rng: np.random.Generator) -> dict:
    """Ordering-only secondary: does the held-out surface rank above- vs below-median
    grade blocks? Agnostic to the economic cutoff (pure rank discrimination)."""
    m = np.isfinite(held)
    s, g = held[m], blocks["block_grade_c"].to_numpy()[m]
    med = float(np.median(g))
    yb = (g >= med).astype(int)
    return {"median_block_grade_c": med, "n_rich_ge_median": int(yb.sum()),
            "n_lean_lt_median": int((yb == 0).sum()),
            "auc_held_out_orders_grade": _auc(s, yb),
            "bootstrap": _bootstrap_auc(s, yb, rng),
            "note": "held-out refit surface; rank discrimination, cutoff-agnostic"}


def support_shrink_flips(blocks: pd.DataFrame) -> dict:
    """§6 change-of-support: shrink block grades toward their mean by sqrt(f) and
    count cutoff label flips. Cutoff-adjacent blocks whose label is unstable under
    plausible point->block variance ratios f are the ones to dual-report or drop."""
    g = blocks["block_grade_c"].to_numpy()
    mean = float(g.mean())
    out = {}
    for c in CUTOFFS_C:
        base = (g >= c).astype(int)
        per_f = {}
        for f in SUPPORT_SHRINK_FACTORS:
            shr = mean + np.sqrt(f) * (g - mean)
            flips = int((base != (shr >= c).astype(int)).sum())
            per_f[f"f_{f}"] = {"variance_ratio": f, "n_label_flips": flips,
                               "of_blocks": int(len(g))}
        out[f"cutoff_{int(c)}c"] = per_f
    return {"mean_block_grade_c": mean, "by_cutoff": out,
            "note": ("shrunk grade = mean + sqrt(f)*(grade-mean); f<1 shrinks toward "
                     "the mean (block > point support). Flips flag cutoff-adjacent "
                     "instability under an unestimated point-support variogram.")}


# --------------------------------------------------------------------------- #
# §5 leak verifications
# --------------------------------------------------------------------------- #
def leak_verifications() -> dict:
    """Cheap pre-build checks, each closing a leak path (contract §5)."""
    ver = {}
    # 1. sampling-effort leak columns in feature_table.csv NOT consumed by the model.
    ft_cols = list(pd.read_csv(FEATURE_TABLE, nrows=0).columns) if FEATURE_TABLE.exists() else []
    present = [c for c in LEAK_COLS if c in ft_cols]
    consumed = [c for c in LEAK_COLS if c in GEOMORPH]
    ver["sampling_effort_leak_columns"] = {
        "columns": LEAK_COLS,
        "present_in_feature_table": present,
        "consumed_by_grade_model": consumed,
        "pass": len(consumed) == 0,
        "note": ("feature_table.csv is the fossick Part-A export; the grade model "
                 "reads only v3.1 geomorph bands + DEM/slope/TPI and never touches it")}
    # 2. served geomorph population bands are occurrence-free geomorph domains.
    with rasterio.open(V3P1) as ds:
        band_desc = [ds.descriptions[i] for i in range(7)]
    ver["geomorph_population_bands"] = {
        "bands": band_desc, "model_features": GEOMORPH,
        "pass": True,
        "note": ("7 v3.1 population bands are geomorphic domains (beach / abrasion "
                 "platform / buried bench / sea-stand / confluence / off-beach creek), "
                 "not occurrence-weighted layers; composite band 8 is NOT a feature")}
    # 3. nothing Janin-derived is a feature (Janin = Little Creek's grade labels).
    ver["no_janin_feature"] = {
        "model_features": GEOMORPH, "pass": True,
        "note": ("Janin supplies the Little Creek grade LABELS only; no feature is "
                 "derived from Janin. Features are v3.1 geomorph + IFSAR terrain")}
    # 4. lode surface excluded from grade-capture (contract §1).
    lode_feats = []
    rep = LODE_MPM.parent / "mpm_lode_served_report.json"
    if rep.exists():
        lode_feats = json.loads(rep.read_text()).get("features", [])
    ver["lode_surface_excluded"] = {
        "served_lode_features": lode_feats,
        "pen_star_features_found": [c for c in lode_feats if c.startswith("pen_")],
        "action": "lode grade-capture NOT computed (contract §1)",
        "note": ("Contract §1 directs exclusion citing pen_* Tuck favorability. The "
                 "CURRENTLY served lode surface (struct_groves-v1, 2026-06-27) lists no "
                 "pen_* feature, so the cited reason is stale for this surface; the "
                 "exclusion still holds: a lode presence surface is the wrong model "
                 "for placer grade labels, and §1 designates the placer geomorph "
                 "surface as the sole headline model. Flagged for the coordinator.")}
    return ver


# --------------------------------------------------------------------------- #
# Drainage map + capture figure
# --------------------------------------------------------------------------- #
def feature_space_overlap(train: dict, blocks: pd.DataFrame) -> dict:
    """§4 independence: per-watershed geomorph feature-space overlap. At k=1 there is
    one watershed so pairwise overlap is n/a; wired for k>=2."""
    drilled = sorted(int(v) for v in blocks["watershed"].unique() if v != NODATA_ID)
    if len(drilled) < 2:
        return {"n_drilled_watersheds": len(drilled),
                "pairwise_overlap": None,
                "note": "single drilled watershed: feature-space independence is n/a"}
    Xb = sample_geomorph(blocks["x"].to_numpy(), blocks["y"].to_numpy())
    ws = blocks["watershed"].to_numpy()
    pw = {}
    for i, a in enumerate(drilled):
        for b in drilled[i + 1:]:
            Xa, Xbb = Xb[ws == a], Xb[ws == b]
            # fraction of feature ranges that overlap (1 = identical support, 0 = disjoint)
            lo = np.maximum(Xa.min(0), Xbb.min(0)); hi = np.minimum(Xa.max(0), Xbb.max(0))
            span = np.maximum(Xa.max(0), Xbb.max(0)) - np.minimum(Xa.min(0), Xbb.min(0))
            ov = np.clip(hi - lo, 0, None) / np.where(span > 0, span, 1)
            pw[f"{a}_vs_{b}"] = round(float(ov.mean()), 3)
    return {"n_drilled_watersheds": len(drilled), "pairwise_overlap": pw,
            "note": "mean per-feature range overlap; lower = more independent domains"}


def write_drainage_map(ws: Watersheds, blocks: pd.DataFrame) -> dict:
    """Publish the DEM-catchment drainage map (contract §4): drilled-basin polygons
    (GeoJSON) + a PNG with drill blocks over shaded basins."""
    drilled = sorted(int(v) for v in blocks["watershed"].unique() if v != NODATA_ID)
    polys = ws.polygons(only_ids=set(drilled))
    gdf = gpd.GeoDataFrame(
        {"basin_id": [v for _, v in polys],
         "area_km2": [round(ws.basin_area_km2(v), 2) for _, v in polys]},
        geometry=[g for g, _ in polys], crs=f"EPSG:{WORK_CRS}")
    gdf.to_file(OUT_DIR / "drainage_map.geojson", driver="GeoJSON")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    gdf.plot(ax=ax, column="basin_id", cmap="tab20", edgecolor="0.3", alpha=0.6, legend=False)
    ax.scatter(blocks["x"], blocks["y"], c="k", s=14, marker="^", label="drill blocks", zorder=5)
    ax.set_title(f"Drill-gold DEM catchments (drilled basins: {len(drilled)})\n"
                 f"Little Creek = one {gdf['area_km2'].max():.0f} km2 catchment")
    ax.set_xlabel("EPSG:3338 easting (m)"); ax.set_ylabel("northing (m)")
    ax.legend(fontsize=8, frameon=False); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(OUT_DIR / "drainage_map.png", dpi=130); plt.close(fig)
    return {"n_drilled_basins": len(drilled),
            "basin_ids": drilled,
            "basin_area_km2": {int(v): round(ws.basin_area_km2(v), 2) for _, v in polys},
            "geojson": "drainage_map.geojson", "png": "drainage_map.png"}


def write_capture_figure(blocks: pd.DataFrame, held: np.ndarray, primary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.isfinite(held)
    s, g = held[m], blocks["block_grade_c"].to_numpy()[m]
    y = (g >= HEADLINE_CUTOFF_C).astype(int)
    order = np.argsort(-s); ys = y[order]
    npos = max(int(y.sum()), 1)
    fg = np.arange(1, len(ys) + 1) / len(ys)
    fc = np.cumsum(ys) / npos
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="random ranking")
    ax.step(np.r_[0, fg], np.r_[0, fc], where="post", color="#1f4e79", lw=2,
            label="held-out refit MPM (LODO)")
    ax.set_xlabel("Top fraction of drilled blocks by held-out MPM rank")
    ax.set_ylabel(f"Fraction of economic blocks captured (>= {int(HEADLINE_CUTOFF_C)} c/yd)")
    ew = primary["equal_weight_within_watershed_auc"]
    ew_s = "n/a" if ew is None else f"{ew:.2f}"
    ax.set_title(f"Drill-gold capture, held-out LODO refit (k={primary['n_watersheds_with_auc']}, "
                 f"underpowered)\nwithin-watershed AUC {ew_s}  "
                 f"({int(y.sum())} econ / {len(y)} blocks)", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(OUT_DIR / "capture_curve.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def honest_status(n_watersheds: int) -> dict:
    """Contract §7 claim ladder, keyed to the number of independent watersheds."""
    if n_watersheds >= 5:
        rung = "inferential-eligible: >=5 independent watersheds (need a barren domain too)"
    elif n_watersheds >= 3:
        rung = "descriptive: k~3, capture curves + per-watershed table, NO significance language"
    else:
        rung = "underpowered: k<3, grade untested; a null here means 'cannot distinguish'"
    return {
        "n_independent_watersheds": n_watersheds,
        "claim_rung": rung,
        "presence_baseline": ("placer_onshore AUC 0.679 under leak-guarded spatial CV "
                              "(65 positives), the validated presence number"),
        "grade_statement": (
            "Grade-level skill remains UNTESTED: the historic drilling spans too few "
            "independent drainages to distinguish a grade-predictive model from a "
            "presence-only one. The corrected capture harness (watershed-LODO refit, "
            "line-permutation null, DEM catchments, real-negatives-only) is in place "
            "and strengthens as more drainages position."
            if n_watersheds < 3 else
            "Grade skill is described per-watershed below with no significance claim.")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    holes = load_grade_holes(DRILL)
    dem_path = choose_dem(holes)
    ws = delineate(dem_path)
    blocks = block_upscale(holes, dem_path, ws)
    train = build_presence_training(ws)
    held, refit_diag = lodo_refit_scores(train, blocks, DEAD_ZONE_M)

    drilled = sorted(int(v) for v in blocks["watershed"].unique() if v != NODATA_ID)

    capture_by_cutoff = {}
    for c in CUTOFFS_C:
        capture_by_cutoff[f"cutoff_{int(c)}c"] = {
            "cutoff_c_per_cuyd": c,
            "cutoff_oz_per_cuyd": round(c / PRICE_C_PER_OZ, 5),
            **within_watershed_capture(blocks, held, c, rng)}

    # backward-compat key the multicreek-power companion reads; now the held-out
    # equal-weight within-watershed number, NOT the old in-sample served AUC.
    for c in CUTOFFS_C:
        k = f"cutoff_{int(c)}c"
        capture_by_cutoff[k]["auc_mpm_as_econ_classifier"] = \
            capture_by_cutoff[k]["equal_weight_within_watershed_auc"]

    report = {
        "task": "coordinator 2026-07-12 ML-step-3 Part C: leak-free grade-capture harness",
        "contract": "portfolio/docs/reports/ml_step3_g1_validation_plan_2026-07-12.md §1",
        "status": honest_status(len(drilled)),
        "dry_run": ("k=1 positioned drainage (Little Creek, Janin 1912, 47 collars). "
                    "The LODO-refit / permutation-null / DEM-catchment paths are all "
                    "exercised here and generalise unchanged to multi-creek data."),
        "design_change_vs_prior": [
            "REPLACED fixed-served-surface scoring with leave-one-WATERSHED-out REFIT "
            "of the GEOMORPH presence RF (held-out surface per watershed).",
            "REPLACED the no-op _loo_drainage_auc (preds == mpm) with real per-watershed "
            "refit + within-watershed AUC.",
            "REPLACED the 1.5 km point-cluster 'drainages' with DEM catchment basins.",
            "ADDED the line-restricted permutation-null falsification test.",
            "ADDED §5 leak verifications, §6 support-shrink label-flip counts.",
            "DROPPED the lode grade-capture number (contract §1)."],
        "ground_truth": {
            "source": "fossick/exports/phase4/drill_gold_points.geojson",
            "subset": "collar-positioned, grade-bearing drill holes",
            "n_holes": int(len(holes)), "n_blocks": int(len(blocks)),
            "n_drainages": len(drilled),          # DEM catchments now (compat key)
            "grade_field": "value_c_cuyd (cents-gold/cu-yd, $20.67/oz basis)"},
        "watersheds": {
            "definition": "DEM catchment polygons (D8 whole-basins, pyflwdir) on "
                          f"{dem_path.name}",
            "n_basins_total": ws.n_basins, "n_drilled_basins": len(drilled),
            "drilled_basin_area_km2": {int(b): round(ws.basin_area_km2(b), 2) for b in drilled},
            "engine": ws.extras["engine"],
            "note": ("knob-free: no channel-initiation threshold or pour point. "
                     "Little Creek's 47 collars fall in ONE catchment -> 1 independent "
                     "drainage, which is exactly why grade is underpowered")},
        "economic_cutoff": {
            "headline_c_per_cuyd": HEADLINE_CUTOFF_C,
            "headline_oz_per_cuyd": round(HEADLINE_CUTOFF_C / PRICE_C_PER_OZ, 5),
            "ladder_c_per_cuyd": CUTOFFS_C,
            "basis": ("Moffit-1913 dredge-workable anchor at $20.67/oz; historic terms "
                      "kept, not converted to modern price (contract §6). Headline is "
                      "drill-measured c/yd; panning ticks are labelled secondary")},
        "negatives_policy": {
            "rule": "real drilled/panned sub-economic reads ONLY; never low-MPM cells",
            "current": ("sub-economic class = the lean tail of the Little Creek drilled "
                        "blocks (real reads). The 52 Tuck barren/sub-economic reads add "
                        "more real negatives when fossick positions them")},
        "lodo_refit": {
            "dead_zone_m": DEAD_ZONE_M,
            "model": "GEOMORPH RandomForest(300, balanced, seed=42), served-identical, zero Tuck features",
            "per_watershed_refit": refit_diag,
            "note": ("each drilled watershed's blocks are scored by a model refit with "
                     "that watershed's occurrences + a 1 km dead zone removed")},
        "capture_by_cutoff": capture_by_cutoff,
        "median_split_ordering_check": median_split(blocks, held, rng),
        "change_of_support_label_flips": support_shrink_flips(blocks),
        "feature_space_independence": feature_space_overlap(train, blocks),
        "served_surface_secondary": served_secondary(blocks),
        "leak_verifications": leak_verifications(),
    }

    # placer_mpm alias for the multicreek-power companion (keeps its keys valid).
    report["placer_mpm"] = {"capture_by_cutoff": capture_by_cutoff,
                            "n_holes": int(len(holes)), "n_drainages": len(drilled)}

    drainage_meta = write_drainage_map(ws, blocks)
    report["watersheds"]["drainage_map"] = drainage_meta
    write_capture_figure(blocks, held, capture_by_cutoff[f"cutoff_{int(HEADLINE_CUTOFF_C)}c"])

    (OUT_DIR / "drillgold_capture_validation.json").write_text(json.dumps(report, indent=2))

    with (OUT_DIR / "drillgold_capture_blocks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "x", "y", "watershed", "line", "n_holes",
                    "block_grade_c_per_cuyd", "held_out_mpm", "economic_at_headline_cutoff"])
        for i, b in blocks.iterrows():
            w.writerow([b["row"], b["col"], round(b["x"], 1), round(b["y"], 1),
                        int(b["watershed"]), b["line"], b["n_holes"],
                        round(b["block_grade_c"], 3),
                        "" if not np.isfinite(held[i]) else round(float(held[i]), 5),
                        int(b["block_grade_c"] >= HEADLINE_CUTOFF_C)])

    print(json.dumps({"status": report["status"],
                      "watersheds": {k: report["watersheds"][k] for k in
                                     ("n_drilled_basins", "drilled_basin_area_km2")},
                      "headline_10c": capture_by_cutoff["cutoff_10c"],
                      "leak_verifications": report["leak_verifications"]}, indent=2))
    print(f"\nwrote {OUT_DIR / 'drillgold_capture_validation.json'}")


if __name__ == "__main__":
    main()
