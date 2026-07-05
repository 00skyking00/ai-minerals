"""F3 leak-free arm: does the KG help once a cell cannot see its own record?

The seven arms in ``f3_kg_marginal_lift.py`` all share the coverage confound the
F3 report diagnoses: every KG feature is non-null (or near-zero distance) exactly
where a record exists, and the model's positives ARE those records, so the +0.19
to +0.34 AUC change is target leakage, not signal. This script runs the one arm
that can speak to a real signal:

  kg_loo : fold-aware, leave-one-out spatial fields ONLY
           (kg_dist_occurrence_m, kg_dist_claim_m, kg_claim_density).

For each CV fold the visible ground set excludes every ground inside the held-out
fold's blocks, and each cell's spatial fields are computed from that fold's
training-region grounds only, never from a ground in the cell's own grid cell. A
held-out positive therefore cannot see its own record, and neither can a training
positive (the leave-one-out). The coverage-confounded features (kg_present, the
ground-attribute fields, the prose embedding) are dropped: they cannot be made
leak-free for a presence model whose positives are the records.

Everything else matches F1 / the leaky run so the marginal change is comparable:
contiguous folds, r = 1 km dead-zone, residual-variogram block size taken from
the base model (so the folds are identical to the base arm), RandomForest(300,
balanced), seed 42. A paired, class-stratified bootstrap of the pooled OOF gives
a 95% interval on the change, because the positive counts are small (30 to 93).

Run: PYTHONPATH=. .venv/bin/python -m scripts.nome_placer.f3_kg_loo
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from ai_minerals import kg_rasterize as kr
from ai_minerals.spatial_cv import (
    assign_block_folds,
    block_size_from_range,
    default_rf_factory,
    residual_variogram,
    subset_auc,
)
from scripts.nome_placer.f1_leak_guarded_rebaseline import (
    load_lode_district,
    load_lode_inbox,
    load_placer,
)
from scripts.nome_placer.f3_paths import KG_FEATURES_DIR as FEAT, KG_JSONLD

RNG = 42
N_FOLDS = 10
DEAD_ZONE_M = 1000.0
CLAIM_RADIUS_M = 400.0
N_BOOT = 2000
IN_BOX_TPL = Path("data/derived/nome_placer/prospectivity_v1p5/"
                  "nome_placer_prospectivity_v1p5_v3p1_3338.tif")
DISTRICT_TPL = Path("data/derived/nome_placer/covariates_mpm/_district_grid_template_3338.tif")
OUT_DIR = Path("data/derived/nome_placer/f3_kg_marginal_lift")
LEAKY_JSON = OUT_DIR / "f3_kg_marginal_lift.json"


def _base_block_size(X_base: np.ndarray, y: np.ndarray, coords: np.ndarray) -> float:
    """Same block-size recipe as ``leak_guarded_evaluate`` on the base features, so
    kg_loo and the committed base arm share identical folds."""
    factory = default_rf_factory(seed=RNG)
    vfit = residual_variogram(X_base, y, coords, model_factory=factory, seed=RNG)
    bs = block_size_from_range(vfit.range_m, multiplier=2.0)
    ext = coords.max(0) - coords.min(0)
    cap = float(np.sqrt(max(ext.prod(), 1.0) / (2.0 * N_FOLDS)))
    if bs > cap:
        bs = float(np.ceil(cap / 100.0) * 100.0)
    return bs


def _paired_bootstrap(y: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray) -> dict:
    """Class-stratified paired bootstrap of (AUC_b - AUC_a) on the pooled OOF."""
    scored = ~np.isnan(oof_a) & ~np.isnan(oof_b)
    ys, a, b = y[scored], oof_a[scored], oof_b[scored]
    pos, neg = np.where(ys == 1)[0], np.where(ys == 0)[0]
    rng = np.random.default_rng(RNG)
    da, db, dd = [], [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        aa, bb = roc_auc_score(ys[idx], a[idx]), roc_auc_score(ys[idx], b[idx])
        da.append(aa); db.append(bb); dd.append(bb - aa)
    pct = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"base_auc_ci95": pct(da), "loo_auc_ci95": pct(db),
            "delta_ci95": pct(dd), "delta_mean": round(float(np.mean(dd)), 4),
            "delta_p_gt_0": round(float(np.mean(np.array(dd) > 0)), 4)}


def run_dataset(name: str, scope: str, loader, template_path: Path,
                ground: dict[str, np.ndarray]) -> dict:
    X_df, y, coords, names, in_box = loader()
    X_base = X_df.to_numpy(np.float32)
    y = np.asarray(y, int)
    coords = np.asarray(coords, float)
    tpl = kr.GridTemplate.from_raster(template_path)

    samp_xy, samp_rc = kr.sample_cell_centroids(coords, tpl)
    bs = _base_block_size(X_base, y, coords)
    block_id, fold = assign_block_folds(coords, y, block_size_m=bs, n_folds=N_FOLDS,
                                        seed=RNG, strategy="contiguous")
    x0, y0 = float(coords[:, 0].min()), float(coords[:, 1].min())
    s_bx, s_by = kr.block_xy(coords, x0, y0, bs)
    test_xy = {f: set() for f in range(N_FOLDS)}
    for bx, by, fl in zip(s_bx, s_by, fold):
        test_xy[fl].add((int(bx), int(by)))

    occ_bx, occ_by = kr.block_xy(ground["occ_xy"], x0, y0, bs) if len(ground["occ_xy"]) \
        else (np.empty(0, int), np.empty(0, int))
    clm_bx, clm_by = kr.block_xy(ground["claim_xy"], x0, y0, bs) if len(ground["claim_xy"]) \
        else (np.empty(0, int), np.empty(0, int))
    fill_dist = float(np.hypot(*(coords.max(0) - coords.min(0))))

    factory = default_rf_factory(seed=RNG)
    oof_base = np.full(len(y), np.nan)
    oof_loo = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        test = fold == f
        if not test.any():
            continue
        train = ~test
        d, _ = cKDTree(coords[test]).query(coords, k=1)
        train &= d >= DEAD_ZONE_M
        if len(np.unique(y[train])) < 2:
            continue
        tset = test_xy[f]
        occ_vis = np.array([(int(a), int(b)) not in tset for a, b in zip(occ_bx, occ_by)], bool)
        clm_vis = np.array([(int(a), int(b)) not in tset for a, b in zip(clm_bx, clm_by)], bool)
        feats = kr.loo_spatial_features(
            samp_xy, samp_rc,
            ground["occ_xy"][occ_vis], ground["occ_rc"][occ_vis],
            ground["claim_xy"][clm_vis], ground["claim_rc"][clm_vis],
            claim_radius_m=CLAIM_RADIUS_M, fill_dist=fill_dist,
        ).to_numpy(np.float32)
        mb = factory().fit(X_base[train], y[train])
        oof_base[test] = mb.predict_proba(X_base[test])[:, 1]
        Xl = np.column_stack([X_base, feats]).astype(np.float32)
        ml = factory().fit(Xl[train], y[train])
        oof_loo[test] = ml.predict_proba(Xl[test])[:, 1]

    auc_base = subset_auc(y, oof_base)
    auc_loo = subset_auc(y, oof_loo)
    auc_base_in = subset_auc(y, oof_base, mask=in_box) if in_box is not None else None
    auc_loo_in = subset_auc(y, oof_loo, mask=in_box) if in_box is not None else None
    delta = None if (auc_base is None or auc_loo is None) else round(auc_loo - auc_base, 3)
    boot = _paired_bootstrap(y, oof_base, oof_loo)
    scored = ~np.isnan(oof_base)

    print(f"[{name}] base={auc_base:.3f} kg_loo={auc_loo:.3f} "
          f"d={delta:+.3f} ci95={boot['delta_ci95']} "
          f"(n_pos_scored={int(y[scored].sum())})")
    return {
        "name": name, "scope": scope, "n": int(len(y)), "n_pos": int(y.sum()),
        "block_size_m": bs, "n_occ_grounds": int(len(ground["occ_xy"])),
        "n_claim_grounds": int(len(ground["claim_xy"])),
        "n_scored": int(scored.sum()), "n_pos_scored": int(y[scored].sum()),
        "base": {"auc_district": auc_base, "auc_in_box": auc_base_in},
        "kg_loo": {"auc_district": auc_loo, "auc_in_box": auc_loo_in},
        "auc_delta": delta, "bootstrap": boot,
    }


def _consistency_note(datasets: list[dict]) -> dict:
    """Confirm the recomputed base AUC matches the committed leaky-run base, so the
    kg_loo folds are the same folds the +0.19..+0.34 arms were graded on."""
    if not LEAKY_JSON.exists():
        return {"checked": False, "reason": "leaky-run JSON absent"}
    leaky = {d["name"]: d["arms"]["base"]["auc_district"]
             for d in json.loads(LEAKY_JSON.read_text())["datasets"]}
    out = {}
    for d in datasets:
        prev, now = leaky.get(d["name"]), d["base"]["auc_district"]
        out[d["name"]] = {"leaky_run_base": prev, "kg_loo_run_base": now,
                          "abs_diff": None if prev is None else round(abs(prev - now), 4)}
    return {"checked": True, "by_dataset": out}


def write_csv(datasets: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scope", "arm", "auc_district", "auc_in_box",
                    "auc_delta_vs_base", "delta_ci95_lo", "delta_ci95_hi", "n_pos_scored"])
        for d in datasets:
            w.writerow([d["name"], d["scope"], "base", d["base"]["auc_district"],
                        d["base"]["auc_in_box"], "", "", "", d["n_pos_scored"]])
            lo, hi = d["bootstrap"]["delta_ci95"]
            w.writerow([d["name"], d["scope"], "kg_loo", d["kg_loo"]["auc_district"],
                        d["kg_loo"]["auc_in_box"], d["auc_delta"], lo, hi, d["n_pos_scored"]])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading grounds (occurrence + claim points, deduped to cells) ...")
    ents = kr.load_feature_entities(FEAT, KG_JSONLD)
    in_box_tpl = kr.GridTemplate.from_raster(IN_BOX_TPL)
    district_tpl = kr.GridTemplate.from_raster(DISTRICT_TPL)
    g_inbox = kr.ground_points_by_extent(ents, in_box_tpl)
    g_district = kr.ground_points_by_extent(ents, district_tpl)

    datasets = [
        run_dataset("placer_onshore", "in_box", load_placer, IN_BOX_TPL, g_inbox),
        run_dataset("lode_inbox", "in_box", load_lode_inbox, IN_BOX_TPL, g_inbox),
        run_dataset("lode_district", "district", load_lode_district, DISTRICT_TPL, g_district),
    ]
    out = {
        "arm": "kg_loo",
        "design": ("fold-aware leave-one-out spatial fields only; visible grounds "
                   "exclude the held-out fold's blocks and the cell's own record; "
                   "coverage-confounded + prose features dropped"),
        "scheme": {"dead_zone_m": DEAD_ZONE_M, "n_folds": N_FOLDS, "seed": RNG,
                   "block_sizing": "2x residual-variogram range (from base features)",
                   "estimator": "RandomForest(300, balanced, seed=42)",
                   "fold_strategy": "contiguous", "claim_radius_m": CLAIM_RADIUS_M,
                   "bootstrap_resamples": N_BOOT, "bootstrap": "class-stratified paired"},
        "base_consistency": _consistency_note(datasets),
        "datasets": datasets,
    }
    (OUT_DIR / "f3_kg_loo.json").write_text(json.dumps(out, indent=2, default=str))
    write_csv(datasets, OUT_DIR / "f3_kg_loo.csv")
    print(f"\nwrote {OUT_DIR / 'f3_kg_loo.json'}")
    print(f"wrote {OUT_DIR / 'f3_kg_loo.csv'}")
    chk = out["base_consistency"]
    if chk.get("checked"):
        print("base-consistency vs leaky run:",
              {k: v["abs_diff"] for k, v in chk["by_dataset"].items()})


if __name__ == "__main__":
    main()
