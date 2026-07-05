"""F3: does the fossick knowledge graph help the Nome models, measured straight?

Re-runs the placer and lode presence/background models under F1's leak-guarded
spatial CV (``ai_minerals.spatial_cv``), with and without the F2 KG covariate
stack (``ai_minerals.kg_rasterize``) and with and without the prose embedding,
and reports the marginal AUC change. The plan says a null or negative result is a
valid outcome, so the number is reported straight; the arms are built to tell a
real signal apart from the coverage confound that haunts any occurrence-derived
covariate.

Arms (each = the F1 base features + a delta), graded contiguous-fold (the
stricter test) at both in-box and district scope:

  base          F1 features only (reproduces the F1 re-baseline AUC).
  +kg_present   a single "a KG record exists in this cell" flag. The CONTROL:
                how much apparent gain is just coverage, not geology.
  +kg_spatial   distance-to-claim, claim density, distance-to-occurrence. Real
                fields defined at every cell; what F1's 1 km dead-zone grades.
  +kg           the full KG stack (spatial + ground attributes).  <- the +KG headline
  +prose        the 96-d prose embedding.
  +kg+prose     both.
  +kg_label     the full stack PLUS the label-restating columns (is_placer/
                is_lode/deposit confidence/commodity flags). The LEAKAGE DEMO:
                what naive KG fusion scores when the target is fed back in.

A weight sensitivity (source_grade x confidence as sample weights, per the F2
manifest) is run once on the +kg arm.

Run: PYTHONPATH=. .venv/bin/python -m scripts.nome_placer.f3_kg_marginal_lift
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals import kg_rasterize as kr
from ai_minerals.spatial_cv import (
    LeakGuardedSpatialCV,
    block_size_from_range,
    default_rf_factory,
    leak_guarded_evaluate,
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
IN_BOX_TPL = Path("data/derived/nome_placer/prospectivity_v1p5/"
                  "nome_placer_prospectivity_v1p5_v3p1_3338.tif")
DISTRICT_TPL = Path("data/derived/nome_placer/covariates_mpm/_district_grid_template_3338.tif")
OUT_DIR = Path("data/derived/nome_placer/f3_kg_marginal_lift")

# band groups (resolved against whatever the stack actually rendered)
SPATIAL = ["kg_dist_claim_m", "kg_claim_density", "kg_dist_occurrence_m"]
CONTENT = ["source_grade_numeric", "mean_confidence", "n_geoterm_families", *kr.PROSE_CONTEXT_FLAGS]
LABEL = ["is_placer", "is_lode", "deposit_model_confidence", "commodity_main_confidence",
         *kr.PROSE_LABEL_FLAGS]
ALL_CONTENT = [*CONTENT, *LABEL]


def build_stack(template_path: Path) -> kr.KGStack:
    ents = kr.load_feature_entities(FEAT, KG_JSONLD)
    tpl = kr.GridTemplate.from_raster(template_path)
    return kr.build_kg_stack(ents, tpl, content_columns=ALL_CONTENT, embedding=True)


def _emb_cols(sampled: pd.DataFrame) -> list[str]:
    return [c for c in sampled.columns if c.startswith("prose_emb_")]


def arm_columns(sampled: pd.DataFrame) -> dict[str, list[str]]:
    present = set(sampled.columns)
    spatial = [c for c in SPATIAL if c in present]
    content = [c for c in CONTENT if c in present]
    label = [c for c in LABEL if c in present]
    emb = _emb_cols(sampled)
    kg_full = spatial + content
    return {
        "base": [],
        "kg_present": ["kg_present"] if "kg_present" in present else [],
        "kg_spatial": spatial,
        "kg": kg_full,
        "prose": emb,
        "kg_prose": kg_full + emb,
        "kg_label": kg_full + label,
    }


def evaluate_arm(X_base, y, coords, in_box, add: pd.DataFrame | None, cols: list[str],
                 *, strategy: str = "contiguous") -> dict:
    X = X_base if not cols else np.column_stack([X_base, add[cols].to_numpy(np.float32)])
    res = leak_guarded_evaluate(
        X.astype(np.float32), y, coords, in_box_mask=in_box,
        model_factory=default_rf_factory(seed=RNG), n_folds=N_FOLDS,
        dead_zone_m=DEAD_ZONE_M, fold_strategy=strategy, seed=RNG,
    )
    return {"auc_district": res.auc_district, "auc_in_box": res.auc_in_box,
            "variogram_range_m": res.variogram_range_m, "block_size_m": res.block_size_m,
            "n_features": int(X.shape[1])}


def _cv_for(X, y, coords, strategy="contiguous") -> LeakGuardedSpatialCV:
    """Same block-sizing recipe as leak_guarded_evaluate, exposed so the weight
    sensitivity grades weighted and unweighted fits on identical folds."""
    factory = default_rf_factory(seed=RNG)
    vfit = residual_variogram(X, y, coords, model_factory=factory, seed=RNG)
    bs = block_size_from_range(vfit.range_m, multiplier=2.0)
    ext = coords.max(0) - coords.min(0)
    cap = float(np.sqrt(max(ext.prod(), 1.0) / (2.0 * N_FOLDS)))
    if bs > cap:
        bs = float(np.ceil(cap / 100.0) * 100.0)
    return LeakGuardedSpatialCV(bs, n_folds=N_FOLDS, dead_zone_m=DEAD_ZONE_M,
                                seed=RNG, strategy=strategy)


def weighted_auc(X, y, coords, in_box, sample_weight) -> dict:
    cv = _cv_for(X, y, coords)
    factory = default_rf_factory(seed=RNG)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(coords, y):
        if len(np.unique(y[tr])) < 2:
            continue
        m = factory()
        if sample_weight is None:
            m.fit(X[tr], y[tr])
        else:
            m.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return {"auc_district": subset_auc(y, oof),
            "auc_in_box": subset_auc(y, oof, mask=in_box) if in_box is not None else None}


def run_dataset(name: str, scope: str, loader, stack: kr.KGStack, *,
                scatter_arms=("base", "kg"), weight_arm="kg") -> dict:
    X_df, y, coords, names, in_box = loader()
    X_base = X_df.to_numpy(np.float32)
    sampled = stack.sample_at(coords[:, 0], coords[:, 1])
    arms = arm_columns(sampled)
    print(f"\n[{name}] n={len(y)} pos={int(y.sum())} base_feats={len(names)} "
          f"kg_bands={len(sampled.columns)}")

    # Headline AUC is always auc_district = the AUC over this dataset's own extent
    # (the placer-core grid for placer/lode_inbox, the full district for
    # lode_district). auc_in_box is the placer-core subset, meaningful only for
    # lode_district, where it is the restriction-of-range alarm.
    results: dict[str, dict] = {}
    base = evaluate_arm(X_base, y, coords, in_box, sampled, arms["base"])
    base_score = base["auc_district"]
    results["base"] = base
    print(f"  {'base':12s} auc={base_score}  (in_box_subset={base['auc_in_box']})")
    for arm, cols in arms.items():
        if arm == "base":
            continue
        r = evaluate_arm(X_base, y, coords, in_box, sampled, cols)
        score = r["auc_district"]
        r["auc_delta"] = None if (score is None or base_score is None) else round(score - base_score, 3)
        results[arm] = r
        delta_s = "n/a" if r["auc_delta"] is None else f"{r['auc_delta']:+.3f}"
        print(f"  {arm:12s} auc={score}  d_auc={delta_s}  (in_box_subset={r['auc_in_box']})")

    # scatter sensitivity on a couple of arms (the F1 strategy finding persists?)
    scatter = {}
    for arm in scatter_arms:
        r = evaluate_arm(X_base, y, coords, in_box, sampled, arms[arm], strategy="balanced_scatter")
        scatter[arm] = {"auc_district": r["auc_district"], "auc_in_box": r["auc_in_box"]}
    results["_scatter"] = scatter

    # weight sensitivity on one arm: source_grade x confidence on the positives
    w = np.ones(len(y), np.float32)
    sg = sampled["source_grade_numeric"].to_numpy(np.float32)
    mc = sampled["mean_confidence"].to_numpy(np.float32)
    wpos = np.clip(sg * mc, 0.1, None)
    w[y == 1] = wpos[y == 1]
    Xw = np.column_stack([X_base, sampled[arms[weight_arm]].to_numpy(np.float32)]).astype(np.float32)
    results["_weight_sensitivity"] = {
        "arm": weight_arm,
        "unweighted": weighted_auc(Xw, y, coords, in_box, None),
        "weighted": weighted_auc(Xw, y, coords, in_box, w),
    }
    return {"name": name, "scope": scope, "n": int(len(y)), "n_pos": int(y.sum()),
            "base_features": names, "arms": results}


def write_csv(datasets: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scope", "arm", "auc_district", "auc_in_box", "auc_delta_vs_base"])
        for d in datasets:
            for arm, r in d["arms"].items():
                if arm.startswith("_"):
                    continue
                w.writerow([d["name"], d["scope"], arm, r.get("auc_district"),
                            r.get("auc_in_box"), r.get("auc_delta")])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building KG stacks (in-box 25 m, district 100 m) ...")
    in_box_stack = build_stack(IN_BOX_TPL)
    district_stack = build_stack(DISTRICT_TPL)
    cap_note = {
        "n_entities": int(len(in_box_stack.winner_idx[in_box_stack.winner_idx >= 0]) and
                          (in_box_stack.winner_idx >= 0).sum()),
        "in_box_cells_multi_ground_uncapped": int((in_box_stack.raw_count > 1).sum()),
        "in_box_max_raw_count": int(in_box_stack.raw_count.max()),
        "capped_coverage_max": int(in_box_stack.capped_coverage.max()),
        "bands": list(in_box_stack.bands)[:30],
        "n_bands": len(in_box_stack.bands),
    }

    datasets = [
        run_dataset("placer_onshore", "in_box", load_placer, in_box_stack),
        run_dataset("lode_inbox", "in_box", load_lode_inbox, in_box_stack),
        run_dataset("lode_district", "district", load_lode_district, district_stack),
    ]

    out = {
        "scheme": {"dead_zone_m": DEAD_ZONE_M, "n_folds": N_FOLDS,
                   "block_sizing": "2x residual-variogram range",
                   "estimator": "RandomForest(300, balanced, seed=42)",
                   "headline_fold_strategy": "contiguous"},
        "grids": {"in_box_res_m": 25, "district_res_m": 100, "nominal_plan_res_m": 250,
                  "note": ("KG covariates aligned to the SERVED model grids (25 m in-box, "
                           "100 m district), not the nominal 250 m: that is the grid F1 "
                           "graded on, and 250 m collapses most claim polygons to one cell, "
                           "defeating the extent-aware guard.")},
        "kg_stack": {**cap_note,
                     "area_footprint_geometry": ("centroid + 200 m disc; the F2 export "
                                                 "resolves KARDEX area_footprint grounds to "
                                                 "centroids, not surveyed polygons (export gap)")},
        "datasets": datasets,
    }
    (OUT_DIR / "f3_kg_marginal_lift.json").write_text(json.dumps(out, indent=2, default=str))
    write_csv(datasets, OUT_DIR / "f3_kg_marginal_lift.csv")
    print(f"\nwrote {OUT_DIR / 'f3_kg_marginal_lift.json'}")
    print(f"wrote {OUT_DIR / 'f3_kg_marginal_lift.csv'}")


if __name__ == "__main__":
    main()
