"""Powered retest: add the Tuck-1942 MS-joined coastal positives to the KG-8 and
re-run the per-subtype local-source CV.

Round-4B addendum, step 2. ``placer_subtype_local_source.py`` graded the KG
placer positives and returned a null set by sample size: only 5 abrasion-platform
and 3 strandline-beach (true-beach) positives carried a clean coastal type, too
few to grade whether distance-to-lode predicts the abrasion-platform subtype while
staying null for true-beach.

fossick's Tuck-1942 extraction (181 typed areas, 72 coastal) was joined to
authoritative MS-survey coordinates by ``tuck_placer_join_coords.py``. The join is
precision-bound and the result is asymmetric: of 72 coastal areas only 9 carry an
MS-survey coordinate (8 true-beach, 1 abrasion-platform). The abrasion-platform
paystreaks (submarine, offshore, buried strandlines, Monroeville / Intermediate /
Present / Center beaches) were never patented as individual mineral surveys, so
they do not join and stay flagged pinpoint-needed. This script adds the coastal
joins that ARE placed and new (de-duplicated against the KG positives), then runs
the identical two-test machinery on the baseline (KG only) and the powered
(KG + Tuck) positive sets, so the delta is attributable to the added positives.

Run: PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_subtype_local_source_tuck
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.nome_placer.f1_leak_guarded_rebaseline import (
    DEM, GEOMORPH, POP, SLOPE, TPI, V3P1, samp)
from scripts.nome_placer.placer_subtype_local_source import (
    MILE_M, build_covariates, distributional, newton_axis, subset_cv,
    write_summary_csv, load_typed)

OUT_DIR = Path("data/derived/nome_placer/subtype_local_source")
TUCK_POS = OUT_DIR / "tuck_placer_positives.csv"
KG_DEDUP_M = 250.0      # a Tuck join nearer than this to a KG positive is the same ground
TUCK_DISTINCT_M = 150.0  # collapse Tuck areas that resolve to the same claim/coord


def feats_at(ex: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Sample the GEOMORPH base-feature stack (V3P1 population priors + DEM/slope/TPI)
    at arbitrary coords, identical to load_placer()'s construction."""
    feat = pd.concat([
        samp(V3P1, ex, ny, list(range(1, 8)), POP),
        samp(DEM, ex, ny, [1], ["dem"]),
        samp(SLOPE, ex, ny, [1], ["slope"]),
        samp(TPI, ex, ny, [1], ["tpi"]),
    ], axis=1).fillna(-999.0)
    return feat[GEOMORPH].to_numpy(np.float32)


def load_tuck_coastal_new(kg_pos_xy: np.ndarray) -> pd.DataFrame:
    """Tuck-joined coastal positives that are new ground (>KG_DEDUP_M from any KG
    positive) and de-duplicated to distinct claim coordinates."""
    rows = [r for r in csv.DictReader(TUCK_POS.open())
            if r["coordinate_source"] == "ms_join"
            and r["deposit_type"] in ("true-beach", "abrasion-platform")]
    tree = cKDTree(kg_pos_xy)
    kept: list[dict] = []
    kept_xy: list[tuple[float, float]] = []
    for r in rows:
        x, y = float(r["x_3338"]), float(r["y_3338"])
        if tree.query([x, y])[0] < KG_DEDUP_M:
            continue
        if any(np.hypot(x - sx, y - sy) < TUCK_DISTINCT_M for sx, sy in kept_xy):
            continue
        kept_xy.append((x, y))
        kept.append({"name": r["name"], "type": r["type_canon"], "x": x, "y": y,
                     "matched_ms": r["matched_ms"], "basis": f"Tuck MS-join ({r['best_confidence']})"})
    return pd.DataFrame(kept)


def assemble(typ_kg: pd.DataFrame, X_kg: np.ndarray, y_kg: np.ndarray, coords_kg: np.ndarray,
             tuck: pd.DataFrame):
    """Insert the Tuck positive block after the KG positive block so positives stay
    contiguous at the front: [KG pos | Tuck pos | background]."""
    n_kg = len(typ_kg)
    if tuck.empty:
        return typ_kg, X_kg, y_kg, coords_kg
    tx, ty = tuck["x"].to_numpy(), tuck["y"].to_numpy()
    X_t = feats_at(tx, ty)
    X = np.vstack([X_kg[:n_kg], X_t, X_kg[n_kg:]])
    coords = np.vstack([coords_kg[:n_kg], np.column_stack([tx, ty]), coords_kg[n_kg:]])
    y = np.concatenate([y_kg[:n_kg], np.ones(len(tuck), int), y_kg[n_kg:]])
    typ = pd.concat([
        typ_kg[["name", "type", "basis", "x", "y"]],
        tuck[["name", "type", "basis", "x", "y"]],
    ], ignore_index=True)
    return typ, X.astype(np.float32), y, coords


def run_arm(typ: pd.DataFrame, X_base: np.ndarray, y: np.ndarray, coords: np.ndarray) -> dict:
    cov, cov_meta = build_covariates(coords)
    dist = distributional(typ, cov, y)
    cvres = subset_cv(typ, X_base, y, coords, cov)
    counts = typ["type"].value_counts().to_dict()
    return {"counts": {g: int(counts.get(g, 0)) for g in
                       ("abrasion_platform", "strandline_beach", "upland_residual", "broad_ambiguous")},
            "n_pos": len(typ), "cov_meta": cov_meta,
            "test1_distributional": dist, "test2_subset_leak_guarded_cv": cvres}


def fmt_cv(arm: dict, cov: str) -> str:
    block = arm["test2_subset_leak_guarded_cv"][cov]
    out = []
    for m in ("pooled", "coastal_beach_all", "abrasion_platform", "strandline_beach"):
        r = block.get(m, {})
        out.append(f"    {m:18s} d={r.get('point')} CI={r.get('ci95')} "
                   f"P(d>0)={r.get('p_gt_0')} n_pos={r.get('n_pos')}")
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    typ_kg, X_kg, y_kg, coords_kg, base_names = load_typed()
    n_kg = len(typ_kg)
    tuck = load_tuck_coastal_new(coords_kg[:n_kg])

    print(f"KG positives: {n_kg}  | Tuck coastal joins added (new, distinct): {len(tuck)}")
    if not tuck.empty:
        print(tuck.groupby("type").size().to_string())
        for _, r in tuck.iterrows():
            print(f"  +[{r['type'][:5]}] {r['name'][:46]:46s} MS={r['matched_ms']}")

    typ_p, X_p, y_p, coords_p = assemble(typ_kg, X_kg, y_kg, coords_kg, tuck)

    print("\n--- BASELINE (KG only) ---")
    baseline = run_arm(typ_kg, X_kg, y_kg, coords_kg)
    print("--- POWERED (KG + Tuck coastal) ---")
    powered = run_arm(typ_p, X_p, y_p, coords_p)

    npt, _, az = newton_axis()
    out = {
        "question": "With the Tuck-1942 MS-joined coastal positives added to the KG-8, does "
                    "distance-to-lode predict the abrasion-platform subtype while staying null "
                    "for true-beach (strandline)? Was the pooled null masking a local-source signal?",
        "join_summary": {
            "source": "fossick tuck1942_areas.json -> MS-survey join (tuck_placer_join_coords.py)",
            "coastal_areas_total": 72, "coastal_ms_joined": 9,
            "coastal_ms_joined_true_beach": 8, "coastal_ms_joined_abrasion_platform": 1,
            "coastal_pinpoint_needed": 63,
            "tuck_coastal_added_new_distinct": int(len(tuck)),
            "tuck_added_counts": (tuck.groupby("type").size().to_dict() if not tuck.empty else {}),
            "note": "Abrasion-platform paystreaks (submarine/offshore/buried strandlines, "
                    "Monroeville/Intermediate/Present/Center beaches) were not patented as "
                    "MS claims, so they do not join and stay pinpoint-needed. The added power "
                    "lands almost entirely on the true-beach (strandline) arm.",
        },
        "type_synonyms": {"strandline_beach": "Tuck 'true-beach' (winnowed marine drift; Hudson "
                          "strandline-beach)", "abrasion_platform": "Tuck/Hudson abrasion-platform "
                          "(local-source bedrock-platform paystreak)"},
        "baseline_kg_only": baseline,
        "powered_kg_plus_tuck": powered,
        "newton_belt_axis_azimuth_deg": az,
        "scheme": {"cv": "F1 leak-guarded spatial CV (contiguous folds, 1 km dead zone), "
                   "RandomForest(300, balanced, seed=42), 2000-resample bootstrap",
                   "base_features": base_names,
                   "dedup": {"kg_dedup_m": KG_DEDUP_M, "tuck_distinct_m": TUCK_DISTINCT_M}},
    }
    (OUT_DIR / "placer_subtype_local_source_tuck.json").write_text(json.dumps(out, indent=2, default=str))

    # combined typing table actually used (positives only; no raw paystreak data)
    typ_p.assign(coordinate=lambda d: ["kg"] * n_kg + ["tuck_ms_join"] * (len(d) - n_kg)) \
        .to_csv(OUT_DIR / "placer_typing_tuck.csv", index=False)
    write_summary_csv(powered["test1_distributional"], powered["test2_subset_leak_guarded_cv"],
                      OUT_DIR / "placer_subtype_local_source_tuck.csv")

    print("\n=== per-subtype counts (baseline -> powered) ===")
    for g in ("abrasion_platform", "strandline_beach", "upland_residual"):
        print(f"  {g:18s} {baseline['counts'][g]:3d} -> {powered['counts'][g]:3d}")
    for cov in ("dist_to_36a_lode", "dist_to_newton_axis"):
        print(f"\nPOWERED CV marginal AUC delta ({cov}):")
        print(fmt_cv(powered, cov))
    print(f"\nwrote {OUT_DIR / 'placer_subtype_local_source_tuck.json'}")
    print(f"wrote {OUT_DIR / 'placer_typing_tuck.csv'}")
    print(f"wrote {OUT_DIR / 'placer_subtype_local_source_tuck.csv'}")


if __name__ == "__main__":
    main()
