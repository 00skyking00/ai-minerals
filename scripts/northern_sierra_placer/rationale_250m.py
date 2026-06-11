"""Phase I.2 + v3 Phase E.4: per-cell SHAP rationale + feature importance.

For each population (placer_tertiary, placer_quaternary):

  1. Load the joblib bundle saved by train_predict_250m
     (rf_full, lgbm_full, optionally xgb_full, cal, feature_cols).
  2. Load the feature parquet for the placer AOI.
  3. Pick the top-K cells by p_cal from pop_calibrated_<pop>_250m.parquet.
  4. Run shap.TreeExplainer on every available base learner at those K
     cells; for each cell, take the top-3 features by the row-wise max
     of |SHAP| across the available learners.
  5. Write rationale_<pop>_250m.parquet with a single string column
     `top_shap_features` summarizing per-cell drivers.
  6. Write feature_importance_<pop>.csv: per-learner mean(|SHAP|) and a
     combined mean across whichever learners were available, sorted
     descending by the combined column. The columns
     mean_abs_shap_rf, mean_abs_shap_lgbm, mean_abs_shap_xgb are
     always present; mean_abs_shap_xgb is NaN when the bundle predates
     v3 Phase C.1 and does not carry xgb_full.
  7. Write recipe_match_<pop>.json with the top-5 features, the
     expected recipe list per E.4, the match count, and the gate-pass
     boolean (>=3 of 5).

SHAP is computed only for the top-K cells (default 20,000) because
TreeExplainer at full grid resolution (~800k cells) on a 400-tree RF
takes hours.

Usage:
    .venv/bin/python scripts/northern_sierra_placer/rationale_250m.py
    .venv/bin/python scripts/northern_sierra_placer/rationale_250m.py --population placer_tertiary --top-k 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
OUT_DIR = DATA_DERIVED / REGION.data_prefix
IN_FEATURES = DATA_DERIVED / f"features_{REGION.data_prefix}_250m.parquet"

POPULATIONS = ("placer_tertiary", "placer_quaternary")
DEFAULT_TOP_K = 20_000

# v3 Phase E.4 recipe-match expected features per population. The plan
# names lithology dummies as `lithology_class_*`; the v2 feature stack
# names them `litho_*`. Either prefix counts as a recipe match.
RECIPE_EXPECTED: dict[str, tuple[str, ...]] = {
    "placer_tertiary": (
        "paleochannel_likelihood",
        "distance_to_lode_m",
        "hydraulic_pit_proximity_m_buffered",
        "tertiary_terrace_likelihood",
        "lithology_class_*",
    ),
    "placer_quaternary": (
        "distance_downstream_from_lode_m",
        "catchment_au_hawkes",
        "spi_band",
        "ksn",
        "distance_to_lode_m",
        "lithology_class_*",
    ),
}

# Lithology one-hots accepted as matches to `lithology_class_*` (or the
# plan's exact `lithology_class_*` token). The v2 build emits litho_*;
# v3 may switch to lithology_class_*. Match either.
_LITHO_PREFIXES = ("lithology_class_", "litho_")


def _matches_expected(feature: str, expected: str) -> bool:
    """True if `feature` (an actual column) satisfies the recipe entry."""
    if expected == feature:
        return True
    # Glob-style trailing wildcard.
    if expected.endswith("_*"):
        prefix = expected[:-1]  # drop the '*', keep the trailing '_'
        if feature.startswith(prefix):
            return True
        # Recipe says lithology_class_*; accept litho_* as the same family.
        if prefix in ("lithology_class_",):
            return any(feature.startswith(p) for p in _LITHO_PREFIXES)
    return False


def _count_recipe_matches(top_features: list[str], expected: tuple[str, ...]) -> int:
    """How many of `top_features` (top-5) match any entry in `expected`.

    Each lithology family (litho_* / lithology_class_*) counts once, not
    once per one-hot dummy.
    """
    matched = 0
    litho_credit_used = False
    for feat in top_features:
        is_litho = any(feat.startswith(p) for p in _LITHO_PREFIXES)
        if is_litho:
            if not litho_credit_used and any(
                e.endswith("_*") and e[:-1] in ("lithology_class_", "litho_")
                for e in expected
            ):
                matched += 1
                litho_credit_used = True
            continue
        if any(_matches_expected(feat, e) for e in expected if not e.endswith("_*")):
            matched += 1
    return matched


def _build_X(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Build the feature matrix used at training time, with the same NaN sentinel."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"features parquet missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    # train_predict_250m fills NaN with -9999.0 before fit; reproduce here.
    return df[feature_cols].fillna(-9999.0).to_numpy(dtype=np.float32)


def _top_k_indices(p_cal: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k cells by p_cal (NaN-safe; NaN ranks last)."""
    valid = np.isfinite(p_cal)
    n_valid = int(valid.sum())
    k = min(k, n_valid)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    score = np.where(valid, p_cal, -np.inf)
    return np.argpartition(-score, k - 1)[:k]


def _per_cell_top3(
    shap_values: np.ndarray,
    feature_cols: list[str],
) -> list[str]:
    """For each row in shap_values, format the top-3 features by |SHAP|."""
    n_rows = shap_values.shape[0]
    abs_shap = np.abs(shap_values)
    top3_idx = np.argsort(-abs_shap, axis=1)[:, :3]
    out: list[str] = []
    for i in range(n_rows):
        parts = []
        for j in top3_idx[i]:
            val = shap_values[i, j]
            parts.append(f"{feature_cols[j]}({val:+.3f})")
        out.append(";".join(parts))
    return out


def _shap_for_estimator(estimator, X: np.ndarray):
    """TreeExplainer on a tree-based estimator; returns positive-class SHAP values."""
    import shap

    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(X)
    # sklearn RF: returns a list [neg_class_shap, pos_class_shap]
    # LightGBM: returns a single array (positive class for binary)
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        # newer SHAP returns (n_samples, n_features, n_classes)
        sv = sv[:, :, 1]
    return np.asarray(sv, dtype=np.float64)


def rationale_for_population(
    pop: str,
    *,
    top_k: int,
    df_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute SHAP rationale + feature-importance + recipe-match for one population.

    Returns (rationale_df, importance_df, recipe_match_dict).
    """
    bundle_path = OUT_DIR / f"pop_estimator_{pop}_250m.joblib"
    cal_path = OUT_DIR / f"pop_calibrated_{pop}_250m.parquet"
    if not bundle_path.exists() or not cal_path.exists():
        raise FileNotFoundError(
            f"Missing {bundle_path.name} or {cal_path.name}; run "
            f"scripts/northern_sierra_placer/train_predict_250m.py first."
        )

    bundle = joblib.load(bundle_path)
    rf_full = bundle["rf_full"]
    lgbm_full = bundle["lgbm_full"]
    xgb_full = bundle.get("xgb_full")  # v3 Phase C.1; absent in v2 bundles.
    feature_cols = list(bundle["feature_cols"])
    cal_df = pd.read_parquet(cal_path)

    # Align cal_df rows to df_features by (row, col); cal_df is the same
    # AOI-clipped grid the training step wrote.
    keyed = df_features.merge(
        cal_df[["row", "col", "p_cal"]], on=["row", "col"], how="left",
    )
    if keyed["p_cal"].isna().all():
        raise ValueError(
            f"merge produced all-NaN p_cal for {pop}; check (row,col) alignment."
        )

    top_idx = _top_k_indices(keyed["p_cal"].to_numpy(), top_k)
    print(f"[{pop}] selecting top-{len(top_idx)} cells by p_cal "
          f"(p_cal range in top-K: {keyed['p_cal'].iloc[top_idx].min():.3f} - "
          f"{keyed['p_cal'].iloc[top_idx].max():.3f})")

    X_top = _build_X(keyed.iloc[top_idx], feature_cols)
    t0 = time.time()
    sv_rf = _shap_for_estimator(rf_full, X_top)
    print(f"[{pop}]   RF SHAP done in {(time.time()-t0)/60:.1f} min")
    t1 = time.time()
    sv_lgbm = _shap_for_estimator(lgbm_full, X_top)
    print(f"[{pop}]   LGBM SHAP done in {(time.time()-t1)/60:.1f} min")
    if xgb_full is not None:
        t2 = time.time()
        sv_xgb = _shap_for_estimator(xgb_full, X_top)
        print(f"[{pop}]   XGB SHAP done in {(time.time()-t2)/60:.1f} min")
    else:
        sv_xgb = None
        print(f"[{pop}]   xgb_full not present in bundle (pre-v3); "
              f"mean_abs_shap_xgb will be NaN.")

    # Per-cell top-3 by row-wise max(|SHAP|) across available learners.
    abs_rf = np.abs(sv_rf)
    abs_lgbm = np.abs(sv_lgbm)
    if sv_xgb is not None:
        abs_xgb = np.abs(sv_xgb)
        # Pick the learner with the largest |SHAP| per (cell, feature) and
        # keep that signed value, so the printed driver string preserves
        # direction.
        stacked = np.stack([sv_rf, sv_lgbm, sv_xgb], axis=0)
        abs_stacked = np.abs(stacked)
        which = np.argmax(abs_stacked, axis=0)
        combined = np.take_along_axis(stacked, which[None, :, :], axis=0)[0]
    else:
        combined = np.where(abs_rf >= abs_lgbm, sv_rf, sv_lgbm)
    top3_strings = _per_cell_top3(combined, feature_cols)

    rationale = pd.DataFrame({
        "row": keyed.iloc[top_idx]["row"].to_numpy(),
        "col": keyed.iloc[top_idx]["col"].to_numpy(),
        "x":   keyed.iloc[top_idx]["x"].to_numpy(),
        "y":   keyed.iloc[top_idx]["y"].to_numpy(),
        "p_cal": keyed.iloc[top_idx]["p_cal"].to_numpy(),
        "top_shap_features": top3_strings,
    })

    # Feature importance: mean(|SHAP|) per feature across the top-K cells.
    mean_abs_rf = np.mean(np.abs(sv_rf), axis=0)
    mean_abs_lgbm = np.mean(np.abs(sv_lgbm), axis=0)
    if sv_xgb is not None:
        mean_abs_xgb = np.mean(np.abs(sv_xgb), axis=0)
        combined_mean = (mean_abs_rf + mean_abs_lgbm + mean_abs_xgb) / 3.0
    else:
        mean_abs_xgb = np.full_like(mean_abs_rf, np.nan, dtype=np.float64)
        combined_mean = (mean_abs_rf + mean_abs_lgbm) / 2.0
    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap_rf": mean_abs_rf,
        "mean_abs_shap_lgbm": mean_abs_lgbm,
        "mean_abs_shap_xgb": mean_abs_xgb,
        "mean_abs_shap_combined": combined_mean,
    }).sort_values("mean_abs_shap_combined", ascending=False).reset_index(drop=True)

    # Recipe-match check (v3 Phase E.4).
    expected = RECIPE_EXPECTED.get(pop, ())
    top_5 = importance["feature"].head(5).tolist()
    n_matched = _count_recipe_matches(top_5, expected)
    recipe = {
        "population": pop,
        "top_5": top_5,
        "expected": list(expected),
        "n_matched": int(n_matched),
        "gate_passes": bool(n_matched >= 3),
    }

    return rationale, importance, recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=("both", *POPULATIONS),
        default="both",
        help="Which population to compute SHAP for (default: both).",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"How many top-p_cal cells to explain (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args(argv)

    if not IN_FEATURES.exists():
        print(f"ERROR: features parquet not found at {IN_FEATURES}; "
              "run scripts/northern_sierra_placer/assemble_250m.py first.",
              file=sys.stderr)
        return 2

    print(f"==> Loading features from {IN_FEATURES}")
    df_features = pd.read_parquet(IN_FEATURES)
    print(f"    cells: {len(df_features):,}  columns: {len(df_features.columns)}")

    pops = POPULATIONS if args.population == "both" else (args.population,)

    for pop in pops:
        print(f"\n==> {pop}")
        try:
            rationale, importance, recipe = rationale_for_population(
                pop, top_k=args.top_k, df_features=df_features,
            )
        except FileNotFoundError as e:
            print(f"    SKIP: {e}", file=sys.stderr)
            continue
        rat_path = OUT_DIR / f"rationale_{pop}_250m.parquet"
        imp_path = OUT_DIR / f"feature_importance_{pop}.csv"
        rcp_path = OUT_DIR / f"recipe_match_{pop}.json"
        rationale.to_parquet(rat_path, index=False)
        importance.to_csv(imp_path, index=False)
        rcp_path.write_text(json.dumps(recipe, indent=2))
        print(f"    wrote {rat_path}")
        print(f"    wrote {imp_path}")
        print(f"    wrote {rcp_path}")
        print(f"    top-5 features by mean|SHAP|:")
        for _, r in importance.head(5).iterrows():
            print(f"      {r['feature']:30s}  combined={r['mean_abs_shap_combined']:.4f}")
        print(f"    recipe match: {recipe['n_matched']}/5 expected "
              f"({'PASS' if recipe['gate_passes'] else 'FAIL'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
