"""Groups C + D for v3.1.

D1 — MRDS Cox-Singer cleanup. The MRDS adapter was updated to filter
     out Au records flagged as placer (oper_type == "Placer" or
     dep_type matching placer/alluvial regex). Rebuilding the feature
     frame applies the cleanup. Re-runs RF spatial-block CV with the
     cleaned labels and reports AUC delta vs Group B.

C1 — Weighted-PU port (Hajihosseinlou et al. 2025).
     `model_weighted_pu.fit_weighted_pu` with TPE-tuned w_pos_mult /
     w_unl_mult on RF base classifier. Reports capture-curve numbers
     vs the bagging-PU baseline.

C2 — Geology-only model + Klamath retransfer.
     Train RF on Mother Lode using only lithology + age + distance
     to fault + topography (no magnetic, no gravity, no NGDB, no
     Sentinel-2). Score Klamath cells. Compare lift to v3's full-
     feature transfer test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.regions.motherlode import MOTHERLODE
from ai_minerals.regions.klamath import KLAMATH
from ai_minerals.features.assemble import build_feature_frame
from ai_minerals.model import (
    add_lithology_onehot, build_training_set, sample_pseudo_negatives,
    NON_FEATURE_COLUMNS,
)
from ai_minerals.model_rf import (
    count_feature_columns, make_rf, spatial_block_scores_tree,
)
from ai_minerals.model_weighted_pu import fit_weighted_pu, WeightedPUConfig

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
V3P1_DIR = ML_DIR / "v3p1"
V3P1_DIR.mkdir(parents=True, exist_ok=True)


# Geology-only feature columns (keep these, drop everything else
# beyond the labels/identity columns).
GEOLOGY_ONLY_FEATURES = {
    "elevation", "slope", "tri",
    "distance_to_fault_m",
}


def step_d1_rebuild_and_verify():
    """Rebuild feature frames with Cox-Singer-cleaned labels."""
    print("\n=== D1: rebuilding feature frames with Cox-Singer cleanup ===")
    for region, label in [(MOTHERLODE, "motherlode"), (KLAMATH, "klamath")]:
        df = build_feature_frame(region, resolution_m=500)
        n_pos = int(df["is_orogenic_gold"].sum())
        print(f"  {label}: {df.shape}, positives now {n_pos}")
        df.to_parquet(DATA_DERIVED / f"features_{label}_500m.parquet")

    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    n_pos = int(df["is_orogenic_gold"].sum())
    print(f"  Mother Lode positives after Cox-Singer cleanup: {n_pos}")

    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")
    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    X, y = build_training_set(
        df, top_classes, n_per_positive=30, random_state=42,
        label_col=label_col, label_cols=label_cols,
    )
    X = X.drop(columns=count_feature_columns(list(X.columns)))
    print(f"  D1 training set: {X.shape}")

    negs = sample_pseudo_negatives(df, n_per_positive=30, random_state=42, label_col=label_col)
    rows = pd.concat(
        [df[df[label_col] == 1][["row", "col", "x", "y"]], negs[["row", "col", "x", "y"]]],
        ignore_index=True,
    )
    cv = spatial_block_scores_tree(X, y, rows, model_factory=make_rf, block_size_m=20_000)
    aucs = cv["roc_auc"].dropna().to_numpy()
    pr_aucs = cv["pr_auc"].dropna().to_numpy()
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(aucs, len(aucs)).mean() for _ in range(2000)])
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    print(f"  D1 RF AUC: {aucs.mean():.3f} ± {aucs.std():.3f}  CI95={ci}")
    print(f"  D1 RF PR-AUC: {pr_aucs.mean():.3f}")

    return {
        "n_positives_after_cleanup": int(n_pos),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "auc_ci95": list(ci),
        "pr_auc_mean": float(pr_aucs.mean()),
        "n_folds": int(len(aucs)),
    }


def step_c1_weighted_pu():
    """Weighted-PU on the (Cox-Singer-cleaned) feature frame."""
    print("\n=== C1: weighted-PU + RF + TPE ===")
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    cfg = WeightedPUConfig(n_trials=20, seed=42)
    p_wpu, summary, _rf = fit_weighted_pu(
        df, top_classes, label_col="is_orogenic_gold", config=cfg,
    )
    pos = (df["is_orogenic_gold"] == 1).to_numpy()
    n = len(p_wpu)
    n_pos = pos.sum()
    order = np.argsort(-p_wpu)
    sorted_pos = pos[order]
    capture = {}
    print(f"  weighted-PU capture curves on {n_pos:,} positives:")
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        rate = sorted_pos[:k].sum() / n_pos
        capture[f"top_{p}_pct"] = float(rate)
        print(f"    top {p:>3}%: {rate*100:.1f}%")
    return {
        "tpe_summary": summary,
        "capture_curves": capture,
    }


def step_c2_geology_only_klamath():
    """Geology-only model trained on Sierra, scored on Klamath."""
    print("\n=== C2: geology-only model + Klamath retransfer ===")
    ml = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    kl = pd.read_parquet(DATA_DERIVED / "features_klamath_500m.parquet")
    print(f"  Sierra ML: {ml.shape}, positives {int(ml['is_orogenic_gold'].sum())}")
    print(f"  Klamath:   {kl.shape}, positives {int(kl['is_orogenic_gold'].sum())}")

    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")
    top_classes = ml["lithology_class"].value_counts().head(10).index.tolist()

    # Identify columns to drop: everything except geology-only.
    DROP_PREFIXES = ("magnetic", "gravity", "s2_", "au_", "as_", "sb_", "hg_",
                     "w_", "ag_", "cu_", "pb_", "zn_", "mo_", "bi_", "te_")
    geology_keep = lambda c: not any(c.startswith(p) for p in DROP_PREFIXES)

    X_full, y_full = build_training_set(
        ml, top_classes, n_per_positive=30, random_state=42,
        label_col=label_col, label_cols=label_cols,
    )
    X_geo = X_full[[c for c in X_full.columns if geology_keep(c)]]
    X_geo = X_geo.drop(columns=count_feature_columns(list(X_geo.columns)))
    print(f"  geology-only train: {X_geo.shape}")
    print(f"    columns kept: {list(X_geo.columns)}")

    rf = make_rf(random_state=42)
    rf.fit(X_geo.fillna(-9999).to_numpy(), y_full)

    # Score Klamath using only the same geology columns.
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in ml.columns:
            extra[col] = ml[col][ml[col] >= 0].value_counts().head(10).index.tolist()
    kl_oh = add_lithology_onehot(kl, top_classes, extra_class_columns=extra or None)
    feat_cols = list(X_geo.columns)
    for c in feat_cols:
        if c not in kl_oh.columns:
            kl_oh[c] = 0.0
    X_kl = kl_oh[feat_cols].fillna(-9999).to_numpy()
    p_kl = rf.predict_proba(X_kl)[:, 1]

    pos = (kl["is_orogenic_gold"] == 1).to_numpy()
    n_pos = pos.sum()
    n = len(p_kl)
    order = np.argsort(-p_kl)
    sorted_pos = pos[order]
    rng = np.random.default_rng(0)
    rand_order = np.argsort(-rng.random(n))
    rand_sorted = pos[rand_order]
    print(f"  Klamath scores: median {np.median(p_kl):.3f}, "
          f"range [{p_kl.min():.3f}, {p_kl.max():.3f}]")
    capture = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        rf_cap = sorted_pos[:k].sum() / n_pos
        rand_cap = rand_sorted[:k].sum() / n_pos
        lift = rf_cap / max(rand_cap, 1e-6)
        capture[f"top_{p}_pct"] = {
            "rf_capture": float(rf_cap),
            "random_capture": float(rand_cap),
            "lift": float(lift),
        }
        print(f"    top {p:>3}% (k={k:>5}):  random {rand_cap*100:>5.1f}%  "
              f"RF {rf_cap*100:>5.1f}%  lift {lift:.2f}x")
    return {
        "n_train_positives": int((y_full == 1).sum()),
        "n_klamath_positives": int(n_pos),
        "klamath_score_stats": {
            "min": float(p_kl.min()), "max": float(p_kl.max()),
            "mean": float(p_kl.mean()), "median": float(np.median(p_kl)),
        },
        "capture_curves": capture,
        "feature_columns": feat_cols,
    }


def main() -> None:
    out = {}
    out["D1_cox_singer_cleanup"] = step_d1_rebuild_and_verify()
    out["C1_weighted_pu"] = step_c1_weighted_pu()
    out["C2_geology_only_klamath"] = step_c2_geology_only_klamath()
    (V3P1_DIR / "group_cd_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {V3P1_DIR / 'group_cd_metrics.json'}")


if __name__ == "__main__":
    main()
