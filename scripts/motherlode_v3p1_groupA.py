"""Group A v3.1 validation tightenings.

Four experiments, all working off the v3 predictions parquet + the
Mother Lode feature frame. No retraining of the base model; these
re-evaluate the existing v3 model under different measurement choices.

A1 — Bootstrap 95% CI on capture-at-top-k%.
     Resample held-out positives with replacement; recompute
     capture rate at top-k%; report 2.5th and 97.5th percentile of
     2,000 resamples.

A2 — Within-Sierra N/S split (V10).
     Train RF on northern Mother Lode (latitude > 38.5° N), test on
     southern. Report capture-at-top-k% on the held-out south. Then
     swap (train south, test north). Report both.

A3 — V6 density-filtered NGDB Au correlation.
     Pearson r between RF prospectivity and 5-km-aggregated NGDB Au,
     restricted to cells with at least three NGDB samples reporting
     Au within 5 km.

A4 — DEEP-SEAM-style measurement comparator (already run separately).
     Just consolidates the result into the v3.1 metrics JSON.

All outputs land in `data/derived/motherlode/v3p1/group_a_metrics.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.regions.motherlode import MOTHERLODE
from ai_minerals.model import (
    add_lithology_onehot, build_training_set, NON_FEATURE_COLUMNS,
)
from ai_minerals.model_rf import count_feature_columns, make_rf

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
V3P1_DIR = ML_DIR / "v3p1"
V3P1_DIR.mkdir(parents=True, exist_ok=True)


def a1_bootstrap_capture_ci(pred: pd.DataFrame) -> dict:
    """A1: bootstrap CIs on capture-at-top-k% rates."""
    pos = (pred["is_orogenic_gold"] == 1).to_numpy()
    score = pred["p_rf_no_count"].fillna(-1e9).to_numpy()
    n = len(score)
    n_pos = pos.sum()
    pos_idx = np.where(pos)[0]
    rng = np.random.default_rng(0)

    out = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        # Identify which cells fall in top-k% of full-AOI ranking.
        top_k_mask = np.zeros(n, dtype=bool)
        top_k_mask[np.argsort(-score)[:k]] = True
        # Bootstrap: resample positive indices with replacement.
        boot = []
        for _ in range(2000):
            sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            captured = top_k_mask[sample].sum()
            boot.append(captured / len(sample))
        boot = np.array(boot)
        rate = top_k_mask[pos].sum() / n_pos
        out[f"top_{p}_pct"] = {
            "rate": float(rate),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
        }
    return out


def a2_ns_split(df: pd.DataFrame) -> dict:
    """A2: N/S within-Sierra split. Train one half, score the other."""
    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")
    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    # Project x/y to lon/lat to split by latitude.
    import geopandas as gpd
    pts = gpd.GeoDataFrame(
        df[["row", "col"]].copy(),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=MOTHERLODE.working_crs,
    )
    lat = pts.to_crs("EPSG:4326").geometry.y.values
    df = df.copy()
    df["lat"] = lat
    cut_lat = 38.5
    north_mask = df["lat"] > cut_lat
    south_mask = ~north_mask
    print(f"[A2] North cells: {north_mask.sum():,}  positives: {int(df.loc[north_mask, label_col].sum())}")
    print(f"[A2] South cells: {south_mask.sum():,}  positives: {int(df.loc[south_mask, label_col].sum())}")

    out = {}
    for direction, train_mask, test_mask in [
        ("train_N_test_S", north_mask, south_mask),
        ("train_S_test_N", south_mask, north_mask),
    ]:
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        X_train, y_train = build_training_set(
            train_df, top_classes,
            n_per_positive=30, random_state=42,
            label_col=label_col, label_cols=label_cols,
        )
        X_train = X_train.drop(columns=count_feature_columns(list(X_train.columns)))
        feat_cols = list(X_train.columns)
        rf = make_rf(random_state=42)
        rf.fit(X_train.fillna(-9999).to_numpy(), y_train)

        # Score test cells.
        test_oh = add_lithology_onehot(test_df, top_classes)
        for c in feat_cols:
            if c not in test_oh.columns:
                test_oh[c] = 0.0
        X_test = test_oh[feat_cols].fillna(-9999).to_numpy()
        p_test = rf.predict_proba(X_test)[:, 1]

        pos_test = (test_df[label_col] == 1).to_numpy()
        n_pos = pos_test.sum()
        n = len(p_test)
        if n_pos == 0:
            print(f"[A2] {direction}: no test positives, skipping")
            continue

        order = np.argsort(-p_test)
        sorted_pos = pos_test[order]
        rng = np.random.default_rng(0)
        rand_order = np.argsort(-rng.random(n))
        rand_sorted = pos_test[rand_order]

        capture = {}
        for p in [1, 2, 5, 10, 30]:
            k = int(np.ceil(n * p / 100))
            rf_cap = sorted_pos[:k].sum() / n_pos
            rand_cap = rand_sorted[:k].sum() / n_pos
            capture[f"top_{p}_pct"] = {
                "rf_capture": float(rf_cap),
                "random_capture": float(rand_cap),
                "lift": float(rf_cap / max(rand_cap, 1e-6)),
            }
            print(f"  {direction} top {p:>3}% (k={k:>5}):  random {rand_cap*100:>5.1f}%  "
                  f"RF {rf_cap*100:>5.1f}%  lift {rf_cap/max(rand_cap,1e-6):.2f}x")

        out[direction] = {
            "n_test_cells": int(n),
            "n_test_positives": int(n_pos),
            "capture_curves": capture,
        }
    return out


def a3_density_filtered_v6(pred: pd.DataFrame, df_features: pd.DataFrame) -> dict:
    """A3: V6 NGDB Au correlation, filtered to cells with at least 3 NGDB samples nearby."""
    merged = pred[["row", "col", "p_rf_no_count"]].merge(
        df_features[["row", "col", "au_mean_5km", "au_count_5km"]],
        on=["row", "col"], how="inner",
    )

    print(f"[A3] cells in merged frame: {len(merged):,}")
    print(f"[A3] cells with any NGDB Au within 5km: {merged['au_mean_5km'].notna().sum():,}")
    out = {"all_cells_with_au": {}, "filtered": {}}
    for thresh in [1, 3, 5, 10]:
        sub = merged[merged["au_count_5km"] >= thresh].dropna(subset=["p_rf_no_count", "au_mean_5km"])
        if len(sub) >= 30:
            r = sub[["p_rf_no_count", "au_mean_5km"]].corr().iloc[0, 1]
            print(f"[A3] count_5km >= {thresh}: n={len(sub):,}  Pearson r = {r:.3f}")
            out[f"thresh_{thresh}"] = {"n": int(len(sub)), "pearson_r": float(r)}
        else:
            print(f"[A3] count_5km >= {thresh}: only {len(sub):,} cells, skipping")
    return out


def a4_deepseam_comparator() -> dict:
    """A4: DEEP-SEAM-style comparator. Already computed in Phase 6 prep."""
    src = ML_DIR / "deepseam_style_metrics.json"
    if not src.exists():
        print(f"[A4] {src} not present; rerun deep_seam_replicate.py-equivalent if needed")
        return {}
    out = json.loads(src.read_text())
    print(f"[A4] DEEP-SEAM-style capture (loaded from {src.name}):")
    for k, v in out.items():
        print(f"  {k}: {v*100:.1f}%")
    return out


def main() -> None:
    pred = pd.read_parquet(ML_DIR / "model_predictions_motherlode.parquet")
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    print(f"v3 predictions: {pred.shape}, features: {df.shape}")
    print(f"v3 positives: {int(df['is_orogenic_gold'].sum()):,}")
    print()

    print("=" * 60)
    print("A1: bootstrap 95% CI on capture-at-top-k%")
    print("=" * 60)
    a1 = a1_bootstrap_capture_ci(pred)
    for k, v in a1.items():
        print(f"  {k}: rate={v['rate']*100:.1f}%  CI95=[{v['ci95_low']*100:.1f}%, {v['ci95_high']*100:.1f}%]")

    print()
    print("=" * 60)
    print("A2: within-Sierra N/S split (V10)")
    print("=" * 60)
    a2 = a2_ns_split(df)

    print()
    print("=" * 60)
    print("A3: V6 NGDB Au correlation, density-filtered")
    print("=" * 60)
    a3 = a3_density_filtered_v6(pred, df)

    print()
    print("=" * 60)
    print("A4: DEEP-SEAM-style comparator (already computed)")
    print("=" * 60)
    a4 = a4_deepseam_comparator()

    metrics = {
        "A1_bootstrap_capture_ci": a1,
        "A2_ns_split": a2,
        "A3_density_filtered_v6": a3,
        "A4_deepseam_comparator": a4,
    }
    out_path = V3P1_DIR / "group_a_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
