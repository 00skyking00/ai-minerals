"""DEEP-SEAM Cell C v3: DevNet on Mother Lode with PCA preprocessing.

Tests whether the original Cell C failure (broken on raw features) is
fixed by adding the dominant preprocessing move from DEEP-SEAM's
pipeline: PCA-style decomposition of correlated feature groups.

Approach:
  1. Group Mother Lode features into types: geochem, geophysics,
     Sentinel-2, DEM, lithology one-hots, faults.
  2. Apply PCA to each numeric group (top 5 PCs per group, or fewer
     if the group has fewer columns).
  3. Concatenate all PCs + lithology one-hots + distance-to-fault.
  4. Median-impute, min-max scale.
  5. Train DevNet.
  6. Compute capture-at-top-k% (full-data scoring, comparable to
     DEEP-SEAM's headline measurement).

If this produces sensible capture rates (top 5% well above 5%), the
methodology gap was preprocessing. If it still fails, the gap is
deeper (positive rate, anomaly-detection framing mismatch, etc.).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from ai_minerals.model import (
    add_lithology_onehot, sample_pseudo_negatives, non_feature_columns,
)
from ai_minerals.model_rf import count_feature_columns
from ai_minerals.model_devnet import fit_devnet, DevNetConfig

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
FEATURE_PARQUET = DATA_DERIVED / "features_motherlode_500m.parquet"
OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/deep_seam_cell_c_v3_pca.json")

LABEL_COL = "is_orogenic_gold"
LABEL_COLS = ("is_orogenic_gold", "is_low_sulfidation")


def group_features(feat_cols: list[str]) -> dict[str, list[str]]:
    """Partition feature columns into groups for per-group PCA."""
    groups: dict[str, list[str]] = {
        "geochem": [],
        "geophysics": [],
        "sentinel2": [],
        "dem": [],
        "lith_onehot": [],
        "other": [],
    }
    for c in feat_cols:
        if "_mean_5km" in c or "_max_5km" in c or "_min_5km" in c:
            groups["geochem"].append(c)
        elif c in ("magnetic", "gravity", "gravity_isostatic") or c.startswith("magnetic_"):
            groups["geophysics"].append(c)
        elif c.startswith("s2_"):
            groups["sentinel2"].append(c)
        elif c in ("elevation", "slope", "tri"):
            groups["dem"].append(c)
        elif c.startswith("lith_") or c.startswith("major1_") or c.startswith("major2_") or c.startswith("major3_"):
            groups["lith_onehot"].append(c)
        else:
            groups["other"].append(c)
    return {k: v for k, v in groups.items() if v}


def main() -> None:
    df = pd.read_parquet(FEATURE_PARQUET)
    n_pos = int((df[LABEL_COL] == 1).sum())
    print(f"feature frame: {df.shape}, positives: {n_pos:,}", flush=True)

    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()
    pos_mask = (df[LABEL_COL] == 1).to_numpy()
    pos_indices = np.where(pos_mask)[0]
    neg_df = sample_pseudo_negatives(df, n_per_positive=30, random_state=42, label_col=LABEL_COL)
    df_keys = list(zip(df["row"].to_numpy().tolist(), df["col"].to_numpy().tolist()))
    df_key_to_idx = {k: i for i, k in enumerate(df_keys)}
    neg_indices = np.array(
        [df_key_to_idx[(int(r), int(c))] for r, c in zip(neg_df["row"], neg_df["col"])]
    )
    train_indices = np.concatenate([pos_indices, neg_indices])
    y_train = np.concatenate(
        [np.ones(len(pos_indices), dtype=np.int64),
         np.zeros(len(neg_indices), dtype=np.int64)]
    )
    print(f"training set: {len(train_indices):,} ({len(pos_indices):,} pos, {len(neg_indices):,} neg)", flush=True)

    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df.columns:
            extra[col] = df[col][df[col] >= 0].value_counts().head(10).index.tolist()
    df_oh = add_lithology_onehot(df, top_classes, extra_class_columns=extra or None)

    non_feat = non_feature_columns(label_cols=LABEL_COLS)
    feat_cols = [c for c in df_oh.columns if c not in non_feat]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    leak_check = [c for c in feat_cols if c in LABEL_COLS]
    assert not leak_check, f"label leak: {leak_check}"

    groups = group_features(feat_cols)
    print(f"\nfeature groups:", flush=True)
    for g, cols in groups.items():
        print(f"  {g}: {len(cols)} cols", flush=True)

    # Median-impute each group, then PCA on numeric groups.
    pca_blocks: list[np.ndarray] = []
    pca_names: list[str] = []
    imp = SimpleImputer(strategy="median")
    for group_name, cols in groups.items():
        X_g = df_oh[cols].to_numpy(dtype=np.float32)
        X_imp = imp.fit_transform(X_g)
        if group_name in ("lith_onehot",):
            # one-hots stay as-is; PCA on binary features is wasteful.
            pca_blocks.append(X_imp)
            pca_names.extend(cols)
        else:
            n_comp = min(5, X_imp.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            X_pca = pca.fit_transform(X_imp)
            ev = pca.explained_variance_ratio_.sum()
            print(f"  {group_name}: {X_imp.shape[1]} -> {n_comp} PCs, "
                  f"explained variance ratio = {ev:.3f}", flush=True)
            pca_blocks.append(X_pca.astype(np.float32))
            pca_names.extend([f"{group_name}_pc{i+1}" for i in range(n_comp)])

    X_all = np.concatenate(pca_blocks, axis=1)
    print(f"\npreprocessed feature shape: {X_all.shape} ({len(pca_names)} columns)", flush=True)

    # Min-max scale to [0, 1].
    col_min = X_all.min(axis=0)
    col_max = X_all.max(axis=0)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)
    X_all = ((X_all - col_min) / col_range).astype(np.float32)
    print(f"  range after scale: [{X_all.min():.3f}, {X_all.max():.3f}]", flush=True)

    train_df = pd.DataFrame(X_all[train_indices], columns=pca_names)
    train_df["y"] = y_train

    cfg = DevNetConfig(
        hidden=(24, 12),
        learning_rate=0.005,
        batch_size=128,
        n_epochs=500,
        n_ref=5000,
        confidence_margin=5.0,
        seed=42,
    )

    print("\nfitting DevNet on PCA-preprocessed Mother Lode...", flush=True)
    t0 = time.time()
    train_scores, cfg_used, model = fit_devnet(
        train_df, feat_cols=pca_names, label_col="y", config=cfg,
    )
    elapsed = time.time() - t0
    print(f"trained in {elapsed:.0f}s", flush=True)

    print("scoring full grid...", flush=True)
    Xtf = X_all[train_indices]
    mu = Xtf.mean(axis=0)
    sd = Xtf.std(axis=0) + 1e-8
    Xn = (X_all - mu) / sd
    model.eval()
    with torch.no_grad():
        scores_all = model(torch.from_numpy(Xn.astype(np.float32))).squeeze(-1).numpy()
    print(f"  scores: range [{scores_all.min():.3f}, {scores_all.max():.3f}], "
          f"median {np.median(scores_all):.3f}", flush=True)

    n = len(scores_all)
    print("\n=== Cell C v3: capture-at-top-k% (DevNet + PCA on Mother Lode) ===", flush=True)
    capture = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * p / 100))
        top_k_idx = np.argsort(-scores_all)[:k]
        captured = int(pos_mask[top_k_idx].sum())
        rate = captured / max(n_pos, 1)
        lift = rate / (p / 100)
        capture[f"top_{p}_pct"] = {
            "rate": float(rate), "lift": float(lift),
            "captured": captured, "n_top": int(k),
        }
        print(f"  top {p:>3}%: rate={rate*100:>5.1f}%  lift={lift:.2f}x  "
              f"captured={captured}/{n_pos}", flush=True)

    out = {
        "methodology": "Cell C v3: our data (Mother Lode) + per-group PCA + DevNet",
        "n_cells": int(n),
        "n_positives": int(n_pos),
        "n_features_after_pca": len(pca_names),
        "training_set_size": len(train_indices),
        "feature_groups": {g: len(cols) for g, cols in groups.items()},
        "score_stats": {
            "min": float(scores_all.min()),
            "max": float(scores_all.max()),
            "mean": float(scores_all.mean()),
            "median": float(np.median(scores_all)),
        },
        "capture_at_top_k": capture,
        "elapsed_seconds": float(elapsed),
        "comparison": {
            "cell_c_v1_no_pca": "0% top 1-10% (broken)",
            "cell_d_rf_oof": "19.5% top 5%, lift 3.91x (held-out RF)",
            "cell_a_devnet_curnamona": "100% top 5% (their data)",
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
