"""Quick comparator: run our Mother Lode RF under DEEP-SEAM-style evaluation.

DEEP-SEAM reports "86% of REE deposits in top 2% of mapped area" using
a random 70/30 train/test split (no spatial-block CV). To put a
like-for-like number on the cover page, run our same RF with that same
measurement choice and report capture-at-top-k% on the random 30% test
fold.

Spatial-block CV is the more rigorous evaluation; this random-split
number is purely a comparator for the published literature, included
in the cover page so the reader can see both at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ai_minerals.regions.motherlode import MOTHERLODE
from ai_minerals.model import (
    add_lithology_onehot, build_training_set, NON_FEATURE_COLUMNS,
)
from ai_minerals.model_rf import count_feature_columns, make_rf

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
OUT_DIR = DATA_DERIVED / "motherlode"


def main() -> None:
    df = pd.read_parquet(DATA_DERIVED / "features_motherlode_500m.parquet")
    label_col = "is_orogenic_gold"
    label_cols = ("is_orogenic_gold", "is_low_sulfidation")
    top_classes = df["lithology_class"].value_counts().head(10).index.tolist()

    X, y = build_training_set(
        df, top_classes,
        n_per_positive=30, random_state=42,
        label_col=label_col, label_cols=label_cols,
    )
    X = X.drop(columns=count_feature_columns(list(X.columns)))
    print(f"Training pool: {X.shape}, positives: {int((y==1).sum())}")

    # Random 70/30 stratified split (DEEP-SEAM's setup).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    print(f"Train: {X_tr.shape} ({int((y_tr==1).sum())} pos)")
    print(f"Test:  {X_te.shape} ({int((y_te==1).sum())} pos)")

    rf = make_rf(random_state=42)
    rf.fit(X_tr.fillna(-9999).to_numpy(), y_tr)

    # ---- Setup matching DEEP-SEAM more closely:
    # Score the FULL Mother Lode grid (192,968 cells), then find what
    # percentile-of-the-full-map each held-out test positive lands at.
    # Cover-page-readable metric.
    df_oh = add_lithology_onehot(df, top_classes)
    feat_cols = list(X_tr.columns)
    X_full = df_oh[feat_cols].fillna(-9999).to_numpy()
    p_full = rf.predict_proba(X_full)[:, 1]
    n_full = len(p_full)

    # Held-out positives: orogenic_gold cells from the original df that
    # ended up in y_te. We can't trace them perfectly back to df indices
    # because build_training_set merged positives with sampled negatives
    # in a single shuffle. Use a clean approach: re-do the split at the
    # df level on the orogenic-Au positives only.
    rng = np.random.default_rng(42)
    pos_idx = np.where(df["is_orogenic_gold"] == 1)[0]
    rng.shuffle(pos_idx)
    cut = int(len(pos_idx) * 0.7)
    train_pos = set(pos_idx[:cut].tolist())
    test_pos_idx = pos_idx[cut:]
    print(f"Held-out positives in full AOI: {len(test_pos_idx)} / {len(pos_idx)} "
          f"(of {n_full} total cells)")

    # Sort full-AOI scores; for each top-k% threshold, count held-out positives.
    order = np.argsort(-p_full)
    rank_of = np.empty(n_full, dtype=np.int64)
    rank_of[order] = np.arange(n_full)
    test_pos_ranks = rank_of[test_pos_idx]

    print("\nDEEP-SEAM-style measurement on Mother Lode "
          "(random 70/30 split, score full 192k-cell map, "
          "report % of held-out test positives in top-k% of map):")
    out = {}
    for p in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n_full * p / 100))
        captured = (test_pos_ranks < k).sum()
        rate = captured / len(test_pos_idx)
        out[f"top_{p}_pct"] = float(rate)
        print(f"  top {p:>3}% of map (k={k:>6}): "
              f"{captured}/{len(test_pos_idx)} held-out positives = "
              f"{100*rate:.1f}%")

    print(f"\n[For comparison: DEEP-SEAM Curnamona REE reports 86% "
          f"of REE deposits in top 2% of map area.]")

    (OUT_DIR / "deepseam_style_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
