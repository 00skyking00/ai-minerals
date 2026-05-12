"""DEEP-SEAM Cell B: Random Forest LOO on Curnamona REE.

Same setup as Cell A but uses our standard Random Forest (with class
balancing for the 7-positive imbalance) instead of DevNet. Tests whether
DEEP-SEAM's results are achievable with a simpler, classical model on
the same engineered features.

Output: data/derived/deep_seam_cell_b_loo.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DEEP_SEAM_DIR = Path(
    "/home/sky/src/learning/ai-minerals/data/raw/deep_seam/"
    "EarthByte-MPM_Curnamona_REE-4bb4139/Files needed for runing/"
    "Output_Features_Generated"
)
OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/deep_seam_cell_b_loo.json")


def main() -> None:
    train = pd.read_csv(DEEP_SEAM_DIR / "Xy_all_minmax_7.11_d0.0115_all_random_0.65_all_109.csv")
    feat_cols = [c for c in train.columns if c != "label"]
    pos_idx = np.where(train["label"].to_numpy() == 1)[0]
    n_pos = len(pos_idx)
    n = len(train)
    print(f"training: {train.shape}, positives: {n_pos}", flush=True)

    held_out_ranks = []
    held_out_pcts = []

    t0 = time.time()
    for fold, held in enumerate(pos_idx, 1):
        keep = np.ones(n, dtype=bool)
        keep[held] = False

        X_train = train.loc[keep, feat_cols].fillna(-9999.0).to_numpy()
        y_train = train.loc[keep, "label"].to_numpy().astype(int)
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )
        rf.fit(X_train, y_train)

        # Score the held-out positive plus all unlabeled (full 870 rows).
        Xt = train[feat_cols].fillna(-9999.0).to_numpy()
        scores = rf.predict_proba(Xt)[:, 1]
        held_score = float(scores[held])
        n_above = int((scores > held_score).sum())
        rank_pct = (n_above / n) * 100
        held_out_ranks.append(n_above)
        held_out_pcts.append(rank_pct)
        elapsed = time.time() - t0
        print(f"  fold {fold}/{n_pos}  held={held}  score={held_score:.4f}  "
              f"rank={n_above}/{n}  top_{rank_pct:.2f}%  elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nLOO complete in {elapsed:.0f}s", flush=True)

    print("\n=== Cell B: capture-at-top-k% (RF on Curnamona REE LOO) ===")
    capture = {}
    for p in [1, 2, 5, 10, 30]:
        threshold_rank = n * p / 100
        captured = sum(1 for r in held_out_ranks if r < threshold_rank)
        rate = captured / n_pos
        capture[f"top_{p}_pct"] = {
            "captured": captured, "rate": float(rate),
            "rate_pct": float(rate * 100),
            "n_top_cells": int(np.ceil(threshold_rank)),
        }
        print(f"  top {p:>3}%: {captured}/{n_pos} = {rate*100:.1f}%")

    out = {
        "methodology": "Cell B: their data (Curnamona REE features) + Random Forest",
        "n_total_rows": int(n),
        "n_positives": int(n_pos),
        "held_out_ranks": [int(r) for r in held_out_ranks],
        "held_out_pcts": [float(p) for p in held_out_pcts],
        "capture_at_top_k": capture,
        "elapsed_seconds": float(elapsed),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
