"""DEEP-SEAM Cell A: leave-one-out replication on Curnamona REE.

Methodology, faithful to the paper (Pang 2019 + DEEP-SEAM Eq. 7),
without consulting their code:

  - DevNet ported from the paper's architecture (24 -> 12 -> 1, ReLU).
  - Pre-processed feature CSVs from their Zenodo deposit (we use their
    feature engineering as-is; the methods question is about the model
    + validation, not the feature pipeline).
  - Leave-one-out over the 7 positive deposits: 7 training runs each
    holding out one positive. Score the held-out positive plus the 863
    unlabeled training rows. Record where the held-out positive ranks.

Headline metric to compare to DEEP-SEAM's "86% of deposits in top 2%":
  - Aggregate held-out positive ranks across the 7 LOO runs.
  - Capture-at-top-k% = fraction of held-out positives in top k% of
    their respective scoring runs.

Output: data/derived/deep_seam_cell_a_loo.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.model_devnet import fit_devnet, DevNetConfig

DEEP_SEAM_DIR = Path(
    "/home/sky/src/learning/ai-minerals/data/raw/deep_seam/"
    "EarthByte-MPM_Curnamona_REE-4bb4139/Files needed for runing/"
    "Output_Features_Generated"
)
OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/deep_seam_cell_a_loo.json")
OUT.parent.mkdir(exist_ok=True, parents=True)


def main() -> None:
    train_csv = DEEP_SEAM_DIR / "Xy_all_minmax_7.11_d0.0115_all_random_0.65_all_109.csv"
    train = pd.read_csv(train_csv)
    print(f"training: {train.shape}, label dist: {train['label'].value_counts().to_dict()}")

    feat_cols = [c for c in train.columns if c != "label"]
    pos_idx = np.where(train["label"].to_numpy() == 1)[0]
    n_pos = len(pos_idx)
    print(f"positives: {n_pos}, indices: {pos_idx.tolist()}")

    cfg = DevNetConfig(
        hidden=(24, 12),
        learning_rate=0.005,
        batch_size=128,
        n_epochs=500,
        n_ref=5000,
        confidence_margin=5.0,
        seed=42,
    )

    # LOO: hold out each positive, train on remaining 6 positives + 863 unlabeled.
    held_out_ranks = []
    held_out_pcts = []
    n = len(train)

    t0 = time.time()
    for fold, held in enumerate(pos_idx, 1):
        keep = np.ones(n, dtype=bool)
        keep[held] = False
        train_fold = train[keep].copy().reset_index(drop=True)
        # Drop NaN rows (defensive).
        train_fold = train_fold[train_fold[feat_cols].notna().all(axis=1)]

        # Fit DevNet on this fold.
        train_aug = train_fold[feat_cols + ["label"]].rename(columns={"label": "y"})
        scores_train, cfg_used, model = fit_devnet(
            train_aug, feat_cols=feat_cols, label_col="y", config=cfg,
        )

        # Score the held-out positive + all unlabeled (i.e., the full 870
        # training rows; the 6 in-fold positives will rank near the top).
        import torch
        model.eval()
        Xt = train[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
        # Use TRAIN-FOLD's mean/std for normalization (paper-faithful;
        # avoids leaking held-out positive's stats into normalization).
        Xtf = train_fold[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
        mu = Xtf.mean(axis=0)
        sd = Xtf.std(axis=0) + 1e-8
        Xt_n = (Xt - mu) / sd
        with torch.no_grad():
            scores_all = model(torch.from_numpy(Xt_n)).squeeze(-1).numpy()

        # Rank of the held-out positive.
        held_score = float(scores_all[held])
        n_above = int((scores_all > held_score).sum())
        rank_pct = (n_above / n) * 100  # smaller = better
        held_out_ranks.append(n_above)
        held_out_pcts.append(rank_pct)
        elapsed = time.time() - t0
        print(f"  fold {fold}/{n_pos}  held_idx={held}  score={held_score:.3f}  "
              f"rank={n_above}/{n}  top_{rank_pct:.2f}%  elapsed={elapsed:.0f}s",
              flush=True)

    elapsed = time.time() - t0
    print(f"\nLOO complete in {elapsed/60:.1f} min")

    # Capture-at-top-k%: fraction of held-out positives that landed in
    # top k% of their scoring run.
    print("\n=== Cell A: capture-at-top-k% on Curnamona REE LOO ===")
    capture = {}
    for p in [1, 2, 5, 10, 30]:
        threshold_rank = n * p / 100
        captured = sum(1 for r in held_out_ranks if r < threshold_rank)
        rate = captured / n_pos
        capture[f"top_{p}_pct"] = {
            "captured": captured,
            "rate": float(rate),
            "rate_pct": float(rate * 100),
            "n_top_cells": int(np.ceil(threshold_rank)),
        }
        print(f"  top {p:>3}%: {captured}/{n_pos} = {rate*100:.1f}%")

    print(f"\n  DEEP-SEAM published headline (Curnamona REE):")
    print(f"    top 2% contains 86% of deposits = 6/7")
    print(f"    top 30% delineates all deposits = 7/7")

    out = {
        "methodology": "Cell A: their data (Curnamona REE features) + DevNet from paper",
        "training_csv": str(train_csv.name),
        "n_total_rows": int(n),
        "n_positives": int(n_pos),
        "n_unlabeled": int(n - n_pos),
        "held_out_ranks": [int(r) for r in held_out_ranks],
        "held_out_pcts": [float(p) for p in held_out_pcts],
        "capture_at_top_k": capture,
        "deep_seam_published": {
            "top_2_pct": "86% (6/7)",
            "top_30_pct": "100% (7/7)",
        },
        "config": {
            "hidden": list(cfg.hidden),
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "n_epochs": cfg.n_epochs,
            "confidence_margin": cfg.confidence_margin,
            "seed": cfg.seed,
        },
        "elapsed_minutes": float(elapsed / 60),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
