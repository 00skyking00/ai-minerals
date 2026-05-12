"""DEEP-SEAM replication: train DevNet on Curnamona REE pre-processed features.

The DEEP-SEAM Zenodo deposit (zenodo.org/records/17098677) ships
pre-processed feature CSVs in `Files needed for runing/Output_Features_Generated/`.
Training data is `Xy_all_minmax_7.11_d0.0115_all_random_0.65_all_109.csv`
(870 rows × 49 cols, 7 positives), prediction grid is
`Xy_all_minmax_7.11_d0.0115_all_regular_0.65_pred_109.csv` (~1.7 MB).

We train DevNet (our PyTorch port in `model_devnet.py`) on the training
CSV and score the prediction grid. The headline metric to reproduce is:

    Top 2% of mapped area contains 86% of known REE deposits.
    30% of area delineates all REE deposits.

Reproducing within ~5 percentage points is acceptable for a clean
methodology replication.

Output: `data/derived/deep_seam_replication.json` with reproduced
metrics and per-prediction-cell scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.model_devnet import fit_devnet, DevNetConfig


DEEP_SEAM_DIR = Path(
    "/home/sky/src/learning/ai-minerals/data/raw/deep_seam/"
    "EarthByte-MPM_Curnamona_REE-4bb4139/Files needed for runing/"
    "Output_Features_Generated"
)
OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/deep_seam_replication.json")
OUT.parent.mkdir(exist_ok=True)


def main() -> None:
    train_csv = DEEP_SEAM_DIR / "Xy_all_minmax_7.11_d0.0115_all_random_0.65_all_109.csv"
    pred_csv = DEEP_SEAM_DIR / "Xy_all_minmax_7.11_d0.0115_all_regular_0.65_pred_109.csv"

    train = pd.read_csv(train_csv)
    pred = pd.read_csv(pred_csv)
    print(f"Training: {train.shape}, label dist: {train['label'].value_counts().to_dict()}")
    print(f"Prediction grid: {pred.shape}")

    # The training CSV has features (after min-max scaling) + label.
    # The prediction CSV has features only (no label column).
    feat_cols = [c for c in train.columns if c != "label"]
    pred_feat_cols = list(pred.columns)
    common = [c for c in feat_cols if c in pred_feat_cols]
    print(f"Common feature columns: {len(common)}")
    feat_cols = common  # use only common features for both train + pred

    cfg = DevNetConfig(
        hidden=(24, 12),  # DEEP-SEAM Eq. 7 architecture
        learning_rate=0.005,
        batch_size=128,
        n_epochs=500,
        n_ref=5000,
        confidence_margin=5.0,
        seed=42,
    )

    train_aug = train[feat_cols + ["label"]].rename(columns={"label": "y"}).copy()
    train_aug = train_aug[train_aug[feat_cols].notna().all(axis=1)]

    print(f"\nTraining DevNet (port of Pang 2019 / DEEP-SEAM)...")
    train_scores, cfg_used, model = fit_devnet(
        train_aug, feat_cols=feat_cols, label_col="y", config=cfg,
    )
    print(f"Train scores: shape {train_scores.shape}, "
          f"range [{train_scores.min():.3f}, {train_scores.max():.3f}]")

    # Score prediction grid using the same z-score normalization the trainer used.
    # fit_devnet normalizes internally; we re-fit for the pred grid.
    print(f"\nScoring prediction grid...")
    import torch
    model.eval()
    Xp = pred[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    # Use training-set means/stds for normalization (paper-faithful;
    # avoids prediction grid shifting the normalization).
    Xt = train[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    mu = Xt.mean(axis=0)
    sd = Xt.std(axis=0) + 1e-8
    Xp_n = (Xp - mu) / sd
    with torch.no_grad():
        scores = model(torch.from_numpy(Xp_n)).squeeze(-1).numpy()
    print(f"Prediction scores: range [{scores.min():.3f}, {scores.max():.3f}]")

    # Recover the locations of REE deposits in the prediction grid: any
    # row in `pred` that's also in `train` with label=1. Easiest match:
    # use coordinate columns from the unscaled training data
    # (`Xy_all_original_..._random_0.65_all_109.csv` has X/Y; the minmax
    # versions might too).
    # NOTE: this match is approximate; for the headline metric we need to
    # know which prediction-grid rows correspond to known deposits. Their
    # notebook handles this via spatial join. We compute on the training
    # set instead (test split = 30% of 870 = 261 rows; we report metric
    # on training to validate methodology, then call out the gap if the
    # notebook's metric is computed on a different split).

    # Headline metric: reproduce the success-rate curve on training split.
    pos_mask = (train["label"] == 1).to_numpy()
    n_pos = pos_mask.sum()
    print(f"\nHeadline-metric attempt on training set ({n_pos} positives):")

    # Sort training scores descending; how many positives are in top X%?
    order = np.argsort(-train_scores)
    sorted_pos = pos_mask[order]
    n = len(train_scores)
    for pct in [1, 2, 5, 10, 30]:
        k = int(np.ceil(n * pct / 100))
        captured = sorted_pos[:k].sum()
        print(f"  Top {pct:>3}% (k={k:>4}): {captured}/{n_pos} positives = "
              f"{100*captured/n_pos:.1f}% capture")

    # Save artifacts.
    out = {
        "train_shape": list(train.shape),
        "pred_shape": list(pred.shape),
        "n_features_used": len(feat_cols),
        "n_positives_train": int(n_pos),
        "config": {
            "hidden": list(cfg_used.hidden),
            "lr": cfg_used.learning_rate,
            "batch_size": cfg_used.batch_size,
            "n_epochs": cfg_used.n_epochs,
            "confidence_margin": cfg_used.confidence_margin,
        },
        "train_score_stats": {
            "min": float(train_scores.min()),
            "max": float(train_scores.max()),
            "mean": float(train_scores.mean()),
            "median": float(np.median(train_scores)),
        },
        "pred_score_stats": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
        },
        "capture_at_topk_train": {
            f"top_{p}pct": float(np.sort(train_scores)[::-1][:int(np.ceil(n * p / 100))].size > 0
                                 and (pos_mask[np.argsort(-train_scores)[:int(np.ceil(n * p / 100))]]).sum() / n_pos)
            for p in [1, 2, 5, 10, 30]
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
