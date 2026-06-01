"""Calibration metrics: ECE and reliability tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: weighted average of |observed - predicted| per bin."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).astype(np.float64).ravel()
    if scores.shape != y_true.shape:
        raise ValueError(
            f"scores and y_true shape mismatch: {scores.shape} vs {y_true.shape}"
        )
    n = scores.shape[0]
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(scores[mask].mean())
        pos_rate = float(y_true[mask].mean())
        ece += (count / n) * abs(pos_rate - mean_pred)
    return float(ece)


def reliability_table(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability-diagram data: per bin, count + mean predicted P + observed pos rate."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).astype(np.float64).ravel()
    if scores.shape != y_true.shape:
        raise ValueError(
            f"scores and y_true shape mismatch: {scores.shape} vs {y_true.shape}"
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "count": count,
                "mean_pred": float(scores[mask].mean()),
                "pos_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["bin_left", "bin_right", "count", "mean_pred", "pos_rate"],
    )
