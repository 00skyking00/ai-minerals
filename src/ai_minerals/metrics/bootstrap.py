"""Bootstrap confidence intervals for ranking and prospectivity metrics."""

from __future__ import annotations

import numpy as np


def _capture_at_k(
    scores: np.ndarray,
    positives: np.ndarray,
    k_percent: float,
) -> float:
    n = scores.shape[0]
    top_n = int(np.ceil(n * k_percent / 100.0))
    if top_n <= 0 or positives.sum() == 0:
        return 0.0
    top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
    return float(positives[top_idx].sum() / positives.sum())


def capture_rate(
    scores: np.ndarray,
    positives: np.ndarray,
    k_pct: float,
) -> float:
    """Fraction of positives in the top k% of cells ranked by score (descending).

    Inputs may be any array-like; positives is coerced to a boolean / 0-1 mask.
    Returns 0.0 if the input is empty or contains no positives.
    """
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    positives_arr = np.asarray(positives).astype(np.int64).ravel()
    if scores_arr.shape != positives_arr.shape:
        raise ValueError(
            f"scores and positives shape mismatch: "
            f"{scores_arr.shape} vs {positives_arr.shape}"
        )
    if scores_arr.size == 0:
        return 0.0
    return _capture_at_k(scores_arr, positives_arr, k_pct)


def bootstrap_capture_ci(
    scores: np.ndarray,
    positives: np.ndarray,
    ks_percent: tuple[float, ...] = (1.0, 5.0, 10.0),
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[float, tuple[float, float, float]]:
    """Per-k bootstrap (point, lo, hi) capture rate at each k%."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = np.asarray(positives).astype(bool).ravel()
    if scores.shape != positives.shape:
        raise ValueError(
            f"scores and positives shape mismatch: {scores.shape} vs {positives.shape}"
        )

    pos_idx = np.flatnonzero(positives)
    n_pos = pos_idx.size

    out: dict[float, tuple[float, float, float]] = {}

    for k in ks_percent:
        point = _capture_at_k(scores, positives.astype(np.int64), k)

        if n_pos == 0:
            out[k] = (point, 0.0, 0.0)
            continue

        rng = np.random.default_rng(seed)
        n = scores.shape[0]
        top_n = int(np.ceil(n * k / 100.0))
        top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
        in_top = np.zeros(n, dtype=bool)
        in_top[top_idx] = True

        boots = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            resampled = rng.choice(pos_idx, size=n_pos, replace=True)
            boots[i] = in_top[resampled].mean()

        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[k] = (point, float(lo), float(hi))

    return out


def _auc_pa(scores: np.ndarray, positives: np.ndarray) -> float:
    n = scores.shape[0]
    total_pos = positives.sum()
    if total_pos == 0 or n == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_pos = positives[order].astype(np.float64)
    cum_pos = np.cumsum(sorted_pos)
    frac_area = np.arange(1, n + 1, dtype=np.float64) / n
    frac_pos = cum_pos / total_pos
    frac_area = np.concatenate(([0.0], frac_area))
    frac_pos = np.concatenate(([0.0], frac_pos))
    return float(np.trapezoid(frac_pos, frac_area))


def bootstrap_auc_pa_ci(
    scores: np.ndarray,
    positives: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) for the area under the P-A (success-rate) curve."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = np.asarray(positives).astype(bool).ravel()
    if scores.shape != positives.shape:
        raise ValueError(
            f"scores and positives shape mismatch: {scores.shape} vs {positives.shape}"
        )

    pos_arr = positives.astype(np.int64)
    point = _auc_pa(scores, pos_arr)

    pos_idx = np.flatnonzero(positives)
    n_pos = pos_idx.size
    if n_pos == 0:
        return (point, 0.0, 0.0)

    n = scores.shape[0]
    order = np.argsort(-scores, kind="stable")
    rank_of = np.empty(n, dtype=np.int64)
    rank_of[order] = np.arange(n)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=np.float64)
    frac_area_full = np.arange(1, n + 1, dtype=np.float64) / n

    for i in range(n_resamples):
        resampled = rng.choice(pos_idx, size=n_pos, replace=True)
        sorted_pos = np.zeros(n, dtype=np.float64)
        ranks = rank_of[resampled]
        np.add.at(sorted_pos, ranks, 1.0)
        cum_pos = np.cumsum(sorted_pos)
        frac_pos = cum_pos / n_pos
        boots[i] = float(
            np.trapezoid(
                np.concatenate(([0.0], frac_pos)),
                np.concatenate(([0.0], frac_area_full)),
            )
        )

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (point, float(lo), float(hi))
