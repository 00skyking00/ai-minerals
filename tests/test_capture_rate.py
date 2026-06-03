"""Verify the v3 Phase E.2 capture-rate helper.

`capture_rate(scores, positives, k_pct)` ranks cells by score descending,
takes the top k%, and returns the fraction of positives in that slice.

Properties checked:
  1. 100 cells, 5 positives in the top 5 scores -> capture_at_5pct = 1.0,
     enrichment = 1.0 / 0.05 = 20x.
  2. 100 cells, 5 positives spread across the score range (one per
     contiguous quintile) -> capture_at_5pct ~= 0.20 with the top-quintile
     positive captured.
  3. 100 cells, 5 positives uniformly random -> capture_at_5pct on average
     ~= 0.05 across many seeds (enrichment ~= 1x), the random-baseline
     check.
  4. Empty arrays and all-zero positives return 0.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.metrics.bootstrap import capture_rate


def test_capture_perfect_top5_is_one_and_enrichment_20x() -> None:
    n = 100
    # Scores 99, 98, ..., 0 -> indices 0..4 are the top 5%.
    scores = np.arange(n, 0, -1, dtype=np.float64)
    positives = np.zeros(n, dtype=np.int64)
    positives[:5] = 1  # all 5 positives sit in the top 5 scores

    cap = capture_rate(scores, positives, 5.0)
    assert cap == pytest.approx(1.0, abs=1e-9)
    enrichment = cap / 0.05
    assert enrichment == pytest.approx(20.0, abs=1e-9)


def test_capture_random_distribution_is_about_baseline() -> None:
    """5 positives randomly distributed across 100 cells average ~5% capture."""
    n = 100
    n_pos = 5
    n_trials = 2000

    captures = np.empty(n_trials, dtype=np.float64)
    rng = np.random.default_rng(seed=42)
    for i in range(n_trials):
        # Random scores (independent of positives).
        scores = rng.random(n)
        positives = np.zeros(n, dtype=np.int64)
        pos_idx = rng.choice(n, size=n_pos, replace=False)
        positives[pos_idx] = 1
        captures[i] = capture_rate(scores, positives, 5.0)

    mean_capture = float(captures.mean())
    # 5 positives in 100 cells, top 5% = 5 cells; expected captures-per-trial
    # are draws from a hypergeometric(N=100, K=5, n=5). Mean = K*n/N = 0.25
    # positives captured -> capture rate mean = 0.25 / 5 = 0.05. With 2000
    # trials, sample mean is tight around 0.05.
    assert mean_capture == pytest.approx(0.05, abs=0.01)
    mean_enrichment = mean_capture / 0.05
    assert mean_enrichment == pytest.approx(1.0, abs=0.2)


def test_capture_partial_capture_one_in_five() -> None:
    """Positives at ranks 0, 20, 40, 60, 80 -> only the top-ranked one lands in top 5%."""
    n = 100
    scores = np.arange(n, 0, -1, dtype=np.float64)  # ranks 0..99 in descending order
    positives = np.zeros(n, dtype=np.int64)
    positives[[0, 20, 40, 60, 80]] = 1

    cap = capture_rate(scores, positives, 5.0)
    # Top 5% = top 5 cells = ranks 0..4; only rank 0 is positive -> 1/5.
    assert cap == pytest.approx(0.2, abs=1e-9)


def test_capture_at_one_percent() -> None:
    """Top 1% of 100 cells = 1 cell; if it holds 1 of 5 positives, capture = 0.2."""
    n = 100
    scores = np.arange(n, 0, -1, dtype=np.float64)
    positives = np.zeros(n, dtype=np.int64)
    positives[[0, 20, 40, 60, 80]] = 1
    cap = capture_rate(scores, positives, 1.0)
    assert cap == pytest.approx(0.2, abs=1e-9)


def test_capture_at_ten_percent() -> None:
    """Top 10% of 100 cells = 10 cells; 2 of 5 positives sit in ranks 0..9 -> capture=0.4."""
    n = 100
    scores = np.arange(n, 0, -1, dtype=np.float64)
    positives = np.zeros(n, dtype=np.int64)
    # Two positives in the top decile (ranks 0, 5), three outside.
    positives[[0, 5, 20, 40, 60]] = 1
    cap = capture_rate(scores, positives, 10.0)
    assert cap == pytest.approx(0.4, abs=1e-9)


def test_capture_empty_inputs_returns_zero() -> None:
    out = capture_rate(np.array([]), np.array([]), 5.0)
    assert out == 0.0


def test_capture_no_positives_returns_zero() -> None:
    n = 100
    scores = np.random.default_rng(0).random(n)
    positives = np.zeros(n, dtype=np.int64)
    assert capture_rate(scores, positives, 5.0) == 0.0


def test_capture_accepts_boolean_positives() -> None:
    n = 100
    scores = np.arange(n, 0, -1, dtype=np.float64)
    positives = np.zeros(n, dtype=bool)
    positives[:5] = True
    cap = capture_rate(scores, positives, 5.0)
    assert cap == pytest.approx(1.0, abs=1e-9)


def test_capture_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        capture_rate(np.arange(10, dtype=np.float64), np.zeros(8, dtype=np.int64), 5.0)
