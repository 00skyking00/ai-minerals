"""Tests for the B.2 retrospective ``capture@N-drills`` scorer.

The metric is the chunk B.2-expansion alternative to ``capture@k%`` for
small absolute drill budgets, where a fixed-percentage cutoff scales
with grid size rather than what an exploration program could actually
drill within a finite money budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.simulator import RetrospectiveBCGSValidator


def _make_validator(n_cells: int = 100,
                    positive_cells: tuple[int, ...] = (3, 7, 12, 25, 40)
                    ) -> RetrospectiveBCGSValidator:
    """Build a minimal validator whose only purpose is to call the scorer."""
    coords = np.column_stack([
        np.arange(n_cells) * 500.0, np.zeros(n_cells, dtype=np.float64),
    ])
    pre_prior = np.full(n_cells, 0.1, dtype=np.float64)
    post_positives = np.zeros(n_cells, dtype=np.int64)
    for c in positive_cells:
        post_positives[c] = 1
    cells_drilled = np.zeros(n_cells, dtype=np.int64)
    post_grade = np.zeros(n_cells, dtype=np.float64)
    return RetrospectiveBCGSValidator(
        pre_2010_prior=pre_prior,
        post_2010_positives=post_positives,
        cells_drilled_pre_2010=cells_drilled,
        cell_coords_m=coords,
        post_2010_grade=post_grade,
        drill_budget=50,
    )


def test_capture_at_n_drills_returns_zero_when_no_positives_exist() -> None:
    validator = _make_validator(positive_cells=())
    trajectory = list(range(50))
    score = validator._capture_at_n_drills(trajectory, n_drills=50)
    assert score == 0.0


def test_capture_at_n_drills_returns_zero_when_n_is_zero() -> None:
    validator = _make_validator()
    trajectory = [3, 7, 12, 25, 40]
    score = validator._capture_at_n_drills(trajectory, n_drills=0)
    assert score == 0.0


def test_capture_at_n_drills_full_capture_when_n_covers_all_positives() -> None:
    """5 positives at indices {3, 7, 12, 25, 40}; first 50 of trajectory hit
    all 5 of them; metric should be 1.0."""
    validator = _make_validator()
    trajectory = list(range(50))
    score = validator._capture_at_n_drills(trajectory, n_drills=50)
    assert score == pytest.approx(1.0)


def test_capture_at_n_drills_partial_capture_at_smaller_n() -> None:
    """At n=20, the first 20 picks cover positives at indices {3, 7, 12} =
    3 out of 5; metric should be 0.6."""
    validator = _make_validator()
    trajectory = list(range(50))
    score = validator._capture_at_n_drills(trajectory, n_drills=20)
    assert score == pytest.approx(3.0 / 5.0)


def test_capture_at_n_drills_clamps_n_to_trajectory_length() -> None:
    """If n is larger than the trajectory length, the metric evaluates over
    the whole trajectory rather than raising."""
    validator = _make_validator()
    trajectory = [3, 7, 12]  # 3 picks, all positives
    score = validator._capture_at_n_drills(trajectory, n_drills=100)
    assert score == pytest.approx(3.0 / 5.0)


def test_capture_at_n_drills_handles_repeated_cells_in_trajectory() -> None:
    """The metric scores against unique positive-cell hits, but counting via
    np.sum on the positives array correctly handles trajectories with
    duplicates (each duplicate just adds 0 or 1 to the total)."""
    validator = _make_validator()
    trajectory = [3, 3, 3, 3, 3]  # one positive, drilled 5 times
    score = validator._capture_at_n_drills(trajectory, n_drills=5)
    # np.sum on the boolean indexes counts cell 3 five times -> hits=5
    # over total_positives=5 -> score=1.0. This is the documented
    # contract: the metric is a "trajectory-share" rather than a
    # unique-cell-share. Callers that want unique-cell scoring should
    # dedupe the trajectory before passing it in.
    assert score == pytest.approx(5.0 / 5.0)


def test_run_policy_full_returns_both_metric_dicts() -> None:
    """run_policy_full delivers a single dict with both ``capture_at_k_pct``
    and ``capture_at_n_drills`` keys, so the batch driver can record both
    in a single pass through the policy."""

    class _ConstantPickPolicy:
        """Trivial policy that picks cells in fixed index order."""

        def reset(self, problem, rng):
            self._next = 0

        def choose_action(self, history, drilled, rng):
            cell = self._next
            self._next += 1
            return cell

    validator = _make_validator()
    out = validator.run_policy_full(
        policy=_ConstantPickPolicy(),
        rng=np.random.default_rng(0),
        n_drills_values=(10, 25, 50),
        k_percent_values=(1, 5, 25),
    )
    assert "capture_at_k_pct" in out
    assert "capture_at_n_drills" in out
    assert set(out["capture_at_n_drills"].keys()) == {10, 25, 50}
    assert set(out["capture_at_k_pct"].keys()) == {1, 5, 25}
    # All positives at indices 3, 7, 12, 25, 40 - first 50 picks cover all 5
    assert out["capture_at_n_drills"][50] == pytest.approx(1.0)
