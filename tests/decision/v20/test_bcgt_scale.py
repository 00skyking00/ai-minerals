"""Tests for the D.1 BCGT-scale multi-hypothesis machinery.

Covers the factory, the expected-deposit aggregation, and the
per-step top-K SARSOP solve. The end-to-end solve test skips when
the `pomdpsol` binary is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_minerals.decision.v20.bcgt_scale import (
    BcgtScaleSARSOPPolicy,
    DEFAULT_TOP_K,
    expected_deposit_per_cell,
    make_bcgt_synthetic_hypothesis_set,
    realize_deposit_sets,
)

REPO = Path(__file__).resolve().parents[3]
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"


def test_factory_builds_three_hypotheses_on_30x30() -> None:
    hset, coords = make_bcgt_synthetic_hypothesis_set(n_side=30)
    assert hset.n_hypotheses == 3
    assert coords.shape == (900, 2)
    for h in hset.hypotheses:
        assert h.n_cells == 900
        assert h.prior_mean_field.shape == (900,)
    # NW and SE peaks should both be positive and roughly equal in magnitude
    nw_peak = float(np.max(hset.hypotheses[0].prior_mean_field))
    se_peak = float(np.max(hset.hypotheses[1].prior_mean_field))
    assert nw_peak > 0.0 and se_peak > 0.0
    assert abs(nw_peak - se_peak) < 0.01


def test_factory_can_drop_null() -> None:
    hset, _ = make_bcgt_synthetic_hypothesis_set(include_null=False)
    assert hset.n_hypotheses == 2
    assert hset.include_null is False


def test_realize_deposit_sets_null_is_empty() -> None:
    hset, _ = make_bcgt_synthetic_hypothesis_set()
    rng = np.random.default_rng(20260613)
    deposit_sets = realize_deposit_sets(hset, rng)
    # Three keys: 0 (NW), 1 (SE), 2 (null)
    assert set(deposit_sets.keys()) == {0, 1, 2}
    # Null hypothesis: no deposit anywhere
    assert deposit_sets[2] == set()
    # The two synthetic hypotheses should both produce a non-empty
    # deposit region (their prior means clear the 0.10 cutoff over a
    # meaningful patch).
    assert len(deposit_sets[0]) > 50
    assert len(deposit_sets[1]) > 50


def test_expected_deposit_under_uniform_belief_peaks_at_blob_centers() -> None:
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=30)
    belief = hset.initial_prior()
    ep = expected_deposit_per_cell(hset, belief)
    assert ep.shape == (900,)
    # Two peaks expected; top-20 should split between NW and SE
    top20 = set(np.argsort(-ep)[:20].tolist())
    nw_quadrant = {
        r * 30 + c for r in range(15, 30) for c in range(0, 15)
    }
    se_quadrant = {
        r * 30 + c for r in range(0, 15) for c in range(15, 30)
    }
    nw_hits = len(top20 & nw_quadrant)
    se_hits = len(top20 & se_quadrant)
    assert nw_hits >= 5
    assert se_hits >= 5


@pytest.mark.skipif(
    not POMDPSOL.exists(),
    reason="pomdpsol binary not built (run scripts/build_pomdpsol.sh)",
)
def test_per_step_solve_returns_a_top_k_candidate() -> None:
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=30)
    rng = np.random.default_rng(20260613)
    deposit_sets = realize_deposit_sets(hset, rng)
    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset,
        deposit_sets=deposit_sets,
        pomdpsol_path=POMDPSOL,
        top_k=DEFAULT_TOP_K,
        sarsop_timeout_sec=10,
        sarsop_precision=0.5,
    )
    chosen = policy.choose_action()
    # The chosen cell must be one of the K candidates the policy
    # surfaced, which under uniform belief sit at one of the two blob
    # centers.
    candidates = policy._top_k_candidates()
    assert chosen in candidates


@pytest.mark.skipif(
    not POMDPSOL.exists(),
    reason="pomdpsol binary not built",
)
def test_observe_updates_belief_toward_truth_hypothesis() -> None:
    """A positive reading at a cell that is in H_NW's deposit set but
    not in H_SE's should shift the categorical belief toward H_NW.
    """
    hset, _ = make_bcgt_synthetic_hypothesis_set()
    rng = np.random.default_rng(20260613)
    deposit_sets = realize_deposit_sets(hset, rng)
    nw_only = deposit_sets[0] - deposit_sets[1]
    assert len(nw_only) > 0, "test setup expects some NW-only cells"
    test_cell = next(iter(nw_only))

    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset,
        deposit_sets=deposit_sets,
        pomdpsol_path=POMDPSOL,
    )
    initial_belief = policy.belief.copy()
    policy.observe(test_cell, observation=1)
    new_belief = policy.belief
    assert new_belief[0] > initial_belief[0]  # NW up
    assert new_belief[1] < initial_belief[1]  # SE down
    assert np.isclose(new_belief.sum(), 1.0)
