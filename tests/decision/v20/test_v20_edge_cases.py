"""Edge-case tests for the v20 multi-hypothesis stack.

Covers four risk areas:
    1. SARSOP top-K when there are fewer un-drilled cells than K.
    2. ESS particle filter under a high observation count (degeneracy).
    3. Hypothesis set with the null only (all paper hypotheses dropped).
    4. Top-K equal to or greater than the full grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.belief_ess import MultiHypothesisESSParticleFilter
from ai_minerals.decision.v20.bcgt_scale import (
    BcgtScaleSARSOPPolicy,
    make_bcgt_synthetic_hypothesis_set,
    realize_deposit_sets,
)
from ai_minerals.decision.v20.hypotheses import (
    HypothesisSet, NullHypothesis,
)


def test_topk_larger_than_grid_returns_only_undrilled():
    """If top_k > available cells, the policy returns whatever it can."""
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=5)
    canonical = realize_deposit_sets(hset, np.random.default_rng(0))
    n_cells = hset.hypotheses[0].n_cells
    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset,
        deposit_sets=canonical,
        pomdpsol_path="bogus",
        top_k=n_cells + 50,
    )
    cands = policy._top_k_candidates()
    assert len(cands) == n_cells


def test_topk_equal_to_grid_after_partial_drilling():
    """When most cells have been drilled, top-K shrinks to available cells."""
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=5)
    canonical = realize_deposit_sets(hset, np.random.default_rng(0))
    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset, deposit_sets=canonical,
        pomdpsol_path="bogus", top_k=20,
    )
    for c in range(20):
        policy._drilled.add(c)
    cands = policy._top_k_candidates()
    assert len(cands) <= 5  # 25 cells total, 20 drilled → 5 remaining
    assert all(c not in policy._drilled for c in cands)


def test_ess_pf_high_observation_count_keeps_belief_valid():
    """50 sequential observations should not produce NaN or degenerate belief."""
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=8)
    rng = np.random.default_rng(123)
    pf = MultiHypothesisESSParticleFilter(
        hypothesis_set=hset, n_particles=20, ess_refresh_steps=1,
    )
    pf.initialize(rng=rng)
    n_cells = hset.hypotheses[0].n_cells
    for step in range(50):
        cell = rng.integers(0, n_cells)
        obs = float(rng.normal(0.0, 0.1))
        pf.update(
            cell_idx=int(cell), observation=obs,
            sensor_noise_sigma=0.05, rng=rng,
        )
    cat = pf.categorical_belief
    assert cat.shape == (hset.n_hypotheses,)
    assert np.all(np.isfinite(cat))
    assert abs(cat.sum() - 1.0) < 1e-6
    assert np.all(cat >= 0.0)


def test_null_only_hypothesis_set():
    """A null-only HypothesisSet is constructable and has the expected size."""
    null = NullHypothesis(marginal_std=0.1)
    hset = HypothesisSet(hypotheses=(), null=null, include_null=True)
    assert hset.n_hypotheses == 1
    init = hset.initial_prior()
    assert init.shape == (1,)
    assert init[0] == pytest.approx(1.0)
