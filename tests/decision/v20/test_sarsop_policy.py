"""Tests for the C.2 part 2 SARSOP-backed multi-hypothesis policy.

Covers GitHub issue #11. The end-to-end SARSOP solve requires the
external `pomdpsol` binary at `vendor/sarsop/pomdpsol`; tests that
need it skip when the binary is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_minerals.decision.v20.sarsop_policy import (
    BinaryObs,
    CellAction,
    HypothesisState,
    MultiHypothesisSARSOPPolicy,
    MultiHypothesisSmallGridPOMDP,
    solve_sarsop,
)


REPO = Path(__file__).resolve().parents[3]
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"


def _make_pomdp() -> MultiHypothesisSmallGridPOMDP:
    deposit_map = {0: 0, 1: 4, 2: 20, 3: 24, 4: None}
    return MultiHypothesisSmallGridPOMDP(
        n_cells=25,
        hypothesis_names=["H1_NW", "H2_NE", "H3_SW", "H4_SE", "H_null"],
        deposit_cell_by_hypothesis=deposit_map,
        initial_prior=np.full(5, 0.2),
        alpha_fp=0.05,
        beta_fn=0.10,
    )


def test_pomdp_construction_state_action_obs_counts() -> None:
    p = _make_pomdp()
    assert len(p.states) == 5
    assert len(p.actions) == 25
    assert len(p.observations) == 2


def test_initial_prior_must_sum_to_one() -> None:
    deposit_map = {0: 0, 1: 4}
    with pytest.raises(ValueError, match="initial_prior must sum"):
        MultiHypothesisSmallGridPOMDP(
            n_cells=25,
            hypothesis_names=["A", "B"],
            deposit_cell_by_hypothesis=deposit_map,
            initial_prior=np.array([0.3, 0.3]),
        )


def test_bayesian_update_concentrates_on_true_hypothesis() -> None:
    """A positive observation at H1's deposit cell should sharpen the
    posterior toward H1."""
    p = _make_pomdp()
    new = p.update_belief(p.initial_prior, cell_idx=0, observation=1)
    # H1 (idx 0) is the only hypothesis whose deposit is cell 0;
    # everyone else has alpha=0.05 of producing obs=1, H1 has 1-beta=0.9.
    assert new[0] > new[1]
    assert new[0] > 0.5
    assert np.isclose(new.sum(), 1.0)


def test_bayesian_update_negative_obs_downweights_hypothesis() -> None:
    """A negative observation at H1's deposit cell should reduce H1's
    posterior weight."""
    p = _make_pomdp()
    new = p.update_belief(p.initial_prior, cell_idx=0, observation=0)
    # H1 has beta=0.10 chance of producing obs=0 at its own deposit cell;
    # everyone else has 1-alpha=0.95.
    assert new[0] < p.initial_prior[0]
    assert np.isclose(new.sum(), 1.0)


def test_agent_build_returns_pomdp_py_agent() -> None:
    """The built agent should have the SARSOP-compatible enumerable spaces."""
    import pomdp_py
    p = _make_pomdp()
    agent = p.build_agent()
    assert isinstance(agent, pomdp_py.Agent)
    assert len(list(agent.all_states)) == 5
    assert len(list(agent.all_actions)) == 25
    assert len(list(agent.all_observations)) == 2


@pytest.mark.skipif(
    not POMDPSOL.exists(),
    reason="pomdpsol binary not built (run scripts/build_pomdpsol.sh)",
)
def test_sarsop_solve_returns_alpha_vectors_and_picks_corner() -> None:
    """SARSOP should converge on the 5x5 / 5-hypothesis POMDP, and after
    observing a positive at a corner the policy should drill that corner.
    """
    import pomdp_py

    p = _make_pomdp()
    alpha_policy = solve_sarsop(
        p,
        pomdpsol_path=POMDPSOL,
        discount=0.95,
        timeout_sec=15,
        precision=0.5,
    )
    assert isinstance(alpha_policy, pomdp_py.AlphaVectorPolicy)
    assert len(alpha_policy.alphas) > 0

    sp = MultiHypothesisSARSOPPolicy(pomdp=p, alpha_policy=alpha_policy)
    # Positive reading at the NW corner concentrates belief on H1_NW;
    # the optimal action under that belief is to re-drill cell 0 to
    # confirm the deposit (and collect the discovery reward).
    sp.observe(cell_idx=0, observation=1)
    assert sp.belief[0] > 0.5
    chosen = sp.choose_action()
    assert chosen == 0, f"expected drill at NW corner; got cell {chosen}"
