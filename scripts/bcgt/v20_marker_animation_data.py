"""Generate episode trajectory data for the interactive Tiger-style
animation in Chapter 7.

Writes a JSON file with full per-step state for every (episode, policy):
belief, action, observation, reward, cumulative reward, drilled-set.
The chapter widget reads this JSON inline and renders a step-through
visualization.

Output: data/derived/bcgt/v20_marker_animation_data.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pomdp_py

from ai_minerals.decision.v20.sarsop_policy import (
    MultiHypothesisSARSOPPolicy,
    MultiHypothesisSmallGridPOMDP,
    solve_sarsop,
)

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "data/derived/bcgt/v20_marker_animation_data.json"
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"

DRILL_BUDGET = 5
DISCOUNT = 0.95
POMCP_N_SIMS = 500


def make_pomdp() -> MultiHypothesisSmallGridPOMDP:
    return MultiHypothesisSmallGridPOMDP(
        n_cells=4,
        hypothesis_names=["H_A", "H_B"],
        deposit_cell_by_hypothesis={0: 0, 1: 1},
        signal_cells_by_hypothesis={0: {2}, 1: {3}},
        initial_prior=np.array([0.5, 0.5]),
        alpha_fp=0.10,
        beta_fn=0.10,
        drill_cost=1.0,
        discovery_value=50.0,
        wrong_commitment_penalty=30.0,
    )


def sample_observation(pomdp, true_h_idx, cell_idx, rng):
    signal = pomdp.signal_cells_by_hypothesis.get(true_h_idx, set())
    p1 = (1.0 - pomdp.beta_fn) if cell_idx in signal else pomdp.alpha_fp
    return int(rng.random() < p1)


def realized_reward(pomdp, true_h_idx, cell_idx, drilled):
    if cell_idx in drilled:
        return -pomdp.drill_cost
    true_deposit = pomdp.deposit_cell_by_hypothesis.get(true_h_idx)
    if true_deposit is not None and true_deposit == cell_idx:
        return -pomdp.drill_cost + pomdp.discovery_value
    if cell_idx in pomdp.claimed_cells:
        return -pomdp.drill_cost - pomdp.wrong_commitment_penalty
    return -pomdp.drill_cost


def make_policy(name, pomdp, alpha_policy, rng):
    """Returns (choose, observe, reset)."""

    if name == "random":
        def choose():
            return int(rng.integers(0, pomdp.n_cells))

        def observe(cell, obs):
            pass

        def reset():
            pass

        def get_belief():
            return pomdp.initial_prior.tolist()
        return choose, observe, reset, get_belief

    if name == "greedy_MAP":
        state = {"belief": pomdp.initial_prior.copy()}

        def choose():
            ep = np.zeros(pomdp.n_cells)
            for i, dc in pomdp.deposit_cell_by_hypothesis.items():
                if dc is not None:
                    ep[dc] += state["belief"][i]
            return int(np.argmax(ep))

        def observe(cell, obs):
            state["belief"] = pomdp.update_belief(state["belief"], cell, obs)

        def reset():
            state["belief"] = pomdp.initial_prior.copy()

        def get_belief():
            return state["belief"].tolist()
        return choose, observe, reset, get_belief

    if name == "pomcp":
        state = {"belief": pomdp.initial_prior.copy(), "agent": None, "planner": None}

        def _rebuild():
            agent = pomdp.build_agent(belief=state["belief"])
            state["agent"] = agent
            state["planner"] = pomdp_py.POUCT(
                max_depth=DRILL_BUDGET,
                discount_factor=DISCOUNT,
                num_sims=POMCP_N_SIMS,
                exploration_const=50.0,
                rollout_policy=agent.policy_model,
            )

        def choose():
            if state["planner"] is None:
                _rebuild()
            action = state["planner"].plan(state["agent"])
            return int(action.cell_idx)

        def observe(cell, obs):
            state["belief"] = pomdp.update_belief(state["belief"], cell, obs)
            _rebuild()

        def reset():
            state["belief"] = pomdp.initial_prior.copy()
            state["agent"] = None
            state["planner"] = None

        def get_belief():
            return state["belief"].tolist()
        return choose, observe, reset, get_belief

    if name == "sarsop":
        sp = MultiHypothesisSARSOPPolicy(pomdp=pomdp, alpha_policy=alpha_policy)

        def choose():
            return sp.choose_action()

        def observe(cell, obs):
            sp.observe(cell, obs)

        def reset():
            sp.reset()

        def get_belief():
            return sp.belief.tolist()
        return choose, observe, reset, get_belief

    raise ValueError(f"unknown policy {name}")


def trace_episode(pomdp, policy_name, alpha_policy, true_h_idx, ep_seed):
    rng = np.random.default_rng(ep_seed)
    choose, observe, reset, get_belief = make_policy(
        policy_name, pomdp, alpha_policy, np.random.default_rng(ep_seed + 1000),
    )
    reset()

    drilled: set[int] = set()
    total = 0.0
    steps = []
    for t in range(DRILL_BUDGET):
        belief_before = get_belief()
        cell = choose()
        obs = sample_observation(pomdp, true_h_idx, cell, rng)
        r = realized_reward(pomdp, true_h_idx, cell, drilled)
        discounted = (DISCOUNT ** t) * r
        total += discounted
        already = cell in drilled
        drilled.add(cell)
        observe(cell, obs)
        belief_after = get_belief()
        steps.append({
            "step": t,
            "belief_before": belief_before,
            "action": cell,
            "observation": obs,
            "reward": r,
            "discounted_reward": discounted,
            "cumulative_reward": total,
            "already_drilled": already,
            "drilled": sorted(drilled),
            "belief_after": belief_after,
        })
    return steps


def main() -> int:
    pomdp = make_pomdp()
    print("Solving SARSOP...")
    alpha_policy = solve_sarsop(
        pomdp, pomdpsol_path=POMDPSOL,
        discount=DISCOUNT, timeout_sec=20, precision=0.1,
    )

    episodes_spec = [
        {"id": 0, "label": "Truth: Hypothesis A", "truth": 0, "seed": 7001},
        {"id": 1, "label": "Truth: Hypothesis B", "truth": 1, "seed": 7002},
        {"id": 2, "label": "Truth: Hypothesis A (different seed)",
         "truth": 0, "seed": 7003},
    ]
    policies = ["random", "greedy_MAP", "pomcp", "sarsop"]

    out = {
        "experiment": {
            "n_cells": pomdp.n_cells,
            "cell_labels": [
                "claim A", "claim B", "marker A", "marker B",
            ],
            "cell_layout": [
                {"row": 0, "col": 0, "kind": "claim", "hypothesis": "A"},
                {"row": 0, "col": 1, "kind": "claim", "hypothesis": "B"},
                {"row": 1, "col": 0, "kind": "marker", "hypothesis": "A"},
                {"row": 1, "col": 1, "kind": "marker", "hypothesis": "B"},
            ],
            "hypothesis_names": list(pomdp.hypothesis_names),
            "deposit_cells": {
                str(i): pomdp.deposit_cell_by_hypothesis[i] for i in range(2)
            },
            "signal_cells": {
                str(i): sorted(pomdp.signal_cells_by_hypothesis[i])
                for i in range(2)
            },
            "discovery_value": pomdp.discovery_value,
            "drill_cost": pomdp.drill_cost,
            "wrong_commitment_penalty": pomdp.wrong_commitment_penalty,
            "alpha_fp": pomdp.alpha_fp,
            "beta_fn": pomdp.beta_fn,
            "discount": DISCOUNT,
            "drill_budget": DRILL_BUDGET,
        },
        "episodes": [],
    }

    for ep in episodes_spec:
        ep_out = {
            "id": ep["id"],
            "label": ep["label"],
            "truth_index": ep["truth"],
            "truth_label": pomdp.hypothesis_names[ep["truth"]],
            "policies": {},
        }
        for policy in policies:
            steps = trace_episode(
                pomdp, policy, alpha_policy,
                true_h_idx=ep["truth"], ep_seed=ep["seed"],
            )
            ep_out["policies"][policy] = {"steps": steps}
        out["episodes"].append(ep_out)
        print(f"episode {ep['id']} ({ep['label']}): traced {len(policies)} policies")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
