"""bcgt-v2.0 C.2 follow-up: where SARSOP and POMCP beat greedy.

The earlier `v20_sarsop_vs_pomcp.py` benchmark put hypotheses at distinct
corner cells with no exploration-vs-exploitation tradeoff; greedy-MAP
tied with both planners. This script constructs the Tiger-style problem
the multi-hypothesis machinery is supposed to handle:

  - Two hypotheses, A and B, each with a "claim" cell that pays +50 if
    drilled when the matching hypothesis is true and incurs a
    wrong-commitment penalty (-30) if drilled under the other.
  - One "listen" cell that produces an informative observation (high
    signal under A, low under B) but no reward.
  - Uniform prior over A and B.

Greedy-MAP under a uniform prior commits to one claim immediately and
eats the penalty on the wrong half of episodes. SARSOP and POMCP can
spend a drill or two on the listen cell, concentrate the posterior,
then commit. Discounting at 0.95 over a tight 5-drill budget makes
the planning advantage visible.

Output: data/derived/bcgt/fig_v20_sarsop_marker_advantage.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pomdp_py

from ai_minerals.decision.v20.sarsop_policy import (
    MultiHypothesisSARSOPPolicy,
    MultiHypothesisSmallGridPOMDP,
    solve_sarsop,
)

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_sarsop_marker_advantage.png"
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"

N_EPISODES = 200
DRILL_BUDGET = 5
DISCOUNT = 0.95


def make_pomdp() -> MultiHypothesisSmallGridPOMDP:
    """4-cell Tiger problem:

      cell 0 = claim A: +50 if H_A is true, -wrong_penalty if H_B is true,
               but gives no informative sensor reading either way.
      cell 1 = claim B: symmetric to claim A.
      cell 2 = marker A: no reward, sensor fires high under H_A.
      cell 3 = marker B: no reward, sensor fires high under H_B.

    The claim cells are pure "commit" actions: they pay the discovery
    reward only when you are right and the wrong-commitment penalty
    when you are wrong, and they tell you nothing about which
    hypothesis is true. The marker cells are pure information: drilling
    them costs the drill_cost, returns a sensor reading correlated with
    the hypothesis, and pays no reward. To make this concrete, the
    signal_cells set for each hypothesis lists only its marker cell
    (the claim cells are NOT in the signal set).
    """
    deposit_map = {
        0: 0,   # H_A pays out at cell 0
        1: 1,   # H_B pays out at cell 1
    }
    signal_map = {
        0: {2},  # H_A's only informative cell is marker A
        1: {3},  # H_B's only informative cell is marker B
    }
    return MultiHypothesisSmallGridPOMDP(
        n_cells=4,
        hypothesis_names=["H_A", "H_B"],
        deposit_cell_by_hypothesis=deposit_map,
        signal_cells_by_hypothesis=signal_map,
        initial_prior=np.array([0.5, 0.5]),
        alpha_fp=0.10,
        beta_fn=0.10,
        drill_cost=1.0,
        discovery_value=50.0,
        wrong_commitment_penalty=30.0,
    )


def sample_observation(
    pomdp: MultiHypothesisSmallGridPOMDP,
    true_h_idx: int,
    cell_idx: int,
    rng: np.random.Generator,
) -> int:
    signal = pomdp.signal_cells_by_hypothesis.get(true_h_idx, set())
    p1 = (1.0 - pomdp.beta_fn) if cell_idx in signal else pomdp.alpha_fp
    return int(rng.random() < p1)


def realized_reward(
    pomdp: MultiHypothesisSmallGridPOMDP,
    true_h_idx: int,
    cell_idx: int,
    drilled: set[int],
) -> float:
    if cell_idx in drilled:
        return -pomdp.drill_cost
    true_deposit = pomdp.deposit_cell_by_hypothesis.get(true_h_idx)
    if true_deposit is not None and true_deposit == cell_idx:
        return -pomdp.drill_cost + pomdp.discovery_value
    if cell_idx in pomdp.claimed_cells:
        return -pomdp.drill_cost - pomdp.wrong_commitment_penalty
    return -pomdp.drill_cost


def run_episode(
    pomdp: MultiHypothesisSmallGridPOMDP,
    choose_fn,
    observe_fn,
    true_h_idx: int,
    rng: np.random.Generator,
) -> tuple[float, bool, bool]:
    """Returns (discounted_reward, found_deposit, wrong_commitment_fired)."""
    drilled: set[int] = set()
    total = 0.0
    found = False
    wrong = False
    true_deposit = pomdp.deposit_cell_by_hypothesis.get(true_h_idx)
    for t in range(DRILL_BUDGET):
        cell = choose_fn()
        obs = sample_observation(pomdp, true_h_idx, cell, rng)
        r = realized_reward(pomdp, true_h_idx, cell, drilled)
        total += (DISCOUNT ** t) * r
        if cell not in drilled:
            if true_deposit is not None and cell == true_deposit:
                found = True
            elif cell in pomdp.claimed_cells:
                wrong = True
        drilled.add(cell)
        observe_fn(cell, obs)
    return total, found, wrong


def policy_random(pomdp: MultiHypothesisSmallGridPOMDP, rng: np.random.Generator):
    state = {"rng": rng}

    def choose():
        return int(state["rng"].integers(0, pomdp.n_cells))

    def observe(cell, obs):
        pass

    def reset():
        pass

    return choose, observe, reset


def policy_greedy_map_cell(pomdp: MultiHypothesisSmallGridPOMDP,
                           rng: np.random.Generator):
    """Drill the cell with the highest expected deposit probability under
    the current belief. Uses the categorical Bayesian update."""
    state = {"belief": pomdp.initial_prior.copy()}

    def expected_deposit_per_cell() -> np.ndarray:
        ep = np.zeros(pomdp.n_cells)
        for i, dc in pomdp.deposit_cell_by_hypothesis.items():
            if dc is not None:
                ep[dc] += state["belief"][i]
        return ep

    def choose():
        return int(np.argmax(expected_deposit_per_cell()))

    def observe(cell, obs):
        state["belief"] = pomdp.update_belief(state["belief"], cell, obs)

    def reset():
        state["belief"] = pomdp.initial_prior.copy()

    return choose, observe, reset


def policy_sarsop(pomdp: MultiHypothesisSmallGridPOMDP,
                  alpha_policy: pomdp_py.AlphaVectorPolicy,
                  rng: np.random.Generator):
    sp = MultiHypothesisSARSOPPolicy(pomdp=pomdp, alpha_policy=alpha_policy)

    def choose():
        return sp.choose_action()

    def observe(cell, obs):
        sp.observe(cell, obs)

    def reset():
        sp.reset()

    return choose, observe, reset


def policy_pomcp(pomdp: MultiHypothesisSmallGridPOMDP,
                 rng: np.random.Generator,
                 n_sims: int = 500, c: float = 50.0):
    state = {"belief": pomdp.initial_prior.copy(), "agent": None, "planner": None}

    def _rebuild():
        agent = pomdp.build_agent(belief=state["belief"])
        state["agent"] = agent
        state["planner"] = pomdp_py.POUCT(
            max_depth=DRILL_BUDGET,
            discount_factor=DISCOUNT,
            num_sims=n_sims,
            exploration_const=c,
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

    return choose, observe, reset


def run_benchmark() -> dict[str, dict[str, float]]:
    pomdp = make_pomdp()

    print("Solving SARSOP...")
    t0 = time.perf_counter()
    alpha_policy = solve_sarsop(
        pomdp,
        pomdpsol_path=POMDPSOL,
        discount=DISCOUNT,
        timeout_sec=20,
        precision=0.1,
    )
    solve_sec = time.perf_counter() - t0
    print(f"  {len(alpha_policy.alphas)} alpha vectors ({solve_sec:.1f}s)")

    rng = np.random.default_rng(20260612)
    true_h_per_episode = rng.choice(
        len(pomdp.states), size=N_EPISODES, p=pomdp.initial_prior,
    )

    policies = {
        "random": lambda: policy_random(pomdp, np.random.default_rng(42)),
        "greedy_MAP": lambda: policy_greedy_map_cell(pomdp, np.random.default_rng(43)),
        "pomcp": lambda: policy_pomcp(pomdp, np.random.default_rng(44), n_sims=500),
        "sarsop": lambda: policy_sarsop(pomdp, alpha_policy, np.random.default_rng(45)),
    }

    results: dict[str, dict[str, float]] = {}
    for name, factory in policies.items():
        rewards, found_flags, wrong_flags = [], [], []
        for ep_idx in range(N_EPISODES):
            choose, observe, reset = factory()
            reset()
            ep_rng = np.random.default_rng(1000 + ep_idx)
            r, found, wrong = run_episode(
                pomdp, choose, observe,
                true_h_idx=int(true_h_per_episode[ep_idx]),
                rng=ep_rng,
            )
            rewards.append(r)
            found_flags.append(found)
            wrong_flags.append(wrong)
        mean_r = float(np.mean(rewards))
        sem_r = float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
        results[name] = dict(
            mean_reward=mean_r,
            sem_reward=sem_r,
            discovery_rate=float(np.mean(found_flags)),
            wrong_commitment_rate=float(np.mean(wrong_flags)),
        )
        print(
            f"  {name:>11s}: reward={mean_r:+6.2f} +/- {sem_r:.2f}  "
            f"discovery={np.mean(found_flags):.2f}  "
            f"wrong-commit={np.mean(wrong_flags):.2f}"
        )

    results["_meta"] = {
        "sarsop_solve_sec": solve_sec,
        "alpha_count": float(len(alpha_policy.alphas)),
        "n_episodes": float(N_EPISODES),
        "drill_budget": float(DRILL_BUDGET),
        "discount": DISCOUNT,
        "drill_cost": pomdp.drill_cost,
        "discovery_value": pomdp.discovery_value,
        "wrong_commitment_penalty": pomdp.wrong_commitment_penalty,
        "alpha_fp": pomdp.alpha_fp,
        "beta_fn": pomdp.beta_fn,
    }
    return results


def make_chart(results: dict[str, dict[str, float]]) -> None:
    names = ["random", "greedy_MAP", "pomcp", "sarsop"]
    colors = ["#9aa0a6", "#1f77b4", "#ff7f0e", "#2ca02c"]

    rewards = [results[n]["mean_reward"] for n in names]
    sems = [results[n]["sem_reward"] for n in names]
    wrong = [results[n]["wrong_commitment_rate"] for n in names]
    discovery = [results[n]["discovery_rate"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    bars = ax.bar(names, rewards, color=colors, edgecolor="black",
                  linewidth=0.6, yerr=sems, capsize=4)
    for bar, val in zip(bars, rewards):
        y = val + (1.2 if val >= 0 else -2.5)
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:+.1f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=10)
    ax.set_ylabel("Mean discounted reward (gamma=0.95)")
    ax.set_title("Cumulative reward")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(names, wrong, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, wrong):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wrong-commitment rate")
    ax.set_title("How often the policy drilled the wrong claim")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    bars = ax.bar(names, discovery, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, discovery):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Discovery rate")
    ax.set_title("Right claim eventually drilled")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "C.2 follow-up: where SARSOP and POMCP beat greedy.\n"
        "Tiger-style problem: 2 claim cells (+50 right / -30 wrong) "
        "and 1 listen cell (0). Uniform prior, 5-drill budget, "
        f"{N_EPISODES} episodes.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


def main() -> int:
    results = run_benchmark()
    make_chart(results)
    out_json = OUT_PNG.with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
