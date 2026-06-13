"""bcgt-v2.0 C.2 part 2: SARSOP vs POMCP on a small multi-hypothesis POMDP.

Builds a discretized multi-hypothesis POMDP (5x5 grid, 4 paper hypotheses
+ 1 null), solves it once offline with SARSOP, and compares the
resulting alpha-vector policy against POMCP (online tree search),
greedy-MAP-cell, and random over `n_episodes` Monte Carlo runs at the
C.1 sensor settings (alpha=0.05, beta=0.10).

Score: discovery rate (fraction of episodes where the policy drills the
true deposit cell within the drill budget) and mean cumulative reward.

Output: data/derived/bcgt/fig_v20_sarsop_vs_pomcp.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pomdp_py

from ai_minerals.decision.v20.sarsop_policy import (
    BinaryObs,
    CellAction,
    HypothesisState,
    MultiHypothesisSARSOPPolicy,
    MultiHypothesisSmallGridPOMDP,
    solve_sarsop,
)

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_sarsop_vs_pomcp.png"
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"

N_EPISODES = 50
DRILL_BUDGET = 9
DISCOUNT = 0.95

GRID_W = 5
GRID_H = 5
N_CELLS = GRID_W * GRID_H


def make_pomdp() -> MultiHypothesisSmallGridPOMDP:
    deposit_map = {
        0: 0,                       # H1 NW corner
        1: GRID_W - 1,              # H2 NE corner
        2: N_CELLS - GRID_W,        # H3 SW corner
        3: N_CELLS - 1,             # H4 SE corner
        4: None,                    # H_null
    }
    prior = np.full(5, 0.2)
    return MultiHypothesisSmallGridPOMDP(
        n_cells=N_CELLS,
        hypothesis_names=["H1_NW", "H2_NE", "H3_SW", "H4_SE", "H_null"],
        deposit_cell_by_hypothesis=deposit_map,
        initial_prior=prior,
        alpha_fp=0.05,
        beta_fn=0.10,
        drill_cost=1.0,
        discovery_value=50.0,
    )


def update_belief(
    pomdp: MultiHypothesisSmallGridPOMDP,
    belief: np.ndarray,
    cell_idx: int,
    observation: int,
) -> np.ndarray:
    return pomdp.update_belief(belief, cell_idx, observation)


def sample_observation(
    pomdp: MultiHypothesisSmallGridPOMDP,
    true_h_idx: int,
    cell_idx: int,
    rng: np.random.Generator,
) -> int:
    deposit_cell = pomdp.deposit_cell_by_hypothesis.get(true_h_idx)
    is_deposit_cell = (deposit_cell is not None) and (deposit_cell == cell_idx)
    p1 = (1.0 - pomdp.beta_fn) if is_deposit_cell else pomdp.alpha_fp
    return int(rng.random() < p1)


def realized_reward(
    pomdp: MultiHypothesisSmallGridPOMDP,
    true_h_idx: int,
    cell_idx: int,
    drilled: set[int],
) -> float:
    deposit_cell = pomdp.deposit_cell_by_hypothesis.get(true_h_idx)
    if cell_idx in drilled:
        return -pomdp.drill_cost
    if deposit_cell is not None and deposit_cell == cell_idx:
        return -pomdp.drill_cost + pomdp.discovery_value
    return -pomdp.drill_cost


def run_episode(
    pomdp: MultiHypothesisSmallGridPOMDP,
    choose_fn,
    observe_fn,
    true_h_idx: int,
    rng: np.random.Generator,
) -> tuple[float, bool, float]:
    """Returns (discounted_cumulative_reward, found_deposit, mean_choose_ms)."""
    drilled: set[int] = set()
    total = 0.0
    found = False
    choose_ms: list[float] = []
    for t in range(DRILL_BUDGET):
        t0 = time.perf_counter()
        cell = choose_fn()
        choose_ms.append((time.perf_counter() - t0) * 1000.0)
        obs = sample_observation(pomdp, true_h_idx, cell, rng)
        r = realized_reward(pomdp, true_h_idx, cell, drilled)
        total += (DISCOUNT ** t) * r
        drilled.add(cell)
        deposit_cell = pomdp.deposit_cell_by_hypothesis.get(true_h_idx)
        if deposit_cell is not None and cell == deposit_cell:
            found = True
        observe_fn(cell, obs)
    return total, found, float(np.mean(choose_ms))


def policy_random(pomdp: MultiHypothesisSmallGridPOMDP, rng: np.random.Generator):
    state = {"rng": rng}

    def choose():
        return int(state["rng"].integers(0, pomdp.n_cells))

    def observe(cell, obs):
        pass

    def reset():
        pass

    return choose, observe, reset


def policy_greedy_map_cell(pomdp: MultiHypothesisSmallGridPOMDP, rng: np.random.Generator):
    """Greedy: drill the most-likely deposit cell under the current belief.

    For the null hypothesis (no deposit), assign zero mass to any cell.
    """
    state = {"belief": pomdp.initial_prior.copy()}

    def expected_deposit_per_cell() -> np.ndarray:
        ep = np.zeros(pomdp.n_cells)
        for i, dc in pomdp.deposit_cell_by_hypothesis.items():
            if dc is not None:
                ep[dc] += state["belief"][i]
        return ep

    def choose():
        ep = expected_deposit_per_cell()
        return int(np.argmax(ep))

    def observe(cell, obs):
        state["belief"] = update_belief(pomdp, state["belief"], cell, obs)

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


def policy_pomcp(pomdp: MultiHypothesisSmallGridPOMDP, rng: np.random.Generator,
                 n_sims: int = 200, c: float = 50.0):
    """Online POMCP on the same small POMDP."""
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
        state["belief"] = update_belief(pomdp, state["belief"], cell, obs)
        # Rebuild agent so POMCP uses the freshly-updated particle belief.
        _rebuild()

    def reset():
        state["belief"] = pomdp.initial_prior.copy()
        state["agent"] = None
        state["planner"] = None

    return choose, observe, reset


def run_benchmark() -> dict[str, dict[str, float]]:
    pomdp = make_pomdp()
    rng_solve = np.random.default_rng(20260612)

    print("Solving SARSOP...")
    t0 = time.perf_counter()
    alpha_policy = solve_sarsop(
        pomdp,
        pomdpsol_path=POMDPSOL,
        discount=DISCOUNT,
        timeout_sec=30,
        precision=0.5,
    )
    sarsop_solve_sec = time.perf_counter() - t0
    print(f"  {len(alpha_policy.alphas)} alpha vectors "
          f"({sarsop_solve_sec:.1f}s offline solve)")

    rng = np.random.default_rng(20260613)
    true_h_per_episode = rng.choice(
        len(pomdp.states), size=N_EPISODES,
        p=pomdp.initial_prior,
    )

    policies = {
        "random": lambda: policy_random(pomdp, np.random.default_rng(42)),
        "greedy_MAP": lambda: policy_greedy_map_cell(pomdp, np.random.default_rng(43)),
        "pomcp": lambda: policy_pomcp(pomdp, np.random.default_rng(44), n_sims=200),
        "sarsop": lambda: policy_sarsop(pomdp, alpha_policy, np.random.default_rng(45)),
    }

    results: dict[str, dict[str, float]] = {}
    for name, factory in policies.items():
        rewards = []
        found_flags = []
        per_step_ms = []
        for ep_idx in range(N_EPISODES):
            choose, observe, reset = factory()
            reset()
            ep_rng = np.random.default_rng(1000 + ep_idx)
            r, found, choose_ms = run_episode(
                pomdp, choose, observe,
                true_h_idx=int(true_h_per_episode[ep_idx]),
                rng=ep_rng,
            )
            rewards.append(r)
            found_flags.append(found)
            per_step_ms.append(choose_ms)
        mean_r = float(np.mean(rewards))
        sem_r = float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
        discovery = float(np.mean(found_flags))
        decision_ms = float(np.mean(per_step_ms))
        results[name] = dict(
            mean_reward=mean_r,
            sem_reward=sem_r,
            discovery_rate=discovery,
            decision_ms=decision_ms,
        )
        print(
            f"  {name:>11s}: discovery={discovery:.2f}  "
            f"reward={mean_r:+.2f} +/- {sem_r:.2f}  "
            f"decision={decision_ms:.2f} ms/step"
        )

    results["_meta"] = {
        "sarsop_solve_sec": sarsop_solve_sec,
        "alpha_count": float(len(alpha_policy.alphas)),
        "n_episodes": float(N_EPISODES),
        "drill_budget": float(DRILL_BUDGET),
        "discount": DISCOUNT,
    }
    return results


def make_chart(results: dict[str, dict[str, float]]) -> None:
    names = ["random", "greedy_MAP", "pomcp", "sarsop"]
    colors = ["#9aa0a6", "#1f77b4", "#ff7f0e", "#2ca02c"]
    discovery = [results[n]["discovery_rate"] for n in names]
    rewards = [results[n]["mean_reward"] for n in names]
    sems = [results[n]["sem_reward"] for n in names]
    decision_ms = [results[n]["decision_ms"] for n in names]
    meta = results.get("_meta", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    bars = ax.bar(names, discovery, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, discovery):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Discovery rate")
    ax.set_title(f"Discovery rate ({N_EPISODES} episodes, "
                 f"budget {DRILL_BUDGET} drills)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(names, rewards, color=colors,
                  edgecolor="black", linewidth=0.6,
                  yerr=sems, capsize=4)
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (1.5 if val >= 0 else -2.5),
                f"{val:+.1f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=10)
    ax.set_ylabel("Mean discounted reward (gamma=0.95)")
    ax.set_title("Cumulative reward "
                 "(alpha=0.05, beta=0.10)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    bars = ax.bar(names, decision_ms, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, decision_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.1 + 0.02,
                (f"{val:.2f} ms" if val < 1 else f"{val:.1f} ms"),
                ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("Decision time (ms/step, log scale)")
    solve_label = f"SARSOP offline solve: {meta.get('sarsop_solve_sec', 0):.1f}s once"
    ax.set_title(f"Online planning latency\n({solve_label})")
    ax.grid(axis="y", alpha=0.3, which="both")

    fig.suptitle(
        "C.2 part 2: SARSOP alpha-vector policy vs POMCP on multi-hypothesis POMDP\n"
        "5x5 grid, 4 corner hypotheses + null; "
        "Bernoulli sensor (alpha, beta) = (0.05, 0.10)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


def main() -> int:
    results = run_benchmark()
    make_chart(results)

    import json
    out_json = OUT_PNG.with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
