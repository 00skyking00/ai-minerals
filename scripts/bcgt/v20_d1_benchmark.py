"""bcgt-v2.0 D.1.C: synthetic Monte Carlo benchmark at BCGT scale.

Runs four policies on the 30x30 multi-hypothesis BCGT problem from
D.1.A. The two planners (POMCP and SARSOP) operate on the same top-K
candidate cells per drill step, computed from the policy's current
categorical belief, so we are comparing decision quality rather than
candidate-pool composition.

Policies:
  - random:       drill a uniformly-random un-drilled cell
  - greedy:       drill the cell with highest expected deposit prob
                  under the current belief (Bayesian update on
                  observations; argmax over all 900 cells)
  - pomcp_topK:   per-step POMCP on the K=20 belief-conditioned
                  candidate subproblem (POUCT with 200 sims/step)
  - sarsop_topK:  per-step SARSOP on the K=20 subproblem
                  (alpha-vector policy)

Episode setup:
  - N_EPISODES=30, truth uniformly drawn from {H_NW, H_SE, H_null}
  - 9-drill budget, gamma=0.95
  - sensor (alpha, beta) = (0.10, 0.10) Bernoulli
  - drill_cost=1.0, discovery_value=50.0 (no wrong-commitment penalty
    at this scale; multi-cell deposit sets soften the Tiger-style
    penalty story)

Output: data/derived/bcgt/fig_v20_d1_benchmark.png
        data/derived/bcgt/fig_v20_d1_benchmark.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pomdp_py

from ai_minerals.decision.v20.bcgt_scale import (
    BcgtScaleSARSOPPolicy,
    DEFAULT_TOP_K,
    expected_deposit_per_cell,
    make_bcgt_synthetic_hypothesis_set,
    realize_deposit_sets,
)
from ai_minerals.decision.v20.sarsop_policy import MultiHypothesisSmallGridPOMDP

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_d1_benchmark.png"
OUT_JSON = REPO / "data/derived/bcgt/fig_v20_d1_benchmark.json"
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"

N_EPISODES = 30
DRILL_BUDGET = 9
DISCOUNT = 0.95
ALPHA_FP = 0.10
BETA_FN = 0.10
DRILL_COST = 1.0
DISCOVERY_VALUE = 50.0
POMCP_N_SIMS = 200


def sample_observation(deposit_set: set[int], cell_idx: int,
                       rng: np.random.Generator) -> int:
    p1 = (1.0 - BETA_FN) if cell_idx in deposit_set else ALPHA_FP
    return int(rng.random() < p1)


def realized_reward(deposit_set: set[int], cell_idx: int,
                    drilled: set[int]) -> float:
    if cell_idx in drilled:
        return -DRILL_COST
    if cell_idx in deposit_set:
        return -DRILL_COST + DISCOVERY_VALUE
    return -DRILL_COST


def update_belief_at_scale(hset, deposit_sets, belief, cell, obs):
    n = hset.n_hypotheses
    lik = np.empty(n)
    for i in range(len(hset.hypotheses)):
        in_signal = cell in deposit_sets.get(i, set())
        p1 = (1.0 - BETA_FN) if in_signal else ALPHA_FP
        lik[i] = p1 if obs == 1 else (1.0 - p1)
    if hset.include_null and hset.null is not None:
        lik[-1] = ALPHA_FP if obs == 1 else (1.0 - ALPHA_FP)
    unnorm = belief * lik
    s = unnorm.sum()
    return unnorm / s if s > 0 else belief.copy()


def build_subproblem(hset, deposit_sets, belief, candidates):
    names = [h.name for h in hset.hypotheses]
    if hset.include_null and hset.null is not None:
        names = names + ["null"]
    deposit_cell_by_hypothesis: dict[int, int | None] = {}
    signal_cells_by_hypothesis: dict[int, set[int]] = {}
    for h_idx, deposit_set in deposit_sets.items():
        overlap = deposit_set.intersection(candidates)
        if not overlap:
            deposit_cell_by_hypothesis[h_idx] = None
            signal_cells_by_hypothesis[h_idx] = set()
            continue
        best_global = max(
            overlap,
            key=lambda c: (
                hset.hypotheses[h_idx].prior_mean_field[c]
                if h_idx < len(hset.hypotheses) else 0.0
            ),
        )
        deposit_cell_by_hypothesis[h_idx] = candidates.index(best_global)
        signal_cells_by_hypothesis[h_idx] = {
            candidates.index(c) for c in overlap
        }
    return MultiHypothesisSmallGridPOMDP(
        n_cells=len(candidates),
        hypothesis_names=names,
        deposit_cell_by_hypothesis=deposit_cell_by_hypothesis,
        signal_cells_by_hypothesis=signal_cells_by_hypothesis,
        initial_prior=belief.copy(),
        alpha_fp=ALPHA_FP, beta_fn=BETA_FN,
        drill_cost=DRILL_COST, discovery_value=DISCOVERY_VALUE,
        wrong_commitment_penalty=0.0,
    )


def top_k_candidates(hset, belief, drilled, k=DEFAULT_TOP_K):
    ep = expected_deposit_per_cell(hset, belief)
    order = np.argsort(-ep)
    out = []
    for c in order:
        c = int(c)
        if c in drilled:
            continue
        out.append(c)
        if len(out) >= k:
            break
    return out


def make_random(hset, deposit_sets, rng):
    state = {"belief": hset.initial_prior(), "drilled": set()}

    def choose():
        n_cells = hset.hypotheses[0].n_cells
        available = [c for c in range(n_cells) if c not in state["drilled"]]
        return int(rng.choice(available))

    def observe(cell, obs):
        state["belief"] = update_belief_at_scale(
            hset, deposit_sets, state["belief"], cell, obs,
        )
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_greedy(hset, deposit_sets):
    state = {"belief": hset.initial_prior(), "drilled": set()}

    def choose():
        ep = expected_deposit_per_cell(hset, state["belief"])
        order = np.argsort(-ep)
        for c in order:
            if int(c) not in state["drilled"]:
                return int(c)
        raise RuntimeError("no cells left")

    def observe(cell, obs):
        state["belief"] = update_belief_at_scale(
            hset, deposit_sets, state["belief"], cell, obs,
        )
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_pomcp_topk(hset, deposit_sets):
    state = {"belief": hset.initial_prior(), "drilled": set()}

    def choose():
        candidates = top_k_candidates(
            hset, state["belief"], state["drilled"], k=DEFAULT_TOP_K,
        )
        sub = build_subproblem(hset, deposit_sets, state["belief"], candidates)
        agent = sub.build_agent(belief=state["belief"])
        planner = pomdp_py.POUCT(
            max_depth=DRILL_BUDGET,
            discount_factor=DISCOUNT,
            num_sims=POMCP_N_SIMS,
            exploration_const=50.0,
            rollout_policy=agent.policy_model,
        )
        sub_action = planner.plan(agent)
        return candidates[sub_action.cell_idx]

    def observe(cell, obs):
        state["belief"] = update_belief_at_scale(
            hset, deposit_sets, state["belief"], cell, obs,
        )
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_sarsop_topk(hset, deposit_sets):
    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset,
        deposit_sets=deposit_sets,
        pomdpsol_path=POMDPSOL,
        top_k=DEFAULT_TOP_K,
    )

    def choose():
        return policy.choose_action()

    def observe(cell, obs):
        policy.observe(cell, obs)

    def reset():
        policy.reset()

    def get_belief():
        return policy.belief

    return choose, observe, reset, get_belief


def run_episode(true_h_idx, deposit_sets_for_truth, policy_funcs, ep_rng):
    choose, observe, reset, get_belief = policy_funcs
    reset()
    drilled: set[int] = set()
    total = 0.0
    discovery_steps = []
    decision_ms = []
    for t in range(DRILL_BUDGET):
        t0 = time.perf_counter()
        cell = choose()
        decision_ms.append((time.perf_counter() - t0) * 1000.0)
        obs = sample_observation(deposit_sets_for_truth, cell, ep_rng)
        r = realized_reward(deposit_sets_for_truth, cell, drilled)
        total += (DISCOUNT ** t) * r
        if cell not in drilled and cell in deposit_sets_for_truth:
            discovery_steps.append(t)
        drilled.add(cell)
        observe(cell, obs)
    final_belief = get_belief()
    final_belief_truth = float(final_belief[true_h_idx])
    return dict(
        cumulative_reward=total,
        first_discovery_step=(
            int(discovery_steps[0]) if discovery_steps else -1
        ),
        n_discoveries=len(discovery_steps),
        final_belief_truth=final_belief_truth,
        mean_decision_ms=float(np.mean(decision_ms)),
    )


def run_benchmark():
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=30)
    rng_canon = np.random.default_rng(20260613)
    canonical_deposit_sets = realize_deposit_sets(hset, rng_canon)
    print(f"canonical deposit cell counts per hypothesis: "
          f"{[len(canonical_deposit_sets[i]) for i in canonical_deposit_sets]}")

    rng_truth = np.random.default_rng(7777)
    true_h_per_episode = rng_truth.choice(
        hset.n_hypotheses, size=N_EPISODES, p=hset.initial_prior(),
    )

    policy_factories = {
        "random": lambda: make_random(hset, canonical_deposit_sets,
                                      np.random.default_rng(42)),
        "greedy_MAP": lambda: make_greedy(hset, canonical_deposit_sets),
        "pomcp_topK": lambda: make_pomcp_topk(hset, canonical_deposit_sets),
        "sarsop_topK": lambda: make_sarsop_topk(hset, canonical_deposit_sets),
    }

    results = {name: {
        "rewards": [], "first_discovery": [], "n_discoveries": [],
        "final_belief_truth": [], "decision_ms": [],
    } for name in policy_factories}
    per_episode_truth = []

    for ep_idx in range(N_EPISODES):
        true_h = int(true_h_per_episode[ep_idx])
        per_episode_truth.append(true_h)
        for name, factory in policy_factories.items():
            ep_rng = np.random.default_rng(9000 + ep_idx * 7)
            metrics = run_episode(
                true_h_idx=true_h,
                deposit_sets_for_truth=canonical_deposit_sets[true_h],
                policy_funcs=factory(),
                ep_rng=ep_rng,
            )
            results[name]["rewards"].append(metrics["cumulative_reward"])
            results[name]["first_discovery"].append(metrics["first_discovery_step"])
            results[name]["n_discoveries"].append(metrics["n_discoveries"])
            results[name]["final_belief_truth"].append(metrics["final_belief_truth"])
            results[name]["decision_ms"].append(metrics["mean_decision_ms"])
        print(f"  episode {ep_idx + 1:2d}/{N_EPISODES} "
              f"(truth=h{true_h}): "
              + ", ".join([
                  f"{n}={results[n]['rewards'][-1]:+5.1f}"
                  for n in policy_factories
              ]))

    summary = {}
    for name in policy_factories:
        rewards = np.array(results[name]["rewards"])
        first_disc = np.array(results[name]["first_discovery"])
        found = (first_disc >= 0)
        summary[name] = {
            "mean_reward": float(rewards.mean()),
            "sem_reward": float(rewards.std(ddof=1) / np.sqrt(len(rewards))),
            "discovery_rate": float(found.mean()),
            "mean_first_discovery_step": (
                float(first_disc[found].mean()) if found.any() else -1.0
            ),
            "mean_decision_ms": float(np.mean(results[name]["decision_ms"])),
            "mean_final_belief_truth": float(
                np.mean(results[name]["final_belief_truth"])
            ),
        }
    summary["_meta"] = {
        "n_episodes": N_EPISODES,
        "drill_budget": DRILL_BUDGET,
        "discount": DISCOUNT,
        "alpha_fp": ALPHA_FP, "beta_fn": BETA_FN,
        "drill_cost": DRILL_COST, "discovery_value": DISCOVERY_VALUE,
        "top_k": DEFAULT_TOP_K,
        "pomcp_n_sims": POMCP_N_SIMS,
        "truth_distribution": [
            int(np.sum(np.array(per_episode_truth) == i))
            for i in range(hset.n_hypotheses)
        ],
    }
    print("\nsummary:")
    for name, s in summary.items():
        if name == "_meta":
            continue
        print(f"  {name:>12s}: reward={s['mean_reward']:+5.2f}+/-{s['sem_reward']:.2f}  "
              f"disc-rate={s['discovery_rate']:.2f}  "
              f"truth-belief={s['mean_final_belief_truth']:.2f}  "
              f"decision={s['mean_decision_ms']:.1f}ms")
    return summary, results, per_episode_truth


def make_chart(summary, results, per_episode_truth):
    names = ["random", "greedy_MAP", "pomcp_topK", "sarsop_topK"]
    colors = ["#9aa0a6", "#1f77b4", "#ff7f0e", "#2ca02c"]
    truth_labels = {0: "H_NW", 1: "H_SE", 2: "H_null"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    ax = axes[0]
    rewards = [summary[n]["mean_reward"] for n in names]
    sems = [summary[n]["sem_reward"] for n in names]
    bars = ax.bar(names, rewards, color=colors,
                  edgecolor="black", linewidth=0.6,
                  yerr=sems, capsize=4)
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (3 if val >= 0 else -5),
                f"{val:+.0f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=10)
    ax.set_ylabel("Mean discounted reward (gamma=0.95)")
    ax.set_title(f"Overall cumulative reward "
                 f"({N_EPISODES} episodes, {DRILL_BUDGET} drills)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    truths = np.array(per_episode_truth)
    xs = np.arange(3)
    width = 0.2
    for i, name in enumerate(names):
        rewards_arr = np.array(results[name]["rewards"])
        per_truth_means = []
        per_truth_sems = []
        for t in range(3):
            mask = (truths == t)
            if mask.sum() == 0:
                per_truth_means.append(0.0)
                per_truth_sems.append(0.0)
                continue
            vals = rewards_arr[mask]
            per_truth_means.append(float(vals.mean()))
            per_truth_sems.append(
                float(vals.std(ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1 else 0.0
            )
        ax.bar(xs + (i - 1.5) * width, per_truth_means, width,
               color=colors[i], edgecolor="black", linewidth=0.5,
               yerr=per_truth_sems, capsize=2, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([
        f"{truth_labels[t]}\n(n={int((truths == t).sum())})" for t in range(3)
    ])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean discounted reward")
    ax.set_title("Reward broken down by ground truth")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    truth_belief = [summary[n]["mean_final_belief_truth"] for n in names]
    bars = ax.bar(names, truth_belief, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, truth_belief):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(1.0 / 3.0, color="gray", linewidth=0.7, linestyle="--",
               label="uniform prior (1/3)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Final P(true hypothesis)")
    ax.set_title("Posterior weight on the true hypothesis\nat end of episode")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "D.1.C: BCGT-scale benchmark. "
        "30x30 grid, 3 hypotheses (NW deposit, SE deposit, null), "
        f"top-K={DEFAULT_TOP_K} candidate cells per step. "
        "Bernoulli sensor (alpha, beta) = (0.10, 0.10).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


def main() -> int:
    summary, results, truths = run_benchmark()
    make_chart(summary, results, truths)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
