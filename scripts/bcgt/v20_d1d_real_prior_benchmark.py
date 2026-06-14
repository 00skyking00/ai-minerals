"""bcgt-v2.0 D.1.D: real BCGS deposit-type prior benchmark.

Mirrors v20_d1_benchmark.py but replaces the synthetic NW/SE-blob
prior with the 4-deposit-type real-prior set built from the BCGT 500 m
feature parquet (porphyry, skarn, epithermal, VMS) plus the null. Uses
the porphyry-Cu economics regime: TRUTH_CUTOFF=0.15, WRONG_COMMITMENT_
PENALTY=30.

The PF variant is omitted to keep wall-clock under run_capped.sh's
cap; the canonical SARSOP is enough to demonstrate the ordering
question on the real-prior hypothesis set.

Outputs:
    data/derived/bcgt/fig_v20_d1d_benchmark.png
    data/derived/bcgt/fig_v20_d1d_benchmark.json
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
    realize_deposit_sets,
)
from ai_minerals.decision.v20.hypotheses import (
    make_bcgt_deposit_type_hypothesis_set,
)
from ai_minerals.decision.v20.sarsop_policy import MultiHypothesisSmallGridPOMDP

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_d1d_benchmark.png"
OUT_JSON = REPO / "data/derived/bcgt/fig_v20_d1d_benchmark.json"
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"

N_EPISODES = 30
DRILL_BUDGET = 9
DISCOUNT = 0.95
ALPHA_FP = 0.10
BETA_FN = 0.10
DRILL_COST = 1.0
DISCOVERY_VALUE = 50.0
WRONG_COMMITMENT_PENALTY = 30.0
TRUTH_CUTOFF = 0.15
POMCP_N_SIMS = 1000

TRUTH_LABELS = ["H_porphyry", "H_skarn", "H_epithermal", "H_vms", "H_null"]


def sample_truth_deposit_set(hset, true_idx, rng, cutoff=TRUTH_CUTOFF):
    n_paper = len(hset.hypotheses)
    if true_idx >= n_paper:
        return set()
    draw = hset.hypotheses[true_idx].sample_realization(rng, n_samples=1)[0]
    return set(int(c) for c in np.where(draw > cutoff)[0])


def sample_observation(deposit_set, cell, rng):
    p_one = (1.0 - BETA_FN) if cell in deposit_set else ALPHA_FP
    return int(rng.random() < p_one)


def realized_reward(deposit_set, cell, drilled, already_found):
    if cell in drilled:
        return -DRILL_COST, False
    if already_found:
        return -DRILL_COST, False
    if cell in deposit_set:
        return -DRILL_COST + DISCOVERY_VALUE, True
    return -DRILL_COST, False


def update_belief(hset, signal_cells, belief, cell, obs):
    n = hset.n_hypotheses
    likelihoods = np.empty(n)
    for i in range(len(hset.hypotheses)):
        in_signal = cell in signal_cells.get(i, set())
        p_one = (1.0 - BETA_FN) if in_signal else ALPHA_FP
        likelihoods[i] = p_one if obs == 1 else (1.0 - p_one)
    if hset.include_null and hset.null is not None:
        likelihoods[-1] = ALPHA_FP if obs == 1 else (1.0 - ALPHA_FP)
    unnorm = belief * likelihoods
    total = unnorm.sum()
    return unnorm / total if total > 0 else belief.copy()


def build_subproblem(hset, signal_cells, belief, candidates):
    names = [h.name for h in hset.hypotheses]
    if hset.include_null and hset.null is not None:
        names = names + ["null"]
    deposit_cell_by_h = {}
    signal_by_h = {}
    for h_idx, sig in signal_cells.items():
        overlap = sig.intersection(candidates)
        if not overlap:
            deposit_cell_by_h[h_idx] = None
            signal_by_h[h_idx] = set()
            continue
        best = max(
            overlap,
            key=lambda c: (
                hset.hypotheses[h_idx].prior_mean_field[c]
                if h_idx < len(hset.hypotheses) else 0.0
            ),
        )
        deposit_cell_by_h[h_idx] = candidates.index(best)
        signal_by_h[h_idx] = {candidates.index(c) for c in overlap}
    return MultiHypothesisSmallGridPOMDP(
        n_cells=len(candidates),
        hypothesis_names=names,
        deposit_cell_by_hypothesis=deposit_cell_by_h,
        signal_cells_by_hypothesis=signal_by_h,
        initial_prior=belief.copy(),
        alpha_fp=ALPHA_FP, beta_fn=BETA_FN,
        drill_cost=DRILL_COST, discovery_value=DISCOVERY_VALUE,
        wrong_commitment_penalty=WRONG_COMMITMENT_PENALTY,
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


def make_random_policy(hset, signal_cells, rng):
    state = {"belief": hset.initial_prior(), "drilled": set()}

    def choose():
        n = hset.hypotheses[0].n_cells
        avail = [c for c in range(n) if c not in state["drilled"]]
        return int(rng.choice(avail))

    def observe(cell, obs):
        state["belief"] = update_belief(hset, signal_cells, state["belief"], cell, obs)
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_greedy_policy(hset, signal_cells):
    state = {"belief": hset.initial_prior(), "drilled": set()}

    def choose():
        ep = expected_deposit_per_cell(hset, state["belief"])
        for c in np.argsort(-ep):
            if int(c) not in state["drilled"]:
                return int(c)
        raise RuntimeError("no cells left")

    def observe(cell, obs):
        state["belief"] = update_belief(hset, signal_cells, state["belief"], cell, obs)
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_pomcp_policy(hset, signal_cells, n_sims=POMCP_N_SIMS):
    state = {"belief": hset.initial_prior(), "drilled": set(),
             "agent": None, "planner": None, "candidates": None}

    def _rebuild():
        cands = top_k_candidates(hset, state["belief"], state["drilled"])
        sub = build_subproblem(hset, signal_cells, state["belief"], cands)
        agent = sub.build_agent(belief=state["belief"])
        planner = pomdp_py.POUCT(
            max_depth=DRILL_BUDGET, discount_factor=DISCOUNT,
            num_sims=n_sims, exploration_const=50.0,
            rollout_policy=agent.policy_model,
        )
        state["agent"] = agent
        state["planner"] = planner
        state["candidates"] = cands

    def choose():
        _rebuild()
        action = state["planner"].plan(state["agent"])
        return state["candidates"][action.cell_idx]

    def observe(cell, obs):
        state["belief"] = update_belief(hset, signal_cells, state["belief"], cell, obs)
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()
        state["agent"] = None
        state["planner"] = None

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_sarsop_policy(hset, signal_cells):
    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset, deposit_sets=signal_cells,
        pomdpsol_path=POMDPSOL, top_k=DEFAULT_TOP_K,
        cutoff=TRUTH_CUTOFF,
        wrong_commitment_penalty=WRONG_COMMITMENT_PENALTY,
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


def run_episode(truth_set, funcs, episode_rng):
    choose, observe, reset, get_belief = funcs
    reset()
    drilled = set()
    cumulative = 0.0
    discovery_step = -1
    found = False
    decision_ms = []
    for t in range(DRILL_BUDGET):
        t0 = time.perf_counter()
        cell = choose()
        decision_ms.append((time.perf_counter() - t0) * 1000.0)
        obs = sample_observation(truth_set, cell, episode_rng)
        reward, found_now = realized_reward(truth_set, cell, drilled, found)
        cumulative += (DISCOUNT ** t) * reward
        if found_now:
            discovery_step = t
            found = True
        drilled.add(cell)
        observe(cell, obs)
    return dict(
        cumulative_reward=cumulative,
        discovery_step=discovery_step,
        found=found,
        final_belief=get_belief().tolist(),
        mean_decision_ms=float(np.mean(decision_ms)),
    )


def run_benchmark():
    hset, _ = make_bcgt_deposit_type_hypothesis_set(n_side=30)

    canonical_rng = np.random.default_rng(20260613)
    signal_cells = realize_deposit_sets(hset, canonical_rng, cutoff=TRUTH_CUTOFF)
    print("canonical signal-cell counts per hypothesis: "
          f"{[len(signal_cells[i]) for i in signal_cells]}")

    truth_rng = np.random.default_rng(7777)
    truth_per_episode = truth_rng.choice(
        hset.n_hypotheses, size=N_EPISODES, p=hset.initial_prior(),
    )

    factories = {
        "random": lambda: make_random_policy(
            hset, signal_cells, np.random.default_rng(42),
        ),
        "greedy_MAP": lambda: make_greedy_policy(hset, signal_cells),
        "pomcp_topK": lambda: make_pomcp_policy(hset, signal_cells),
        "sarsop_topK": lambda: make_sarsop_policy(hset, signal_cells),
    }

    results = {n: {"rewards": [], "discovery_step": [], "found": [],
                   "final_belief_truth": [], "decision_ms": []}
               for n in factories}
    per_episode_truth = []

    for ep_idx in range(N_EPISODES):
        truth_idx = int(truth_per_episode[ep_idx])
        per_episode_truth.append(truth_idx)
        real_rng = np.random.default_rng(80000 + ep_idx)
        truth_set = sample_truth_deposit_set(hset, truth_idx, real_rng)

        for name, factory in factories.items():
            sensor_rng = np.random.default_rng(90000 + ep_idx * 17)
            m = run_episode(truth_set, factory(), sensor_rng)
            results[name]["rewards"].append(m["cumulative_reward"])
            results[name]["discovery_step"].append(m["discovery_step"])
            results[name]["found"].append(m["found"])
            results[name]["final_belief_truth"].append(
                float(m["final_belief"][truth_idx])
            )
            results[name]["decision_ms"].append(m["mean_decision_ms"])

        truth_name = TRUTH_LABELS[truth_idx]
        print(f"  episode {ep_idx + 1:2d}/{N_EPISODES} "
              f"(truth={truth_name}, |deposit|={len(truth_set)}): "
              + ", ".join([
                  f"{n}={results[n]['rewards'][-1]:+5.1f}"
                  for n in factories
              ]))

    summary = {}
    for name in factories:
        rewards = np.array(results[name]["rewards"])
        found = np.array(results[name]["found"])
        ds = np.array(results[name]["discovery_step"])
        summary[name] = {
            "mean_reward": float(rewards.mean()),
            "sem_reward": float(rewards.std(ddof=1) / np.sqrt(len(rewards))),
            "discovery_rate": float(found.mean()),
            "mean_discovery_step_when_found": (
                float(ds[found].mean()) if found.any() else -1.0
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
        "wrong_commitment_penalty": WRONG_COMMITMENT_PENALTY,
        "truth_cutoff": TRUTH_CUTOFF,
        "top_k": DEFAULT_TOP_K,
        "pomcp_n_sims": POMCP_N_SIMS,
        "truth_distribution": [
            int(np.sum(np.array(per_episode_truth) == i))
            for i in range(hset.n_hypotheses)
        ],
        "reward_semantic": "one_shot_discovery",
        "prior_source": "BCGS_MINFILE_deposit_type_per_cell_labels",
    }
    print("\nsummary:")
    for name, s in summary.items():
        if name == "_meta":
            continue
        print(f"  {name:>12s}: reward={s['mean_reward']:+5.2f}+/-{s['sem_reward']:.2f}  "
              f"disc-rate={s['discovery_rate']:.2f}  "
              f"avg-disc-step={s['mean_discovery_step_when_found']:.1f}  "
              f"truth-belief={s['mean_final_belief_truth']:.2f}  "
              f"decision={s['mean_decision_ms']:.1f}ms")
    return summary, results, per_episode_truth


def make_chart(summary, results, per_episode_truth):
    names = ["random", "greedy_MAP", "pomcp_topK", "sarsop_topK"]
    colors = ["#9aa0a6", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    rewards = [summary[n]["mean_reward"] for n in names]
    sems = [summary[n]["sem_reward"] for n in names]
    bars = ax.bar(names, rewards, color=colors, edgecolor="black",
                  linewidth=0.6, yerr=sems, capsize=4)
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (1.5 if val >= 0 else -2),
                f"{val:+.1f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=10)
    ax.set_ylabel("Mean discounted reward (gamma=0.95)")
    ax.set_title(f"Overall cumulative reward "
                 f"({N_EPISODES} episodes, {DRILL_BUDGET} drills)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    truths = np.array(per_episode_truth)
    n_truths = len(TRUTH_LABELS)
    xs = np.arange(n_truths)
    width = 0.2
    for i, name in enumerate(names):
        ra = np.array(results[name]["rewards"])
        per_truth_means = []
        per_truth_sems = []
        for t in range(n_truths):
            mask = (truths == t)
            if mask.sum() == 0:
                per_truth_means.append(0.0)
                per_truth_sems.append(0.0)
                continue
            vals = ra[mask]
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
        f"{TRUTH_LABELS[t]}\n(n={int((truths == t).sum())})"
        for t in range(n_truths)
    ], fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean discounted reward")
    ax.set_title("Reward broken down by ground truth")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    discovery_rates = [summary[n]["discovery_rate"] for n in names]
    bars = ax.bar(names, discovery_rates, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, discovery_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Discovery rate")
    ax.set_title("Found the deposit within budget\n"
                 "(non-null episodes have a deposit to find)")
    ax.axhline(
        float(np.sum(np.array(per_episode_truth) != (len(TRUTH_LABELS) - 1)))
        / N_EPISODES,
        color="gray", linewidth=0.7, linestyle="--",
        label="ceiling (non-null episode fraction)",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "D.1.D: BCGT real-deposit-type benchmark. "
        "30x30 grid, 4 BCGS deposit-type hypotheses (porphyry, skarn, "
        "epithermal, VMS) + null, per-episode truth realization, "
        f"cutoff={TRUTH_CUTOFF}, wrong-commit penalty={WRONG_COMMITMENT_PENALTY}, "
        f"top-K={DEFAULT_TOP_K}, POMCP={POMCP_N_SIMS} sims/step.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

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
