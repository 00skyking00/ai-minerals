"""bcgt-v2.0 D.1.C: synthetic Monte Carlo benchmark at BCGT scale.

Runs four policies on the 30x30 multi-hypothesis BCGT problem from
D.1.A. Both planners (POMCP and SARSOP) operate on the same top-K
candidate cells per drill step, computed from the policy's current
categorical belief, so we are comparing decision quality rather than
candidate-pool composition.

Policies:
    - random          drill a uniformly-random un-drilled cell.
    - greedy_MAP      drill the cell with highest expected deposit
                      probability under the current belief. Updates
                      its categorical posterior after every drill,
                      then picks argmax across all 900 cells.
    - pomcp_topK      per-step POMCP (POUCT) on the K = 20
                      belief-conditioned candidate subproblem.
                      `POMCP_N_SIMS` Monte Carlo simulations per step.
    - sarsop_topK     per-step SARSOP on the K = 20 subproblem.
                      Alpha-vector policy.

Methodology choices that matter:

    Per-episode truth realization
        Each episode samples a fresh Gaussian-process draw under the
        truth hypothesis and thresholds it for deposit cells. This
        replaces the earlier "one canonical realization replayed
        across episodes" shortcut. Episode-to-episode variance now
        reflects both sensor noise AND GP-prior variability, which
        is what a Monte Carlo over the prior is supposed to measure.

    Stable policy model
        The policy's likelihood model uses a separate "canonical"
        realization for its signal-cell representation. Real
        exploration never has access to the actual truth realization;
        the policy operates on its prior model and any observations
        it gathers along the way. The canonical signal-cell set is
        a reasonable stand-in for the policy's marginal expectations
        under each hypothesis.

    One-shot discovery reward
        Each episode pays the +discovery_value bonus ONCE, on the
        first deposit cell drilled. Subsequent drills (in or out of
        the deposit region) only pay -drill_cost. This matches the
        Tiger experiment's semantic ("did you find the deposit") and
        the real-exploration intuition that finding the deposit is
        the milestone, not the count of deposit-region cells you
        drilled. Replaces the earlier "each new deposit cell pays
        +50" aggregate reward.

Episode setup:
    N_EPISODES           60
    truth distribution   uniform over {H_NW, H_SE, H_null}
    drill budget         9 cells per episode
    discount factor      0.95
    sensor (alpha, beta) (0.10, 0.10) Bernoulli
    drill cost           1.0
    discovery value      50.0 (paid once per episode)

Output:
    data/derived/bcgt/fig_v20_d1_benchmark.png
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
    DEFAULT_CUTOFF,
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

N_EPISODES = 60
DRILL_BUDGET = 9
DISCOUNT = 0.95
ALPHA_FP = 0.10
BETA_FN = 0.10
DRILL_COST = 1.0
DISCOVERY_VALUE = 50.0
POMCP_N_SIMS = 10000


def sample_truth_deposit_set(
    hset, true_hypothesis_index: int, rng: np.random.Generator,
    cutoff: float = DEFAULT_CUTOFF,
) -> set[int]:
    """Sample one fresh GP realization for the truth hypothesis and
    threshold it for deposit cells.

    For the null hypothesis the deposit set is empty (a null world
    has no deposit anywhere).
    """
    n_paper_hypotheses = len(hset.hypotheses)
    if true_hypothesis_index >= n_paper_hypotheses:
        return set()
    truth_hypothesis = hset.hypotheses[true_hypothesis_index]
    draw = truth_hypothesis.sample_realization(rng, n_samples=1)[0]
    return set(int(c) for c in np.where(draw > cutoff)[0])


def sample_observation(
    deposit_set: set[int], cell_index: int,
    rng: np.random.Generator,
) -> int:
    """One Bernoulli sensor reading at the chosen cell, against the
    episode's truth deposit set."""
    p_one = (1.0 - BETA_FN) if cell_index in deposit_set else ALPHA_FP
    return int(rng.random() < p_one)


def realized_reward(
    deposit_set: set[int], cell_index: int,
    drilled: set[int], already_found: bool,
) -> tuple[float, bool]:
    """One-shot discovery reward. Returns (reward, found_this_step).

    Subsequent drills after the first discovery only pay -drill_cost.
    Re-drilling an already-drilled cell only pays -drill_cost.
    """
    if cell_index in drilled:
        return -DRILL_COST, False
    if already_found:
        return -DRILL_COST, False
    if cell_index in deposit_set:
        return -DRILL_COST + DISCOVERY_VALUE, True
    return -DRILL_COST, False


def update_belief(
    hset, policy_signal_cells, belief: np.ndarray,
    cell_index: int, observation: int,
) -> np.ndarray:
    """Bayesian update of the categorical posterior given one binary
    observation. Uses the policy's (canonical) signal-cell map, not
    the episode's truth realization.
    """
    n = hset.n_hypotheses
    likelihoods = np.empty(n)
    for i in range(len(hset.hypotheses)):
        in_signal = cell_index in policy_signal_cells.get(i, set())
        p_one = (1.0 - BETA_FN) if in_signal else ALPHA_FP
        likelihoods[i] = p_one if observation == 1 else (1.0 - p_one)
    if hset.include_null and hset.null is not None:
        likelihoods[-1] = ALPHA_FP if observation == 1 else (1.0 - ALPHA_FP)
    unnormalized = belief * likelihoods
    total = unnormalized.sum()
    return unnormalized / total if total > 0 else belief.copy()


def build_subproblem(
    hset, policy_signal_cells, belief: np.ndarray,
    candidate_cells: list[int],
) -> MultiHypothesisSmallGridPOMDP:
    """Compress the 30x30 problem onto the top-K candidate cells.

    The subproblem inherits the policy's canonical signal-cell model
    (NOT the truth realization). Each hypothesis's signal-cell set
    in the small POMDP is the intersection of the candidate set and
    the policy's signal cells for that hypothesis.
    """
    names = [h.name for h in hset.hypotheses]
    if hset.include_null and hset.null is not None:
        names = names + ["null"]
    deposit_cell_by_hypothesis: dict[int, int | None] = {}
    signal_cells_by_hypothesis: dict[int, set[int]] = {}
    for hypothesis_index, signal_cells in policy_signal_cells.items():
        overlap = signal_cells.intersection(candidate_cells)
        if not overlap:
            deposit_cell_by_hypothesis[hypothesis_index] = None
            signal_cells_by_hypothesis[hypothesis_index] = set()
            continue
        best_global = max(
            overlap,
            key=lambda cell: (
                hset.hypotheses[hypothesis_index].prior_mean_field[cell]
                if hypothesis_index < len(hset.hypotheses) else 0.0
            ),
        )
        deposit_cell_by_hypothesis[hypothesis_index] = (
            candidate_cells.index(best_global)
        )
        signal_cells_by_hypothesis[hypothesis_index] = {
            candidate_cells.index(c) for c in overlap
        }
    return MultiHypothesisSmallGridPOMDP(
        n_cells=len(candidate_cells),
        hypothesis_names=names,
        deposit_cell_by_hypothesis=deposit_cell_by_hypothesis,
        signal_cells_by_hypothesis=signal_cells_by_hypothesis,
        initial_prior=belief.copy(),
        alpha_fp=ALPHA_FP, beta_fn=BETA_FN,
        drill_cost=DRILL_COST, discovery_value=DISCOVERY_VALUE,
        wrong_commitment_penalty=0.0,
    )


def top_k_candidates(
    hset, belief: np.ndarray, drilled: set[int], k: int = DEFAULT_TOP_K,
) -> list[int]:
    """Top-K un-drilled cells ranked by expected deposit probability
    under the current belief."""
    expected_per_cell = expected_deposit_per_cell(hset, belief)
    order = np.argsort(-expected_per_cell)
    out: list[int] = []
    for cell in order:
        cell = int(cell)
        if cell in drilled:
            continue
        out.append(cell)
        if len(out) >= k:
            break
    return out


def make_random_policy(hset, policy_signal_cells, rng):
    state = {"belief": hset.initial_prior(), "drilled": set()}

    def choose():
        n_cells = hset.hypotheses[0].n_cells
        available = [c for c in range(n_cells) if c not in state["drilled"]]
        return int(rng.choice(available))

    def observe(cell, obs):
        state["belief"] = update_belief(
            hset, policy_signal_cells, state["belief"], cell, obs,
        )
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_greedy_policy(hset, policy_signal_cells):
    state = {"belief": hset.initial_prior(), "drilled": set()}

    def choose():
        expected_per_cell = expected_deposit_per_cell(hset, state["belief"])
        order = np.argsort(-expected_per_cell)
        for cell in order:
            if int(cell) not in state["drilled"]:
                return int(cell)
        raise RuntimeError("no cells left")

    def observe(cell, obs):
        state["belief"] = update_belief(
            hset, policy_signal_cells, state["belief"], cell, obs,
        )
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_pomcp_policy(
    hset, policy_signal_cells, n_sims: int = POMCP_N_SIMS,
):
    state = {"belief": hset.initial_prior(), "drilled": set(),
             "agent": None, "planner": None}

    def _rebuild():
        candidates = top_k_candidates(
            hset, state["belief"], state["drilled"], k=DEFAULT_TOP_K,
        )
        sub = build_subproblem(
            hset, policy_signal_cells, state["belief"], candidates,
        )
        agent = sub.build_agent(belief=state["belief"])
        planner = pomdp_py.POUCT(
            max_depth=DRILL_BUDGET,
            discount_factor=DISCOUNT,
            num_sims=n_sims,
            exploration_const=50.0,
            rollout_policy=agent.policy_model,
        )
        state["agent"] = agent
        state["planner"] = planner
        state["candidates"] = candidates

    def choose():
        _rebuild()
        action = state["planner"].plan(state["agent"])
        return state["candidates"][action.cell_idx]

    def observe(cell, obs):
        state["belief"] = update_belief(
            hset, policy_signal_cells, state["belief"], cell, obs,
        )
        state["drilled"].add(cell)

    def reset():
        state["belief"] = hset.initial_prior()
        state["drilled"] = set()
        state["agent"] = None
        state["planner"] = None

    def get_belief():
        return state["belief"].copy()

    return choose, observe, reset, get_belief


def make_sarsop_policy(hset, policy_signal_cells):
    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset,
        deposit_sets=policy_signal_cells,
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


def run_episode(
    truth_deposit_set: set[int], policy_funcs, episode_rng,
) -> dict:
    """One (policy, ground_truth) episode."""
    choose, observe, reset, get_belief = policy_funcs
    reset()
    drilled: set[int] = set()
    cumulative = 0.0
    discovery_step = -1
    already_found = False
    decision_ms = []
    for t in range(DRILL_BUDGET):
        t0 = time.perf_counter()
        cell = choose()
        decision_ms.append((time.perf_counter() - t0) * 1000.0)
        obs = sample_observation(truth_deposit_set, cell, episode_rng)
        reward, found_now = realized_reward(
            truth_deposit_set, cell, drilled, already_found,
        )
        cumulative += (DISCOUNT ** t) * reward
        if found_now:
            discovery_step = t
            already_found = True
        drilled.add(cell)
        observe(cell, obs)
    final_belief = get_belief()
    return dict(
        cumulative_reward=cumulative,
        discovery_step=discovery_step,
        found=already_found,
        final_belief=final_belief.tolist(),
        mean_decision_ms=float(np.mean(decision_ms)),
    )


def run_benchmark():
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=30)

    canonical_rng = np.random.default_rng(20260613)
    policy_signal_cells = realize_deposit_sets(hset, canonical_rng)
    print("policy's canonical signal-cell counts per hypothesis: "
          f"{[len(policy_signal_cells[i]) for i in policy_signal_cells]}")

    truth_assignment_rng = np.random.default_rng(7777)
    truth_per_episode = truth_assignment_rng.choice(
        hset.n_hypotheses, size=N_EPISODES, p=hset.initial_prior(),
    )

    policy_factories = {
        "random": lambda: make_random_policy(
            hset, policy_signal_cells, np.random.default_rng(42),
        ),
        "greedy_MAP": lambda: make_greedy_policy(hset, policy_signal_cells),
        "pomcp_topK": lambda: make_pomcp_policy(
            hset, policy_signal_cells, n_sims=POMCP_N_SIMS,
        ),
        "sarsop_topK": lambda: make_sarsop_policy(hset, policy_signal_cells),
    }

    results = {name: {
        "rewards": [], "discovery_step": [], "found": [],
        "final_belief_truth": [], "decision_ms": [],
    } for name in policy_factories}
    per_episode_truth = []

    for episode_index in range(N_EPISODES):
        truth_idx = int(truth_per_episode[episode_index])
        per_episode_truth.append(truth_idx)

        truth_realization_rng = np.random.default_rng(80000 + episode_index)
        truth_deposit_set = sample_truth_deposit_set(
            hset, truth_idx, truth_realization_rng,
        )

        for name, factory in policy_factories.items():
            sensor_rng = np.random.default_rng(90000 + episode_index * 17)
            metrics = run_episode(
                truth_deposit_set=truth_deposit_set,
                policy_funcs=factory(),
                episode_rng=sensor_rng,
            )
            results[name]["rewards"].append(metrics["cumulative_reward"])
            results[name]["discovery_step"].append(metrics["discovery_step"])
            results[name]["found"].append(metrics["found"])
            results[name]["final_belief_truth"].append(
                float(metrics["final_belief"][truth_idx])
            )
            results[name]["decision_ms"].append(metrics["mean_decision_ms"])

        truth_name = ["H_NW", "H_SE", "H_null"][truth_idx]
        print(f"  episode {episode_index + 1:2d}/{N_EPISODES} "
              f"(truth={truth_name}, |deposit|={len(truth_deposit_set)}): "
              + ", ".join([
                  f"{n}={results[n]['rewards'][-1]:+5.1f}"
                  for n in policy_factories
              ]))

    summary = {}
    for name in policy_factories:
        rewards = np.array(results[name]["rewards"])
        found = np.array(results[name]["found"])
        discovery_step = np.array(results[name]["discovery_step"])
        summary[name] = {
            "mean_reward": float(rewards.mean()),
            "sem_reward": float(rewards.std(ddof=1) / np.sqrt(len(rewards))),
            "discovery_rate": float(found.mean()),
            "mean_discovery_step_when_found": (
                float(discovery_step[found].mean()) if found.any() else -1.0
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
        "reward_semantic": "one_shot_discovery",
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
    truth_labels = {0: "H_NW", 1: "H_SE", 2: "H_null"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    ax = axes[0]
    rewards = [summary[n]["mean_reward"] for n in names]
    sems = [summary[n]["sem_reward"] for n in names]
    bars = ax.bar(
        names, rewards, color=colors, edgecolor="black", linewidth=0.6,
        yerr=sems, capsize=4,
    )
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
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    discovery_rates = [summary[n]["discovery_rate"] for n in names]
    bars = ax.bar(
        names, discovery_rates, color=colors,
        edgecolor="black", linewidth=0.6,
    )
    for bar, val in zip(bars, discovery_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Discovery rate")
    ax.set_title("Found the deposit within budget\n"
                 "(non-null episodes have a deposit to find)")
    ax.axhline(
        float(np.sum(np.array(per_episode_truth) != 2)) / N_EPISODES,
        color="gray", linewidth=0.7, linestyle="--",
        label="ceiling (non-null episode fraction)",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "D.1.C: BCGT-scale benchmark. 30x30 grid, 3 hypotheses "
        "(NW deposit, SE deposit, null), per-episode truth realization, "
        f"top-K={DEFAULT_TOP_K} candidate cells per step, "
        f"POMCP={POMCP_N_SIMS} sims/step.",
        fontsize=10,
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
