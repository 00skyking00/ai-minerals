"""bcgt-v2.0 D.1.C sensitivity sweeps.

Two small experiments to check whether the main-benchmark result
("all four policies tie at BCGT scale under one-shot reward and
per-episode truth realization") is robust to two parameter choices:

  K sweep
      Top-K action-pruning size. Default in the main benchmark is
      K = 20. We re-run at K in {5, 10, 20, 50} to see whether the
      policy ranking depends on the candidate-pool size.

  POMCP num_sims sweep
      POMCP simulation count per step. Default in the main benchmark
      is 10000. We re-run at num_sims in {200, 1000, 5000, 10000}
      to see where POMCP plateaus.

Both sweeps use 30 episodes per condition with per-episode truth
realization and one-shot discovery reward, matching the main
benchmark's methodology.

Outputs:
    data/derived/bcgt/fig_v20_d1_k_sweep.png
    data/derived/bcgt/fig_v20_d1_pomcp_sims_sweep.png
    data/derived/bcgt/fig_v20_d1_sensitivity.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v20_d1_benchmark as bench
from ai_minerals.decision.v20.bcgt_scale import (
    make_bcgt_synthetic_hypothesis_set,
    realize_deposit_sets,
)

REPO = Path(__file__).resolve().parents[2]
OUT_K_PNG = REPO / "data/derived/bcgt/fig_v20_d1_k_sweep.png"
OUT_SIMS_PNG = REPO / "data/derived/bcgt/fig_v20_d1_pomcp_sims_sweep.png"
OUT_JSON = REPO / "data/derived/bcgt/fig_v20_d1_sensitivity.json"

N_EPISODES_PER_CONDITION = 30
DRILL_BUDGET = 9
K_VALUES = [5, 10, 20, 50]
POMCP_SIMS_VALUES = [200, 1000, 5000, 10000]


def setup_problem():
    hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=30)
    policy_signal_cells = realize_deposit_sets(
        hset, np.random.default_rng(20260613),
    )
    truth_per_episode = np.random.default_rng(7777).choice(
        hset.n_hypotheses, size=N_EPISODES_PER_CONDITION,
        p=hset.initial_prior(),
    )
    return hset, policy_signal_cells, truth_per_episode


def run_policies_for_condition(
    hset, policy_signal_cells, truth_per_episode,
    top_k: int, pomcp_n_sims: int,
):
    """Run all four policies on N_EPISODES_PER_CONDITION episodes with
    the given (top_k, pomcp_n_sims) configuration."""
    # Monkey-patch the benchmark module's globals so the make_*
    # factories pick up the new parameters.
    saved_top_k = bench.DEFAULT_TOP_K
    saved_n_sims = bench.POMCP_N_SIMS
    try:
        bench.DEFAULT_TOP_K = top_k
        # Also patch in the bcgt_scale module since SARSOP factory reads it
        import ai_minerals.decision.v20.bcgt_scale as bcgt_scale_mod
        saved_scale_k = bcgt_scale_mod.DEFAULT_TOP_K
        bcgt_scale_mod.DEFAULT_TOP_K = top_k

        policy_factories = {
            "random": lambda: bench.make_random_policy(
                hset, policy_signal_cells, np.random.default_rng(42),
            ),
            "greedy_MAP": lambda: bench.make_greedy_policy(
                hset, policy_signal_cells,
            ),
            "pomcp_topK": lambda: bench.make_pomcp_policy(
                hset, policy_signal_cells, n_sims=pomcp_n_sims,
            ),
            "sarsop_topK": lambda: bench.make_sarsop_policy(
                hset, policy_signal_cells,
            ),
        }

        results = {name: {"rewards": []} for name in policy_factories}

        for episode_index in range(N_EPISODES_PER_CONDITION):
            truth_idx = int(truth_per_episode[episode_index])
            truth_deposit_set = bench.sample_truth_deposit_set(
                hset, truth_idx,
                np.random.default_rng(80000 + episode_index),
            )
            for name, factory in policy_factories.items():
                metrics = bench.run_episode(
                    truth_deposit_set=truth_deposit_set,
                    policy_funcs=factory(),
                    episode_rng=np.random.default_rng(
                        90000 + episode_index * 17,
                    ),
                )
                results[name]["rewards"].append(metrics["cumulative_reward"])

        summary = {}
        for name in policy_factories:
            rewards = np.array(results[name]["rewards"])
            summary[name] = {
                "mean_reward": float(rewards.mean()),
                "sem_reward": float(
                    rewards.std(ddof=1) / np.sqrt(len(rewards))
                ),
            }
        return summary
    finally:
        bench.DEFAULT_TOP_K = saved_top_k
        bench.POMCP_N_SIMS = saved_n_sims
        import ai_minerals.decision.v20.bcgt_scale as bcgt_scale_mod
        bcgt_scale_mod.DEFAULT_TOP_K = saved_scale_k


def run_k_sweep(hset, policy_signal_cells, truth_per_episode):
    """K sweep at fixed POMCP_N_SIMS=1000 (moderate; faster than 10000)."""
    pomcp_sims = 1000
    out = {}
    for k in K_VALUES:
        t0 = time.perf_counter()
        out[k] = run_policies_for_condition(
            hset, policy_signal_cells, truth_per_episode,
            top_k=k, pomcp_n_sims=pomcp_sims,
        )
        elapsed = time.perf_counter() - t0
        print(f"K = {k:3d} done in {elapsed:.1f} s: "
              + ", ".join([
                  f"{n}={out[k][n]['mean_reward']:+5.2f}"
                  for n in out[k]
              ]))
    return out, pomcp_sims


def run_pomcp_sims_sweep(hset, policy_signal_cells, truth_per_episode):
    """POMCP sims sweep at fixed K=20."""
    top_k = 20
    out = {}
    for n_sims in POMCP_SIMS_VALUES:
        t0 = time.perf_counter()
        out[n_sims] = run_policies_for_condition(
            hset, policy_signal_cells, truth_per_episode,
            top_k=top_k, pomcp_n_sims=n_sims,
        )
        elapsed = time.perf_counter() - t0
        print(f"num_sims = {n_sims:5d} done in {elapsed:.1f} s: "
              + ", ".join([
                  f"{n}={out[n_sims][n]['mean_reward']:+5.2f}"
                  for n in out[n_sims]
              ]))
    return out, top_k


def chart_k_sweep(k_results, pomcp_sims):
    names = ["random", "greedy_MAP", "pomcp_topK", "sarsop_topK"]
    colors = ["#9aa0a6", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = K_VALUES
    for name, color in zip(names, colors):
        means = [k_results[k][name]["mean_reward"] for k in xs]
        sems = [k_results[k][name]["sem_reward"] for k in xs]
        ax.errorbar(xs, means, yerr=sems, fmt="-o", color=color,
                    label=name, capsize=3, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in xs])
    ax.set_xlabel("Top-K candidate cells per step (log scale)")
    ax.set_ylabel("Mean discounted reward")
    ax.set_title(
        f"K-sensitivity sweep. {N_EPISODES_PER_CONDITION} episodes per condition. "
        f"POMCP = {pomcp_sims} sims/step. One-shot reward, per-episode truth."
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    OUT_K_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_K_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_K_PNG}")


def chart_pomcp_sims_sweep(sims_results, top_k):
    names = ["random", "greedy_MAP", "pomcp_topK", "sarsop_topK"]
    colors = ["#9aa0a6", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = POMCP_SIMS_VALUES
    for name, color in zip(names, colors):
        means = [sims_results[s][name]["mean_reward"] for s in xs]
        sems = [sims_results[s][name]["sem_reward"] for s in xs]
        ax.errorbar(xs, means, yerr=sems, fmt="-o", color=color,
                    label=name, capsize=3, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in xs])
    ax.set_xlabel("POMCP simulations per step (log scale)")
    ax.set_ylabel("Mean discounted reward")
    ax.set_title(
        f"POMCP num_sims sensitivity. {N_EPISODES_PER_CONDITION} episodes per condition. "
        f"K = {top_k}. random / greedy / SARSOP shown for context "
        "(they are independent of POMCP sims)."
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    OUT_SIMS_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_SIMS_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_SIMS_PNG}")


def main() -> int:
    hset, policy_signal_cells, truth_per_episode = setup_problem()

    print("K sweep (POMCP_N_SIMS=1000):")
    k_results, pomcp_sims_in_k_sweep = run_k_sweep(
        hset, policy_signal_cells, truth_per_episode,
    )
    chart_k_sweep(k_results, pomcp_sims_in_k_sweep)

    print("\nPOMCP num_sims sweep (K=20):")
    sims_results, k_in_sims_sweep = run_pomcp_sims_sweep(
        hset, policy_signal_cells, truth_per_episode,
    )
    chart_pomcp_sims_sweep(sims_results, k_in_sims_sweep)

    output = {
        "k_sweep": {
            "pomcp_n_sims": pomcp_sims_in_k_sweep,
            "n_episodes": N_EPISODES_PER_CONDITION,
            "results": {str(k): k_results[k] for k in K_VALUES},
        },
        "pomcp_sims_sweep": {
            "top_k": k_in_sims_sweep,
            "n_episodes": N_EPISODES_PER_CONDITION,
            "results": {str(s): sims_results[s] for s in POMCP_SIMS_VALUES},
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
