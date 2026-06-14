"""bcgt-v2.0 C.2 part 3: falsification demo on real BCGS deposit-type priors.

Mirrors v20_c2_falsification_demo.py but swaps the toy left-half /
bottom-half synthetic hypothesis set for the 4-deposit-type real-
prior set built from the BCGT 500 m feature parquet (porphyry, skarn,
epithermal, VMS) plus null. Runs three ground-truth scenarios and plots
the categorical posterior trajectory under random drilling.

Three scenarios:
    truth = porphyry  expect posterior to concentrate on H_porphyry
    truth = epithermal expect posterior to concentrate on H_epithermal
                       (this is the harder case because H_epithermal
                       and H_vms are highly correlated, r=0.7)
    truth = null      expect the null-argmax falsification trigger to fire

Output: data/derived/bcgt/fig_v20_c2_real_falsification.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_minerals.decision.v20.hypotheses import (
    make_bcgt_deposit_type_hypothesis_set,
)
from ai_minerals.decision.v20.policies import MultiHypothesisFalsificationPolicy
from ai_minerals.decision.v20.pomdp import MultiHypothesisDrillingProblem

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_c2_real_falsification.png"

N_GRID = 30
N_DRILLS = 25
SENSOR_NOISE_SIGMA = 0.05


def _sample_ground_truth(scenario, hs, rng):
    if scenario == "porphyry":
        return hs.hypotheses[0].sample_realization(rng, n_samples=1)[0]
    if scenario == "epithermal":
        return hs.hypotheses[2].sample_realization(rng, n_samples=1)[0]
    if scenario == "null":
        return hs.null.sample_realization(
            rng, n_cells=N_GRID * N_GRID, n_samples=1,
        )[0]
    raise ValueError(f"Unknown scenario: {scenario}")


def _scenario_truth_idx(scenario):
    return {"porphyry": 0, "epithermal": 2, "null": -1}[scenario]


def _run_scenario(scenario, hs, coords, rng):
    true_grade = _sample_ground_truth(scenario, hs, rng)
    problem = MultiHypothesisDrillingProblem(
        hypotheses=hs,
        true_hypothesis_idx=_scenario_truth_idx(scenario),
        x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=true_grade,
        sensor_noise_sigma=SENSOR_NOISE_SIGMA,
    )
    policy = MultiHypothesisFalsificationPolicy(
        problem=problem, hypothesis_set=hs,
        sensor_noise_sigma=SENSOR_NOISE_SIGMA,
        falsification_threshold_likelihood=0.5,
    )
    policy.reset(rng)

    posteriors = [policy._hypothesis_posterior.copy()]
    falsifications = [policy.falsification_fired()]
    drilled_cells = []
    drilled: frozenset[int] = frozenset()
    for _ in range(N_DRILLS):
        unvisited = [c for c in range(coords.shape[0]) if c not in drilled]
        cell = int(rng.choice(unvisited))
        obs, _, drilled = problem.step(cell, drilled, rng)
        policy.step_posterior(cell_idx=cell, observation=float(obs))
        posteriors.append(policy._hypothesis_posterior.copy())
        falsifications.append(policy.falsification_fired())
        drilled_cells.append(cell)
    return np.array(posteriors), falsifications, np.array(drilled_cells)


def main() -> int:
    hs, coords = make_bcgt_deposit_type_hypothesis_set(n_side=N_GRID)

    scenarios = ["porphyry", "epithermal", "null"]
    titles = {
        "porphyry": "Ground truth: porphyry hypothesis",
        "epithermal": "Ground truth: epithermal hypothesis",
        "null": "Ground truth: null (no spatial structure)",
    }
    series_labels = {
        0: "porphyry", 1: "skarn", 2: "epithermal", 3: "vms", 4: "null",
    }
    series_colors = {
        0: "#1f77b4", 1: "#d97b00", 2: "#2ca02c",
        3: "#9467bd", 4: "#888888",
    }

    results = {}
    for sc in scenarios:
        rng = np.random.default_rng(20260613 + abs(hash(sc)) % (2**31))
        post_traj, fals, drilled = _run_scenario(sc, hs, coords, rng)
        results[sc] = (post_traj, fals, drilled)
        print(f"\n[{sc}] final posterior: " + "  ".join([
            f"{series_labels[i]}={post_traj[-1, i]:.3f}"
            for i in range(post_traj.shape[1])
        ]))
        fal_step = next((i for i, f in enumerate(fals) if f), None)
        print(f"      falsification: "
              f"{'step ' + str(fal_step) if fal_step is not None else 'never'}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    steps = np.arange(N_DRILLS + 1)
    for ax, sc in zip(axes, scenarios):
        post_traj, fals, _ = results[sc]
        for i, label in series_labels.items():
            ax.plot(steps, post_traj[:, i], marker="o", markersize=3,
                    label=label, color=series_colors[i], linewidth=1.6)
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.6,
                   label="falsification threshold")
        fal_step = next((i for i, f in enumerate(fals) if f), None)
        if fal_step is not None:
            ax.axvline(fal_step, color="red", linestyle="--", alpha=0.5)
            ax.text(fal_step + 0.4, 0.92, f"falsified\n@ drill {fal_step}",
                    color="red", fontsize=9)
        ax.set_title(titles[sc])
        ax.set_xlabel("Drill step")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Posterior probability")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle(
        "BCGT C.2 falsification on real BCGS deposit-type priors. "
        "Categorical posterior over {porphyry, skarn, epithermal, vms, null} "
        f"across {N_DRILLS} random drills on a 30x30 grid "
        f"(sensor sigma={SENSOR_NOISE_SIGMA}; 0.5 likelihood + null-argmax trigger).",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PNG} ({OUT_PNG.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
