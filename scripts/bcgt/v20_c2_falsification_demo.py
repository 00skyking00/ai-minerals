"""bcgt-v2.0 C.2 part 2: falsification-check demo across 3 ground-truth scenarios.

Sets up a small 3-hypothesis test (two paper hypotheses + null) and simulates
sequential drilling under three ground-truth choices:

    Scenario A     ground truth drawn from hypothesis A
    Scenario B     ground truth drawn from hypothesis B
    Scenario null  ground truth drawn from the null (no spatial structure)

For each, runs 25 sequential drills, updates the categorical posterior over
hypothesis indices, and records whether the falsification flag fires. Plots
the per-scenario posterior trajectories side-by-side with the falsification
trigger annotated.

Output: data/derived/bcgt/fig_v20_c2_falsification_demo.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_minerals.decision.v20.hypotheses import (
    Hypothesis, HypothesisSet, NullHypothesis,
)
from ai_minerals.decision.v20.policies import MultiHypothesisFalsificationPolicy
from ai_minerals.decision.v20.pomdp import MultiHypothesisDrillingProblem

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_c2_falsification_demo.png"

N_GRID = 20            # 20x20 cells = 400 cells
N_DRILLS = 25
SENSOR_NOISE_SIGMA = 0.05


def _build_hypothesis_set() -> tuple[HypothesisSet, np.ndarray]:
    """Two paper hypotheses with distinct spatial means + the null."""
    spacing = 500.0
    x = np.arange(N_GRID) * spacing
    y = np.arange(N_GRID) * spacing
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Hypothesis A: high mean in the left half
    mean_a = np.where(coords[:, 0] < (N_GRID * spacing / 2), 0.35, 0.05)
    # Hypothesis B: high mean in the bottom half
    mean_b = np.where(coords[:, 1] < (N_GRID * spacing / 2), 0.35, 0.05)

    h_a = Hypothesis(
        name="A (left-half anomaly)",
        n_grabens=1, n_domains=1,
        cell_coords_m=coords,
        prior_mean_field=mean_a,
        gp_marginal_std=0.05,
        gp_lengthscale_m=1500.0,
    )
    h_b = Hypothesis(
        name="B (bottom-half anomaly)",
        n_grabens=1, n_domains=1,
        cell_coords_m=coords,
        prior_mean_field=mean_b,
        gp_marginal_std=0.05,
        gp_lengthscale_m=1500.0,
    )
    null = NullHypothesis(marginal_std=0.1)
    return (
        HypothesisSet(hypotheses=(h_a, h_b), null=null, include_null=True),
        coords,
    )


def _sample_ground_truth(
    scenario: str,
    hs: HypothesisSet,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one full grade field given the chosen ground-truth scenario."""
    if scenario == "A":
        return hs.hypotheses[0].sample_realization(rng, n_samples=1)[0]
    if scenario == "B":
        return hs.hypotheses[1].sample_realization(rng, n_samples=1)[0]
    if scenario == "null":
        return hs.null.sample_realization(rng, n_cells=N_GRID * N_GRID, n_samples=1)[0]
    raise ValueError(f"Unknown scenario: {scenario}")


def _run_scenario(
    scenario: str,
    hs: HypothesisSet,
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[bool], np.ndarray]:
    """Simulate N_DRILLS drills under the given ground truth.

    Returns (posterior_trajectory, falsification_flags, drill_cells).
    posterior_trajectory has shape (N_DRILLS + 1, n_hypotheses) including the
    uniform initial step.
    """
    true_grade = _sample_ground_truth(scenario, hs, rng)
    problem = MultiHypothesisDrillingProblem(
        hypotheses=hs,
        true_hypothesis_idx={"A": 0, "B": 1, "null": -1}[scenario],
        x_m=coords[:, 0], y_m=coords[:, 1],
        true_grade=true_grade,
        sensor_noise_sigma=SENSOR_NOISE_SIGMA,
    )
    policy = MultiHypothesisFalsificationPolicy(
        problem=problem,
        hypothesis_set=hs,
        sensor_noise_sigma=SENSOR_NOISE_SIGMA,
        falsification_threshold_likelihood=0.5,
    )
    policy.reset(rng)

    posteriors = [policy._hypothesis_posterior.copy()]
    falsifications = [policy.falsification_fired()]
    drilled_cells = []

    drilled: frozenset[int] = frozenset()
    for _ in range(N_DRILLS):
        # Pick a random unvisited cell uniformly; the falsification demo is
        # action-agnostic so the chart isolates the inference effect.
        unvisited = [c for c in range(coords.shape[0]) if c not in drilled]
        cell = int(rng.choice(unvisited))
        obs, _, drilled = problem.step(cell, drilled, rng)
        policy.step_posterior(cell_idx=cell, observation=float(obs))
        posteriors.append(policy._hypothesis_posterior.copy())
        falsifications.append(policy.falsification_fired())
        drilled_cells.append(cell)

    return np.array(posteriors), falsifications, np.array(drilled_cells)


def main() -> int:
    hs, coords = _build_hypothesis_set()

    scenarios = ["A", "B", "null"]
    titles = {
        "A": "Ground truth: hypothesis A",
        "B": "Ground truth: hypothesis B",
        "null": "Ground truth: null (no spatial structure)",
    }
    series_labels = {0: "h_A", 1: "h_B", 2: "h_0 (null)"}
    series_colors = {0: "#1f77b4", 1: "#d97b00", 2: "#888888"}

    results: dict[str, tuple] = {}
    for sc in scenarios:
        rng = np.random.default_rng(20260612 + hash(sc) % (2**31))
        post_traj, falsifications, drilled = _run_scenario(sc, hs, coords, rng)
        results[sc] = (post_traj, falsifications, drilled)

        print(f"\n[{sc}] final posterior: "
              f"h_A={post_traj[-1, 0]:.3f}  "
              f"h_B={post_traj[-1, 1]:.3f}  "
              f"h_0={post_traj[-1, 2]:.3f}")
        fal_step = next(
            (i for i, f in enumerate(falsifications) if f), None,
        )
        print(f"      falsification fired: "
              f"{'step ' + str(fal_step) if fal_step is not None else 'never'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    steps = np.arange(N_DRILLS + 1)
    for ax, sc in zip(axes, scenarios):
        post_traj, falsifications, _ = results[sc]
        for i, label in series_labels.items():
            ax.plot(
                steps, post_traj[:, i], marker="o", markersize=3,
                label=label, color=series_colors[i], linewidth=1.8,
            )
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.6,
                   label="falsification threshold")
        fal_step = next(
            (i for i, f in enumerate(falsifications) if f), None,
        )
        if fal_step is not None:
            ax.axvline(fal_step, color="red", linestyle="--", alpha=0.5)
            ax.text(
                fal_step + 0.4, 0.92, f"falsified\n@ drill {fal_step}",
                color="red", fontsize=9,
            )
        ax.set_title(titles[sc])
        ax.set_xlabel("Drill step")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Posterior probability")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle(
        "BCGT C.2 falsification demo: categorical posterior over "
        "{h_A, h_B, h_0} across 25 random drills\n"
        f"(20x20 synthetic grid; sensor sigma={SENSOR_NOISE_SIGMA}; "
        "0.5 likelihood + null-argmax trigger)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
