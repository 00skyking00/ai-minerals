"""Render the B.2 multi-subarea benchmark results.

Reads ``data/derived/bcgt/v20_b2_multi_subarea_results.json`` produced
by ``v20_b2_multi_subarea_benchmark.py`` and produces multi-panel
charts visualizing capture@N curves by district.

Two output charts:

- ``fig_v20_b2_multi_subarea.png``
    Headline chart. Single row of 7 panels, one per BCGT mining
    district, informative prior only. X-axis: drill budget N. Y-axis:
    capture rate (% of post-2010 Cu+ cells captured). One line per
    policy (random, static_prior, bayes_greedy, pomcp).

- ``fig_v20_b2_multi_subarea_leakfree.png``
    Same layout for the pre-2010 leak-free prior. The leak-free panel
    typically has flat priors for most districts (no pre-2010 Cu+
    cells nearby), so this chart is mostly about KSM + Red_Chris.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
IN_JSON = REPO / "data/derived/bcgt/v20_b2_multi_subarea_results.json"
OUT_INFORMATIVE = REPO / "data/derived/bcgt/fig_v20_b2_multi_subarea.png"
OUT_LEAKFREE = REPO / "data/derived/bcgt/fig_v20_b2_multi_subarea_leakfree.png"

POLICY_ORDER = ("random", "static_prior", "bayes_greedy", "pomcp")
POLICY_LABELS = {
    "random": "Random",
    "static_prior": "Static prior",
    "bayes_greedy": "BayesianGreedy",
    "pomcp": "POMCP",
}
POLICY_COLORS = {
    "random": "#888888",
    "static_prior": "#d97b00",
    "bayes_greedy": "#1f77b4",
    "pomcp": "#2c7c2c",
}
POLICY_MARKERS = {
    "random": "o",
    "static_prior": "s",
    "bayes_greedy": "^",
    "pomcp": "D",
}


def render_one_prior(
    results: list[dict],
    prior_variant: str,
    out_path: Path,
    n_drills_values: list[int],
) -> None:
    """Render the per-district capture@N curves for one prior variant."""
    by_district = {
        r["district"]: r for r in results if r["prior_variant"] == prior_variant
    }
    if not by_district:
        print(f"  no results for prior_variant={prior_variant}; skipping {out_path}")
        return
    districts = list(by_district.keys())
    n_panels = len(districts)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(2.6 * n_panels, 3.4),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, district in zip(axes, districts):
        record = by_district[district]
        n_pos = record["n_positives"]
        for policy_name in POLICY_ORDER:
            cap_dict = record["policy_results"][policy_name]["capture_at_n_drills"]
            y = [cap_dict[str(n)] * 100 for n in n_drills_values]
            ax.plot(
                n_drills_values, y,
                marker=POLICY_MARKERS[policy_name],
                color=POLICY_COLORS[policy_name],
                label=POLICY_LABELS[policy_name],
                linewidth=1.4, markersize=5,
            )
        ax.set_xscale("log")
        ax.set_xticks(n_drills_values)
        ax.set_xticklabels(
            [str(n) for n in n_drills_values], rotation=45, fontsize=8,
        )
        ax.set_xlabel("Drill budget N (log)", fontsize=8)
        ax.set_title(
            f"{district}\n({n_pos} post-2010 Cu+ cells)",
            fontsize=9,
        )
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Capture rate (% of Cu+ cells found)", fontsize=9)
    axes[-1].legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.4, 0))

    prior_label_map = {
        "informative": "informative MINFILE-derived prior",
        "pre2010_only": "pre-2010 leak-free prior",
        "uniform": "uniform 0.1 prior",
    }
    prior_label = prior_label_map.get(prior_variant, prior_variant)

    fig.suptitle(
        f"BCGT B.2 retrospective expansion: capture@N across 7 mining "
        f"districts ({prior_label})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    with open(IN_JSON) as f:
        data = json.load(f)
    n_drills_values = list(data["n_drills_values"])
    results = data["results"]

    render_one_prior(results, "informative", OUT_INFORMATIVE, n_drills_values)
    render_one_prior(results, "pre2010_only", OUT_LEAKFREE, n_drills_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
