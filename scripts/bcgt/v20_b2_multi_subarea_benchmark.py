"""B.2 retrospective benchmark across all 7 BCGT mining districts.

Drives the four policies (Random / Static prior / BayesianGreedy /
POMCP) through a RetrospectiveBCGSValidator for each district in
``src.ai_minerals.regions.bcgt.BCGT_B2_CLUSTERS``, scoring both
capture@k% (legacy metric, useful for KSM continuity) and capture@N
(new metric, more interpretable for small actual-budget regimes).

Methodology:

- One run per (district, prior, policy) at the maximum drill budget
  (625 cells = top 25% of the 2500-cell working area). The validator
  records the policy's full ranked recommendation list as a trajectory.
- capture@N evaluates over the first N cells of that trajectory at each
  N in {10, 25, 50, 100, 250, 625}; capture@k% evaluates at
  k in {1, 5, 10, 25}. Both pulled from the same single run via
  ``RetrospectiveBCGSValidator.run_policy_full``.
- Three prior variants tested per district: informative (smoothed
  MINFILE), pre-2010 leak-free, and uniform.
- Single master seed per (district, prior) for consistency; capture rate
  has wide variance for districts with only 2-5 positives but the
  multi-budget curves make the variance visible.

Outputs:

- ``data/derived/bcgt/v20_b2_multi_subarea_results.json``
- ``data/derived/bcgt/fig_v20_b2_multi_subarea.png``
- Per-district CSV summaries (optional, controlled by ``--per-district-csv``)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ai_minerals.decision.v20.policies import (
    BayesianGreedyPolicy,
    CorrelatedPriorPOMCPPolicy,
    GreedyMeanPolicy,
    RandomPolicy,
)
from ai_minerals.decision.v20.simulator import RetrospectiveBCGSValidator
from ai_minerals.regions.bcgt import BCGT_B2_CLUSTERS

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data/derived/bcgt"
OUT_JSON = OUT_DIR / "v20_b2_multi_subarea_results.json"

PRIOR_VARIANTS = ("informative", "pre2010_only", "uniform")
N_DRILLS_VALUES = (10, 25, 50, 100, 250, 625)
K_PERCENT_VALUES = (1, 5, 10, 25)
DRILL_BUDGET = 625
SENSOR_NOISE_SIGMA = 0.05
POMCP_N_PARTICLES = 400
POMCP_N_ROLLOUTS = 60
BAYES_N_PARTICLES = 400
POLICY_ORDER = ("random", "static_prior", "bayes_greedy", "pomcp")


def npz_path_for(district: str, prior_variant: str) -> Path:
    """Return the NPZ path for (district, prior_variant)."""
    stem_by_variant = {
        "informative": "b2_inputs",
        "pre2010_only": "b2_inputs_pre2010_only",
        "uniform": "b2_inputs_uniform",
    }
    base = stem_by_variant[prior_variant]
    # KSM has the legacy un-suffixed paths from prepare_b2_inputs.py
    # backward-compat behavior; everything else carries the district suffix.
    if district == "KSM":
        path_legacy = OUT_DIR / f"{base}.npz"
        path_suffixed = OUT_DIR / f"{base}_{district}.npz"
        return path_suffixed if path_suffixed.exists() else path_legacy
    return OUT_DIR / f"{base}_{district}.npz"


def make_policies() -> dict[str, object]:
    """Fresh policy instances per (district, prior, seed) run."""
    return {
        "random": RandomPolicy(),
        "static_prior": GreedyMeanPolicy(),
        "bayes_greedy": BayesianGreedyPolicy(n_particles=BAYES_N_PARTICLES),
        "pomcp": CorrelatedPriorPOMCPPolicy(
            n_particles=POMCP_N_PARTICLES, n_rollouts=POMCP_N_ROLLOUTS,
        ),
    }


def run_district_prior(
    district: str, prior_variant: str,
) -> dict | None:
    """Run all four policies for one (district, prior_variant) cell."""
    npz_path = npz_path_for(district, prior_variant)
    if not npz_path.exists():
        print(f"  [skip] missing input: {npz_path}")
        return None
    npz = np.load(npz_path)
    n_positives = int(npz["post_2010_positive"].sum())
    n_pre_drilled = int(npz["pre_2010_drilled"].sum())

    validator = RetrospectiveBCGSValidator(
        pre_2010_prior=npz["prior_mean"],
        post_2010_positives=npz["post_2010_positive"],
        cells_drilled_pre_2010=npz["pre_2010_drilled"],
        cell_coords_m=npz["cell_coords_m"],
        post_2010_grade=npz["post_2010_grade"],
        sensor_noise_sigma=SENSOR_NOISE_SIGMA,
        drill_budget=DRILL_BUDGET,
    )

    master_rng = np.random.default_rng(
        20260613 + (hash(district) % 2**31) + (hash(prior_variant) % 2**31)
    )
    policy_results: dict[str, dict] = {}
    policies = make_policies()
    seeds = master_rng.integers(0, 2**31 - 1, size=len(policies))
    for (policy_name, policy), seed in zip(policies.items(), seeds):
        t0 = time.perf_counter()
        metrics = validator.run_policy_full(
            policy=policy,
            rng=np.random.default_rng(int(seed)),
            n_drills_values=N_DRILLS_VALUES,
            k_percent_values=K_PERCENT_VALUES,
        )
        elapsed = time.perf_counter() - t0
        policy_results[policy_name] = {
            "capture_at_k_pct": {str(k): v for k, v in metrics["capture_at_k_pct"].items()},
            "capture_at_n_drills": {str(n): v for n, v in metrics["capture_at_n_drills"].items()},
            "elapsed_sec": elapsed,
        }
    return {
        "district": district,
        "prior_variant": prior_variant,
        "n_positives": n_positives,
        "n_pre_drilled": n_pre_drilled,
        "n_cells": int(len(npz["prior_mean"])),
        "policy_results": policy_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--districts", type=str, nargs="*", default=None,
        help="Districts to run. Default: all 7 in BCGT_B2_CLUSTERS.",
    )
    parser.add_argument(
        "--priors", type=str, nargs="*", default=list(PRIOR_VARIANTS),
        help=f"Prior variants. Default: {list(PRIOR_VARIANTS)}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    districts = list(args.districts) if args.districts else list(BCGT_B2_CLUSTERS.keys())
    priors = list(args.priors)
    print(f"Running B.2 multi-subarea benchmark:")
    print(f"  {len(districts)} districts: {districts}")
    print(f"  {len(priors)} prior variants: {priors}")
    print(f"  drill budget: {DRILL_BUDGET} cells")
    print(f"  capture@N values: {list(N_DRILLS_VALUES)}")
    print(f"  capture@k% values: {list(K_PERCENT_VALUES)}")

    all_results: list[dict] = []
    t_start = time.perf_counter()
    for district in districts:
        cluster = BCGT_B2_CLUSTERS[district]
        print(f"\n[{district}] {cluster['description']}")
        print(f"  cluster size: {cluster['n_post2010_cuplus_cells']} cells, "
              f"{cluster['n_post2010_cuplus_holes']} holes")
        for prior_variant in priors:
            print(f"  prior={prior_variant}")
            record = run_district_prior(district, prior_variant)
            if record is None:
                continue
            all_results.append(record)
            print(f"    n_positives_in_subarea={record['n_positives']}, "
                  f"n_pre_drilled={record['n_pre_drilled']}")
            for policy_name in POLICY_ORDER:
                cap = record["policy_results"][policy_name]["capture_at_n_drills"]
                print(f"      {policy_name:>13s}  "
                      "capture@N: " + ", ".join([
                          f"N={n}:{cap[str(n)]*100:5.1f}%" for n in N_DRILLS_VALUES
                      ]))

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {elapsed_total:.1f} s ({elapsed_total/60:.1f} min)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "districts": districts,
            "prior_variants": priors,
            "n_drills_values": list(N_DRILLS_VALUES),
            "k_percent_values": list(K_PERCENT_VALUES),
            "drill_budget": DRILL_BUDGET,
            "sensor_noise_sigma": SENSOR_NOISE_SIGMA,
            "pomcp_n_particles": POMCP_N_PARTICLES,
            "pomcp_n_rollouts": POMCP_N_ROLLOUTS,
            "elapsed_sec_total": elapsed_total,
            "results": all_results,
        }, f, indent=2)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
