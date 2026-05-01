"""Decision-theoretic drill planning on the BCGT prior.

A v1.0-style POMDP demo (per Mern et al. 2023, *Intelligent Prospector*),
adapted onto the BCGT 500m feature frame. The prior over deposit
indicator per cell is the RF posterior; drilling is modeled as a
noiseless point measurement; rewards are −cost per drill plus a
discovery reward per hit.

Public API:
    load_subarea_prior          → restrict the BCGT feature frame to a
                                   working subarea, build the RF prior,
                                   and return per-cell `(x, y, p_prior,
                                   true_label)` for the subarea
    DrillingProblem             → problem container (cells, prior, true
                                   labels, costs)
    simulate_policy             → run a policy on a single ground-truth
                                   realization; returns the trace + reward
    RandomPolicy / GreedyPolicy → baseline policies
    pomcp_plan                  → POMCP wrapper from pomdp_py
"""

from .pomdp import (
    DrillingProblem,
    load_subarea_prior,
    sample_ground_truth,
    simulate_policy,
)
from .policies import (
    GreedyPolicy,
    POMCPPolicy,
    RandomPolicy,
    pomcp_plan,
)

__all__ = [
    "DrillingProblem",
    "GreedyPolicy",
    "POMCPPolicy",
    "RandomPolicy",
    "load_subarea_prior",
    "pomcp_plan",
    "sample_ground_truth",
    "simulate_policy",
]
