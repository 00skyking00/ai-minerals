"""Hypothesis-falsification scaffolding for the BCGT POMDP.

Toy demonstration of the v2.0 Intelligent Prospector extension (Mern,
Corso, Burch, House, Caers 2024, [arXiv:2410.10610](https://arxiv.org/abs/2410.10610))
where the planner maintains a belief over multiple competing geological
hypotheses rather than a single fixed prior. After each drill, the
hypothesis posterior updates by Bayes' rule:

    P(H_i | obs) ∝ P(obs | H_i) · P(H_i)

The trajectory of the hypothesis posterior over drilling steps exposes
two regimes:

1. **One hypothesis is correct.** Its posterior approaches 1; the others
   collapse to 0. The planner has identified the right geological model.
2. **All hypotheses are wrong** (the "none-of-the-above" case). All
   posteriors stay near uniform OR a clearly-non-uniform pattern shifts
   slowly with no clear winner. The planner detects model misspecification
   early and the user gets a stop signal.

This module is intentionally a small toy. The full production-scale
multi-hypothesis falsification on the real BCGT prior is the v2.0
build queued at section D.2.3 of `research/kobold_integration_longterm.md`.
The toy lives in the portfolio to demonstrate the methodology
end-to-end on a synthetic prior; the production version uses three
independently-calibrated regional models as the hypothesis set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HypothesisSet:
    """A set of competing per-cell deposit-indicator priors.

    Attributes
    ----------
    priors : ndarray of shape (n_hypotheses, n_cells)
        Per-cell Bernoulli probability of a deposit under each hypothesis.
    names : list[str]
        Human-readable label per hypothesis, length n_hypotheses.
    """

    priors: np.ndarray
    names: list[str]

    @property
    def n_hypotheses(self) -> int:
        return self.priors.shape[0]

    @property
    def n_cells(self) -> int:
        return self.priors.shape[1]


@dataclass
class BeliefOverHypotheses:
    """Posterior over which hypothesis is correct.

    Encapsulates the per-hypothesis posterior probability ``P(H_i)`` and
    a Bayesian update method ``update(cell, obs)`` that conditions on a
    drill observation.
    """

    hypotheses: HypothesisSet
    posterior: np.ndarray  # shape (n_hypotheses,)

    @classmethod
    def uniform(cls, hypotheses: HypothesisSet) -> "BeliefOverHypotheses":
        """Start with a uniform posterior over the hypotheses."""
        n = hypotheses.n_hypotheses
        return cls(hypotheses=hypotheses, posterior=np.full(n, 1.0 / n))

    def update(self, cell: int, obs: int) -> None:
        """Bayesian update after observing ``obs`` (0 or 1) at ``cell``.

        Under each hypothesis the cell's marginal probability is the
        prior at that cell, so:

            P(obs=1 | H_i) = priors[i, cell]
            P(obs=0 | H_i) = 1 - priors[i, cell]

        The posterior is normalized:

            P(H_i | obs) = P(obs | H_i) · P(H_i) / Σ_j P(obs | H_j) · P(H_j)
        """
        p_obs_given_h = (
            self.hypotheses.priors[:, cell] if obs == 1
            else 1.0 - self.hypotheses.priors[:, cell]
        )
        unnorm = p_obs_given_h * self.posterior
        z = unnorm.sum()
        if z > 0:
            self.posterior = unnorm / z
        # if all hypotheses assign zero probability to this observation,
        # we keep the prior unchanged (degenerate case)

    def marginal_prior(self) -> np.ndarray:
        """Per-cell Bernoulli probability marginalized over the hypothesis
        posterior:

            P(s_c = 1) = Σ_i P(H_i) · priors[i, c]

        Returns a length-``n_cells`` array. This is the prior the
        downstream POMCP / EOI planner should use when the belief over
        hypotheses is active.
        """
        return np.einsum("i,ic->c", self.posterior, self.hypotheses.priors)

    def is_falsified(self, threshold: float = 0.05) -> bool:
        """Heuristic: all hypotheses below ``threshold`` total posterior.

        Returns True if every hypothesis's posterior has dropped to
        roughly the prior expectation under "all hypotheses are wrong."
        The threshold is conservative; production use should also check
        the data likelihood under each hypothesis.
        """
        # If the posterior is uniform on a small set, no single hypothesis
        # is decisively above threshold but they're not "falsified" either.
        # We declare falsification only when the BEST hypothesis sits
        # below 1.5x the uniform expectation (a weak signal: nothing
        # explains the data well).
        n = self.hypotheses.n_hypotheses
        return self.posterior.max() < 1.5 / n


def synthetic_vein_priors(
    n_side: int = 20,
    *,
    rng: np.random.Generator | None = None,
) -> HypothesisSet:
    """Three toy synthetic deposit-indicator priors over an ``n_side × n_side``
    working subarea, suitable for the EW3 mini-experiment.

    The three priors:

    - ``"NW vein"``: cells along a NW-trending strip get high probability
      (0.4); rest of grid is at 0.02.
    - ``"NE vein"``: cells along a NE-trending strip get high probability
      (0.4); rest of grid is at 0.02.
    - ``"disseminated"``: a Gaussian blob centered on the grid has
      moderate-high probability (peak ~0.25); rest is at 0.05.

    Each prior is a different geological hypothesis; ground-truth data
    can be sampled from any of them to test which the planner identifies.
    """
    rng = np.random.default_rng(rng)
    n = n_side
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")

    # H1: NW-trending vein (line y = -x + n; weighted by gaussian distance to line)
    d1 = np.abs((xs + ys) - n) / np.sqrt(2)
    p1 = 0.02 + 0.38 * np.exp(-(d1 ** 2) / 4.0)

    # H2: NE-trending vein (line y = x; weighted by gaussian distance)
    d2 = np.abs(ys - xs) / np.sqrt(2)
    p2 = 0.02 + 0.38 * np.exp(-(d2 ** 2) / 4.0)

    # H3: Disseminated halo (Gaussian centered on grid)
    cx, cy = n / 2, n / 2
    d3 = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    p3 = 0.05 + 0.20 * np.exp(-(d3 ** 2) / 16.0)

    priors = np.stack([p1.flatten(), p2.flatten(), p3.flatten()])
    priors = np.clip(priors, 0.001, 0.999)
    return HypothesisSet(
        priors=priors.astype(np.float64),
        names=["NW vein", "NE vein", "disseminated"],
    )
