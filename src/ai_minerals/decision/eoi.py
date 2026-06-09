"""Efficacy of Information (EOI) per Caers, Scheidt, Yin, Wang, Mukerji,
House (2022), DOI 10.1007/s11053-022-10030-1.

EOI quantifies how much a candidate measurement is expected to reduce
uncertainty on a discrete property of interest. For a candidate drill at
cell ``c``:

    EOI(c) = H[ S_c | belief ] − E_o[ H[ S_c | belief, observation o ] ]

where ``S_c`` is the binary deposit indicator at cell ``c`` and ``H[·]``
is Shannon entropy in bits.

In the Mern v1.0-faithful setup this BCGT POMDP uses (noiseless sensor,
independent-Bernoulli prior), EOI collapses to a clean closed form:

    EOI(c) = h2(p_c)

the binary entropy of the prior probability at ``c``, since the
observation ``o ∈ {0, 1}`` is delivered noiselessly and the posterior
entropy is 0 regardless of the realization. The argmax-EOI policy then
becomes **uncertainty sampling**: drill the cell whose prior is closest
to 0.5.

This is structurally different from the greedy policy (drill argmax p),
and exposes the methodological point clearly: maximizing per-drill hits
(greedy) and maximizing per-drill information (EOI) are different
objectives, and disagree on which cell to drill next.

The contrast is the v1.0 → v2.0 gap. With noisy sensors and correlated
priors, EOI no longer collapses to a local entropy and the EOI argmax
diverges from both greedy and uncertainty sampling. The function below
is built so that a v2.0 extension just provides a richer
``posterior_after_observation`` callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .pomdp import DrillingProblem


# ---------------------------------------------------------------------------
# Binary entropy
# ---------------------------------------------------------------------------


def binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
    """Shannon binary entropy h2(p) in bits.

    h2(p) = -p log2(p) - (1-p) log2(1-p), with 0 log 0 ≡ 0.

    Maximized at p=0.5 (h2(0.5) = 1 bit); zero at p ∈ {0, 1}.
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    pm = p[mask]
    out[mask] = -(pm * np.log2(pm) + (1.0 - pm) * np.log2(1.0 - pm))
    return out if out.ndim else float(out)


# ---------------------------------------------------------------------------
# EOI under the v1.0 setup (noiseless sensor, independent prior)
# ---------------------------------------------------------------------------


def efficacy_of_information(
    posterior: np.ndarray,
    drilled: frozenset[int],
) -> np.ndarray:
    """Per-cell EOI of drilling that cell next.

    In the noiseless-independent v1.0 setup:

        EOI(c) = h2(posterior[c])     for c not in drilled
        EOI(c) = 0                    for c in drilled

    Returns a length-N array of EOI values in bits.
    """
    eoi = binary_entropy(posterior)
    if drilled:
        eoi = eoi.copy()
        idx = np.fromiter(drilled, dtype=int)
        eoi[idx] = 0.0
    return eoi


# ---------------------------------------------------------------------------
# Policy: argmax-EOI (uncertainty sampling under v1.0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EOIPolicy:
    """Drill the un-drilled cell with highest expected information yield.

    Under the v1.0 noiseless-independent setup this is equivalent to
    uncertainty sampling: drill the cell whose prior is closest to 0.5.
    Under a v2.0 setup with noisy sensors and correlated priors, EOI is
    a strict superset of uncertainty sampling and the argmax shifts
    accordingly.

    The policy is intentionally a thin wrapper over
    ``efficacy_of_information`` so the same scoring function feeds both
    the policy and the per-step EOI annotation on the simulator output.
    """

    name: str = "eoi"

    def choose(
        self,
        problem: "DrillingProblem",
        posterior: np.ndarray,
        drilled: frozenset[int],
        *,
        rng: np.random.Generator | None = None,
    ) -> int | None:
        eoi = efficacy_of_information(posterior, drilled)
        if drilled:
            mask = np.ones_like(eoi, dtype=bool)
            idx = np.fromiter(drilled, dtype=int)
            mask[idx] = False
            if not mask.any():
                return None
            # Restrict to un-drilled cells for the argmax
            available = np.where(mask)[0]
            return int(available[np.argmax(eoi[available])])
        return int(np.argmax(eoi))


# ---------------------------------------------------------------------------
# Optional v2.0 extension point
# ---------------------------------------------------------------------------


def efficacy_of_information_general(
    posterior: np.ndarray,
    drilled: frozenset[int],
    *,
    observation_likelihoods: Callable[[int, int], np.ndarray] | None = None,
    posterior_after_observation: Callable[[int, int, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """General-form EOI with explicit observation likelihoods and update.

    Default behavior (both callables None) reduces to
    ``efficacy_of_information`` (the v1.0 closed form).

    Parameters
    ----------
    observation_likelihoods :
        Callable ``(cell, obs) → ndarray[N]`` returning P(observation =
        obs | underlying state at cell) for each cell. Used to model
        noisy sensors. None ⇒ deterministic.
    posterior_after_observation :
        Callable ``(cell, obs, posterior) → ndarray[N]`` returning the
        posterior after observing ``obs`` at ``cell``. Used to model
        correlated priors where one observation updates beliefs about
        other cells. None ⇒ collapse only the observed cell.

    Returns
    -------
    eoi : ndarray[N]
        Per-cell EOI in bits.

    Notes
    -----
    This is the entry point for the long-term "closing the POMDP" work
    (research/kobold_integration_longterm.md section D.2). The function
    is stubbed at v1.0 fidelity; v2.0 plumbing requires only injection of
    the two callables.
    """
    if observation_likelihoods is None and posterior_after_observation is None:
        return efficacy_of_information(posterior, drilled)
    raise NotImplementedError(
        "General-form EOI with noisy sensors and correlated priors is the "
        "v2.0 extension; the v1.0 closed form is the current portfolio scope. "
        "See research/kobold_integration_longterm.md section D.2.1 + D.2.2."
    )
