"""BCGT-scale multi-hypothesis POMDP machinery (D.1).

This module scales the C.2 multi-hypothesis SARSOP machinery from the
hand-built 4-cell Tiger demonstration up to the BC Golden Triangle
working subarea: a 30 by 30 grid (15 km by 15 km at 500 m per cell).
The grid size, Gaussian-process (GP) kernel choice, and kernel
parameters match the v2.0 spec
(Matern smoothness :math:`\\nu = 2.5`, marginal standard deviation
:math:`\\sigma = 0.1`, correlation length :math:`\\ell = 1500` m).

The module exposes three pieces a caller will compose:

`make_bcgt_synthetic_hypothesis_set`
    Build a HypothesisSet with two synthetic GP-correlated hypotheses
    (deposit blob in NW and SE quadrants of the grid) plus a null.
    Returns the HypothesisSet and the per-cell coordinates.

`expected_deposit_per_cell`
    Per-cell marginal expected deposit probability under any
    categorical belief over the hypothesis set. The aggregation that
    drives top-K candidate selection.

`BcgtScaleSARSOPPolicy`
    The belief-conditioned top-K SARSOP planner. At each drill step
    it computes the top-K candidate cells under the current belief,
    compresses them into a small POMDP, solves it with SARSOP, and
    returns the recommended cell.

Tractability at BCGT scale relies on the top-K subproblem trick: the
full 900-cell action space is intractable for SARSOP, but the K=20
subproblem rebuilt at every step solves in about 0.02 seconds and
preserves SARSOP's alpha-vector machinery.

Examples
--------
Build the hypothesis set and inspect a per-step solve:

>>> import numpy as np
>>> from pathlib import Path
>>> from ai_minerals.decision.v20.bcgt_scale import (
...     make_bcgt_synthetic_hypothesis_set,
...     realize_deposit_sets,
...     BcgtScaleSARSOPPolicy,
... )
>>> hset, coords = make_bcgt_synthetic_hypothesis_set(n_side=30)
>>> deposit_sets = realize_deposit_sets(hset, np.random.default_rng(0))
>>> policy = BcgtScaleSARSOPPolicy(
...     hypothesis_set=hset,
...     deposit_sets=deposit_sets,
...     pomdpsol_path=Path("vendor/sarsop/pomdpsol"),
... )
>>> first_drill = policy.choose_action()
>>> policy.observe(first_drill, observation=1)

See also
--------
ai_minerals.decision.v20.sarsop_policy
    The underlying small-grid POMDP class and SARSOP wrapper this
    module composes onto.
ai_minerals.decision.v20.hypotheses
    Hypothesis and HypothesisSet primitives, including the
    Gaussian-process priors and the null-hypothesis class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .hypotheses import (
    KERNEL_LENGTHSCALE_M_BCGT,
    KERNEL_MARGINAL_STD,
    Hypothesis,
    HypothesisSet,
    NullHypothesis,
)
from .sarsop_policy import (
    MultiHypothesisSARSOPPolicy,
    MultiHypothesisSmallGridPOMDP,
    solve_sarsop,
)

__all__ = [
    "make_bcgt_synthetic_hypothesis_set",
    "realize_deposit_sets",
    "expected_deposit_per_cell",
    "BcgtScaleSARSOPPolicy",
    "DEFAULT_N_SIDE",
    "DEFAULT_SPACING_M",
    "DEFAULT_ANOMALY_PEAK",
    "DEFAULT_ANOMALY_SPREAD_M",
    "DEFAULT_CUTOFF",
    "DEFAULT_TOP_K",
]

DEFAULT_N_SIDE = 30
DEFAULT_SPACING_M = 500.0
DEFAULT_ANOMALY_PEAK = 0.18
DEFAULT_ANOMALY_SPREAD_M = 3000.0
DEFAULT_CUTOFF = 0.10
DEFAULT_TOP_K = 20


def make_bcgt_synthetic_hypothesis_set(
    n_side: int = DEFAULT_N_SIDE,
    spacing_m: float = DEFAULT_SPACING_M,
    anomaly_peak: float = DEFAULT_ANOMALY_PEAK,
    anomaly_spread_m: float = DEFAULT_ANOMALY_SPREAD_M,
    include_null: bool = True,
) -> tuple[HypothesisSet, np.ndarray]:
    """Build a synthetic multi-hypothesis prior on the BCGT subarea.

    Two GP-correlated hypotheses are constructed by placing a Gaussian
    deposit anomaly at different quadrants of the grid:

      H_NW: anomaly centered at (0.25 * extent, 0.75 * extent)
      H_SE: anomaly centered at (0.75 * extent, 0.25 * extent)

    Plus a NullHypothesis with a flat-zero prior mean (no deposit
    anywhere) and the same GP marginal variance.

    Returns
    -------
    hypothesis_set
        HypothesisSet with the two paper hypotheses and the null.
    coords
        (n_side ** 2, 2) array of cell (x, y) coordinates in meters.
    """
    x = np.arange(n_side) * spacing_m
    y = np.arange(n_side) * spacing_m
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    extent = (n_side - 1) * spacing_m
    nw_center = np.array([0.25 * extent, 0.75 * extent])
    se_center = np.array([0.75 * extent, 0.25 * extent])

    def gaussian_blob(center: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(coords - center, axis=1)
        return anomaly_peak * np.exp(-0.5 * (d / anomaly_spread_m) ** 2)

    h_nw = Hypothesis(
        name="NW_deposit",
        n_grabens=1, n_domains=1,
        cell_coords_m=coords,
        prior_mean_field=gaussian_blob(nw_center),
        gp_marginal_std=KERNEL_MARGINAL_STD,
        gp_lengthscale_m=KERNEL_LENGTHSCALE_M_BCGT,
    )
    h_se = Hypothesis(
        name="SE_deposit",
        n_grabens=1, n_domains=1,
        cell_coords_m=coords,
        prior_mean_field=gaussian_blob(se_center),
        gp_marginal_std=KERNEL_MARGINAL_STD,
        gp_lengthscale_m=KERNEL_LENGTHSCALE_M_BCGT,
    )

    if include_null:
        null = NullHypothesis(marginal_std=KERNEL_MARGINAL_STD)
        return HypothesisSet(
            hypotheses=(h_nw, h_se),
            null=null,
            include_null=True,
        ), coords
    return HypothesisSet(
        hypotheses=(h_nw, h_se),
        include_null=False,
    ), coords


def realize_deposit_sets(
    hypothesis_set: HypothesisSet,
    rng: np.random.Generator,
    cutoff: float = DEFAULT_CUTOFF,
) -> dict[int, set[int]]:
    """Sample one GP realization per hypothesis and return the set of
    cells whose realized grade exceeds the cutoff.

    For the null hypothesis the returned set is empty: a null world has
    no deposit cells. The dict is keyed by hypothesis index in
    `hypothesis_set` ordering (paper hypotheses 0..N-1, null last).
    """
    out: dict[int, set[int]] = {}
    for i, h in enumerate(hypothesis_set.hypotheses):
        draw = h.sample_realization(rng, n_samples=1)[0]
        out[i] = set(int(c) for c in np.where(draw > cutoff)[0])
    if hypothesis_set.include_null and hypothesis_set.null is not None:
        out[len(hypothesis_set.hypotheses)] = set()
    return out


def expected_deposit_per_cell(
    hypothesis_set: HypothesisSet,
    belief: np.ndarray,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray:
    """Per-cell expected deposit probability under a categorical belief.

    For each cell `c`, returns the marginal probability that the
    underlying GP draw exceeds the cutoff, averaged across hypotheses
    according to the current belief:

    .. math::
        P(\\text{deposit at } c)
        = \\sum_h \\text{belief}[h] \\cdot P(\\text{GP draw at } c > \\text{cutoff} \\mid h)

    The per-hypothesis term uses the Gaussian-tail probability at each
    cell. Under each hypothesis the GP is :math:`N(\\mu_h(c), \\sigma_h^2)`
    where :math:`\\mu_h(c)` is the hypothesis's prior mean field and
    :math:`\\sigma_h` is its marginal standard deviation. The null
    hypothesis contributes the cutoff tail of :math:`N(0, \\sigma_0^2)`
    at every cell.

    Parameters
    ----------
    hypothesis_set : HypothesisSet
        Multi-hypothesis prior over the working area.
    belief : np.ndarray, shape (n_hypotheses,)
        Categorical belief over hypotheses. Must sum to 1. Paper
        hypotheses come first in `hypothesis_set` ordering; the null
        is at the final index if present.
    cutoff : float, default DEFAULT_CUTOFF
        Threshold separating "this cell has ore" from "this cell does
        not." A GP draw above the cutoff counts as a deposit cell.

    Returns
    -------
    np.ndarray, shape (n_cells,)
        Per-cell expected deposit probability under the belief.

    Examples
    --------
    >>> import numpy as np
    >>> from ai_minerals.decision.v20.bcgt_scale import (
    ...     make_bcgt_synthetic_hypothesis_set,
    ...     expected_deposit_per_cell,
    ... )
    >>> hset, _ = make_bcgt_synthetic_hypothesis_set(n_side=30)
    >>> belief = hset.initial_prior()  # uniform
    >>> deposit_probabilities = expected_deposit_per_cell(hset, belief)
    >>> deposit_probabilities.shape
    (900,)
    """
    n_cells = hypothesis_set.hypotheses[0].n_cells
    expected = np.zeros(n_cells, dtype=np.float64)
    for hypothesis_index, hypothesis in enumerate(hypothesis_set.hypotheses):
        z_score = (cutoff - hypothesis.prior_mean_field) / hypothesis.gp_marginal_std
        probability_above_cutoff = 1.0 - norm.cdf(z_score)
        expected += belief[hypothesis_index] * probability_above_cutoff
    if hypothesis_set.include_null and hypothesis_set.null is not None:
        z_score_null = cutoff / hypothesis_set.null.marginal_std
        probability_above_cutoff_null = 1.0 - norm.cdf(z_score_null)
        expected += belief[-1] * probability_above_cutoff_null
    return expected


@dataclass
class BcgtScaleSARSOPPolicy:
    """Belief-conditioned top-K SARSOP planner for the BCGT-scale
    multi-hypothesis problem.

    At each drill step:
      1. Compute expected deposit probability per cell under the
         current categorical belief over hypotheses.
      2. Pick the top K cells by that expectation, excluding any
         already-drilled cells.
      3. Build a small MultiHypothesisSmallGridPOMDP with K actions:
         each action stands for "drill the candidate cell at this
         position". The signal-cells set per hypothesis is the
         intersection of the candidate set and the hypothesis's
         realized deposit set; the deposit cell per hypothesis is
         the highest-prior cell in that intersection.
      4. Solve the small POMDP with SARSOP and return the
         recommended candidate cell.

    The wrong-commitment penalty is wired in so SARSOP can prefer
    information-gathering over premature commit even on the larger
    grid.
    """
    hypothesis_set: HypothesisSet
    deposit_sets: dict[int, set[int]]
    pomdpsol_path: str | Path
    top_k: int = DEFAULT_TOP_K
    cutoff: float = DEFAULT_CUTOFF
    alpha_fp: float = 0.10
    beta_fn: float = 0.10
    drill_cost: float = 1.0
    discovery_value: float = 50.0
    wrong_commitment_penalty: float = 30.0
    sarsop_timeout_sec: int = 15
    sarsop_precision: float = 0.5

    _belief: np.ndarray = field(init=False, repr=False)
    _drilled: set[int] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._belief = self.hypothesis_set.initial_prior()
        self._drilled = set()

    @property
    def belief(self) -> np.ndarray:
        return self._belief.copy()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._belief = self.hypothesis_set.initial_prior()
        self._drilled = set()

    def _top_k_candidates(self) -> list[int]:
        """Top-K un-drilled cells by expected deposit probability."""
        ep = expected_deposit_per_cell(
            self.hypothesis_set, self._belief, cutoff=self.cutoff,
        )
        order = np.argsort(-ep)
        candidates: list[int] = []
        for c in order:
            c = int(c)
            if c in self._drilled:
                continue
            candidates.append(c)
            if len(candidates) >= self.top_k:
                break
        return candidates

    def _build_subproblem(
        self, candidates: list[int],
    ) -> MultiHypothesisSmallGridPOMDP:
        """Compress the 30x30 problem onto the top-K candidate cells."""
        names = [h.name for h in self.hypothesis_set.hypotheses]
        if self.hypothesis_set.include_null and self.hypothesis_set.null is not None:
            names = names + ["null"]

        deposit_cell_by_hypothesis: dict[int, int | None] = {}
        signal_cells_by_hypothesis: dict[int, set[int]] = {}
        for h_idx, deposit_set in self.deposit_sets.items():
            overlap = deposit_set.intersection(candidates)
            if not overlap:
                deposit_cell_by_hypothesis[h_idx] = None
                signal_cells_by_hypothesis[h_idx] = set()
                continue
            # Action index within the K-cell subproblem; pick the candidate
            # with the highest prior mean under this hypothesis as the
            # reward cell.
            best_global_cell = max(
                overlap,
                key=lambda c: (
                    self.hypothesis_set.hypotheses[h_idx].prior_mean_field[c]
                    if h_idx < len(self.hypothesis_set.hypotheses)
                    else 0.0
                ),
            )
            deposit_cell_by_hypothesis[h_idx] = candidates.index(best_global_cell)
            signal_cells_by_hypothesis[h_idx] = {
                candidates.index(c) for c in overlap
            }

        return MultiHypothesisSmallGridPOMDP(
            n_cells=len(candidates),
            hypothesis_names=names,
            deposit_cell_by_hypothesis=deposit_cell_by_hypothesis,
            signal_cells_by_hypothesis=signal_cells_by_hypothesis,
            initial_prior=self._belief.copy(),
            alpha_fp=self.alpha_fp,
            beta_fn=self.beta_fn,
            drill_cost=self.drill_cost,
            discovery_value=self.discovery_value,
            wrong_commitment_penalty=self.wrong_commitment_penalty,
        )

    def choose_action(
        self,
        history: list[tuple[int, float]] | None = None,
        drilled: frozenset[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Pick the next cell to drill. Builds and solves a subproblem
        SARSOP at each step. Returns the cell index in the BCGT grid."""
        if drilled is not None:
            self._drilled = set(drilled)
        candidates = self._top_k_candidates()
        if not candidates:
            raise RuntimeError("no un-drilled candidate cells left")

        sub = self._build_subproblem(candidates)
        alpha_policy = solve_sarsop(
            sub,
            pomdpsol_path=self.pomdpsol_path,
            discount=0.95,
            timeout_sec=self.sarsop_timeout_sec,
            precision=self.sarsop_precision,
        )
        sub_policy = MultiHypothesisSARSOPPolicy(
            pomdp=sub, alpha_policy=alpha_policy,
        )
        sub_action = sub_policy.choose_action()
        return candidates[sub_action]

    def observe(self, cell_idx: int, observation: int) -> None:
        """Update the categorical belief from one Bernoulli observation."""
        likelihoods = np.empty(len(self._belief))
        for i, h in enumerate(self.hypothesis_set.hypotheses):
            in_signal = cell_idx in self.deposit_sets.get(i, set())
            p1 = (1.0 - self.beta_fn) if in_signal else self.alpha_fp
            likelihoods[i] = p1 if observation == 1 else (1.0 - p1)
        if self.hypothesis_set.include_null and self.hypothesis_set.null is not None:
            likelihoods[-1] = self.alpha_fp if observation == 1 else (1.0 - self.alpha_fp)
        unnorm = self._belief * likelihoods
        s = unnorm.sum()
        if s > 0:
            self._belief = unnorm / s
        self._drilled.add(cell_idx)
