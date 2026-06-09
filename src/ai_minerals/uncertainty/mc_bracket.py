"""AutoBEL-style Monte Carlo uncertainty bracket on a PU-bagging raster.

Per-cell P05 / P50 / P95 across the n_bags-many bag-level predictions of a
PU-bagging ensemble. Cited as the entry-level Monte Carlo precursor to the
full Bayesian Evidential Learning framework in Yin, Strebelle, Caers (2020),
DOI 10.5194/gmd-13-651-2020.

Conceptually: the n_bags Mordelet-Vert bags each see a different random
"unlabeled-as-negative" subsample. Their predictions at any given cell
sample from the model's epistemic uncertainty over which negatives are
actually negatives. The quantiles across that sample form a defensible
±90% (P05–P95) confidence band per cell.

This is intentionally a thin module: the work is in PU-bagging itself
(see ``model_pu.fit_pu_bagging(..., return_per_bag=True)``); MC bracket
is a few NumPy calls on the per-bag array.
"""

from __future__ import annotations

import numpy as np


def monte_carlo_bracket(
    predictions_per_bag: np.ndarray,
    *,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> dict[str, np.ndarray]:
    """Per-cell P05 / P50 / P95 across the bag axis.

    Parameters
    ----------
    predictions_per_bag :
        Float array of shape ``(n_bags, n_cells)``. NaN entries in any bag
        are ignored (``np.nanpercentile`` semantics) so per-bag OOB masks
        do not contaminate the brackets.
    lower_q, upper_q :
        Lower and upper quantiles in [0, 1]. Defaults to a ±90% band
        (P05, P95). Median is always returned alongside.

    Returns
    -------
    dict with keys ``'p_lower'``, ``'p50'``, ``'p_upper'``, ``'spread'``,
    ``'mean'``. All length-``n_cells`` arrays. ``spread = p_upper - p_lower``.

    Notes
    -----
    The PU-bagging point estimate (the OOB-aggregated mean over bags) is
    not necessarily equal to this function's ``'p50'`` median. The mean
    is more stable for the headline raster; the median is included for
    completeness and is what's reported alongside the bracket below.
    """
    if predictions_per_bag.ndim != 2:
        raise ValueError(
            f"predictions_per_bag must be 2D (n_bags, n_cells); "
            f"got shape {predictions_per_bag.shape}"
        )
    lo = np.nanpercentile(predictions_per_bag, lower_q * 100, axis=0)
    md = np.nanpercentile(predictions_per_bag, 50.0, axis=0)
    hi = np.nanpercentile(predictions_per_bag, upper_q * 100, axis=0)
    mn = np.nanmean(predictions_per_bag, axis=0)
    return {
        "p_lower": lo.astype(np.float32),
        "p50": md.astype(np.float32),
        "p_upper": hi.astype(np.float32),
        "spread": (hi - lo).astype(np.float32),
        "mean": mn.astype(np.float32),
    }
