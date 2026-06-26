"""Leak-guarded spatial cross-validation for presence/background MPM.

Why this module exists
----------------------
Naive (random) K-fold CV on spatially autocorrelated samples is optimistic:
a test point usually has a near-duplicate in the training fold a few hundred
metres away, so the model is graded on points it has effectively already seen.
Distance-to-occurrence covariates make this worse: a 400-800 m proximity buffer
written into the features leaks straight across a naive fold boundary. The two
guards here are the standard fixes from the ecology / spatial-ML literature:

  * **Spatial block CV** (Roberts et al. 2017, Ecography 40:913; Valavi et al.
    2019 blockCV, Methods Ecol. Evol. 10:225): assign whole square blocks of the
    map to folds, so a held-out fold is a contiguous chunk of ground rather than
    a random scatter. Block edge is set from the autocorrelation range of the
    **model residuals** (not the raw covariates: an overfit model flattens
    residual structure, so a residual variogram understates the needed block
    size and we deliberately size the block *larger* than it suggests).

  * **Dead-zone buffer** (Airola et al. 2018, "spatial leave-pair-out", Pattern
    Recognition 80:172): drop every training point within radius ``r`` of any
    held-out test point. ``r`` must exceed the largest proximity-feature buffer
    (so r >= ~1 km given our 400-800 m features) or the proximity covariate
    leaks across the fold edge regardless of how blocks are drawn.

This is a documented equivalent of ``verde.BlockKFold`` / ``spacv`` (neither is a
project dependency; pyproject carries a deferred ``spacv`` placeholder). It
reuses the proven in-repo pieces -- the square-block assignment of
``ai_minerals.model.SpatialBlockCV`` and the cKDTree dead-zone first written
inline in the Nome ``mpm_*_presence_cv`` scripts -- and adds the residual
variogram, even-positive fold balancing, and an in-box-vs-district AUC split
that those inline copies lacked.

Counterweight worth stating: spatial CV *overestimates* error on genuinely
independent data (Wadoux et al. 2021, Ecol. Modell. 457:109692). The block size
should match the actual label dependence, not be maximally conservative by
reflex. The residual variogram is how we calibrate it instead of guessing.

CRS contract: all coordinates are projected metres (EPSG:3338 for Nome). Pass
``coords`` as an (n, 2) array of (x, y) in metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FoldStrategy = str  # "contiguous" | "balanced_scatter"

Model = Callable[[], "object"]  # zero-arg factory returning a fresh sklearn-like estimator


def default_rf_factory(*, seed: int = 42, n_jobs: int = -1) -> Model:
    """The RandomForest used by the Nome presence/background scripts (300 trees,
    balanced classes) so leak-guarded AUCs are comparable to the old numbers."""
    return lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=n_jobs
    )


# --------------------------------------------------------------------------- #
# Empirical variogram + range fit (on model residuals)
# --------------------------------------------------------------------------- #
def empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_lags: int = 15,
    max_dist: float | None = None,
    max_points: int = 5000,
    min_pairs_per_lag: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isotropic empirical semivariogram.

    Returns (lag_centre_m, semivariance, n_pairs) over lag bins that clear
    ``min_pairs_per_lag``. Subsamples to ``max_points`` (deterministic) to keep
    the full pairwise distance matrix small.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    n = len(coords)
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        coords, values = coords[idx], values[idx]
    dist = pdist(coords)
    sqdiff = pdist(values.reshape(-1, 1), metric="sqeuclidean")
    if max_dist is None:
        max_dist = 0.5 * dist.max()  # half the domain: longer lags are pair-starved
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centres, semiv, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (dist >= lo) & (dist < hi)
        c = int(in_bin.sum())
        if c < min_pairs_per_lag:
            continue
        centres.append(0.5 * (lo + hi))
        semiv.append(0.5 * float(sqdiff[in_bin].mean()))
        counts.append(c)
    return np.array(centres), np.array(semiv), np.array(counts, int)


def _spherical(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    h = np.asarray(h, float)
    out = nugget + sill * (1.5 * (h / rng) - 0.5 * (h / rng) ** 3)
    return np.where(h >= rng, nugget + sill, out)


@dataclass
class VariogramFit:
    """Fitted (or heuristic) spherical variogram of a residual field."""

    range_m: float
    sill: float
    nugget: float
    method: str  # "spherical_fit" or "effective_range_heuristic"
    lags: np.ndarray
    semivariance: np.ndarray
    counts: np.ndarray


def fit_variogram(
    coords: np.ndarray, values: np.ndarray, **vg_kwargs
) -> VariogramFit:
    """Fit a spherical model to the empirical semivariogram; fall back to the
    effective-range heuristic (first lag reaching 95% of the empirical sill) if
    the non-linear fit does not converge."""
    lags, semiv, counts = empirical_variogram(coords, values, **vg_kwargs)
    if len(lags) < 3:
        rng = float(lags[-1]) if len(lags) else 0.0
        return VariogramFit(rng, float(semiv.max() if len(semiv) else 0.0), 0.0,
                            "effective_range_heuristic", lags, semiv, counts)
    sill0 = float(np.median(semiv[-max(1, len(semiv) // 3):]))
    nugget0 = float(min(semiv[0], sill0))
    eff = lags[np.argmax(semiv >= 0.95 * sill0)] if (semiv >= 0.95 * sill0).any() else lags[-1]
    try:
        popt, _ = curve_fit(
            _spherical, lags, semiv,
            p0=[nugget0, max(sill0 - nugget0, 1e-9), float(eff)],
            bounds=([0.0, 0.0, lags[0]], [sill0 * 2 + 1e-9, sill0 * 5 + 1e-9, lags[-1]]),
            maxfev=10000,
        )
        nugget, sill, rng = (float(v) for v in popt)
        return VariogramFit(rng, sill, nugget, "spherical_fit", lags, semiv, counts)
    except (RuntimeError, ValueError):
        return VariogramFit(float(eff), float(sill0 - nugget0), float(nugget0),
                            "effective_range_heuristic", lags, semiv, counts)


def residual_variogram(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    *,
    model_factory: Model,
    n_splits: int = 5,
    seed: int = 42,
    **vg_kwargs,
) -> VariogramFit:
    """Out-of-fold residuals from a non-spatial K-fold, then a variogram on them.

    OOF (not in-sample) residuals are used so the fitted range reflects true
    prediction error, not the near-zero residuals an overfit model leaves at its
    own training points."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y, int)
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = model_factory().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    resid = y.astype(float) - oof
    return fit_variogram(coords, resid, **vg_kwargs)


def block_size_from_range(
    range_m: float, *, multiplier: float = 2.0, round_to: float = 100.0, floor_m: float = 0.0
) -> float:
    """Block edge = ``multiplier`` x residual range, rounded up. Roberts et al.
    advise blocks >= the range; we go above it (default 2x) because a residual
    variogram understates the range under any overfitting."""
    return float(max(np.ceil(multiplier * range_m / round_to) * round_to, floor_m))


# --------------------------------------------------------------------------- #
# Block -> fold assignment (even positive spread) + the leak-guarded splitter
# --------------------------------------------------------------------------- #
def _grid_blocks(coords: np.ndarray, block_size_m: float) -> np.ndarray:
    bx = np.floor((coords[:, 0] - coords[:, 0].min()) / block_size_m).astype(int)
    by = np.floor((coords[:, 1] - coords[:, 1].min()) / block_size_m).astype(int)
    return bx * (by.max() + 1) + by


def _contiguous_block_folds(block_id, coords, n_folds, seed) -> dict[int, int]:
    """Group whole blocks into ``n_folds`` spatially-compact folds (KMeans on the
    block centroids, weighted by block size). A held-out fold is then a contiguous
    region, which forces extrapolation -- the test that exposes restriction of
    range. Positives end up where they sit, so per-fold counts are uneven by
    design; the pooled-OOF AUC does not need them balanced."""
    blocks = np.unique(block_id)
    if len(blocks) <= n_folds:
        return {int(b): i for i, b in enumerate(blocks)}
    cent = np.array([coords[block_id == b].mean(0) for b in blocks])
    w = np.array([(block_id == b).sum() for b in blocks], float)
    labels = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit(cent, sample_weight=w).labels_
    return {int(b): int(l) for b, l in zip(blocks, labels)}


def _balanced_scatter_folds(block_id, y, n_folds, seed) -> dict[int, int]:
    """Greedily spread whole blocks across folds to equalise the positive count
    (then point count). Even per-fold positives, but folds are spatially
    scattered -- weaker spatial separation than contiguous folds."""
    blocks = np.unique(block_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    pos_per_block = {int(b): int(y[block_id == b].sum()) for b in blocks}
    n_per_block = {int(b): int((block_id == b).sum()) for b in blocks}
    order = sorted((int(b) for b in blocks), key=lambda b: (-pos_per_block[b], -n_per_block[b]))
    fold_pos = np.zeros(n_folds, int)
    fold_n = np.zeros(n_folds, int)
    block_fold: dict[int, int] = {}
    for b in order:
        f = int(np.lexsort((fold_n, fold_pos))[0])  # fewest positives, then fewest points
        block_fold[b] = f
        fold_pos[f] += pos_per_block[b]
        fold_n[f] += n_per_block[b]
    return block_fold


def assign_block_folds(
    coords: np.ndarray,
    y: np.ndarray,
    *,
    block_size_m: float,
    n_folds: int,
    seed: int = 42,
    strategy: FoldStrategy = "contiguous",
) -> tuple[np.ndarray, np.ndarray]:
    """Square-grid blocks (edge ``block_size_m``) assigned to ``n_folds`` folds.

    strategy="contiguous"      : compact regional folds (default). Exposes
                                 restriction-of-range; the stricter test.
    strategy="balanced_scatter": positive-balanced scattered folds (Valavi 2019
                                 "random" assignment). Stable per-fold counts but
                                 a weaker spatial test for large-scale bias.

    Returns (block_id_per_point, fold_per_point). Whole blocks go to one fold.
    """
    coords = np.asarray(coords, float)
    y = np.asarray(y, int)
    block_id = _grid_blocks(coords, block_size_m)
    if strategy == "contiguous":
        block_fold = _contiguous_block_folds(block_id, coords, n_folds, seed)
    elif strategy == "balanced_scatter":
        block_fold = _balanced_scatter_folds(block_id, y, n_folds, seed)
    else:
        raise ValueError(f"unknown fold strategy {strategy!r}")
    fold = np.array([block_fold[int(b)] for b in block_id], int)
    return block_id, fold


@dataclass
class LeakGuardedSpatialCV:
    """Spatial block K-fold with a dead-zone buffer between test and train.

    block_size_m : square-block edge (set from the residual variogram).
    n_folds      : number of spatial folds.
    dead_zone_m  : drop training points within this radius of ANY test point;
                   must exceed the largest proximity-feature buffer in X.
    strategy     : "contiguous" (default, compact folds) or "balanced_scatter".
    """

    block_size_m: float
    n_folds: int = 10
    dead_zone_m: float = 1000.0
    seed: int = 42
    strategy: FoldStrategy = "contiguous"

    def split(self, coords: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        coords = np.asarray(coords, float)
        y = np.asarray(y, int)
        _, fold = assign_block_folds(
            coords, y, block_size_m=self.block_size_m, n_folds=self.n_folds,
            seed=self.seed, strategy=self.strategy,
        )
        for f in range(self.n_folds):
            test = fold == f
            if not test.any():
                continue
            train = ~test
            if self.dead_zone_m > 0:
                d, _ = cKDTree(coords[test]).query(coords, k=1)
                train &= d >= self.dead_zone_m
            yield np.where(train)[0], np.where(test)[0]


def spatial_cv_oof(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    cv: LeakGuardedSpatialCV,
    *,
    model_factory: Model,
) -> np.ndarray:
    """Pooled out-of-fold positive-class probabilities under ``cv`` (NaN where a
    point was never a clean test point)."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y, int)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(coords, y):
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_factory().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def subset_auc(y: np.ndarray, oof: np.ndarray, *, mask: np.ndarray | None = None) -> float | None:
    """Pooled-OOF ROC-AUC over the masked subset; None if a class is absent."""
    y = np.asarray(y, int)
    ok = ~np.isnan(oof)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], oof[ok]))


# --------------------------------------------------------------------------- #
# High-level convenience: variogram -> block size -> AUC (in-box & district)
# --------------------------------------------------------------------------- #
@dataclass
class LeakGuardedResult:
    """Everything the F1 report needs from one leak-guarded evaluation."""

    auc_district: float | None
    auc_in_box: float | None
    n: int
    n_pos: int
    n_pos_in_box: int | None
    n_pos_scored: int
    fold_strategy: FoldStrategy
    variogram_range_m: float
    variogram_method: str
    block_size_m: float
    block_size_multiplier: float
    dead_zone_m: float
    n_folds: int
    n_blocks: int
    n_scored: int
    extras: dict = field(default_factory=dict)


def leak_guarded_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    *,
    in_box_mask: np.ndarray | None = None,
    model_factory: Model | None = None,
    n_folds: int = 10,
    dead_zone_m: float = 1000.0,
    block_size_multiplier: float = 2.0,
    block_size_floor_m: float = 0.0,
    min_blocks_factor: float = 2.0,
    fold_strategy: FoldStrategy = "contiguous",
    seed: int = 42,
) -> LeakGuardedResult:
    """Size blocks from the residual variogram, run leak-guarded CV, and return
    district-wide and in-box AUC plus the chosen scheme parameters.

    in_box_mask : boolean over rows marking the restricted (in-box) subset; its
    AUC is compared to the district AUC and the divergence is the standing
    restriction-of-range alarm.
    fold_strategy : "contiguous" (default, the stricter test) or
    "balanced_scatter" (positive-balanced but spatially weaker).
    """
    X = np.asarray(X, np.float32)
    y = np.asarray(y, int)
    coords = np.asarray(coords, float)
    factory = model_factory or default_rf_factory(seed=seed)

    vfit = residual_variogram(X, y, coords, model_factory=factory, seed=seed)
    block_size = block_size_from_range(
        vfit.range_m, multiplier=block_size_multiplier, floor_m=block_size_floor_m
    )
    # Cap so the domain yields enough blocks to populate n_folds.
    ext = coords.max(0) - coords.min(0)
    cap = float(np.sqrt(max(ext.prod(), 1.0) / (min_blocks_factor * n_folds)))
    capped = False
    if block_size > cap:
        block_size, capped = float(np.ceil(cap / 100.0) * 100.0), True

    cv = LeakGuardedSpatialCV(block_size, n_folds=n_folds, dead_zone_m=dead_zone_m,
                              seed=seed, strategy=fold_strategy)
    block_id, _ = assign_block_folds(coords, y, block_size_m=block_size, n_folds=n_folds,
                                     seed=seed, strategy=fold_strategy)
    oof = spatial_cv_oof(X, y, coords, cv, model_factory=factory)
    scored = ~np.isnan(oof)

    return LeakGuardedResult(
        auc_district=subset_auc(y, oof),
        auc_in_box=subset_auc(y, oof, mask=in_box_mask) if in_box_mask is not None else None,
        n=len(y),
        n_pos=int(y.sum()),
        n_pos_in_box=int(y[in_box_mask].sum()) if in_box_mask is not None else None,
        n_pos_scored=int(y[scored].sum()),
        fold_strategy=fold_strategy,
        variogram_range_m=round(vfit.range_m, 1),
        variogram_method=vfit.method,
        block_size_m=block_size,
        block_size_multiplier=block_size_multiplier,
        dead_zone_m=dead_zone_m,
        n_folds=n_folds,
        n_blocks=int(len(np.unique(block_id))),
        n_scored=int(scored.sum()),
        extras={"block_size_capped_to_fit_folds": capped,
                "variogram_sill": round(vfit.sill, 6),
                "variogram_nugget": round(vfit.nugget, 6)},
    )
