"""Unit tests for the leak-guarded spatial-CV harness (ai_minerals.spatial_cv).

These prove the mechanics independently of the Nome data: the variogram tracks
the autocorrelation scale, the dead-zone actually removes near-by training
points, folds are balanced and cover every point once, and -- the headline
acceptance property -- the harness flags restriction-of-range (in-box AUC much
higher than district AUC) on a synthetic field built to fail that way.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier

from ai_minerals.spatial_cv import (
    LeakGuardedSpatialCV,
    assign_block_folds,
    block_size_from_range,
    fit_variogram,
    leak_guarded_evaluate,
)


def _fast_rf():
    """Small RF so the suite stays quick; mechanics are estimator-independent."""
    return lambda: RandomForestClassifier(
        n_estimators=60, class_weight="balanced", random_state=0, n_jobs=1
    )


def _grid(n_side: int, spacing: float) -> np.ndarray:
    g = np.arange(n_side) * spacing
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_variogram_range_tracks_wavelength():
    """A longer-wavelength (more autocorrelated) field must yield a larger
    fitted range. Deterministic sinusoid field, no RNG."""
    coords = _grid(40, 250.0)  # 40x40 grid, 250 m spacing -> ~10 km square
    x, y = coords[:, 0], coords[:, 1]

    def fitted_range(wavelength: float) -> float:
        k = 2 * np.pi / wavelength
        z = np.sin(k * x) + np.sin(k * y)
        return fit_variogram(coords, z, n_lags=20).range_m

    assert fitted_range(8000.0) > fitted_range(2000.0)


def test_variogram_range_positive_and_bounded():
    coords = _grid(30, 300.0)
    z = np.sin(2 * np.pi * coords[:, 0] / 4000.0)
    vf = fit_variogram(coords, z, n_lags=15)
    assert 0.0 < vf.range_m <= coords[:, 0].max()
    assert vf.method in {"spherical_fit", "effective_range_heuristic"}


def test_dead_zone_removes_nearby_training_points():
    """No training point may sit within dead_zone_m of any test point."""
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 20_000, size=(600, 2))
    y = (rng.random(600) < 0.2).astype(int)
    r = 1000.0
    cv = LeakGuardedSpatialCV(block_size_m=2500.0, n_folds=5, dead_zone_m=r, seed=1)
    n_splits = 0
    for tr, te in cv.split(coords, y):
        n_splits += 1
        d, _ = cKDTree(coords[te]).query(coords[tr], k=1)
        assert d.min() >= r  # strict guard: dead-zone honoured for every fold
    assert n_splits >= 1


def test_every_point_tested_exactly_once():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 30_000, size=(500, 2))
    y = (rng.random(500) < 0.15).astype(int)
    cv = LeakGuardedSpatialCV(block_size_m=4000.0, n_folds=5, dead_zone_m=0.0, seed=2)
    test_idx = np.concatenate([te for _, te in cv.split(coords, y)])
    assert np.array_equal(np.sort(test_idx), np.arange(len(coords)))


def test_balanced_scatter_balances_positives():
    """The balanced_scatter strategy should split positives near-evenly."""
    coords = _grid(20, 1000.0)  # 400 points, 1 km spacing
    rng = np.random.default_rng(11)
    pos_idx = rng.choice(len(coords), size=40, replace=False)
    y = np.zeros(len(coords), int)
    y[pos_idx] = 1
    n_folds = 5
    _, fold = assign_block_folds(coords, y, block_size_m=1500.0, n_folds=n_folds,
                                 seed=4, strategy="balanced_scatter")
    per_fold = np.array([y[fold == f].sum() for f in range(n_folds)])
    assert per_fold.sum() == 40
    assert per_fold.max() - per_fold.min() <= 2  # tight balance


def test_contiguous_folds_are_more_compact_than_scatter():
    """Contiguous folds must be spatially tighter than scattered ones -- that
    compactness is what forces extrapolation and exposes restriction of range."""
    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 50_000, size=(800, 2))
    y = (rng.random(800) < 0.2).astype(int)

    def mean_within_fold_distance(strategy: str) -> float:
        _, fold = assign_block_folds(coords, y, block_size_m=3000.0, n_folds=5,
                                     seed=1, strategy=strategy)
        spreads = [pdist(coords[fold == f]).mean()
                   for f in np.unique(fold) if (fold == f).sum() > 1]
        return float(np.mean(spreads))

    assert mean_within_fold_distance("contiguous") < mean_within_fold_distance("balanced_scatter")


def test_block_size_from_range_goes_larger():
    # 2x multiplier, rounded up to 100 m
    assert block_size_from_range(1234.0, multiplier=2.0, round_to=100.0) == 2500.0
    assert block_size_from_range(900.0, multiplier=2.0, floor_m=3000.0) == 3000.0


def _restriction_of_range_dataset(seed: int = 0):
    """Elevation separates positives from background ONLY inside a low-x box; the
    upland hinterland background shares the positives' elevation, so the signal
    collapses district-wide. Positives are spread across the whole district so
    contiguous folds always retain training positives (no degenerate fold)."""
    rng = np.random.default_rng(seed)
    box = 25_000.0
    # Positives: spread across the 100 km district, all 'upland' (high elevation).
    n_pos = 120
    p_xy = rng.uniform(0, 100_000, size=(n_pos, 2))
    p_elev = rng.normal(80, 5, n_pos)
    # 'Coastal flat' background: only inside the box (low x), low elevation.
    n_a = 400
    a_xy = np.column_stack([rng.uniform(0, box, n_a), rng.uniform(0, 100_000, n_a)])
    a_elev = rng.normal(20, 5, n_a)
    # 'Upland hinterland' background: outside the box, same elevation as positives.
    n_b = 1500
    b_xy = np.column_stack([rng.uniform(box, 100_000, n_b), rng.uniform(0, 100_000, n_b)])
    b_elev = rng.normal(80, 5, n_b)

    coords = np.vstack([p_xy, a_xy, b_xy])
    elev = np.concatenate([p_elev, a_elev, b_elev])
    noise = rng.normal(0, 1, len(coords))
    X = np.column_stack([elev, noise])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_a + n_b)]).astype(int)
    in_box = coords[:, 0] < box
    return X, y, coords, in_box


def test_two_model_collapse_reproduced():
    """Canonical reproduction: a model trained+tested inside the box scores high
    (there, elevation separates the classes); the SAME feature trained+tested
    district-wide collapses. This is the lode-failure signature."""
    X, y, coords, in_box = _restriction_of_range_dataset(seed=0)
    res_box = leak_guarded_evaluate(  # in-box-only: train and test confined to the box
        X[in_box], y[in_box], coords[in_box], model_factory=_fast_rf(),
        n_folds=5, dead_zone_m=1000.0, seed=0,
    )
    res_dist = leak_guarded_evaluate(  # district: train and test across the district
        X, y, coords, model_factory=_fast_rf(), n_folds=5, dead_zone_m=1000.0, seed=0,
    )
    assert res_box.auc_district > 0.85          # in-box looks great
    assert res_dist.auc_district < 0.72         # district collapses
    assert res_box.auc_district - res_dist.auc_district > 0.15  # the collapse


def test_within_run_in_box_alarm_fires():
    """Every-run alarm: one district model, AUC on the in-box subset exceeds the
    district-wide AUC -> restriction-of-range flagged from a single fit."""
    X, y, coords, in_box = _restriction_of_range_dataset(seed=0)
    res = leak_guarded_evaluate(
        X, y, coords, in_box_mask=in_box, model_factory=_fast_rf(),
        n_folds=5, dead_zone_m=1000.0, seed=0,
    )
    assert res.auc_in_box is not None and res.auc_district is not None
    assert res.auc_in_box > res.auc_district + 0.1  # divergence alarm fires
    assert res.n_pos_scored == res.n_pos  # contiguous folds still scored every positive
    assert res.block_size_m > 0 and res.n_blocks >= res.n_folds


def test_result_records_scheme_parameters():
    X, y, coords, in_box = _restriction_of_range_dataset(seed=1)
    res = leak_guarded_evaluate(
        X, y, coords, in_box_mask=in_box, model_factory=_fast_rf(),
        n_folds=5, dead_zone_m=1200.0, seed=1,
    )
    assert res.dead_zone_m == 1200.0
    assert res.variogram_range_m > 0
    assert res.n_pos == 120 and res.n == len(y)
    assert res.fold_strategy == "contiguous"
    assert 0 <= res.n_scored <= len(y)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
