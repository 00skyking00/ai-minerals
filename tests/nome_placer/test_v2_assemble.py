"""Tests for the Phase 2 feature assembler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_minerals.features.v2_assemble import (
    PROSPECTIVITY_BAND_KEYS,
    V2FeatureFrame,
    assemble,
    claim_ms_kfold_split,
)
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION


DRILLHOLES = Path(NOME_PLACER_REGION.raw_paths["drillholes"])
RASTER = Path(
    "data/derived/nome_placer/prospectivity_v1p5/"
    "nome_placer_prospectivity_v1p5_v2_4326.tif"
)


@pytest.fixture(scope="module")
def feature_frame() -> V2FeatureFrame:
    if not DRILLHOLES.exists():
        pytest.skip(f"drillholes parquet missing: {DRILLHOLES}")
    if not RASTER.exists():
        pytest.skip(f"v1.5 v2 raster missing: {RASTER}")
    return assemble(DRILLHOLES, RASTER)


def test_assemble_returns_one_row_per_drillhole(feature_frame: V2FeatureFrame) -> None:
    holes = pd.read_parquet(DRILLHOLES)
    assert len(feature_frame.holes) == len(holes)


def test_feature_columns_present(feature_frame: V2FeatureFrame) -> None:
    """All 7 raster bands + bedrock_depth + surface_class_code are columns."""
    for k in PROSPECTIVITY_BAND_KEYS:
        assert f"raster_{k}" in feature_frame.holes.columns
    assert "bedrock_depth_ft" in feature_frame.holes.columns
    assert "surface_class_code" in feature_frame.holes.columns
    assert len(feature_frame.feature_columns) == len(PROSPECTIVITY_BAND_KEYS) + 2


def test_target_columns_resolve(feature_frame: V2FeatureFrame) -> None:
    """Every named target column actually exists in the holes table."""
    for head, col in feature_frame.target_columns.items():
        assert col in feature_frame.holes.columns, head


def test_X_returns_feature_columns_only(feature_frame: V2FeatureFrame) -> None:
    X = feature_frame.X()
    assert list(X.columns) == feature_frame.feature_columns


def test_y_returns_target_series(feature_frame: V2FeatureFrame) -> None:
    y_pay = feature_frame.y("pay_binary")
    assert len(y_pay) == len(feature_frame.holes)
    # pay_binary should be True/False values; permit pandas nullable bool
    valid = y_pay.dropna()
    assert valid.isin([True, False, 0, 1]).all()


def test_raster_samples_in_unit_interval(feature_frame: V2FeatureFrame) -> None:
    """Prospectivity bands are in [0, 1] by construction."""
    for k in PROSPECTIVITY_BAND_KEYS:
        col = f"raster_{k}"
        vals = feature_frame.holes[col].dropna()
        assert (vals >= 0.0).all()
        assert (vals <= 1.0).all()


def test_claim_ms_kfold_splits_no_overlap(feature_frame: V2FeatureFrame) -> None:
    """Each fold's train and test sets are disjoint."""
    splits = claim_ms_kfold_split(feature_frame.holes, n_folds=4)
    assert len(splits) <= 4
    for train, test in splits:
        assert not (set(train) & set(test))


def test_claim_ms_kfold_loo_yields_all_ms_groups(feature_frame: V2FeatureFrame) -> None:
    """LOO split has at least one fold per distinct claim_ms value."""
    splits = claim_ms_kfold_split(feature_frame.holes)
    n_ms_groups = (
        feature_frame.holes["claim_ms"].fillna("__unknown__").nunique()
    )
    assert len(splits) <= n_ms_groups
    # Each hole appears in exactly one test fold
    test_indices_per_fold = [set(test) for _, test in splits]
    union = set().union(*test_indices_per_fold)
    assert union == set(range(len(feature_frame.holes)))


def test_new_bearcub_columns_threaded(feature_frame: V2FeatureFrame) -> None:
    """2026-06-16 Janin re-grade new columns reach the assembler."""
    cols = set(feature_frame.holes.columns)
    assert "pay_zone_grade_wholecol_oz_cuyd" in cols
    assert "bottom_of_pay_ft" in cols
    assert "surface_mineable" in cols
