"""Tests for the BCGS real-prior factory and the deposit-type hypothesis set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_minerals.decision.v20.hypotheses import (
    make_bcgt_deposit_type_hypothesis_set,
)
from ai_minerals.decision.v20.real_priors import (
    DEFAULT_TYPES,
    bcgs_deposit_type_prior_surfaces,
)


def _tiny_features(tmp_path: Path, n_rows=20, n_cols=10) -> Path:
    """A tiny synthetic features parquet that mimics the BCGT 500m layout."""
    rows = []
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            rows.append({
                "row": r, "col": c,
                "x": float(c * 500), "y": float(r * 500),
                "is_porphyry": 1 if (r, c) in {(5, 5), (6, 5)} else 0,
                "is_skarn": 1 if (r, c) in {(15, 8)} else 0,
                "is_epithermal": 1 if (r, c) in {(3, 2)} else 0,
                "is_vms": 0,
            })
    p = tmp_path / "features_tiny.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return p


def test_surfaces_shape_and_peak(tmp_path):
    p = _tiny_features(tmp_path)
    surfaces, coords = bcgs_deposit_type_prior_surfaces(p, n_side=5)
    assert coords.shape == (25, 2)
    for name, surf in surfaces.items():
        assert surf.shape == (25,)
        assert 0.0 <= surf.min() <= surf.max() <= 0.4 + 1e-9


def test_vms_with_no_occurrences_is_all_zero(tmp_path):
    p = _tiny_features(tmp_path)
    surfaces, _ = bcgs_deposit_type_prior_surfaces(p, n_side=5)
    assert np.all(surfaces["vms"] == 0.0)


def test_missing_label_column_raises(tmp_path):
    df = pd.DataFrame({
        "row": [1, 2], "col": [1, 2], "x": [0.0, 1.0], "y": [0.0, 1.0],
        "is_porphyry": [0, 1],
    })
    p = tmp_path / "no_skarn.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="Missing label column"):
        bcgs_deposit_type_prior_surfaces(p, n_side=2, types=("is_porphyry", "is_skarn"))


def test_n_side_too_large_raises(tmp_path):
    p = _tiny_features(tmp_path, n_rows=4, n_cols=4)
    with pytest.raises(ValueError, match="too large"):
        bcgs_deposit_type_prior_surfaces(p, n_side=10)


def test_deposit_type_hypothesis_set_factory_smoke(tmp_path):
    p = _tiny_features(tmp_path, n_rows=30, n_cols=20)
    hset, coords = make_bcgt_deposit_type_hypothesis_set(
        features_path=str(p), n_side=10,
    )
    assert hset.n_hypotheses == 5
    assert hset.include_null
    names = [h.name for h in hset.hypotheses]
    assert names == ["H_porphyry", "H_skarn", "H_epithermal", "H_vms"]
    assert coords.shape == (100, 2)
    init = hset.initial_prior()
    assert init.shape == (5,)
    assert abs(init.sum() - 1.0) < 1e-9


def test_default_types_constant():
    assert DEFAULT_TYPES == (
        "is_porphyry", "is_skarn", "is_epithermal", "is_vms",
    )
