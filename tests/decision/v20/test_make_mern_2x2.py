"""Tests for the Mern 2024 2x2 hypothesis-set factory.

Verifies the factory produces 4 paper hypotheses indexed by
(n_grabens, n_domains) in {(1,1), (1,2), (2,1), (2,2)} plus an
optional null, with the metadata threaded through correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.hypotheses import (
    Hypothesis,
    HypothesisSet,
    NullHypothesis,
    make_mern_2x2_hypothesis_set,
)


def test_default_set_has_four_paper_hypotheses_plus_null() -> None:
    hset = make_mern_2x2_hypothesis_set(seed=0)
    assert isinstance(hset, HypothesisSet)
    assert hset.n_hypotheses == 5  # 4 paper + 1 null
    assert len(hset.hypotheses) == 4
    assert hset.null is not None
    assert isinstance(hset.null, NullHypothesis)


def test_hypothesis_names_match_paper_2x2_grid_order() -> None:
    hset = make_mern_2x2_hypothesis_set(seed=0)
    names = [h.name for h in hset.hypotheses]
    assert names == ["H_1_1", "H_1_2", "H_2_1", "H_2_2"]


def test_n_grabens_and_n_domains_match_names() -> None:
    hset = make_mern_2x2_hypothesis_set(seed=0)
    expected = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for h, (n_g, n_d) in zip(hset.hypotheses, expected):
        assert (h.n_grabens, h.n_domains) == (n_g, n_d)


def test_include_null_false_returns_4_paper_hypotheses_only() -> None:
    hset = make_mern_2x2_hypothesis_set(seed=0, include_null=False)
    assert hset.n_hypotheses == 4
    assert hset.null is None
    assert hset.include_null is False


def test_grid_size_threads_through() -> None:
    hset = make_mern_2x2_hypothesis_set(grid_n=16, seed=0)
    for h in hset.hypotheses:
        assert h.n_cells == 16 * 16


def test_factory_is_deterministic_given_seed() -> None:
    hset_a = make_mern_2x2_hypothesis_set(seed=42)
    hset_b = make_mern_2x2_hypothesis_set(seed=42)
    for h_a, h_b in zip(hset_a.hypotheses, hset_b.hypotheses):
        np.testing.assert_allclose(h_a.prior_mean_field, h_b.prior_mean_field)


def test_factory_with_different_seeds_produces_distinct_realizations() -> None:
    hset_a = make_mern_2x2_hypothesis_set(seed=0)
    hset_b = make_mern_2x2_hypothesis_set(seed=1)
    # At least one hypothesis should have a different prior_mean_field
    assert not np.allclose(
        hset_a.hypotheses[0].prior_mean_field,
        hset_b.hypotheses[0].prior_mean_field,
    )


def test_explicit_rng_overrides_seed() -> None:
    rng = np.random.default_rng(99)
    hset_a = make_mern_2x2_hypothesis_set(rng=rng, seed=0)
    rng = np.random.default_rng(99)
    hset_b = make_mern_2x2_hypothesis_set(rng=rng, seed=999)
    # Same rng => same realization regardless of seed
    np.testing.assert_allclose(
        hset_a.hypotheses[0].prior_mean_field,
        hset_b.hypotheses[0].prior_mean_field,
    )


def test_initial_prior_is_uniform_across_5_hypotheses() -> None:
    hset = make_mern_2x2_hypothesis_set(seed=0)
    prior = hset.initial_prior()
    assert prior.shape == (5,)
    assert prior == pytest.approx(np.full(5, 1.0 / 5.0))


def test_initial_prior_uniform_across_4_when_null_omitted() -> None:
    hset = make_mern_2x2_hypothesis_set(seed=0, include_null=False)
    prior = hset.initial_prior()
    assert prior == pytest.approx(np.full(4, 1.0 / 4.0))
