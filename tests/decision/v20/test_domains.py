"""Tests for the Mern 2024 structured-prior domain generation.

Covers polygon sampling determinism, mask construction shape and
content, and the masks-to-prior-mean-field combiner.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.decision.v20.domains import (
    DEFAULT_DOMAIN_VERTEX_COUNT,
    DOMAIN_MEAN_BOOST,
    GRABEN_MEAN_BOOST,
    domain_mask_from_polygons,
    prior_mean_field_from_masks,
    sample_geochem_domain_polygons,
    sample_graben_polygons,
)


def test_graben_polygon_count_and_vertex_count() -> None:
    rng = np.random.default_rng(0)
    polys = sample_graben_polygons(n_grabens=2, grid_n=32, rng=rng)
    assert len(polys) == 2
    for p in polys:
        assert len(p) == 4  # rectangular strip


def test_graben_polygon_is_deterministic_given_seed() -> None:
    polys_a = sample_graben_polygons(2, 32, np.random.default_rng(42))
    polys_b = sample_graben_polygons(2, 32, np.random.default_rng(42))
    for pa, pb in zip(polys_a, polys_b):
        for va, vb in zip(pa, pb):
            assert va == pytest.approx(vb)


def test_graben_zero_count_returns_empty_list() -> None:
    polys = sample_graben_polygons(0, 32, np.random.default_rng(0))
    assert polys == []


def test_graben_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="n_grabens"):
        sample_graben_polygons(-1, 32, np.random.default_rng(0))


def test_graben_rejects_tiny_grid() -> None:
    with pytest.raises(ValueError, match="grid_n"):
        sample_graben_polygons(1, 2, np.random.default_rng(0))


def test_geochem_domain_default_vertex_count() -> None:
    polys = sample_geochem_domain_polygons(
        n_domains=2, grid_n=32, rng=np.random.default_rng(0),
    )
    assert len(polys) == 2
    for p in polys:
        assert len(p) == DEFAULT_DOMAIN_VERTEX_COUNT


def test_geochem_domain_custom_vertex_count() -> None:
    polys = sample_geochem_domain_polygons(
        n_domains=1, grid_n=32,
        rng=np.random.default_rng(0),
        n_vertices=12,
    )
    assert len(polys[0]) == 12


def test_geochem_domain_zero_count_returns_empty() -> None:
    polys = sample_geochem_domain_polygons(0, 32, np.random.default_rng(0))
    assert polys == []


def test_domain_mask_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    polys = sample_graben_polygons(1, 32, rng)
    mask = domain_mask_from_polygons(polys, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == bool


def test_domain_mask_no_polygons_returns_all_false() -> None:
    mask = domain_mask_from_polygons([], 16)
    assert mask.shape == (16, 16)
    assert not mask.any()


def test_domain_mask_polygons_with_fewer_than_3_vertices_are_ignored() -> None:
    polys = [
        [(5.0, 5.0)],
        [(5.0, 5.0), (10.0, 10.0)],
    ]
    mask = domain_mask_from_polygons(polys, 16)
    assert not mask.any()


def test_domain_mask_covers_polygon_interior() -> None:
    """A 10x10 square polygon centered at (16, 16) should mask a
    contiguous block of cells inside its interior."""
    polygon = [(11.0, 11.0), (21.0, 11.0), (21.0, 21.0), (11.0, 21.0)]
    mask = domain_mask_from_polygons([polygon], 32)
    # Cells at (15, 15) and (18, 18) are clearly inside.
    assert mask[15, 15]
    assert mask[18, 18]
    # Cells at (0, 0) and (30, 30) are clearly outside.
    assert not mask[0, 0]
    assert not mask[30, 30]


def test_prior_mean_field_default_boosts() -> None:
    graben = np.zeros((4, 4), dtype=bool)
    graben[1, 1] = True
    domain = np.zeros((4, 4), dtype=bool)
    domain[2, 2] = True
    field = prior_mean_field_from_masks(graben, domain)
    assert field.shape == (16,)
    cell_with_graben = 1 * 4 + 1
    cell_with_domain = 2 * 4 + 2
    cell_with_neither = 0 * 4 + 0
    assert field[cell_with_graben] == pytest.approx(GRABEN_MEAN_BOOST)
    assert field[cell_with_domain] == pytest.approx(DOMAIN_MEAN_BOOST)
    assert field[cell_with_neither] == pytest.approx(0.0)


def test_prior_mean_field_overlapping_masks_sum_boosts() -> None:
    overlap = np.zeros((4, 4), dtype=bool)
    overlap[2, 2] = True
    field = prior_mean_field_from_masks(overlap, overlap)
    cell = 2 * 4 + 2
    assert field[cell] == pytest.approx(GRABEN_MEAN_BOOST + DOMAIN_MEAN_BOOST)


def test_prior_mean_field_rejects_mismatched_shapes() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="same shape"):
        prior_mean_field_from_masks(a, b)
