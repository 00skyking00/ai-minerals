"""Tests for the Tuck 1942 Nome District bedrock-contour extractor.

Exercises the real "Nome District Map (Contours).pdf" at /tmp/tuck/.
Skips if the PDF is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_minerals.data.tuck1942_bedrock_contours import (
    extract,
    save_extraction,
)


CONTOURS_PDF = Path("/tmp/tuck/Nome District Map (Contours).pdf")


@pytest.fixture(scope="module")
def contours_extraction(tmp_path_factory):
    if not CONTOURS_PDF.exists():
        pytest.skip(f"Contours PDF not at {CONTOURS_PDF}")
    cache = tmp_path_factory.mktemp("tuck1942_contours_cache")
    return extract(CONTOURS_PDF, cache_dir=cache)


def test_extraction_yields_many_red_segments(contours_extraction):
    """The Nome District Map has hundreds to thousands of red contour
    segments (dashed; each dash is a separate polygon). Expect > 100."""
    assert contours_extraction.n_segments > 100, (
        f"Only {contours_extraction.n_segments} segments; threshold may be off"
    )


def test_pixel_geometry_has_positive_area(contours_extraction):
    assert contours_extraction.total_pixel_area > 100
    assert not contours_extraction.segments_pixel.is_empty


def test_rough_georeferenced_bounds_overlap_nome(contours_extraction):
    """The Nome District Map (Contours) covers roughly the same Cape
    Nome area as Map B1; bounds should overlap our project AOI."""
    from ai_minerals.aoi import NOME_PLACER
    from shapely.geometry import box

    if contours_extraction.segments_4326.is_empty:
        pytest.fail("4326 segments empty")
    minx, miny, maxx, maxy = contours_extraction.segments_4326.bounds
    assert -165.6 <= minx <= -165.3
    assert -165.4 <= maxx <= -165.1
    assert 64.3 <= miny <= 64.6
    assert 64.5 <= maxy <= 64.7

    # Intersection with the AOI box should be non-empty
    aoi_box = box(*NOME_PLACER.bbox)
    assert not contours_extraction.segments_4326.intersection(aoi_box).is_empty


def test_save_writes_metadata_with_elevation_gap_documented(
    contours_extraction, tmp_path,
):
    out = save_extraction(contours_extraction, tmp_path)
    assert out["pixel"].exists()
    assert out["wgs84"].exists()
    assert out["meta"].exists()
    import json
    meta = json.loads(out["meta"].read_text())
    # v1 explicitly documents the elevation-label gap
    assert "elevation_label_gap" in meta
    assert "OCR" in meta["elevation_label_gap"] or "manual" in meta["elevation_label_gap"]
    assert "use_cases_v1" in meta
    assert "use_cases_v2" in meta
