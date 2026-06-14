"""Tests for the Tuck 1942 Map B1 dredge-cleanup polygon adapter.

Exercises the real Map B1 PDF at /tmp/tuck/. Skips if the PDF is not
present (CI environments without the data cache). The test runs the
full pipeline -- render -> colour-threshold -> vectorize -> rough-
georeference -- against the real source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_minerals.data.tuck1942_map_b1 import (
    PIXEL_GCPS_AT_200DPI,
    extract,
    save_extraction,
)


MAP_B1_PDF = Path("/tmp/tuck/Map B1.pdf")


@pytest.fixture(scope="module")
def map_b1_extraction(tmp_path_factory):
    if not MAP_B1_PDF.exists():
        pytest.skip(f"Map B1 PDF not at {MAP_B1_PDF}")
    cache = tmp_path_factory.mktemp("tuck1942_cache")
    return extract(MAP_B1_PDF, cache_dir=cache)


def test_extraction_yields_many_dredge_polygons(map_b1_extraction):
    """Cape Nome's historical dredging area is densely cleaned up; expect
    at least 100 distinct polygons after colour thresholding."""
    assert map_b1_extraction.n_polygons > 100, (
        f"Only {map_b1_extraction.n_polygons} polygons; threshold may be off"
    )


def test_extraction_pixel_geometry_has_positive_area(map_b1_extraction):
    assert map_b1_extraction.total_pixel_area > 1000
    assert not map_b1_extraction.polygons_pixel.is_empty


def test_rough_affine_is_orientation_consistent(map_b1_extraction):
    """Lat should DECREASE with pixel-row (map is north-up; row 0 at
    top of image is the northern edge). Lon should INCREASE with
    pixel-col (col 0 at left is the western edge)."""
    a, b, d, e, *_ = map_b1_extraction.rough_affine_4326
    # a = d(lon)/d(col), e = d(lat)/d(row)
    assert a > 0, f"lon should increase with col; a={a}"
    assert e < 0, f"lat should decrease with row; e={e}"


def test_rough_georeferenced_bounds_overlap_cape_nome(map_b1_extraction):
    """The Cape Nome AOI is (-165.5, 64.45, -165.2, 64.62). The Map B1
    rough-georeferenced bounds should overlap this region."""
    if map_b1_extraction.polygons_4326.is_empty:
        pytest.fail("4326 polygons empty")
    minx, miny, maxx, maxy = map_b1_extraction.polygons_4326.bounds
    # Reasonable Cape Nome area (rough bounds): lon -165.6 to -165.1, lat 64.3 to 64.7
    assert -165.6 <= minx <= -165.3
    assert -165.4 <= maxx <= -165.1
    assert 64.3 <= miny <= 64.6
    assert 64.45 <= maxy <= 64.7


def test_dredge_polygons_overlap_nome_aoi(map_b1_extraction):
    """At least some dredge polygons should fall inside the project's
    Nome AOI. This validates that the rough-georeference is roughly
    centred where we expect."""
    from ai_minerals.aoi import NOME_PLACER
    from shapely.geometry import box

    aoi_box = box(*NOME_PLACER.bbox)
    intersected = map_b1_extraction.polygons_4326.intersection(aoi_box)
    assert not intersected.is_empty, (
        "No dredge polygons inside the Nome AOI; rough-affine may be off"
    )


def test_save_extraction_writes_three_files(map_b1_extraction, tmp_path):
    out = save_extraction(map_b1_extraction, tmp_path)
    assert out["pixel"].exists()
    assert out["wgs84"].exists()
    assert out["meta"].exists()
    # Pixel GeoJSON is the largest because it includes every vertex.
    assert out["pixel"].stat().st_size > 100_000
    # Metadata records the GCP set actually used.
    import json
    meta = json.loads(out["meta"].read_text())
    assert meta["n_polygons"] == map_b1_extraction.n_polygons
    assert len(meta["gcps_pixel_to_4326"]) == len(PIXEL_GCPS_AT_200DPI)
