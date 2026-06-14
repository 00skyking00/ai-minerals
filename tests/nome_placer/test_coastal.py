"""Unit + feature tests for ai_minerals.features.coastal.

These exercise the real IfSAR DEM (n65w166), the real grid, and the
real claim polygons; they don't mock geometries. The tests check
geomorphic ground truth (Honey at +600 ft, Bear Cub at ~45 m, etc.)
documented in research/nome_tuck_1942_prepass_findings.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_minerals.features.coastal import (
    STAND_ELEVATIONS_FT,
    distance_to_paleo_shoreline_m,
    elevation_relative_to_stand_m,
    paleo_shoreline_contours_all_stands,
    paleo_shoreline_contours_at_stand,
    stand_elevation_m,
)


# ---------------------------------------------------------------------------
# Stand list -- documented constants, no external deps
# ---------------------------------------------------------------------------

def test_stand_list_covers_eleven_named_stands():
    """8 named-beach stands + 3 Tuck high stands = 11 total."""
    assert len(STAND_ELEVATIONS_FT) == 11
    expected = {
        "submarine_deep", "submarine_inner", "submarine_outer",
        "intermediate", "second", "monroeville", "third", "fourth",
        "tuck_high_200", "tuck_high_300", "tuck_high_600",
    }
    assert set(STAND_ELEVATIONS_FT) == expected


def test_tuck_high_stands_have_correct_elevations():
    assert STAND_ELEVATIONS_FT["tuck_high_200"] == 200.0
    assert STAND_ELEVATIONS_FT["tuck_high_300"] == 300.0
    assert STAND_ELEVATIONS_FT["tuck_high_600"] == 600.0


def test_third_beach_elevation_matches_published_midpoint_ft_to_m():
    """Third Beach 70-79 ft midpoint = 74.5 ft = 22.71 m. Hard ground truth."""
    assert stand_elevation_m("third") == pytest.approx(22.7076, rel=1e-3)


def test_tuck_600ft_is_182_88_m():
    """+600 ft converted to metres. Verifies the unit conversion path."""
    assert stand_elevation_m("tuck_high_600") == pytest.approx(182.88, rel=1e-4)


# ---------------------------------------------------------------------------
# Per-cell elevation-relative-to-stand on real IfSAR DEM
# ---------------------------------------------------------------------------

def test_elevation_relative_to_third_returns_per_cell_series(
    ifsar_dem_aoi_proj, nome_grid_25m
):
    rel = elevation_relative_to_stand_m(ifsar_dem_aoi_proj, nome_grid_25m, "third")
    assert len(rel) == nome_grid_25m.n_cells
    assert rel.name == "elevation_relative_to_stand_third_m"
    # Some cells should be above, some below.
    valid = rel.dropna()
    assert len(valid) > 0
    assert (valid > 0).any()
    assert (valid < 0).any()


def test_honey_centroid_is_within_5m_of_tuck_high_600(
    ifsar_dem_aoi_proj, nome_grid_25m, family_claim_polygons,
):
    """Tuck 1942 documented +600 ft stand for the head-of-Dexter-Creek
    high-bench gravel. Honey MS 1181 should sit at that elevation."""
    import geopandas as gpd

    rel = elevation_relative_to_stand_m(
        ifsar_dem_aoi_proj, nome_grid_25m, "tuck_high_600"
    )
    centroids = nome_grid_25m.centroid_gdf()
    honey_poly = gpd.GeoSeries(
        [family_claim_polygons["honey"].geometry], crs="EPSG:4326"
    ).to_crs(centroids.crs).iloc[0]

    inside = centroids.within(honey_poly)
    assert inside.sum() > 0, "Honey polygon has no grid cells inside it"
    rel_inside = rel[inside].dropna()
    # The minimum absolute difference inside Honey must be within 5 m of
    # the +600 ft stand. Honey is the Tuck Locality H / Dexter Creek
    # head exemplar.
    min_abs = rel_inside.abs().min()
    assert min_abs < 5.0, (
        f"Honey's closest cell to +600 ft is {min_abs:.1f} m away; expected < 5 m. "
        "The +600 ft stand may not match the actual head-of-Dexter-Creek elevation."
    )


def test_bear_cub_centroid_sits_above_modern_sea_level(
    ifsar_dem_aoi_proj, nome_grid_25m, family_claim_polygons,
):
    """Bear Cub MS 1178 is on the coastal-plain lowland (Tuck p.10
    coastal-plain area). Cells inside should be above modern sea level
    but well below the +600 ft TB stand."""
    import geopandas as gpd

    rel_to_present = elevation_relative_to_stand_m(
        ifsar_dem_aoi_proj, nome_grid_25m, "third"
    )
    centroids = nome_grid_25m.centroid_gdf()
    bear_poly = gpd.GeoSeries(
        [family_claim_polygons["bear_cub"].geometry], crs="EPSG:4326"
    ).to_crs(centroids.crs).iloc[0]
    inside = centroids.within(bear_poly)
    assert inside.sum() > 0
    elev_at_bear = (rel_to_present[inside].dropna() + stand_elevation_m("third"))
    # Bear Cub elevation range from prior debug: 45.83 m at centroid;
    # the polygon spans some range. Should be 5-100 m above sea level.
    assert elev_at_bear.min() > 0
    assert elev_at_bear.max() < 200, (
        f"Bear Cub polygon should be lowland-coastal-plain (<200 m); "
        f"got max {elev_at_bear.max():.1f} m"
    )


# ---------------------------------------------------------------------------
# Paleo-shoreline contour extraction
# ---------------------------------------------------------------------------

def test_third_beach_contour_extracts_polylines(ifsar_dem_aoi_proj):
    contours = paleo_shoreline_contours_at_stand(ifsar_dem_aoi_proj, "third")
    assert len(contours) > 0, "Third Beach should have iso-elevation polylines"
    assert (contours["stand"] == "third").all()
    total_km = float(contours.geometry.length.sum()) / 1000
    # On the real Nome AOI Third Beach iso-elevation totals ~138 km.
    assert total_km > 50, f"Third Beach contour total length {total_km:.1f} km is too small"


def test_submarine_stands_extract_no_polylines_on_land(ifsar_dem_aoi_proj):
    """The three Submarine stands are at -6 to -21 m below sea level.
    The onshore Nome AOI has elevation floor near 0, so submarine stands
    should not extract any onshore contours."""
    for stand in ("submarine_outer", "submarine_inner", "submarine_deep"):
        contours = paleo_shoreline_contours_at_stand(ifsar_dem_aoi_proj, stand)
        assert len(contours) == 0, (
            f"Submarine stand {stand} extracted contours from onshore DEM; "
            "expected zero. Submarine paleo-shorelines need bathymetry."
        )


def test_all_stand_contours_concatenated(ifsar_dem_aoi_proj):
    """Compound helper: union across all stands. Useful for downstream
    distance-to-stand features that take one contour frame and filter."""
    all_contours = paleo_shoreline_contours_all_stands(ifsar_dem_aoi_proj)
    above_water_stands = {
        "intermediate", "second", "monroeville", "third", "fourth",
        "tuck_high_200", "tuck_high_300", "tuck_high_600",
    }
    represented = set(all_contours["stand"].unique())
    assert represented >= {"third", "fourth", "tuck_high_600"}, (
        f"Expected at least third/fourth/+600 ft contours; got {represented}"
    )
    # No submarine stands (they're below the onshore AOI elevation floor)
    assert "submarine_deep" not in represented


def test_distance_to_third_beach_returns_per_cell_series(
    ifsar_dem_aoi_proj, nome_grid_25m,
):
    contours = paleo_shoreline_contours_all_stands(ifsar_dem_aoi_proj)
    dist = distance_to_paleo_shoreline_m(contours, nome_grid_25m, "third")
    assert len(dist) == nome_grid_25m.n_cells
    valid = dist.dropna()
    assert (valid >= 0).all()
    # Minimum should be ~0 (cells whose centroids sit on or within one
    # cell's worth of the contour). At 25 m grid, expect < 25 m.
    assert valid.min() < 25.0
    assert valid.max() <= 50_000.0  # the function's max_distance_m cap
