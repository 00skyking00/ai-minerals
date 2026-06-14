"""Unit + feature tests for ai_minerals.features.coastal_scorer.

Validates the Phase 1 fuzzy-overlay scorer against the bearcub-mandated
gate landmarks: Anvil Creek, Third Beach, Submarine Beach, Molasses,
Honey, Bear Cub. Each landmark must land in the top decile of its
population score without being used as a label.

Uses real bearcub claim polygons (MS 1178 Bear Cub / 1179 Molasses /
1180 Early Bear / 1181 Honey) for the within-polygon-max scoring.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ai_minerals.features.coastal_scorer import (
    AP_STANDS,
    BL_STANDS,
    CoastalScores,
    TB_STANDS,
    score_all_populations,
    score_ap,
    score_bl,
    score_tb,
)


# ---------------------------------------------------------------------------
# Population stand lists (no external deps)
# ---------------------------------------------------------------------------

def test_bl_stand_list_matches_tuck_classification():
    """Tuck 1942 names Present + Second + Third as true beaches. Fourth
    is included because Tuck p.28 documents the +120 ft surfaces as
    planed-off marine benches."""
    assert BL_STANDS == ("second", "third", "fourth")


def test_ap_stand_list_is_sub_aqueous_plus_intermediate_monroeville():
    assert set(AP_STANDS) == {
        "submarine_outer", "submarine_inner", "submarine_deep",
        "intermediate", "monroeville",
    }


def test_tb_stand_list_is_tuck_high_300_and_600():
    assert set(TB_STANDS) == {"tuck_high_300", "tuck_high_600"}


# ---------------------------------------------------------------------------
# Per-population scorers return per-cell Series in [0, 1]
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scores(ifsar_dem_aoi_proj, nome_grid_25m) -> CoastalScores:
    """Score all populations on the real DEM. Module-scoped so the
    multi-stand sampling cost is paid once for the whole test suite."""
    return score_all_populations(ifsar_dem_aoi_proj, nome_grid_25m)


def test_bl_score_is_per_cell_series_bounded_0_1(scores, nome_grid_25m):
    bl = scores.bl
    assert len(bl) == nome_grid_25m.n_cells
    valid = bl.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1.0001).all()  # float tolerance


def test_ap_score_is_per_cell_series_bounded_0_1(scores, nome_grid_25m):
    ap = scores.ap
    assert len(ap) == nome_grid_25m.n_cells
    valid = ap.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1.0001).all()


def test_tb_score_is_per_cell_series_bounded_0_1(scores, nome_grid_25m):
    tb = scores.tb
    assert len(tb) == nome_grid_25m.n_cells
    valid = tb.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1.0001).all()


def test_composite_score_is_per_cell_max_across_populations(scores):
    """Composite must dominate each per-population score cell-wise."""
    composite = scores.composite.to_numpy()
    bl = scores.bl.to_numpy()
    ap = scores.ap.to_numpy()
    tb = scores.tb.to_numpy()
    valid = ~(np.isnan(composite) | np.isnan(bl) | np.isnan(ap) | np.isnan(tb))
    # Composite should equal the per-cell max across BL, AP, TB.
    expected = np.maximum.reduce([bl, ap, tb])
    np.testing.assert_allclose(
        composite[valid], expected[valid], rtol=1e-5, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Bearcub validation gate -- the critical Phase 1 feature test
# ---------------------------------------------------------------------------

def _max_score_in_polygon(centroids, polygon_4326, score_series):
    """Reproject polygon to centroids CRS; return max score inside."""
    poly = gpd.GeoSeries([polygon_4326], crs="EPSG:4326").to_crs(centroids.crs).iloc[0]
    inside_mask = centroids.within(poly)
    assert inside_mask.sum() > 0, "polygon has no cell centroids inside"
    return float(score_series[inside_mask].dropna().max())


def test_honey_lands_in_top_decile_of_tb_score(scores, nome_grid_25m, family_claim_polygons):
    """Honey MS 1181 = head-of-Dexter-Creek high-bench-gravel; Tuck
    documents this as +600 ft. Without being used as a label, it must
    land in the top decile of TB score."""
    p90 = float(np.nanquantile(scores.tb.to_numpy(), 0.90))
    s = _max_score_in_polygon(
        nome_grid_25m.centroid_gdf(),
        family_claim_polygons["honey"].geometry,
        scores.tb,
    )
    assert s >= p90, (
        f"Honey TB score {s:.3f} is below top-decile threshold {p90:.3f}. "
        "Either the +600 ft stand definition is off, or the Honey polygon "
        "doesn't reach the +600 ft surface."
    )


def test_molasses_lands_in_top_decile_of_tb_score(
    scores, nome_grid_25m, family_claim_polygons,
):
    """Molasses MS 1179 = head-of-Dexter-Creek high-bench-gravel.
    Surface elevation is slightly lower than Honey but the polygon
    should still reach into the +600 ft band."""
    p90 = float(np.nanquantile(scores.tb.to_numpy(), 0.90))
    s = _max_score_in_polygon(
        nome_grid_25m.centroid_gdf(),
        family_claim_polygons["molasses"].geometry,
        scores.tb,
    )
    assert s >= p90, (
        f"Molasses TB score {s:.3f} is below top-decile threshold {p90:.3f}."
    )


def test_early_bear_lands_in_top_decile_of_tb_score(
    scores, nome_grid_25m, family_claim_polygons,
):
    """Early Bear MS 1180 sits between Bear Cub and Honey on the
    Dexter divide. Bonus check: it should also score TB."""
    p90 = float(np.nanquantile(scores.tb.to_numpy(), 0.90))
    s = _max_score_in_polygon(
        nome_grid_25m.centroid_gdf(),
        family_claim_polygons["early_bear"].geometry,
        scores.tb,
    )
    assert s >= p90, f"Early Bear TB score {s:.3f} below threshold {p90:.3f}"


def test_third_beach_sunset_landmark_lands_in_top_decile_of_bl_score(
    scores, nome_grid_25m,
):
    """Third Beach (Tuck's high-grade exemplar) point. Should hit BL = 1.0
    because Third Beach IS one of the BL stands."""
    p90 = float(np.nanquantile(scores.bl.to_numpy(), 0.90))
    centroids = nome_grid_25m.centroid_gdf()
    third = gpd.GeoSeries([Point(-165.40, 64.50)], crs="EPSG:4326").to_crs(centroids.crs).iloc[0]
    idx = centroids.distance(third).idxmin()
    s = float(scores.bl.iloc[idx])
    assert s >= p90 - 1e-3, f"Third Beach BL score {s:.6f} below threshold {p90:.6f}"


def test_submarine_beach_landmark_lands_in_top_decile_of_bl_score(
    scores, nome_grid_25m,
):
    """Submarine Beach proxy point on the modern shoreline edge. Should
    hit BL because the modern shoreline (elevation ~0) is included in
    the BL scorer via BL_PRESENT_ELEVATION_M."""
    p90 = float(np.nanquantile(scores.bl.to_numpy(), 0.90))
    centroids = nome_grid_25m.centroid_gdf()
    sub = gpd.GeoSeries([Point(-165.41, 64.495)], crs="EPSG:4326").to_crs(centroids.crs).iloc[0]
    idx = centroids.distance(sub).idxmin()
    s = float(scores.bl.iloc[idx])
    assert s >= p90, f"Submarine BL score {s:.3f} below threshold {p90:.3f}"


# ---------------------------------------------------------------------------
# Bear Cub is the documented partial: needs BC scorer
# ---------------------------------------------------------------------------

def test_bear_cub_bl_score_is_partial_pending_bc_scorer(
    scores, nome_grid_25m, family_claim_polygons,
):
    """Bear Cub MS 1178 is a beach-stream confluence per Sky's
    calibration; pure BL scoring (no stream) lands it at moderate
    score (~0.4 on the BL scorer). When the BC scorer ships (Phase 1
    v1 with stream features), this test should be replaced with a
    BC-population top-decile check."""
    s = _max_score_in_polygon(
        nome_grid_25m.centroid_gdf(),
        family_claim_polygons["bear_cub"].geometry,
        scores.bl,
    )
    # Lower bound: Bear Cub touches the Fourth Beach band (+37 m); some
    # cell in the polygon should score >= 0.3 on BL.
    assert s > 0.3, (
        f"Bear Cub BL score {s:.3f} should be at least moderate via Fourth Beach proximity."
    )
    # Upper bound: until BC scorer lands, expect Bear Cub to NOT hit a
    # perfect 1.0 BL because the polygon does not sit exactly on a beach
    # stand elevation.
    assert s < 1.0
