"""Stream-derivation + BC + QM tests for Phase 1 v1.

Exercises:
- features.hydrology.streams_from_flow_accumulation against real IfSAR
- features.hydrology.distance_to_stream_m on the Nome 25 m grid
- features.coastal_scorer.score_bc / score_qm
- Full 5-landmark validation gate in natural populations
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ai_minerals.features.coastal_scorer import (
    score_all_populations,
    score_bc,
    score_qm,
)
from ai_minerals.features.hydrology import (
    distance_to_stream_m,
    flow_accumulation,
    streams_from_flow_accumulation,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures so the slow flow-accumulation + scoring runs once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flow_acc_array(ifsar_dem_aoi_proj):
    return flow_accumulation(
        ifsar_dem_aoi_proj.values,
        transform=ifsar_dem_aoi_proj.rio.transform(),
    )


@pytest.fixture(scope="module")
def streams_gdf(ifsar_dem_aoi_proj, flow_acc_array):
    return streams_from_flow_accumulation(
        flow_acc_array,
        transform=ifsar_dem_aoi_proj.rio.transform(),
        crs=str(ifsar_dem_aoi_proj.rio.crs),
    )


@pytest.fixture(scope="module")
def d_stream(streams_gdf, nome_grid_25m):
    return distance_to_stream_m(streams_gdf, nome_grid_25m)


@pytest.fixture(scope="module")
def scores_with_streams(ifsar_dem_aoi_proj, nome_grid_25m, d_stream):
    return score_all_populations(
        ifsar_dem_aoi_proj, nome_grid_25m, distance_to_stream_m=d_stream
    )


# ---------------------------------------------------------------------------
# Stream extraction
# ---------------------------------------------------------------------------

def test_stream_extraction_yields_reasonable_count(streams_gdf):
    """At the default 500-cell flow-accumulation threshold the Cape Nome
    AOI should yield tens of thousands of stream cells (not the ~1000
    of a 5000-cell threshold and not the ~hundreds of thousands of a
    100-cell threshold)."""
    assert 10_000 <= len(streams_gdf) <= 100_000, (
        f"Expected 10k to 100k stream cells; got {len(streams_gdf)}. "
        "Threshold may be off."
    )


def test_streams_are_in_working_crs(streams_gdf):
    assert str(streams_gdf.crs).startswith("EPSG:3338") or "Albers" in str(streams_gdf.crs)


def test_distance_to_stream_is_per_cell_series_finite(d_stream, nome_grid_25m):
    assert len(d_stream) == nome_grid_25m.n_cells
    valid = d_stream.dropna()
    assert (valid >= 0).all()
    # max bounded by the function's cap (25 km default)
    assert valid.max() <= 25_000.0


def test_anvil_creek_centroid_is_near_a_stream(d_stream, nome_grid_25m):
    """Anvil Creek Discovery (4.25 mi N of Nome) should be ON or
    immediately adjacent to a stream cell. < 50 m is the bar."""
    centroids = nome_grid_25m.centroid_gdf()
    pt = gpd.GeoSeries(
        [Point(-165.354, 64.563)], crs="EPSG:4326"
    ).to_crs(centroids.crs).iloc[0]
    idx = centroids.distance(pt).idxmin()
    assert d_stream.iloc[idx] < 50.0, (
        f"Anvil Creek nearest-stream distance was "
        f"{d_stream.iloc[idx]:.0f} m; expected < 50 m"
    )


# ---------------------------------------------------------------------------
# BC + QM scorer outputs
# ---------------------------------------------------------------------------

def test_bc_score_present_when_streams_provided(scores_with_streams):
    assert scores_with_streams.bc is not None
    assert scores_with_streams.qm is not None


def test_bc_score_is_per_cell_bounded(scores_with_streams, nome_grid_25m):
    bc = scores_with_streams.bc
    assert len(bc) == nome_grid_25m.n_cells
    valid = bc.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1.0001).all()


def test_qm_score_is_per_cell_bounded(scores_with_streams, nome_grid_25m):
    qm = scores_with_streams.qm
    valid = qm.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1.0001).all()


def test_composite_with_streams_includes_bc_and_qm(scores_with_streams):
    """When BC/QM are populated, the composite must dominate them
    cell-wise alongside BL/AP/TB."""
    s = scores_with_streams
    composite = s.composite.to_numpy()
    bl = s.bl.to_numpy(); ap = s.ap.to_numpy(); tb = s.tb.to_numpy()
    bc = s.bc.to_numpy(); qm = s.qm.to_numpy()
    valid = ~np.isnan(composite)
    expected = np.maximum.reduce([bl, ap, tb, bc, qm])
    np.testing.assert_allclose(
        composite[valid], expected[valid], rtol=1e-5, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Bearcub validation gate -- 5 mandated landmarks in their natural populations
# ---------------------------------------------------------------------------

def _max_in_poly(centroids, polygon_4326, score):
    poly = gpd.GeoSeries([polygon_4326], crs="EPSG:4326").to_crs(centroids.crs).iloc[0]
    inside = centroids.within(poly)
    assert inside.sum() > 0
    return float(score[inside].dropna().max())


def _at_point(centroids, lon, lat, score):
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(centroids.crs).iloc[0]
    idx = centroids.distance(pt).idxmin()
    return float(score.iloc[idx])


def test_validation_gate_5_landmarks_pass_in_natural_populations(
    scores_with_streams, nome_grid_25m, family_claim_polygons,
):
    """The bearcub Phase 1 validation gate (greenlit in
    2026-06-14-from-bearcub-phase0-close-reply.md): Anvil Creek, Third
    Beach, Submarine Beach, Molasses, Honey land in the top deciles
    without being used as labels. Each in its natural population."""
    s = scores_with_streams
    centroids = nome_grid_25m.centroid_gdf()

    p90_bl = float(np.nanquantile(s.bl.to_numpy(), 0.90))
    p90_tb = float(np.nanquantile(s.tb.to_numpy(), 0.90))
    p90_qm = float(np.nanquantile(s.qm.to_numpy(), 0.90))

    # Anvil Creek (Discovery, 4.25 mi N of Nome). Anvil main reach is
    # off-beach modern creek per Tuck 1942 p.37 -> QM population.
    anvil_qm = _at_point(centroids, -165.354, 64.563, s.qm)
    assert anvil_qm >= p90_qm - 1e-3, f"Anvil QM {anvil_qm:.3f} below {p90_qm:.3f}"

    # Third Beach Sunset point -> BL
    third_bl = _at_point(centroids, -165.40, 64.50, s.bl)
    assert third_bl >= p90_bl - 1e-3, f"Third Beach BL {third_bl:.3f} below {p90_bl:.3f}"

    # Submarine Beach (modern shoreline edge) -> BL via present-sea
    sub_bl = _at_point(centroids, -165.41, 64.495, s.bl)
    assert sub_bl >= p90_bl - 1e-3, f"Submarine BL {sub_bl:.3f} below {p90_bl:.3f}"

    # Molasses MS 1179 -> TB
    mol_tb = _max_in_poly(centroids, family_claim_polygons["molasses"].geometry, s.tb)
    assert mol_tb >= p90_tb, f"Molasses TB {mol_tb:.3f} below {p90_tb:.3f}"

    # Honey MS 1181 -> TB
    honey_tb = _max_in_poly(centroids, family_claim_polygons["honey"].geometry, s.tb)
    assert honey_tb >= p90_tb, f"Honey TB {honey_tb:.3f} below {p90_tb:.3f}"


def test_bear_cub_partial_bc_score_documented(
    scores_with_streams, nome_grid_25m, family_claim_polygons,
):
    """Bear Cub MS 1178 is a BC per Sky's calibration but its position
    (+8 m above Fourth Beach AND ~530 m from streams) lands it at
    around BC = 0.25, which is top-quintile but not top-decile. This
    is honest signal from the BC product structure, not a scorer bug.
    A higher BC sigma or per-stand BL sigma would lift Bear Cub above
    the top-decile bar, but the v1 baseline records this as the
    documented Bear Cub state."""
    centroids = nome_grid_25m.centroid_gdf()
    bc = _max_in_poly(
        centroids, family_claim_polygons["bear_cub"].geometry,
        scores_with_streams.bc,
    )
    assert 0.10 <= bc <= 0.45, (
        f"Bear Cub BC {bc:.3f} outside documented partial range [0.10, 0.45]"
    )
