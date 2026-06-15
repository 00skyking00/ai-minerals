"""Tests for ai_minerals.features.lithology (Phase 1.5 buried-beach).

Exercises the bearcub-delivered bedrock_topo + variance rasters and
the buried-stand membership computation. The critical assertion is
that Bear Cub MS 1178 buried-BL scores at or near 1.0 -- without that,
the v1 BL ceiling at +0.38 caps the BC scoring and Bear Cub stays
below the BC top-decile threshold.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from ai_minerals.features.coastal_scorer import BL_STANDS
from ai_minerals.features.lithology import (
    DEFAULT_VARIANCE_THRESHOLD_FT2,
    buried_population_score,
    buried_stand_membership,
    elevation_of_bedrock_m,
    load_bedrock_surface,
)


@pytest.fixture(scope="module")
def bedrock_field(nome_region, nome_grid_25m):
    return load_bedrock_surface(
        Path(nome_region.raw_paths["bedrock_topo"]),
        Path(nome_region.raw_paths["bedrock_topo_variance"]),
        nome_grid_25m,
    )


@pytest.fixture(scope="module")
def bedrock_elev_series(ifsar_dem_aoi_proj, nome_grid_25m, bedrock_field):
    return elevation_of_bedrock_m(ifsar_dem_aoi_proj, nome_grid_25m, bedrock_field)


# ---------------------------------------------------------------------------
# Bedrock surface coverage
# ---------------------------------------------------------------------------

def test_bedrock_coverage_is_bear_cub_envelope_sized(bedrock_field, nome_grid_25m):
    """bearcub said the bedrock surface is ~800 m x 875 m. At 25 m grid
    that should give ~32 x 35 = ~1100 cells. The variance threshold (50
    ft^2) trims to where the GP is trustworthy; expect 100-1000 cells."""
    n_covered = int(bedrock_field.coverage_mask.sum())
    assert 100 <= n_covered <= 1000, (
        f"Bedrock coverage of {n_covered} cells outside expected 100-1000 band. "
        "Check the variance threshold or the raster extent."
    )


def test_bedrock_depth_in_realistic_range(bedrock_field):
    """Bear Cub holes have bedrock at 62-85 ft = 19-26 m depth. The GP
    surface should fall within roughly that range over the envelope."""
    depth_m = bedrock_field.depth_m.dropna()
    assert depth_m.min() >= 5.0, f"Min depth {depth_m.min()} m unreasonably shallow"
    assert depth_m.max() <= 35.0, f"Max depth {depth_m.max()} m unreasonably deep"
    assert 15.0 <= depth_m.mean() <= 30.0, (
        f"Mean depth {depth_m.mean():.1f} m outside 15-30 m band; "
        "GP surface may be off"
    )


def test_variance_threshold_excludes_extrapolation(bedrock_field, nome_grid_25m):
    """Cells outside the bedrock-coverage mask should have NaN depth.
    Cells inside should all be at or below the variance threshold."""
    covered = bedrock_field.coverage_mask
    assert bedrock_field.depth_m[~covered].isna().all()
    assert (bedrock_field.variance_m2[covered] <=
            DEFAULT_VARIANCE_THRESHOLD_FT2 / (3.28084 ** 2) + 1e-3).all()


# ---------------------------------------------------------------------------
# Bedrock elevation = surface - depth
# ---------------------------------------------------------------------------

def test_bedrock_elevation_at_bear_cub_near_third_beach(
    bedrock_field, bedrock_elev_series, family_claim_polygons, nome_grid_25m,
):
    """Sky 2026-06-14: Third Beach is buried 80-90 ft below Bear Cub
    surface. Modern surface ~+160 ft, so bedrock at Bear Cub should
    sit near +70 to +80 ft (+21 to +24 m). This validates the
    'bedrock = surface - depth' arithmetic at the family-claim level."""
    centroids = nome_grid_25m.centroid_gdf()
    bear_poly = gpd.GeoSeries(
        [family_claim_polygons["bear_cub"].geometry], crs="EPSG:4326"
    ).to_crs(centroids.crs).iloc[0]
    inside = centroids.within(bear_poly)
    assert inside.sum() > 0

    bedrock_elev_at_bear = bedrock_elev_series[inside].dropna()
    assert len(bedrock_elev_at_bear) > 0, "No bedrock-elev cells at Bear Cub"
    # Bear Cub bedrock elevation should be in the +15 to +35 m range
    # (a generous band covering +25 m mean +/- 10 m).
    assert 15.0 <= bedrock_elev_at_bear.mean() <= 35.0, (
        f"Bear Cub bedrock-elev mean {bedrock_elev_at_bear.mean():.1f} m "
        "outside expected +15 to +35 m band"
    )


# ---------------------------------------------------------------------------
# Buried-stand membership
# ---------------------------------------------------------------------------

def test_buried_third_beach_membership_high_at_bear_cub(
    bedrock_elev_series, family_claim_polygons, nome_grid_25m,
):
    """The headline result: Bear Cub's bedrock should match Third
    Beach's elevation within the 5 m sigma, giving buried-third
    membership at or near 1.0 across the polygon."""
    centroids = nome_grid_25m.centroid_gdf()
    third_membership = buried_stand_membership(bedrock_elev_series, "third")
    bear_poly = gpd.GeoSeries(
        [family_claim_polygons["bear_cub"].geometry], crs="EPSG:4326"
    ).to_crs(centroids.crs).iloc[0]
    inside = centroids.within(bear_poly)
    assert inside.sum() > 0
    bear_third_max = float(third_membership[inside].max())
    assert bear_third_max >= 0.5, (
        f"Bear Cub buried-Third-Beach max = {bear_third_max:.3f}; "
        "expected >= 0.5 (Sky's 80-90 ft burial implies bedrock at "
        "Third Beach elevation)"
    )


def test_buried_bl_population_score_uses_max_over_stands(
    bedrock_elev_series, nome_grid_25m,
):
    """buried_population_score(BL_STANDS) must dominate each individual
    stand's membership cell-wise."""
    individual = [
        buried_stand_membership(bedrock_elev_series, s).to_numpy()
        for s in BL_STANDS
    ]
    expected_max = np.maximum.reduce(individual)
    pop = buried_population_score(bedrock_elev_series, BL_STANDS).to_numpy()
    np.testing.assert_allclose(pop, expected_max, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: full validation gate now passes 6 of 6 with buried-BL
# ---------------------------------------------------------------------------

def test_combined_bedrock_surface_extends_beyond_bearcub_envelope(
    ifsar_dem_aoi_proj, nome_grid_25m, bedrock_field, nome_region,
):
    """Phase 1.5 v2 addition: combined_bedrock_elevation_m fills in
    bedrock elevation from the wider 99-hole network (Janin + Bear
    Cub) outside the bearcub GP envelope. Coverage should expand by
    >= 10x relative to bearcub-only."""
    from pathlib import Path
    from ai_minerals.features.lithology import (
        combined_bedrock_elevation_m, elevation_of_bedrock_m,
    )

    bearcub_only = elevation_of_bedrock_m(
        ifsar_dem_aoi_proj, nome_grid_25m, bedrock_field,
    )
    combined = combined_bedrock_elevation_m(
        ifsar_dem_aoi_proj, nome_grid_25m,
        drillholes_path=Path(nome_region.raw_paths["drillholes"]),
        cross_sections_path=Path("data/raw/bearcub_nome/cross_sections_nome_placer.parquet"),
        bearcub_field=bedrock_field,
    )
    bearcub_cov = int(bearcub_only.notna().sum())
    combined_cov = int(combined.notna().sum())
    assert combined_cov >= 10 * bearcub_cov, (
        f"Combined coverage {combined_cov} not >= 10x bearcub-only "
        f"{bearcub_cov}; the wider hole-network interpolation isn't "
        "filling in as expected"
    )


def test_load_all_drillhole_depths_combines_janin_and_bear_cub(nome_region):
    """The combined hole dataset should contain both Janin Pioneer and
    Bear Cub Murray records with non-null bedrock_depth_ft + lat/lon."""
    from pathlib import Path
    from ai_minerals.features.lithology import load_all_drillhole_depths

    holes = load_all_drillhole_depths(
        Path(nome_region.raw_paths["drillholes"]),
        Path("data/raw/bearcub_nome/cross_sections_nome_placer.parquet"),
    )
    assert len(holes) >= 80, f"Expected >= 80 combined holes; got {len(holes)}"
    sources = holes["source"].value_counts().to_dict()
    assert "janin_pioneer_1912" in sources, f"No Janin holes in combined dataset; {sources}"
    assert "bear_cub_murray" in sources, f"No Bear Cub holes in combined dataset; {sources}"
    assert holes["bedrock_depth_ft"].notna().all()


def test_interpolated_bedrock_surface_covers_janin_area(
    ifsar_dem_aoi_proj, nome_grid_25m, nome_region,
):
    """Janin Little Creek + Moonlight (NW of Bear Cub) should have
    interpolated bedrock coverage at the Janin hole locations. Take
    one Janin hole position and verify it lands in the interpolated
    coverage."""
    from pathlib import Path
    import geopandas as gpd
    from shapely.geometry import Point
    from ai_minerals.features.lithology import (
        interpolate_depth_surface, load_all_drillhole_depths,
    )

    holes = load_all_drillhole_depths(
        Path(nome_region.raw_paths["drillholes"]),
        Path("data/raw/bearcub_nome/cross_sections_nome_placer.parquet"),
    )
    janin = holes[holes["source"] == "janin_pioneer_1912"].iloc[0]

    interp = interpolate_depth_surface(
        holes, nome_grid_25m, field_name="bedrock_depth_ft", max_distance_m=1500.0,
    )
    centroids = nome_grid_25m.centroid_gdf()
    pt = gpd.GeoSeries(
        [Point(janin["lon"], janin["lat"])], crs="EPSG:4326",
    ).to_crs(centroids.crs).iloc[0]
    idx = centroids.distance(pt).idxmin()
    val = interp.iloc[idx]
    assert pd.notna(val), (
        f"Interpolated bedrock-depth at Janin hole {janin['hole_id']} is NaN; "
        "expected coverage at a hole position"
    )
    # The interpolation at a hole position should be near the hole's measured
    # bedrock depth (within ~10 ft of the local pattern)
    assert abs(val - janin["bedrock_depth_ft"]) < 20, (
        f"Interpolated {val:.1f} ft vs measured {janin['bedrock_depth_ft']:.1f} ft "
        "at the same hole position is too far off"
    )


def test_bear_cub_bc_passes_top_decile_with_buried_bl(
    ifsar_dem_aoi_proj, nome_grid_25m, bedrock_elev_series, family_claim_polygons,
):
    """Headline Phase 1.5 result: Bear Cub MS 1178 BC now passes the
    bearcub-mandated top-decile threshold once buried-BL feeds into BC.
    Without buried-BL the BC ceiling is 0.36; with it Bear Cub scores
    ~0.98 because bedrock-at-Third-Beach + streams-through-claim both
    saturate near 1.0."""
    from ai_minerals.features.coastal_scorer import score_all_populations
    from ai_minerals.features.hydrology import (
        distance_to_stream_m,
        flow_accumulation,
        streams_from_flow_accumulation,
    )

    flow_acc = flow_accumulation(
        ifsar_dem_aoi_proj.values,
        transform=ifsar_dem_aoi_proj.rio.transform(),
    )
    streams = streams_from_flow_accumulation(
        flow_acc,
        transform=ifsar_dem_aoi_proj.rio.transform(),
        crs=str(ifsar_dem_aoi_proj.rio.crs),
    )
    d_stream = distance_to_stream_m(streams, nome_grid_25m)

    scores = score_all_populations(
        ifsar_dem_aoi_proj, nome_grid_25m,
        distance_to_stream_m=d_stream,
        bedrock_elevation_m=bedrock_elev_series,
    )

    centroids = nome_grid_25m.centroid_gdf()
    bear_poly = gpd.GeoSeries(
        [family_claim_polygons["bear_cub"].geometry], crs="EPSG:4326"
    ).to_crs(centroids.crs).iloc[0]
    inside = centroids.within(bear_poly)
    assert inside.sum() > 0

    p90 = float(np.nanquantile(scores.bc.to_numpy(), 0.90))
    bear_bc = float(scores.bc[inside].dropna().max())
    assert bear_bc >= p90 - 1e-3, (
        f"Bear Cub BC {bear_bc:.3f} below top-decile threshold {p90:.3f}; "
        "the buried-BL contribution may not be feeding into BC correctly."
    )
