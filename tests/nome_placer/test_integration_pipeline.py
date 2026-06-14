"""End-to-end integration tests for the Nome placer pipeline.

These tests exercise the full data flow:

  bearcub-delivered artifacts -> ai-minerals scoring -> goldbug-shaped output

They use the real bearcub-delivered files, the real IfSAR DEM, and the
real Tuck 1942 source PDFs. The point is to catch contract regressions
when one repo changes a path, a schema, or a CRS without notifying the
others.

Each test starts at a documented integration point and walks the
pipeline through to the next contract boundary. A failure pinpoints
which boundary broke.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box


# ---------------------------------------------------------------------------
# bearcub -> ai-minerals contract
# ---------------------------------------------------------------------------

def test_integration_bearcub_drillholes_match_documented_schema(bearcub_drillholes):
    """The drillholes parquet must carry the column set bearcub agreed to
    in their 2026-06-14 Phase 0 reply. Schema regressions here would
    silently break the supervised model heads in Phase 2."""
    expected_columns = {
        "hole_id", "lon", "lat", "easting_3338", "northing_3338",
        "pay_zone_average_oz_cuyd", "pay_binary", "deposit_class",
        "bedrock_depth_ft", "pay_zone_thickness_ft", "claim_ms",
        "source", "ocr_confidence", "drill_line_id", "drill_year",
        "operator",
    }
    actual = set(bearcub_drillholes.columns)
    missing = expected_columns - actual
    assert not missing, f"bearcub drillholes missing columns: {missing}"


def test_integration_bearcub_drillholes_row_count_band(bearcub_drillholes):
    """bearcub's 2026-06-14 phase0-reply said 104 holes (67 Janin + 37
    Bear Cub). Allow some grow as AGC ingest lands, but flag if the
    count crashes below 50 or jumps unexpectedly without a documented
    AGC ingest."""
    assert 50 <= len(bearcub_drillholes) <= 200, (
        f"drillhole count {len(bearcub_drillholes)} outside expected band; "
        "may indicate an AGC ingest landed without notification"
    )


def test_integration_bearcub_hole_layers_join_to_drillholes(
    bearcub_drillholes, bearcub_hole_layers,
):
    """The hole_layers sidecar must join cleanly on hole_id."""
    dh_ids = set(bearcub_drillholes["hole_id"].astype(str))
    hl_ids = set(bearcub_hole_layers["hole_id"].astype(str))
    overlap = hl_ids & dh_ids
    assert len(overlap) >= 40, (
        f"hole_layers covers only {len(overlap)} drillholes; "
        "expected >= 40 (bearcub said 80 of 104)"
    )


def test_integration_bearcub_claims_carry_family_set(bearcub_claims):
    """Bear Cub MS 1178 + Molasses MS 1179 + Early Bear MS 1180 +
    Honey MS 1181 must be present as polygons. Phase 1 TB validation
    depends on these."""
    ms_set = set(bearcub_claims["MS"].astype(str))
    for ms in ("1178", "1179", "1180", "1181"):
        assert ms in ms_set, f"family claim MS {ms} missing from claims_by_ms"


# ---------------------------------------------------------------------------
# ai-minerals internal: full scoring pipeline
# ---------------------------------------------------------------------------

def test_integration_full_pipeline_produces_composite_score(
    ifsar_dem_aoi_proj, nome_grid_25m,
):
    """Bringing it all together: DEM + flow accumulation + streams ->
    BL + AP + TB + BC + QM -> composite. The composite must be a valid
    per-cell Series at the grid's cell count."""
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
        ifsar_dem_aoi_proj, nome_grid_25m, distance_to_stream_m=d_stream,
    )
    assert len(scores.composite) == nome_grid_25m.n_cells
    assert scores.bc is not None
    assert scores.qm is not None


# ---------------------------------------------------------------------------
# ai-minerals -> bearcub: beach contour loopback artifact
# ---------------------------------------------------------------------------

def test_integration_beach_contour_artifact_exists_for_bearcub():
    """ai-minerals -> bearcub Phase 1 v1 loopback. The contour GeoJSONs
    must exist in bearcub's inbox (or in the local outbox cache if the
    bearcub repo isn't on this machine)."""
    bearcub_inbox = Path("/home/sky/src/learning/bearcub/handoff/inbox/"
                        "2026-06-14-from-ai-minerals-nome-beach-contours/")
    local_artifact = Path("data/derived/nome_placer/beach_contours/")
    found = bearcub_inbox if bearcub_inbox.exists() else local_artifact
    if not found.exists():
        pytest.skip(
            "Neither bearcub inbox nor local derived/ has the beach contour artifact"
        )
    # At least the Third and tuck_high_600 stands should be there
    assert (found / "third_contour_4326.geojson").exists()
    assert (found / "tuck_high_600_contour_4326.geojson").exists()


def test_integration_third_beach_contour_inside_aoi():
    """Third Beach contour polylines must overlap the Nome AOI."""
    from ai_minerals.aoi import NOME_PLACER

    path = Path("data/derived/nome_placer/beach_contours/third_contour_4326.geojson")
    if not path.exists():
        pytest.skip("Third Beach contour not generated; run export script")
    gdf = gpd.read_file(path)
    aoi_box = box(*NOME_PLACER.bbox)
    assert not gdf.geometry.intersection(aoi_box).is_empty.all()


# ---------------------------------------------------------------------------
# Tuck 1942 -> ai-minerals: Map B1 + bedrock contours
# ---------------------------------------------------------------------------

def test_integration_tuck_map_b1_exists_with_rough_georef():
    """Phase 1 v2 deliverable. The Map B1 derived GeoJSON + metadata
    must be present and the metadata must record the rough-affine
    note."""
    derived = Path("data/derived/tuck1942/")
    pixel = derived / "map_b1_dredge_cleanups_pixel.geojson"
    wgs84 = derived / "map_b1_dredge_cleanups_4326.geojson"
    meta = derived / "map_b1_metadata.json"
    for p in (pixel, wgs84, meta):
        if not p.exists():
            pytest.skip(f"{p} not generated; run tuck1942_map_b1.extract()")
    meta_data = json.loads(meta.read_text())
    assert meta_data["n_polygons"] > 100
    assert "rough_affine_note" in meta_data


def test_integration_tuck_bedrock_contours_exist_with_elevation_gap_documented():
    """Phase 1 v3 deliverable. Metadata must explicitly document the
    elevation-label gap and v1/v2 use cases."""
    derived = Path("data/derived/tuck1942/")
    meta = derived / "bedrock_contours_metadata.json"
    if not meta.exists():
        pytest.skip(f"{meta} not generated; run tuck1942_bedrock_contours.extract()")
    meta_data = json.loads(meta.read_text())
    assert "elevation_label_gap" in meta_data
    assert "use_cases_v1" in meta_data
    assert "use_cases_v2" in meta_data
