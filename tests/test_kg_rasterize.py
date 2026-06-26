"""Unit tests for the extent-aware, capped KG rasterizer (ai_minerals.kg_rasterize).

These prove the F2 over-attribution guards on synthetic geometry, independent of
the fossick export, mirroring the run plan's F2 verification criteria:

  * a point ground lands in exactly one cell;
  * a claim polygon covers several cells;
  * an area footprint with only a centroid spreads across a disc, not one cell;
  * a district-extent ground is NOT pinned to any cell (ecological-fallacy guard);
  * where grounds stack on a cell, the per-cell cap keeps the single highest-
    scoring ground and the capped coverage never exceeds one, while the uncapped
    count still records the pile-up.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, Polygon

from ai_minerals.kg_rasterize import (
    GridTemplate,
    block_xy,
    build_kg_stack,
    ground_points_by_extent,
    loo_spatial_features,
    sample_cell_centroids,
    _cells_for_geometry,
)


def _template(res_m: float = 25.0, n: int = 40) -> GridTemplate:
    """A small north-up grid whose origin is (0, n*res) in a metres CRS."""
    transform = rasterio.transform.from_origin(0.0, n * res_m, res_m, res_m)
    return GridTemplate(transform=transform, shape=(n, n), crs="EPSG:3338")


def _entities(records: list[dict]) -> gpd.GeoDataFrame:
    """Build a minimal entities frame the stack builder accepts."""
    gdf = gpd.GeoDataFrame(records, geometry=[r["geometry"] for r in records], crs="EPSG:3338")
    for col in ("source_grade_numeric", "mean_confidence", "n_geoterm_families"):
        if col not in gdf:
            gdf[col] = 1.0
    return gdf


def test_point_lands_in_single_cell():
    tpl = _template()
    cells = _cells_for_geometry(Point(130.0, 130.0), "point", tpl, area_buffer_m=200.0)
    assert cells.shape == (1, 2)


def test_claim_polygon_covers_many_cells():
    tpl = _template()  # 25 m cells
    poly = Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])  # 200 m square
    cells = _cells_for_geometry(poly, "claim_polygon", tpl, area_buffer_m=200.0)
    assert len(cells) > 1
    # ~ (200/25)^2 = 64 cells, allow for all_touched edge inclusion
    assert 49 <= len(cells) <= 100


def test_area_footprint_point_spreads_not_pinned():
    tpl = _template()
    pinned = _cells_for_geometry(Point(500, 500), "point", tpl, area_buffer_m=200.0)
    spread = _cells_for_geometry(Point(500, 500), "area_footprint", tpl, area_buffer_m=200.0)
    assert len(pinned) == 1
    assert len(spread) > 1  # the over-attribution guard: never one cell


def test_district_not_pinned_to_cells():
    tpl = _template()
    cells = _cells_for_geometry(Point(300, 300), "district", tpl, area_buffer_m=200.0)
    assert len(cells) == 0  # regional covariate, never attributed to a cell


def test_per_cell_cap_keeps_highest_score():
    tpl = _template()
    # two points in the SAME cell, different source grade -> the A-grade wins
    ents = _entities([
        {"geometry": Point(312, 312), "source_grade_numeric": 2.0, "mean_confidence": 0.9,
         "attr_extent": "point", "score": 2.0 * 0.9},
        {"geometry": Point(313, 313), "source_grade_numeric": 5.0, "mean_confidence": 0.9,
         "attr_extent": "point", "score": 5.0 * 0.9},
    ])
    stack = build_kg_stack(ents, tpl, content_columns=["source_grade_numeric"])
    cell = stack.bands["source_grade_numeric"]
    occupied = cell[cell > 0]
    assert occupied.max() == 5.0           # higher-score ground wins the cell
    assert int(stack.capped_coverage.max()) == 1   # cap: at most one per cell
    assert int(stack.raw_count.max()) == 2         # but the pile-up is recorded


def test_sample_fill_outside_coverage():
    tpl = _template()
    ents = _entities([
        {"geometry": Point(100, 100), "source_grade_numeric": 5.0, "mean_confidence": 1.0,
         "attr_extent": "point", "score": 5.0},
    ])
    stack = build_kg_stack(ents, tpl, content_columns=["source_grade_numeric"], fill_value=0.0)
    # a point far from the single ground samples the fill value
    far = stack.sample_at(np.array([900.0]), np.array([100.0]))
    assert float(far["source_grade_numeric"].iloc[0]) == 0.0
    # spatial distance field is finite and positive there
    assert float(far["kg_dist_occurrence_m"].iloc[0]) > 0.0


# --- leak-free, fold-aware spatial fields (kg_loo) -------------------------- #
def test_block_xy_matches_grid_origin():
    coords = np.array([[0.0, 0.0], [150.0, 0.0], [0.0, 150.0]])
    bx, by = block_xy(coords, 0.0, 0.0, 100.0)
    assert list(bx) == [0, 1, 0]
    assert list(by) == [0, 0, 1]


def test_loo_excludes_own_cell_occurrence():
    tpl = _template()  # 25 m cells
    ents = _entities([  # two occurrences 200 m apart
        {"geometry": Point(100, 100), "attr_extent": "point", "score": 1.0},
        {"geometry": Point(300, 100), "attr_extent": "point", "score": 1.0},
    ])
    g = ground_points_by_extent(ents, tpl)
    assert len(g["occ_xy"]) == 2 and len(g["claim_xy"]) == 0
    samp_xy, samp_rc = sample_cell_centroids(np.array([[100.0, 100.0]]), tpl)
    feats = loo_spatial_features(samp_xy, samp_rc, g["occ_xy"], g["occ_rc"],
                                 g["claim_xy"], g["claim_rc"])
    d = float(feats["kg_dist_occurrence_m"].iloc[0])
    assert d > 1.0            # NOT zero: the cell's own record is left out
    assert abs(d - 200.0) < 26.0   # distance to the OTHER occurrence (cell grain)


def test_loo_density_excludes_own_and_counts_within_radius():
    tpl = _template()
    ents = _entities([
        {"geometry": Point(100, 100), "attr_extent": "claim_polygon", "score": 1.0},   # own cell
        {"geometry": Point(130, 100), "attr_extent": "area_footprint", "score": 1.0},   # ~25 m away
        {"geometry": Point(900, 900), "attr_extent": "claim_polygon", "score": 1.0},    # far
    ])
    g = ground_points_by_extent(ents, tpl)
    samp_xy, samp_rc = sample_cell_centroids(np.array([[100.0, 100.0]]), tpl)
    feats = loo_spatial_features(samp_xy, samp_rc, g["occ_xy"], g["occ_rc"],
                                 g["claim_xy"], g["claim_rc"], claim_radius_m=400.0)
    assert float(feats["kg_claim_density"].iloc[0]) == 1.0   # own-cell claim not counted
    assert float(feats["kg_dist_claim_m"].iloc[0]) > 1.0     # nearest OTHER claim, not 0


def test_loo_no_visible_grounds_uses_fill():
    tpl = _template()
    samp_xy, samp_rc = sample_cell_centroids(np.array([[100.0, 100.0]]), tpl)
    empty = np.empty((0, 2))
    feats = loo_spatial_features(samp_xy, samp_rc, empty, np.empty((0, 2), int),
                                 empty, np.empty((0, 2), int), fill_dist=1.0e6)
    assert float(feats["kg_dist_occurrence_m"].iloc[0]) == 1.0e6
    assert float(feats["kg_claim_density"].iloc[0]) == 0.0
