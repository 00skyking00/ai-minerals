"""Tests for the areal lode-distance feature and the aux-positive-labels helper.

Both are Phase 2 prep work: PDF 94-39 schist polygons drive an areal
lode-source-distance feature replacing the point-seeded version, and Qh
placer-tailings polygons feed weighted auxiliary positive labels for the
supervised heads.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from ai_minerals.aoi import AOI
from ai_minerals.features.hydrology import distance_to_lode_areal_m
from ai_minerals.features.labels import aux_positives_from_mask
from ai_minerals.features.surficial import (
    load_surface_geology, schist_mask, tailings_mask,
)
from ai_minerals.grid import build_grid
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION


SURFICIAL_GPKG = Path(NOME_PLACER_REGION.raw_paths["surficial_nome"])


@pytest.fixture(scope="module")
def cape_nome_grid():
    aoi = AOI(
        name="cape_nome_test",
        min_lon=-165.40, min_lat=64.49,
        max_lon=-165.30, max_lat=64.56,
    )
    return build_grid(aoi, resolution_m=100, working_crs="EPSG:3338")


@pytest.fixture(scope="module")
def surface_polys() -> gpd.GeoDataFrame:
    if not SURFICIAL_GPKG.exists():
        pytest.skip(f"PDF 94-39 cache not present at {SURFICIAL_GPKG}")
    from ai_minerals.aoi import NOME_PLACER
    return load_surface_geology(SURFICIAL_GPKG, NOME_PLACER)


def test_distance_inside_a_synthetic_polygon_is_zero():
    """sjoin_nearest returns 0 distance for centroids inside a polygon."""
    aoi = AOI(
        name="t", min_lon=-165.35, min_lat=64.52,
        max_lon=-165.34, max_lat=64.53,
    )
    grid = build_grid(aoi, resolution_m=100, working_crs="EPSG:3338")
    # Polygon clearly larger than the grid AOI (margin: 0.005 deg ~ 500 m).
    poly = box(-165.36, 64.515, -165.33, 64.535)
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, geometry="geometry", crs="EPSG:4326")
    dist = distance_to_lode_areal_m(gdf, grid, max_m=5000.0)
    assert (dist.to_numpy() == 0.0).all()


def test_distance_outside_synthetic_polygon_is_positive():
    """Cells outside the polygon get positive distance up to max_m."""
    aoi = AOI(
        name="t", min_lon=-165.35, min_lat=64.52,
        max_lon=-165.34, max_lat=64.53,
    )
    grid = build_grid(aoi, resolution_m=100, working_crs="EPSG:3338")
    # A polygon well to the east (out of grid)
    poly = box(-165.20, 64.50, -165.19, 64.51)
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, geometry="geometry", crs="EPSG:4326")
    dist = distance_to_lode_areal_m(gdf, grid, max_m=50_000.0)
    arr = dist.to_numpy()
    # All cells in our small grid should be far from the polygon but within max_m
    assert (arr > 0).all()
    assert (arr < 50_000.0).all()


def test_real_schist_polys_yield_nonzero_distance_at_cape_nome(
    surface_polys, cape_nome_grid,
) -> None:
    """Cape Nome has real schist polygons; distance should be finite and bounded."""
    schist = surface_polys[surface_polys["surface_class"] == "bedrock_schist"]
    dist = distance_to_lode_areal_m(schist, cape_nome_grid, max_m=25_000.0)
    arr = dist.to_numpy()
    finite = arr[np.isfinite(arr)]
    assert len(finite) > 0
    assert finite.min() >= 0.0
    assert finite.max() < 25_000.0


def test_aux_positives_from_tailings_mask_returns_table(
    surface_polys, cape_nome_grid,
) -> None:
    """tailings_mask -> aux positives table preserves only mask>0 cells."""
    mask = tailings_mask(surface_polys, cape_nome_grid)
    n_tailings = int(mask.sum())
    table = aux_positives_from_mask(
        mask, cape_nome_grid, weight=0.25, source="qh_tailings",
    )
    assert len(table) == n_tailings
    expected_cols = {"row", "col", "lon", "lat", "label", "weight", "source"}
    assert expected_cols <= set(table.columns)
    assert (table["weight"] == np.float32(0.25)).all()
    assert (table["source"] == "qh_tailings").all()
    assert (table["label"] == 1).all()


def test_aux_positives_dropped_when_mask_empty(cape_nome_grid):
    """An all-zero mask yields an empty aux-positives table."""
    import pandas as pd
    n = cape_nome_grid.n_cells
    mask = pd.Series(np.zeros(n, dtype=np.float32), name="empty")
    table = aux_positives_from_mask(mask, cape_nome_grid)
    assert len(table) == 0
