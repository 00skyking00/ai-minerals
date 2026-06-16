"""Tests for the PDF 94-39 surficial-class masks.

Real-data integration: loads bearcub's cached PDF 94-39 polygons and
rasterizes the three masks (drift, schist, tailings) over a small
grid covering the Bear Cub block. Validates counts and the expected
classification of MS 1178 Bear Cub (sits on Qd2 Nome River drift per
bearcub's note).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from ai_minerals.aoi import AOI, NOME_PLACER
from ai_minerals.features.surficial import (
    DRIFT_CLASS,
    SCHIST_CLASS,
    TAILINGS_CLASS,
    drift_mask,
    load_surface_geology,
    schist_mask,
    surface_class_mask,
    tailings_mask,
)
from ai_minerals.grid import build_grid
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION


SURFICIAL_GPKG = Path(NOME_PLACER_REGION.raw_paths["surficial_nome"])


@pytest.fixture(scope="module")
def surface_polys() -> gpd.GeoDataFrame:
    if not SURFICIAL_GPKG.exists():
        pytest.skip(f"PDF 94-39 cache not present at {SURFICIAL_GPKG}")
    return load_surface_geology(SURFICIAL_GPKG, NOME_PLACER)


@pytest.fixture(scope="module")
def bear_cub_grid():
    """Tight 1 km x 1 km grid centred near the Bear Cub block."""
    aoi = AOI(
        name="bear_cub_test",
        min_lon=-165.345, min_lat=64.520,
        max_lon=-165.325, max_lat=64.535,
    )
    return build_grid(aoi, resolution_m=50, working_crs="EPSG:3338")


def test_load_returns_pdf94_39_schema(surface_polys: gpd.GeoDataFrame) -> None:
    assert "surface_class" in surface_polys.columns
    assert (surface_polys["source"] == "ADGGS_PDF94_39").all()
    counts = surface_polys["surface_class"].value_counts()
    assert counts[DRIFT_CLASS] > 100
    assert counts[SCHIST_CLASS] > 100


def test_drift_mask_hits_bear_cub(surface_polys: gpd.GeoDataFrame, bear_cub_grid) -> None:
    """MS 1178 Bear Cub sits on Qd2 Nome River drift per bearcub's note."""
    mask = drift_mask(surface_polys, bear_cub_grid)
    assert len(mask) == bear_cub_grid.shape[0] * bear_cub_grid.shape[1]
    # The grid is small and centred on Bear Cub. We expect at least some
    # cells in drift; if zero, the loader is broken or coverage is off.
    assert mask.sum() > 0, "drift_mask returned 0 cells over Bear Cub grid"


def test_mask_dtype_and_values(surface_polys: gpd.GeoDataFrame, bear_cub_grid) -> None:
    """Masks are float32 in {0.0, 1.0} so downstream code can multiply."""
    mask = drift_mask(surface_polys, bear_cub_grid)
    assert mask.dtype == np.float32
    unique_vals = np.unique(mask.to_numpy())
    for v in unique_vals:
        assert v in (0.0, 1.0)


def test_three_masks_are_disjoint(
    surface_polys: gpd.GeoDataFrame, bear_cub_grid,
) -> None:
    """A cell sitting on bedrock_schist shouldn't also read as
    glacial_drift_outwash. Allow boundary cells to land on either side."""
    d = drift_mask(surface_polys, bear_cub_grid).to_numpy()
    s = schist_mask(surface_polys, bear_cub_grid).to_numpy()
    t = tailings_mask(surface_polys, bear_cub_grid).to_numpy()
    overlap_ds = float((d * s).sum())
    overlap_dt = float((d * t).sum())
    overlap_st = float((s * t).sum())
    # Some overlap is permissible at polygon boundaries (a cell centroid
    # technically can fall on a shared edge); cap at a small fraction.
    n = len(d)
    assert overlap_ds / n < 0.05
    assert overlap_dt / n < 0.05
    assert overlap_st / n < 0.05


def test_unknown_class_raises_or_zeros(
    surface_polys: gpd.GeoDataFrame, bear_cub_grid,
) -> None:
    """Passing a surface_class that doesn't exist returns an all-zero mask."""
    mask = surface_class_mask(surface_polys, bear_cub_grid, "nonexistent_class")
    assert (mask.to_numpy() == 0.0).all()


def test_missing_surface_class_column_raises(bear_cub_grid) -> None:
    """If we pass a GDF without surface_class, fail clearly."""
    gdf = gpd.GeoDataFrame(
        {"geometry": [], "mapunit": []},
        geometry="geometry",
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="surface_class"):
        drift_mask(gdf, bear_cub_grid)
