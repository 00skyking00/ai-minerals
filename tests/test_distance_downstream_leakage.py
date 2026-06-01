"""Regression test for the distance-downstream-from-lode leakage guard.

If MRDS lode-Au seeds for distance-downstream-from-lode contain a placer
record (mis-classified at MRDS), the feature would encode the positive
label. The function asserts a pre-filter was applied; this test verifies
the assertion fires when a placer-flagged record slips through.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from ai_minerals.features.hydrology import distance_downstream_from_lode
from ai_minerals.grid import Grid


# --- fixtures -------------------------------------------------------------


# Small Grid in EPSG:3310 (California Albers, the northern_sierra_placer
# working CRS). A 4x4 grid at 250 m sits around a synthetic origin near
# Nevada City; absolute location doesn't matter for the leakage assertion.
_X0, _Y0 = -150_000.0, 100_000.0


@pytest.fixture
def grid() -> Grid:
    res = 250
    xs = np.array([_X0 + (i + 0.5) * res for i in range(4)], dtype=float)
    ys = np.array([_Y0 + (j + 0.5) * res for j in range(4)], dtype=float)
    return Grid(xs=xs, ys=ys, resolution_m=res, crs="EPSG:3310")


@pytest.fixture
def nhd_network(grid: Grid) -> gpd.GeoDataFrame:
    """A single LineString reach crossing the grid, in EPSG:4326.

    The adapter (`data/adapters/hydrology/nhdplus_hr.py`) returns EPSG:4326
    flowlines, so the feature function is responsible for projecting to the
    grid CRS internally. We build the line in 3310 first, then reproject.
    """
    coords_3310 = [
        (grid.xs[0] - 50, grid.ys[0]),
        (grid.xs[-1] + 50, grid.ys[-1]),
    ]
    line_3310 = gpd.GeoSeries([LineString(coords_3310)], crs="EPSG:3310")
    line_4326 = line_3310.to_crs("EPSG:4326")
    return gpd.GeoDataFrame(
        {
            "geometry": line_4326,
            "comid": [1],
            "arbolate_sum": [10.0],
            "stream_order": [3],
            "hydroseq": [1],
            "fcode": [46006],
            "source": ["NHDPlus_HR"],
        },
        crs="EPSG:4326",
    )


def _lode_points(records: list[dict]) -> gpd.GeoDataFrame:
    """Build a lode_points GeoDataFrame from row dicts.

    Each row dict supplies at least `x`, `y` (EPSG:3310) and the columns
    the leakage guard inspects. The function under test takes raw points
    in the grid CRS (no reprojection contract on the caller's side here;
    if the real implementation expects EPSG:4326, swap the crs here).
    """
    geoms = [Point(r["x"], r["y"]) for r in records]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "y")} for r in records])
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:3310")


# --- tests ----------------------------------------------------------------


def test_placer_seed_raises(grid: Grid, nhd_network: gpd.GeoDataFrame) -> None:
    """A placer-flagged record in lode_points must trip the leakage guard.

    The seed list mixes one legitimate orogenic-Au record with one record
    whose dep_type matches the MRDS placer regex
    (`adapters/occurrences/mrds.py::_PLACER_DEP_TYPE_RE`). The function
    must refuse the inputs rather than silently encode placer locations.
    """
    seeds = _lode_points(
        [
            {
                "x": grid.xs[1],
                "y": grid.ys[1],
                "dep_type": "Quartz vein",
                "dev_stat": "Past Producer",
                "raw_record_id": "lode-good-1",
            },
            {
                "x": grid.xs[2],
                "y": grid.ys[2],
                "dep_type": "Placer, alluvial",
                "dev_stat": "Past Producer",
                "raw_record_id": "placer-bad-1",
            },
        ]
    )

    with pytest.raises((ValueError, AssertionError)) as excinfo:
        distance_downstream_from_lode(seeds, nhd_network, grid)

    msg = str(excinfo.value).lower()
    # The error must name the offending record so the operator can find
    # it without re-running a query against MRDS by hand.
    assert "placer-bad-1" in str(excinfo.value) or "placer" in msg, (
        f"Leakage-guard error should identify the offending placer record; got: {excinfo.value!r}"
    )


def test_pure_lode_seeds_pass(grid: Grid, nhd_network: gpd.GeoDataFrame) -> None:
    """Clean orogenic-Au seeds must pass and return a length-n_cells Series."""
    seeds = _lode_points(
        [
            {
                "x": grid.xs[0],
                "y": grid.ys[0],
                "dep_type": "Quartz vein",
                "dev_stat": "Past Producer",
                "raw_record_id": "lode-1",
            },
            {
                "x": grid.xs[1],
                "y": grid.ys[1],
                "dep_type": "Lode gold",
                "dev_stat": "Producer",
                "raw_record_id": "lode-2",
            },
            {
                "x": grid.xs[2],
                "y": grid.ys[2],
                "dep_type": "Vein, orogenic",
                "dev_stat": "Past Producer",
                "raw_record_id": "lode-3",
            },
        ]
    )

    out = distance_downstream_from_lode(seeds, nhd_network, grid)

    assert isinstance(out, pd.Series), f"Expected pd.Series; got {type(out)!r}"
    assert len(out) == grid.n_cells, (
        f"Expected length {grid.n_cells} (n_cells); got {len(out)}"
    )


def test_no_dep_type_column_raises(grid: Grid, nhd_network: gpd.GeoDataFrame) -> None:
    """Missing dep_type column means the filter cannot have been applied — refuse."""
    bare = gpd.GeoDataFrame(
        {
            "geometry": [Point(grid.xs[1], grid.ys[1])],
            "raw_record_id": ["lode-1"],
            "dev_stat": ["Past Producer"],
        },
        crs="EPSG:3310",
    )

    with pytest.raises((ValueError, AssertionError)) as excinfo:
        distance_downstream_from_lode(bare, nhd_network, grid)

    msg = str(excinfo.value).lower()
    assert "dep_type" in msg, (
        f"Error should mention the missing dep_type column; got: {excinfo.value!r}"
    )
