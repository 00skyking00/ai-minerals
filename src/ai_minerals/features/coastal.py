"""Coastal / glacial / marine features for the Nome placer model.

Three per-grid-cell features feed the Phase 1 four-population
knowledge-driven index and the Phase 2 supervised models:

- ``elevation_relative_to_stand_m``: signed elevation (m) of each cell
  relative to one of the documented Nome paleo-shoreline stands
  (Submarine -20 / -35 / -70 ft, Intermediate +20, Second +37.5,
  Monroeville +40, Third +74.5, Fourth +122.5). A cell at the same
  elevation as the stand returns 0; a cell above the stand returns a
  positive value; a cell below returns negative.

- ``distance_to_paleo_shoreline_m``: Euclidean distance (m) from each
  cell centroid to the nearest point on the paleo-shoreline contour
  at a given stand. The shoreline contour is the DEM iso-elevation line
  at the stand elevation, extracted from IfSAR.

- ``paleo_shoreline_contours_at_stand``: helper that returns the
  GeoDataFrame of polyline contours at a given stand, derived from
  the DEM. Useful as a debug artifact and as the input geometry for
  the distance feature.

Stand elevations come from the Geological Dynamics of Nome Coastal
Placer Gold Deposits doc in ``research/``. Values are recorded in feet
because the historical literature uses feet; the module converts to
metres internally because the IfSAR DEM is in metres.

These features drive the beach-line (BL) and beach-stream-confluence
(BC) population scores in ``scripts/nome_placer/phase1_nome_index.py``.
The off-beach creek (QM) and buried multi-bench (TB) populations use
``features/hydrology.py`` (stream-channel proximity) and a forthcoming
``features/lithology.py`` module respectively.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  -- registers .rio accessor
import xarray as xr
from shapely.geometry import LineString

from ai_minerals.grid import Grid


_FT_PER_M = 1.0 / 0.3048

#: Documented Nome paleo-shoreline stand elevations (feet above modern sea
#: level). Sources: "Geological Dynamics of Nome Coastal Placer Gold Deposits"
#: in ``research/`` for the named-beach stands (Submarine through Fourth);
#: Metcalfe and Tuck 1942 (``research/nome_tuck_1942_prepass_findings.md``)
#: for the three higher un-named stands at +200 / +300 / +600 ft, evidenced
#: by Anvil Creek Left Limit Channels, the bench cut in limestone at
#: Anvil Creek's left limit, the Newton Creek gravel ridge terrace, the
#: Sledge Island (604 ft) and Cape Nome (620 ft) flat tops, and the
#: Dexter Creek head high-gravel deposits at ~ +600 ft. The +600 ft stand
#: is the documented setting for the Dexter Divide high-gravel deposits
#: (Tuck Localities F / G / H) which are the architectural analog for the
#: Molasses (MS 1179) and Honey (MS 1181) buried multi-bench claims.
#: Values are stand midpoints where the literature gives a range.
STAND_ELEVATIONS_FT: dict[str, float] = {
    "submarine_deep":  -70.0,    # Pliocene marine, -70 ft
    "submarine_inner": -35.0,    # Beringian Transgression inner wave-bench
    "submarine_outer": -20.0,    # Beringian Transgression outer wave-bench
    "intermediate":    +20.0,    # Yarmouth Interglacial intermediate
    "second":          +37.5,    # Sangamon Interglacial Second Beach (35-40 ft midpoint)
    "monroeville":     +40.0,    # Yarmouth Monroeville
    "third":           +74.5,    # Yarmouth Third Beach (70-79 ft midpoint; high-grade exemplar)
    "fourth":         +122.5,    # Aftonian Fourth Beach (120-125 ft midpoint)
    "tuck_high_200":  +200.0,    # Tuck 1942: sea rise to at least +200 ft inferred from Anvil Creek Left Limit Channels
    "tuck_high_300":  +300.0,    # Tuck 1942: Anvil Creek left-limit limestone bench; Newton Creek gravel ridge terrace
    "tuck_high_600":  +600.0,    # Tuck 1942: Sledge Island top (604), Cape Nome top (620), Dexter Divide high-gravel (Molasses / Honey setting)
}


def stand_elevation_m(stand_name: str) -> float:
    """Convert one documented stand elevation from feet to metres."""
    return STAND_ELEVATIONS_FT[stand_name] / _FT_PER_M


def paleo_shoreline_contours_at_stand(
    dem: xr.DataArray,
    stand_name: str,
    *,
    min_contour_length_m: float = 200.0,
) -> gpd.GeoDataFrame:
    """Extract DEM iso-elevation polylines at the named paleo-shoreline stand.

    Uses ``skimage.measure.find_contours`` on the DEM array, transforms
    pixel-coordinate contour vertices back to map coordinates via the
    DEM affine transform, and emits a GeoDataFrame of LineStrings in the
    DEM CRS.

    Short contour fragments (sub-200 m by default) are dropped; they are
    typically transient noise from the DEM rather than paleo-shoreline
    features.
    """
    from skimage import measure

    elev_m = stand_elevation_m(stand_name)
    arr = dem.values
    # find_contours returns (row, col) pixel coordinates of iso-lines.
    raw_contours = measure.find_contours(arr, level=elev_m)
    if not raw_contours:
        return gpd.GeoDataFrame(
            {"stand": [], "elevation_m": [], "geometry": []},
            geometry="geometry",
            crs=dem.rio.crs,
        )

    transform = dem.rio.transform()
    geoms: list[LineString] = []
    for pix_coords in raw_contours:
        # pix_coords is an (N, 2) array of (row, col) -> (y_pix, x_pix).
        xs, ys = transform * (pix_coords[:, 1], pix_coords[:, 0])
        line = LineString(zip(xs, ys))
        if line.length >= min_contour_length_m:
            geoms.append(line)

    gdf = gpd.GeoDataFrame(
        {
            "stand": [stand_name] * len(geoms),
            "elevation_m": [elev_m] * len(geoms),
            "geometry": geoms,
        },
        geometry="geometry",
        crs=dem.rio.crs,
    )
    return gdf


def paleo_shoreline_contours_all_stands(
    dem: xr.DataArray,
    *,
    min_contour_length_m: float = 200.0,
) -> gpd.GeoDataFrame:
    """Extract contours at every documented stand; return one concatenated frame."""
    parts: list[gpd.GeoDataFrame] = []
    for stand in STAND_ELEVATIONS_FT:
        parts.append(
            paleo_shoreline_contours_at_stand(
                dem, stand, min_contour_length_m=min_contour_length_m,
            )
        )
    return gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=dem.rio.crs,
    )


def elevation_relative_to_stand_m(
    dem: xr.DataArray,
    grid: Grid,
    stand_name: str,
) -> pd.Series:
    """Per-grid-cell signed elevation (m) relative to one stand.

    Positive values mean the cell sits above the stand; negative means
    below. Result is indexed like ``grid.centroid_gdf()`` (range index
    0..n_cells-1).

    The DEM is sampled at each centroid via nearest-neighbour
    interpolation in the DEM's native CRS, so the cell may pull from a
    DEM tile a few metres away rather than the cell's exact centre. At
    25 m grid and 10 m DEM that mismatch is below feature noise.
    """
    centroids = grid.centroid_gdf()
    if centroids.crs != dem.rio.crs:
        centroids = centroids.to_crs(dem.rio.crs)
    xs = centroids.geometry.x.to_numpy()
    ys = centroids.geometry.y.to_numpy()
    sampled = dem.sel(x=xr.DataArray(xs), y=xr.DataArray(ys), method="nearest")
    elev_at_centroid = sampled.values.astype(np.float32)
    rel = elev_at_centroid - np.float32(stand_elevation_m(stand_name))
    return pd.Series(
        rel,
        index=grid.centroid_gdf().index,
        name=f"elevation_relative_to_stand_{stand_name}_m",
    )


def distance_to_paleo_shoreline_m(
    contour_polylines: gpd.GeoDataFrame,
    grid: Grid,
    stand_name: str,
    *,
    max_distance_m: float = 50_000.0,
) -> pd.Series:
    """Per-grid-cell distance (m) to the nearest contour polyline at the named stand.

    ``contour_polylines`` is the frame returned by
    ``paleo_shoreline_contours_all_stands`` (or
    ``paleo_shoreline_contours_at_stand`` for a single stand); the
    function filters to the requested stand internally.

    Cells with no contour within ``max_distance_m`` get the cap value.
    """
    target = contour_polylines.loc[contour_polylines["stand"] == stand_name]
    if target.empty:
        return pd.Series(
            np.full(grid.n_cells, max_distance_m, dtype=np.float32),
            index=grid.centroid_gdf().index,
            name=f"distance_to_paleo_shoreline_{stand_name}_m",
        )

    target = target.to_crs(grid.crs)
    centroids = grid.centroid_gdf()

    joined = gpd.sjoin_nearest(
        centroids,
        target[["geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="_dist_m",
    )
    joined = joined.loc[~joined.index.duplicated(keep="first")]
    dist = (
        joined["_dist_m"]
        .fillna(max_distance_m)
        .astype(np.float32)
        .to_numpy()
    )
    return pd.Series(
        dist,
        index=centroids.index,
        name=f"distance_to_paleo_shoreline_{stand_name}_m",
    )
