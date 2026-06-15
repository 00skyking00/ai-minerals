"""Depth-aware lithology features for the Nome placer model.

The v1 coastal scorer (``features/coastal_scorer.py``) assumes the
paleo-beach at a given stand is exposed at the modern surface. That
fails at Bear Cub MS 1178 where Sky 2026-06-14 confirmed Third Beach
is **buried 80-90 ft below** the modern surface (and the bearcub drill
logs show bedrock at 62-85 ft depth, mean 74 ft, matching Tuck's
+70-79 ft Third Beach stand at +75 to +98 ft elevation when measured
from Bear Cub's ~+160 ft surface).

Buried-beach detection needs depth information. bearcub delivers two
artifacts that carry that information:

- **bedrock_topo_nome_placer.tif** + **bedrock_topo_variance_nome_placer.tif**:
  GP-fitted bedrock surface (and variance) over the Bear Cub claim
  envelope (~800 m x 875 m, 25 m grid, EPSG:3338). The variance grows
  at the envelope edges; use it to mask or downweight extrapolation.
- **hole_layers_nome_placer.parquet**: per-hole layered lithology
  (1,390 intervals over 80 holes; GRAVEL / ICE / BEDROCK / CLAY /
  OTHER / MIXED / SAND / MUCK).
- **cross_sections_nome_placer.parquet**: 57 holes across drill lines
  AA-EE with bedrock + pay-zone picks (Janin 1912).

This module loads those artifacts and produces per-cell features:

- ``depth_to_bedrock_m(grid)``: bearcub GP surface sampled at each cell;
  ``None`` (or NaN) outside the variance-acceptable envelope.
- ``elevation_of_bedrock_m(dem, grid)``: ``modern_surface - depth``;
  the absolute bedrock elevation, used by buried-beach scoring.
- ``buried_stand_membership(bedrock_elev, stand)``: per-cell Gaussian
  membership over ``|stand_elev - bedrock_elev|``; cells where bedrock
  sits at a documented paleo-stand elevation score high.

Bear Cub specifically: with bedrock mean 74 ft below the +160 ft
surface, ``bedrock_elev ~ +86 ft (+26 m)`` -- within ~3 m of Third
Beach's +74.5 ft (+22.7 m). The buried-stand membership for Third at
sigma 5 m gives 0.61. Bear Cub's BL ceiling jumps from 0.38 to 0.61
once buried-stand scoring lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  -- registers .rio accessor
import xarray as xr

from ai_minerals.features.coastal import (
    STAND_ELEVATIONS_FT,
    elevation_relative_to_stand_m,
    stand_elevation_m,
)
from ai_minerals.grid import Grid


_FT_PER_M = 1.0 / 0.3048

#: Variance threshold above which the bearcub GP bedrock surface is
#: considered extrapolation and masked out. Bearcub's note: variance
#: grows to 17-121 ft^2 at the envelope edges. 50 ft^2 corresponds to a
#: ~7 ft (1 sigma) bedrock-depth uncertainty, which is sub-stand for
#: Third Beach's 9 ft range. Outside this threshold the bedrock signal
#: is unreliable.
DEFAULT_VARIANCE_THRESHOLD_FT2 = 50.0


@dataclass(frozen=True)
class BedrockField:
    """Sampled bearcub bedrock surface + variance at grid resolution."""
    depth_m: pd.Series       # depth-to-bedrock, metres (NaN outside envelope)
    variance_m2: pd.Series   # GP variance, metres^2 (capped at threshold)
    coverage_mask: pd.Series # bool, True where within envelope


def load_bedrock_surface(
    bedrock_tif: Path,
    variance_tif: Path,
    grid: Grid,
    *,
    variance_threshold_ft2: float = DEFAULT_VARIANCE_THRESHOLD_FT2,
) -> BedrockField:
    """Sample bearcub's GP bedrock surface + variance at grid centroids.

    Returns NaN-valued depth where variance exceeds the threshold; the
    coverage_mask Series is True only where the GP is trustworthy. The
    bedrock raster is in feet (per bearcub's deliverable convention);
    we convert to metres for consistency with the rest of the feature
    stack.
    """
    bedrock_ft = xr.open_dataarray(bedrock_tif, engine="rasterio").squeeze("band", drop=True)
    variance_ft2 = xr.open_dataarray(variance_tif, engine="rasterio").squeeze("band", drop=True)

    centroids = grid.centroid_gdf()
    if centroids.crs != bedrock_ft.rio.crs:
        centroids = centroids.to_crs(bedrock_ft.rio.crs)
    xs = centroids.geometry.x.to_numpy()
    ys = centroids.geometry.y.to_numpy()

    sampled_ft = bedrock_ft.sel(x=xr.DataArray(xs), y=xr.DataArray(ys), method="nearest").values
    sampled_var = variance_ft2.sel(x=xr.DataArray(xs), y=xr.DataArray(ys), method="nearest").values

    # Outside the bearcub envelope (~800 m x 875 m), the GP sample
    # returns the raster's nodata or extrapolates wildly. Mark those
    # cells as out-of-coverage.
    in_coverage = (
        np.isfinite(sampled_ft)
        & np.isfinite(sampled_var)
        & (sampled_var <= variance_threshold_ft2)
    )

    depth_m = sampled_ft.astype(np.float32) / _FT_PER_M
    depth_m = np.where(in_coverage, depth_m, np.nan)

    return BedrockField(
        depth_m=pd.Series(depth_m, index=grid.centroid_gdf().index, name="depth_to_bedrock_m"),
        variance_m2=pd.Series(
            (sampled_var / (_FT_PER_M ** 2)).astype(np.float32),
            index=grid.centroid_gdf().index,
            name="bedrock_variance_m2",
        ),
        coverage_mask=pd.Series(
            in_coverage,
            index=grid.centroid_gdf().index,
            name="bedrock_coverage",
        ),
    )


def elevation_of_bedrock_m(
    dem: xr.DataArray,
    grid: Grid,
    bedrock: BedrockField,
) -> pd.Series:
    """Compute bedrock elevation = modern_surface_elevation - depth_to_bedrock.

    NaN outside the bedrock coverage envelope. Result is in metres
    above modern sea level.
    """
    # Sample the DEM at centroids via the existing helper -- reuse
    # elevation_relative_to_stand_m with a dummy stand of 0 to get
    # the absolute elevation back out.
    dummy_stand = next(iter(STAND_ELEVATIONS_FT))
    rel = elevation_relative_to_stand_m(dem, grid, dummy_stand).to_numpy(np.float32)
    surface_m = rel + stand_elevation_m(dummy_stand)

    depth = bedrock.depth_m.to_numpy()
    bedrock_elev = surface_m - depth
    return pd.Series(
        bedrock_elev.astype(np.float32),
        index=grid.centroid_gdf().index,
        name="bedrock_elevation_m",
    )


def buried_stand_membership(
    bedrock_elev_m: pd.Series,
    stand_name: str,
    *,
    sigma_m: float = 5.0,
) -> pd.Series:
    """Gaussian membership: cell scores high when bedrock sits at the
    named stand's paleo elevation.

    This is the buried-beach analog of
    ``coastal_scorer.score_population_over_stands``: instead of asking
    "does the modern surface match the stand elevation", we ask "does
    the BEDROCK (at depth) match the stand elevation".

    At Bear Cub MS 1178: surface ~+160 ft, bedrock ~ +86 ft, Third Beach
    +74.5 ft. ``bedrock_elev - third_stand = +12 ft = +3.7 m``. Gaussian
    with sigma 5 m gives 0.76. The cell membership is high.
    """
    target_elev_m = stand_elevation_m(stand_name)
    delta = bedrock_elev_m.to_numpy(np.float64) - target_elev_m
    membership = np.exp(-(delta ** 2) / (2.0 * sigma_m ** 2))
    return pd.Series(
        membership.astype(np.float32),
        index=bedrock_elev_m.index,
        name=f"buried_stand_membership_{stand_name}",
    )


def buried_population_score(
    bedrock_elev_m: pd.Series,
    stand_names: tuple[str, ...],
    *,
    sigma_m: float = 5.0,
) -> pd.Series:
    """Max over a population's stand list of buried-stand membership.

    Mirrors the surface-elevation BL/AP/TB scorers in coastal_scorer
    but uses bedrock elevation as the signal source. The Bear Cub
    case where Third Beach is buried at bedrock gets the highest
    score for the BL population this way.
    """
    parts = []
    for stand in stand_names:
        parts.append(buried_stand_membership(bedrock_elev_m, stand, sigma_m=sigma_m).to_numpy())
    stacked = np.stack(parts, axis=0)
    best = stacked.max(axis=0)
    return pd.Series(
        best.astype(np.float32),
        index=bedrock_elev_m.index,
        name="buried_population_score",
    )
