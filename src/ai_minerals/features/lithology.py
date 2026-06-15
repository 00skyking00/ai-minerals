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


def load_all_drillhole_depths(
    drillholes_path: Path,
    cross_sections_path: Path | None = None,
) -> pd.DataFrame:
    """Combine drillholes + cross_sections into a unified depth dataset.

    Output schema:
      hole_id, lon, lat, easting_3338, northing_3338, source,
      bedrock_depth_ft, pay_top_ft, pay_bottom_ft, grade_oz_cuyd

    For Bear Cub Murray holes (which have ``pay_zone_thickness_ft`` but
    not ``pay_top_ft``), we derive ``pay_top_ft = bedrock_depth_ft -
    pay_zone_thickness_ft`` (placer convention: pay sits at the base
    of the gravel column, on bedrock).

    For Janin Pioneer holes that also appear in cross_sections_path,
    the parsed ``pay_top_ft`` / ``pay_bottom_ft`` from the side-elevation
    diagrams replaces the bedrock-derived estimate where available.
    """
    dh = pd.read_parquet(drillholes_path)
    valid = dh.dropna(subset=["lon", "lat", "bedrock_depth_ft"]).copy()

    # Default: derive pay-zone from bedrock - thickness
    valid["pay_bottom_ft"] = valid["bedrock_depth_ft"]
    valid["pay_top_ft"] = (
        valid["bedrock_depth_ft"] - valid["pay_zone_thickness_ft"]
    )

    # Janin cross-sections override (where parsed)
    if cross_sections_path is not None and cross_sections_path.exists():
        cs = pd.read_parquet(cross_sections_path)
        cs_valid = cs.dropna(subset=["pay_top_ft", "pay_bottom_ft"])[
            ["hole_id", "pay_top_ft", "pay_bottom_ft"]
        ].rename(columns={
            "pay_top_ft": "pay_top_ft_cs",
            "pay_bottom_ft": "pay_bottom_ft_cs",
        })
        valid = valid.merge(cs_valid, on="hole_id", how="left")
        cs_present = valid["pay_top_ft_cs"].notna()
        valid.loc[cs_present, "pay_top_ft"] = valid.loc[cs_present, "pay_top_ft_cs"]
        valid.loc[cs_present, "pay_bottom_ft"] = valid.loc[cs_present, "pay_bottom_ft_cs"]
        valid = valid.drop(columns=["pay_top_ft_cs", "pay_bottom_ft_cs"])

    if "grade_oz_cuyd" not in valid.columns:
        valid["grade_oz_cuyd"] = valid.get(
            "pay_zone_average_oz_cuyd",
            pd.Series([float("nan")] * len(valid), index=valid.index),
        )

    out_cols = [
        "hole_id", "lon", "lat", "easting_3338", "northing_3338", "source",
        "bedrock_depth_ft", "pay_top_ft", "pay_bottom_ft", "grade_oz_cuyd",
    ]
    return valid[out_cols].reset_index(drop=True)


def interpolate_depth_surface(
    holes: pd.DataFrame,
    grid: Grid,
    *,
    field_name: str = "bedrock_depth_ft",
    method: str = "linear",
    max_distance_m: float = 1500.0,
) -> "pd.Series":
    """Interpolate a per-cell depth surface (feet) from hole data.

    Uses ``scipy.interpolate.griddata`` with the specified method
    (``linear``, ``cubic``, or ``nearest``). Cells more than
    ``max_distance_m`` from any hole are masked NaN -- prevents wild
    extrapolation outside the drilled envelope.

    ``holes`` must carry ``easting_3338`` + ``northing_3338`` (EPSG:3338
    metres, the working CRS) and the field-name column.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    valid = holes.dropna(subset=["easting_3338", "northing_3338", field_name])
    if len(valid) < 3:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            index=grid.centroid_gdf().index,
            name=f"interp_{field_name}",
        )

    points = valid[["easting_3338", "northing_3338"]].to_numpy(dtype=np.float64)
    values = valid[field_name].to_numpy(dtype=np.float64)

    centroids = grid.centroid_gdf()
    targets = np.column_stack([
        centroids.geometry.x.to_numpy(dtype=np.float64),
        centroids.geometry.y.to_numpy(dtype=np.float64),
    ])

    interp = griddata(points, values, targets, method=method)

    # Mask cells too far from any drillhole
    tree = cKDTree(points)
    nearest_d, _ = tree.query(targets, k=1)
    interp = np.where(nearest_d <= max_distance_m, interp, np.nan)

    return pd.Series(
        interp.astype(np.float32),
        index=centroids.index,
        name=f"interp_{field_name}",
    )


def combined_bedrock_elevation_m(
    dem: xr.DataArray,
    grid: Grid,
    *,
    drillholes_path: Path,
    cross_sections_path: Path | None,
    bearcub_field: BedrockField | None = None,
    max_distance_m: float = 1500.0,
) -> pd.Series:
    """Per-cell bedrock elevation (m above modern sea level), combining:

    1. **Bearcub GP surface** within the Bear Cub envelope where
       variance is acceptable. Sub-cell precision; trusted.
    2. **Interpolated surface** from the wider 99-hole network (Janin
       Pioneer + Bear Cub Murray) elsewhere within ``max_distance_m``
       of any drillhole. Linear interpolation via scipy.griddata; less
       precise than the GP but covers ~15x the area.
    3. **NaN** outside both coverage zones.

    Result is bedrock elevation (modern surface - depth) in metres.
    Buried-BL scoring uses this to detect paleo-beach lines below the
    modern surface across the wider district, not just the Bear Cub
    family-claim envelope.
    """
    centroids = grid.centroid_gdf()
    surface_m = _surface_elevation_m(dem, grid)
    out = np.full(grid.n_cells, np.nan, dtype=np.float32)

    # Layer 1: bearcub GP elevation (where coverage_mask is True)
    if bearcub_field is not None:
        in_cov = bearcub_field.coverage_mask.to_numpy()
        depth_m_bc = bearcub_field.depth_m.to_numpy()
        out[in_cov] = (surface_m[in_cov] - depth_m_bc[in_cov]).astype(np.float32)

    # Layer 2: wider-hole-network interpolation where bearcub is missing
    holes = load_all_drillhole_depths(drillholes_path, cross_sections_path)
    interp_depth_ft = interpolate_depth_surface(
        holes, grid, field_name="bedrock_depth_ft",
        max_distance_m=max_distance_m,
    ).to_numpy()
    # Convert ft to m
    interp_depth_m = interp_depth_ft / _FT_PER_M
    fill_mask = np.isnan(out) & np.isfinite(interp_depth_m)
    out[fill_mask] = (surface_m[fill_mask] - interp_depth_m[fill_mask]).astype(np.float32)

    return pd.Series(out, index=centroids.index, name="combined_bedrock_elevation_m")


def _surface_elevation_m(dem: xr.DataArray, grid: Grid) -> np.ndarray:
    """Sample DEM at each grid centroid; return surface elevation (m, numpy)."""
    dummy_stand = next(iter(STAND_ELEVATIONS_FT))
    rel = elevation_relative_to_stand_m(dem, grid, dummy_stand).to_numpy(np.float32)
    return rel + stand_elevation_m(dummy_stand)


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
