"""Raster-sampled features — one value per grid cell from each input raster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from ai_minerals.grid import Grid


def _target_dataarray(grid: Grid) -> xr.DataArray:
    """An empty xarray DataArray with the exact grid of `grid` for reproject_match."""
    r = grid.resolution_m
    # reproject_match needs a transform + crs. Build a minimal target.
    transform = from_origin(grid.xs[0] - r / 2, grid.ys[-1] + r / 2, r, r)
    target = xr.DataArray(
        np.zeros(grid.shape, dtype=np.float32),
        dims=("y", "x"),
        coords={"x": grid.xs, "y": grid.ys[::-1]},  # y decreasing top->bottom
    )
    target.rio.write_crs(grid.crs, inplace=True)
    target.rio.write_transform(transform, inplace=True)
    return target


def sample_raster(
    raster_path: Path,
    grid: Grid,
    *,
    band: int | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Read a raster and resample to the grid. Return a 2-D array (n_rows, n_cols).

    If `band` is given and the raster has multiple bands, return only that band;
    otherwise the raster is expected to be single-band.
    """
    da = rioxarray.open_rasterio(raster_path, masked=True)
    if "band" in da.dims and da.sizes["band"] > 1:
        if band is None:
            raise ValueError(f"{raster_path.name} has {da.sizes['band']} bands; specify band=")
        da = da.sel(band=band)
    da = da.squeeze(drop=True)
    target = _target_dataarray(grid)
    out = da.rio.reproject_match(target, resampling=resampling)
    arr = out.values.astype(np.float32)
    # rio.reproject_match puts y descending (top->bottom); flip to match grid.ys (ascending).
    return arr[::-1, :]


def slope_and_tri(dem_grid: np.ndarray, resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and Terrain Ruggedness Index from a 2-D DEM array.

    - Slope: arctan(|∇elev| / resolution), standard Horn-style central differences.
    - TRI (Riley 1999): mean absolute elevation difference from 8 neighbors.
    """
    dz_dy, dz_dx = np.gradient(dem_grid, resolution_m)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    # TRI: mean |center - neighbor| over 8-connected neighbors
    pad = np.pad(dem_grid, 1, mode="edge")
    abs_diffs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = pad[1 + dy : 1 + dy + dem_grid.shape[0],
                          1 + dx : 1 + dx + dem_grid.shape[1]]
            abs_diffs.append(np.abs(dem_grid - shifted))
    tri = np.mean(abs_diffs, axis=0)
    return slope, tri


def s2_indices(
    s2_path: Path, grid: Grid
) -> dict[str, np.ndarray]:
    """Compute Sentinel-2 alteration indices + NDVI from the mosaic.

    Assumes the mosaic's band axis order is [B02, B03, B04, B08, B11, B12]
    (as written by `sentinel2_mosaic.py`). rioxarray exposes that axis with
    band coord 1..6.
    """
    da = rioxarray.open_rasterio(s2_path, masked=True)
    target = _target_dataarray(grid)
    # Reproject all 6 bands in one pass.
    resampled = da.rio.reproject_match(target, resampling=Resampling.bilinear)
    # Band axis is 1..N; our canonical order is B02=1, B03=2, B04=3, B08=4, B11=5, B12=6.
    B02 = resampled.sel(band=1).values.astype(np.float32)[::-1, :]
    B03 = resampled.sel(band=2).values.astype(np.float32)[::-1, :]
    B04 = resampled.sel(band=3).values.astype(np.float32)[::-1, :]
    B08 = resampled.sel(band=4).values.astype(np.float32)[::-1, :]
    B11 = resampled.sel(band=5).values.astype(np.float32)[::-1, :]
    B12 = resampled.sel(band=6).values.astype(np.float32)[::-1, :]

    # Guard against divide-by-zero.
    def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan, dtype=np.float32)
        mask = np.isfinite(a) & np.isfinite(b) & (b > 0)
        out[mask] = a[mask] / b[mask]
        return out

    return {
        "s2_iron_oxide": _safe_div(B04, B02),    # gossan / weathered sulfides
        "s2_ferrous":    _safe_div(B11, B08),    # ferrous iron
        "s2_clay":       _safe_div(B11, B12),    # clay / hydroxyl minerals
        "s2_ndvi":       _safe_div(B08 - B04, B08 + B04),  # vegetation mask
    }
