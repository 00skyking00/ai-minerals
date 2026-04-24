"""USGS national-composite geophysics grids → canonical NaN-masked array.

USGS aeromagnetic + gravity composites encode nodata as ~±3.4e38 (float32
sentinel). Mask to NaN so downstream `sample_raster` can use nearest-valid
interpolation without poisoning the neighborhood.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


NODATA_MAGNITUDE = 1e10   # anything larger in absolute value is a float32 sentinel


def mask_nodata(arr: np.ndarray) -> np.ndarray:
    """Replace sentinel nodata values with NaN."""
    return np.where(np.abs(arr) > NODATA_MAGNITUDE, np.nan, arr)


def load(path: Path, arr: np.ndarray | None = None, *, field_name: str, units: str = "nT") -> np.ndarray:
    """Return a NaN-masked numpy array.

    Two calling conventions:
      - load(path, field_name=...)    reads the tif and applies the mask
      - load(path, arr, field_name=...) applies the mask to an already-read
        array (used by `build_feature_frame` which samples the raster at
        grid centroids, not the whole raster).

    Attaching metadata via xarray DataArray would be the cleaner contract
    long-term, but v1 consumes plain numpy arrays; match that to minimize
    refactor surface.
    """
    if arr is None:
        from ai_minerals.features.rasters import sample_raster  # lazy to avoid cycles
        raise ValueError(
            "usgs_geophysics.load needs an already-sampled array. "
            "Call sample_raster(path, grid) then pass arr=... to mask nodata."
        )
    return mask_nodata(arr)
