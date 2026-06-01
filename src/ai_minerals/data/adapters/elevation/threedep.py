"""USGS 3DEP elevation mosaic → canonical xarray DataArray.

Reads the mosaic GeoTIFF written by `data/threedep.py` plus its sidecar
metadata, returns an `xr.DataArray` with the attrs the feature stack
expects. Reprojection to the working CRS happens downstream, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import rioxarray  # noqa: F401  -- registers .rio accessor
import xarray as xr

from ai_minerals.aoi import AOI


def load(path: Path, aoi: AOI) -> xr.DataArray:
    """Read the 3DEP mosaic GeoTIFF; return a tagged DataArray in native CRS.

    Attrs:
      - resolution_m: int  -- approximate native pixel size in metres
      - source: "USGS_3DEP"
      - field_name: "elevation_m"

    The sidecar `*_meta.json` (written by the fetcher) is consulted for the
    resolution when present; otherwise the value is derived from the raster
    transform.
    """
    da = xr.open_dataarray(path, engine="rasterio").squeeze("band", drop=True)
    da.name = "elevation_m"

    meta_path = path.with_name(path.stem + "_meta.json")
    resolution_m: int | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        resolution_m = int(meta.get("actual_resolution_m") or 0) or None

    if resolution_m is None:
        # Fall back to deriving from the transform. Assumes either projected
        # metres or, for geographic CRS, returns degrees * 111_320 at the
        # mosaic centre latitude.
        import math

        tx = da.rio.transform()
        if da.rio.crs and da.rio.crs.is_geographic:
            cy = float(da.y.mean())
            dx_m = abs(tx.a) * 111_320 * math.cos(math.radians(cy))
            dy_m = abs(tx.e) * 111_320
            resolution_m = int(round((dx_m + dy_m) / 2))
        else:
            resolution_m = int(round((abs(tx.a) + abs(tx.e)) / 2))

    da.attrs.update(
        {
            "resolution_m": resolution_m,
            "source": "USGS_3DEP",
            "field_name": "elevation_m",
            "aoi_name": aoi.name,
        }
    )
    return da
