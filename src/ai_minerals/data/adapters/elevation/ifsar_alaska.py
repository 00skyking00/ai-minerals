"""Alaska IfSAR-derived elevation mosaic GeoTIFF -> canonical xarray DataArray.

Reads the mosaic written by `data/ifsar_alaska.py` plus its sidecar
metadata; returns an `xr.DataArray` with the attrs the feature stack
expects. Reprojection to the working CRS happens downstream, not here.

Mirrors the loader for `data/adapters/elevation/threedep.py` because the
underlying TNM product structure is identical between the CONUS 10 m
(1/3 arc-second) and Alaska 10 m (1/3 arc-second IfSAR-derived) layers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import rioxarray  # noqa: F401  -- registers .rio accessor
import xarray as xr

from ai_minerals.aoi import AOI


def load(path: Path, aoi: AOI) -> xr.DataArray:
    """Read the Alaska IfSAR mosaic GeoTIFF; return a tagged DataArray in native CRS.

    Attrs:
      - resolution_m: int  -- approximate native pixel size in metres
      - source: "USGS_TNM_AK_IFSAR"
      - field_name: "elevation_m"
    """
    da = xr.open_dataarray(path, engine="rasterio").squeeze("band", drop=True)
    da.name = "elevation_m"

    meta_path = path.with_name(path.stem + "_meta.json")
    resolution_m: int | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        resolution_m = int(meta.get("actual_resolution_m") or 0) or None

    if resolution_m is None:
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
            "source": "USGS_TNM_AK_IFSAR",
            "field_name": "elevation_m",
            "aoi_name": aoi.name,
        }
    )
    return da
