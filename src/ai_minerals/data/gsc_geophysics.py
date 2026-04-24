"""GSC / NRCan national geophysics grids — aeromagnetic + gravity.

**v2 limitation:** NRCan publishes these grids only as Geosoft `.grd`
files inside interactive-order-form ZIP distributions, with no direct
GeoTIFF endpoint. The CKAN resource URLs redirect to `index-eng.php`
pages that serve HTML ordering workflows, not single downloads.
Additionally the WMS endpoint (`http://wms.agg.nrcan.gc.ca/wms2/`) returns
styled PNG imagery, not raster values.

v2 currently ships *without* a Canadian geophysics grid layer. The
feature pipeline handles this by emitting a NaN-filled magnetic and
gravity array of the correct shape (so the schema remains identical to
EastAK's feature frame). Concrete v2.1 next steps to un-punt this:

1. Build a WMS-to-GeoTIFF fetcher using `owslib` GetMap calls; approximate
   the magnetic field from the rendered PNG using the known colour ramp.
2. Or: negotiate direct `.grd` download via the CKAN `index-eng.php`
   form (`db_project_no=10013` parameter family) using a scripted
   POST with the regional-bbox form fields.
3. Or: convert downloaded Geosoft `.grd` files via `gdal_translate`
   if we can cache them once manually.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
import rioxarray  # noqa: F401
import xarray as xr

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "gsc_geophysics"


def _write_nan_grid(
    out_path: Path, aoi: AOI, working_crs: str, *, field_name: str
) -> Path:
    """Write a NaN-filled GeoTIFF matching the AOI extent + working CRS.

    Placeholder until we wire up real NRCan access. Shape is ~1 km
    resolution so the file is tiny but the projection alignment with other
    rasters is correct.
    """
    import pyproj
    xf = pyproj.Transformer.from_crs(aoi.crs, working_crs, always_xy=True)
    minx, miny = xf.transform(aoi.min_lon, aoi.min_lat)
    maxx, maxy = xf.transform(aoi.max_lon, aoi.max_lat)
    res = 1000.0
    nx = max(int((maxx - minx) / res), 1)
    ny = max(int((maxy - miny) / res), 1)
    arr = np.full((ny, nx), np.nan, dtype=np.float32)
    transform = rasterio.transform.from_origin(minx, maxy, res, res)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", width=nx, height=ny, count=1,
        dtype="float32", crs=working_crs, transform=transform,
        nodata=np.nan, compress="deflate", tiled=True,
    ) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, field_name)
    return out_path


def fetch(aoi: AOI, working_crs: str = "EPSG:3005") -> tuple[Path, Path]:
    """Write NaN-filled placeholder GeoTIFFs for magnetic + gravity.

    Returns (magnetic_path, gravity_path). The canonical geophysics
    adapter (`adapters/geophysics/usgs.py::mask_nodata`) correctly
    handles the resulting all-NaN arrays.
    """
    out_dir = dataset_dir(NAME)
    mag_path = out_dir / f"magnetic_{aoi.name.lower()}.tif"
    grav_path = out_dir / f"gravity_{aoi.name.lower()}.tif"

    print(f"[GSC geophysics] PLACEHOLDER — writing NaN-filled {aoi.name} grids.")
    print("  real NRCan aeromag/gravity integration pending (v2.1); see module docstring.")
    _write_nan_grid(mag_path, aoi, working_crs, field_name="residual_magnetic_nT")
    _write_nan_grid(grav_path, aoi, working_crs, field_name="bouguer_gravity_mGal")

    write_source_md(
        NAME,
        title="GSC / NRCan geophysics grids — placeholder",
        url="http://gdr.agg.nrcan.gc.ca/gdrdap/dap/index-eng.php?dapid=129",
        license="Open Government Licence - Canada (when data arrives)",
        notes=(
            "v2 placeholder: NaN-filled magnetic + gravity GeoTIFFs with correct "
            "CRS + extent so downstream sampling works. Real NRCan .grd → GeoTIFF "
            "pipeline is a v2.1 TODO (see module docstring)."
        ),
    )
    return mag_path, grav_path


if __name__ == "__main__":
    from ai_minerals.regions.bcgt import BCGT
    fetch(BCGT.aoi, working_crs=BCGT.working_crs)
