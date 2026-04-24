"""NOAA EMAG2 v3 — Earth Magnetic Anomaly Grid, 2 arc-minute.

Global aeromagnetic compilation at ~2 km N-S / ~2 km E-W at mid-latitudes.
Coarser than NRCan's 1 km native product, but (a) directly downloadable as
a plain GeoTIFF with stable multi-year URLs, (b) covers the entire globe
so the fetcher trivially works for any AOI, and (c) public domain.

We only fetch the sea-level ("at-surface") version; the 4-km upward-continued
version is noisier and less locally-diagnostic.

NRCan's GDR portal lost programmatic access in 2024 when they migrated from
`gdr.agg.nrcan.gc.ca` to the JS-only `geophysical-data.canada.ca/portal`.
v2.1 could still add USGS NAMAG (1 km, GXF/GRD, needs gdal_translate) as a
finer-resolution fallback over North America.

Gravity: there is no comparable directly-downloadable global Bouguer grid.
WGM2012 sits behind a JS-rendered BGI catalogue that likely requires account
registration. v2 ships without gravity; v2.1 task item.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import requests
import rioxarray  # noqa: F401  -- registers rio accessor
import xarray as xr

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "gsc_geophysics"
EMAG2_URL = (
    "https://www.ngdc.noaa.gov/geomag/data/EMAG2/"
    "EMAG2_V3_20170530/EMAG2_V3_20170530_Sealevel.tif"
)


def _download(dest: Path) -> None:
    print(f"Downloading EMAG2 v3 from {EMAG2_URL} (~240 MB, ~3 min)")
    with requests.get(EMAG2_URL, stream=True, timeout=900) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")


def fetch(aoi: AOI, working_crs: str = "EPSG:3005", *, force: bool = False) -> tuple[Path, Path]:
    """Fetch the global EMAG2 GeoTIFF and clip to the AOI, reprojected to
    `working_crs`. Writes `magnetic_<region>.tif` + a placeholder NaN
    `gravity_<region>.tif` (real gravity integration TODO — v2.1).

    Returns (magnetic_path, gravity_path).
    """
    out_dir = dataset_dir(NAME)
    global_path = out_dir / "EMAG2_V3_20170530_Sealevel.tif"
    mag_path = out_dir / f"magnetic_{aoi.name.lower()}.tif"
    grav_path = out_dir / f"gravity_{aoi.name.lower()}.tif"

    if not global_path.exists() or force:
        _download(global_path)
    else:
        print(f"EMAG2 global grid present ({global_path.stat().st_size:,} B); skipping download.")

    # Clip to AOI + reproject
    print(f"Clipping + reprojecting to {working_crs}...")
    # EMAG2 grid is EPSG:4326 equirectangular, but longitude runs 0–360° E,
    # not -180 to 180. Convert AOI bbox accordingly.
    da = xr.open_dataarray(global_path, engine="rasterio").squeeze("band", drop=True)
    west, south, east, north = aoi.bbox
    west_e, east_e = (west % 360), (east % 360)  # convert -131.5 → 228.5
    pad = 0.1
    aoi_slice = da.sel(
        x=slice(min(west_e, east_e) - pad, max(west_e, east_e) + pad),
        y=slice(north + pad, south - pad),
    )
    print(f"  AOI slice shape (y, x): {aoi_slice.shape}  "
          f"(x range {float(aoi_slice.x.min()):.3f}..{float(aoi_slice.x.max()):.3f} °E)")

    # Reassign x to the -180..180 convention so reproject lands it correctly.
    aoi_slice = aoi_slice.assign_coords(x=((aoi_slice.x + 180) % 360) - 180).sortby("x")
    aoi_slice.rio.write_crs("EPSG:4326", inplace=True)
    # Strip any pre-existing _FillValue attr that rioxarray would refuse to overwrite.
    aoi_slice.attrs.pop("_FillValue", None)
    reproj = aoi_slice.rio.reproject(working_crs, resampling=1)  # 1 = bilinear
    reproj.attrs.pop("_FillValue", None)
    reproj.rio.to_raster(mag_path, compress="deflate", tiled=True)
    print(f"  wrote {mag_path} ({mag_path.stat().st_size:,} bytes)")

    # Gravity placeholder — same NaN approach as before. TODO v2.1.
    from ai_minerals.data.gsc_geophysics import _write_nan_grid
    _write_nan_grid(grav_path, aoi, working_crs, field_name="bouguer_gravity_mGal")
    print(f"  wrote gravity placeholder {grav_path} (all-NaN; v2.1 TODO)")

    write_source_md(
        NAME,
        title="Geophysics — EMAG2 v3 (aeromag) + NaN gravity placeholder",
        url=EMAG2_URL,
        license="EMAG2: US public domain (NOAA). Gravity placeholder pending v2.1.",
        notes=(
            "magnetic_<region>.tif: EMAG2 v3 sea-level (2017-05-30 release), "
            "clipped to AOI + reprojected to working CRS. 2 arc-minute "
            "resolution — coarser than NRCan 1 km native but directly "
            "downloadable. Gravity is a NaN placeholder; WGM2012 integration "
            "is a v2.1 TODO (JS-rendered BGI catalogue, likely needs account)."
        ),
    )
    return mag_path, grav_path


if __name__ == "__main__":
    from ai_minerals.regions.bcgt import BCGT
    fetch(BCGT.aoi, working_crs=BCGT.working_crs)
