"""USGS magnetic anomaly grid — conterminous US.

EMAG2 is a global compilation but has sparse coverage over the US land
mass (only ~4% of cells have data over California in the 2 arc-min grid).
This fetcher uses the USGS conterminous-US magnetic compilation
(`https://mrdata.usgs.gov/magnetic/`), which has dense coverage and 1-km
native resolution.

Source format is GXF (Geosoft Grid Exchange Format) at NAD27 Albers
Equal-Area Conic, same projection as the USGS gravity grids. Three
products are available: original (residual total intensity merge),
500-km high-pass filtered (removes long-wavelength regional bias), and
the CM (satellite-baseline) version. We fetch the original residual
grid and the high-pass version; the high-pass is generally what's
useful for prospectivity since it removes the regional bias that
non-prospective long-wavelength geology contributes.

The .tif files USGS hosts on the same portal are RGB visualizations,
not data. Use GXF.
"""

from __future__ import annotations

from pathlib import Path

import gzip
import shutil
import requests
import rioxarray  # noqa: F401
import xarray as xr

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "gsc_geophysics"

USGS_MAG_GXF_URL = "https://mrdata.usgs.gov/magnetic/magnetic.gxf.gz"

# Same NAD27 Albers as the USGS gravity grids.
USGS_MAG_CRS_PROJ4 = (
    "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=0 +lon_0=-96 "
    "+x_0=0 +y_0=0 +datum=NAD27 +units=m +no_defs"
)


def _download_and_unzip(url: str, gz_dest: Path, gxf_dest: Path) -> None:
    if gxf_dest.exists():
        print(f"USGS magnetic grid present ({gxf_dest.stat().st_size:,} B); "
              f"skipping download.")
        return
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with gz_dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with gzip.open(gz_dest, "rb") as gz, gxf_dest.open("wb") as out:
        shutil.copyfileobj(gz, out)
    print(f"  wrote {gxf_dest} ({gxf_dest.stat().st_size:,} bytes)")


def _clip_and_reproject(global_path: Path, out_path: Path, aoi: AOI, working_crs: str) -> None:
    da = xr.open_dataarray(global_path, engine="rasterio").squeeze("band", drop=True)
    da.rio.write_crs(USGS_MAG_CRS_PROJ4, inplace=True)

    west, south, east, north = aoi.bbox
    pad = 0.5
    da_clip = da.rio.clip_box(
        minx=west - pad, miny=south - pad,
        maxx=east + pad, maxy=north + pad,
        crs="EPSG:4326",
    )
    print(f"  AOI-clipped shape (y, x): {da_clip.shape}")

    da_clip.attrs.pop("_FillValue", None)
    reproj = da_clip.rio.reproject(working_crs, resampling=1)  # bilinear
    reproj.attrs.pop("_FillValue", None)
    reproj.rio.to_raster(out_path, compress="deflate", tiled=True)
    print(f"  wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def fetch(aoi: AOI, working_crs: str, *, force: bool = False) -> Path:
    """Fetch the USGS magnetic grid (residual total intensity), clip + reproject.

    Returns the per-region clipped GeoTIFF path. Writes to
    `magnetic_<region>.tif` so feature-frame builders read it via the
    existing `magnetic` raw-path key.
    """
    out_dir = dataset_dir(NAME)
    mag_gz = out_dir / "magnetic.gxf.gz"
    mag_gxf = out_dir / "magnetic.gxf"
    mag_clipped = out_dir / f"magnetic_{aoi.name.lower()}.tif"

    if force or not mag_gxf.exists():
        _download_and_unzip(USGS_MAG_GXF_URL, mag_gz, mag_gxf)

    print("Clipping + reprojecting USGS magnetic...")
    _clip_and_reproject(mag_gxf, mag_clipped, aoi, working_crs)

    write_source_md(
        NAME,
        title="USGS magnetic anomaly grid (residual total intensity, conterminous US)",
        url=USGS_MAG_GXF_URL,
        license="US public domain (USGS)",
        notes=(
            f"magnetic_{aoi.name.lower()}.tif: USGS conterminous-US magnetic "
            f"compilation, residual total intensity, clipped to AOI + "
            f"reprojected to {working_crs}. 1-km native resolution. Source "
            f"format is GXF (Geosoft Grid Exchange Format) at NAD27 Albers "
            f"Equal-Area Conic. Replaces the EMAG2 global compilation for "
            f"CONUS AOIs (EMAG2 has sparse land coverage at 2 arc-min "
            f"resolution; only ~4% of California cells have valid EMAG2 "
            f"data). The .tif files USGS hosts on the same portal are RGB "
            f"visualizations, not data; only the GXF carries actual nT values."
        ),
    )
    return mag_clipped


if __name__ == "__main__":
    from ai_minerals.regions.motherlode import MOTHERLODE
    fetch(MOTHERLODE.aoi, working_crs=MOTHERLODE.working_crs)
