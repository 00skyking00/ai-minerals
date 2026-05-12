"""USGS gravity grids — Bouguer + isostatic residual, conterminous US.

Replaces the NaN-placeholder gravity raster that emag2_geophysics writes
when used outside of US AOIs that need real gravity data.

Source: https://mrdata.usgs.gov/gravity/ — direct GXF (Geosoft Grid
Exchange Format) downloads. The .tif files USGS hosts on the same
portal are *RGB visualizations*, not data; the actual gridded mGal
values are in the GXF format. GDAL has a GXF driver, so rasterio reads
them directly.

Native projection: NAD27 / Albers Equal-Area Conic with lat_1=29.5,
lat_2=45.5, lat_0=0, lon_0=-96. The GXF header declares NAD27 but
GDAL's auto-detected CRS only carries the geographic datum, not the
Albers projection. We override the CRS with the correct proj string
at load time. NAD27-vs-NAD83 differs by under 10 m in CONUS, well
within the 4 km grid cell.

Bouguer corrects for topographic mass; isostatic additionally corrects
for the deep crustal mass that supports topography. We fetch both and
let the feature-frame builder pick.
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

# Real-data GXF format (the .tif files at the same paths are RGB displays)
USGS_BOUGUER_GXF_URL = "https://mrdata.usgs.gov/gravity/bouguer/bouguer.gxf.gz"
USGS_ISOSTATIC_GXF_URL = "https://mrdata.usgs.gov/gravity/isostatic/isograv.gxf.gz"

# NAD27 Albers Equal-Area Conic, parameters from the GXF header.
USGS_GRAVITY_CRS_PROJ4 = (
    "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=0 +lon_0=-96 "
    "+x_0=0 +y_0=0 +datum=NAD27 +units=m +no_defs"
)


def _download_and_unzip(url: str, gz_dest: Path, gxf_dest: Path) -> None:
    if gxf_dest.exists():
        print(f"USGS gravity grid present ({gxf_dest.stat().st_size:,} B); skipping download.")
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
    # Override the auto-detected CRS with the correct Albers proj string.
    da.rio.write_crs(USGS_GRAVITY_CRS_PROJ4, inplace=True)

    # Reproject AOI bbox into the source CRS so we can clip by extent.
    west, south, east, north = aoi.bbox
    pad = 0.5  # degrees, generous since it's lat/lon -> meters
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


def fetch(aoi: AOI, working_crs: str, *, force: bool = False) -> tuple[Path, Path]:
    """Fetch USGS Bouguer + isostatic gravity grids, clip to AOI, reproject.

    Returns (bouguer_path, isostatic_path). The Bouguer file is also written
    to the path that emag2_geophysics would have written its NaN placeholder
    to, so existing assemble code reads real gravity automatically.
    """
    out_dir = dataset_dir(NAME)
    bouguer_gz = out_dir / "bouguer.gxf.gz"
    bouguer_gxf = out_dir / "bouguer.gxf"
    isostatic_gz = out_dir / "isograv.gxf.gz"
    isostatic_gxf = out_dir / "isograv.gxf"

    bouguer_clipped = out_dir / f"gravity_{aoi.name.lower()}.tif"
    isostatic_clipped = out_dir / f"gravity_isostatic_{aoi.name.lower()}.tif"

    if force or not bouguer_gxf.exists():
        _download_and_unzip(USGS_BOUGUER_GXF_URL, bouguer_gz, bouguer_gxf)
    if force or not isostatic_gxf.exists():
        _download_and_unzip(USGS_ISOSTATIC_GXF_URL, isostatic_gz, isostatic_gxf)

    print("Clipping + reprojecting Bouguer...")
    _clip_and_reproject(bouguer_gxf, bouguer_clipped, aoi, working_crs)
    print("Clipping + reprojecting isostatic...")
    _clip_and_reproject(isostatic_gxf, isostatic_clipped, aoi, working_crs)

    write_source_md(
        NAME,
        title="USGS gravity grids (Bouguer + isostatic residual, conterminous US)",
        url=USGS_BOUGUER_GXF_URL,
        license="US public domain (USGS)",
        notes=(
            f"gravity_{aoi.name.lower()}.tif: USGS Bouguer anomaly grid "
            f"clipped to AOI + reprojected to {working_crs}. "
            f"gravity_isostatic_{aoi.name.lower()}.tif: same clip+reproject "
            "for the isostatic residual grid. Source format is GXF "
            "(Geosoft Grid Exchange Format) at 4-km grid spacing in NAD27 "
            "Albers Equal-Area Conic projection. The .tif files USGS hosts "
            "on the same portal are RGB visualizations, not data; only the "
            "GXF carries actual mGal values. Both grids cover the "
            "conterminous United States only; non-CONUS AOIs (BCGT, EastAK) "
            "keep the EMAG2 NaN gravity placeholder."
        ),
    )
    return bouguer_clipped, isostatic_clipped


if __name__ == "__main__":
    from ai_minerals.regions.motherlode import MOTHERLODE
    fetch(MOTHERLODE.aoi, working_crs=MOTHERLODE.working_crs)
