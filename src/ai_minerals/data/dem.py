"""Copernicus GLO-30 DEM via Microsoft Planetary Computer STAC."""

from __future__ import annotations

from pathlib import Path

import planetary_computer
import pystac_client
import stackstac

from ai_minerals.aoi import AOI, WORKING_CRS
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "dem"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "cop-dem-glo-30"


def fetch(aoi: AOI, working_crs: str = WORKING_CRS) -> Path:
    """Fetch GLO-30 DEM for the AOI, reproject to `working_crs`, save as GeoTIFF.

    `working_crs` defaults to the Alaska Albers constant (EPSG:3338) for v1
    back-compat; pass the BCGT / Mother Lode working CRS when fetching for
    those regions.
    """
    import rioxarray  # noqa: F401  -- registers rio accessor

    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = client.search(collections=[COLLECTION], bbox=aoi.bbox)
    items = list(search.items())
    print(f"DEM: found {len(items)} GLO-30 tile(s) covering {aoi.name}.")
    if not items:
        raise RuntimeError(f"No GLO-30 tiles found for {aoi.name}.")

    stack = stackstac.stack(
        items,
        epsg=int(working_crs.split(":")[1]),
        bounds_latlon=aoi.bbox,
        resolution=30,
    )
    arr = stack.squeeze("band").max("time").compute()
    arr.name = "elevation"

    out_path = dataset_dir(NAME) / f"dem_{aoi.name.lower()}.tif"
    arr.rio.write_crs(working_crs, inplace=True)
    arr.rio.to_raster(out_path, compress="deflate", tiled=True)

    write_source_md(
        NAME,
        title="Copernicus GLO-30 DEM (Planetary Computer)",
        url=f"{STAC_URL}/collections/{COLLECTION}",
        license=(
            "Open, free and perpetual (Copernicus programme) — "
            "https://docs.sentinel-hub.com/api/latest/static/files/data/dem/resources/"
            "license/License-COPDEM-30.pdf"
        ),
        notes=(
            f"GLO-30 tiles covering AOI {aoi.name}, reprojected to {working_crs}, "
            "30 m nominal resolution. Retrieved via Microsoft Planetary Computer STAC."
        ),
    )
    return out_path


if __name__ == "__main__":
    import argparse
    from ai_minerals.regions.eastak import EASTAK
    from ai_minerals.regions.bcgt import BCGT
    from ai_minerals.regions.motherlode import MOTHERLODE

    regions_by_slug = {r.slug: r for r in (EASTAK, BCGT, MOTHERLODE)}
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="eastak", choices=list(regions_by_slug))
    args = p.parse_args()

    region = regions_by_slug[args.region]
    path = fetch(region.aoi, working_crs=region.working_crs)
    print(f"Wrote {path}")
