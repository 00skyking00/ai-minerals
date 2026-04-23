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


def fetch(aoi: AOI) -> Path:
    """Fetch GLO-30 DEM for the AOI, reproject to Alaska Albers, save as GeoTIFF."""
    import rioxarray  # noqa: F401  -- registers rio accessor

    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = client.search(collections=[COLLECTION], bbox=aoi.bbox)
    items = list(search.items())
    print(f"DEM: found {len(items)} GLO-30 tile(s) covering {aoi.name}.")
    if not items:
        raise RuntimeError(f"No GLO-30 tiles found for {aoi.name}.")

    stack = stackstac.stack(
        items,
        epsg=int(WORKING_CRS.split(":")[1]),
        bounds_latlon=aoi.bbox,
        resolution=30,
    )
    # GLO-30 collection has a single asset 'data'; take the first band.
    arr = stack.squeeze("band").max("time").compute()
    arr.name = "elevation"

    out_path = dataset_dir(NAME) / f"dem_{aoi.name.lower()}.tif"
    arr.rio.write_crs(WORKING_CRS, inplace=True)
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
            f"GLO-30 tiles covering AOI {aoi.name}, reprojected to {WORKING_CRS}, "
            "30 m nominal resolution. Retrieved via Microsoft Planetary Computer STAC."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import EASTERN_ALASKA

    path = fetch(EASTERN_ALASKA)
    print(f"Wrote {path}")
