"""Sentinel-2 mosaic via Element 84's Earth Search STAC on AWS.

Parallel fallback to sentinel2_mosaic.py (which hits Microsoft Planetary
Computer). Same shape as the PC mosaic but uses:

  - STAC endpoint: https://earth-search.aws.element84.com/v1
  - Collection: sentinel-2-l2a  (COGs in s3://sentinel-s2-l2a-cogs, us-west-2)
  - No auth, no signing — pure anonymous HTTPS to the public S3 bucket

Rationale for this alternative: the Planetary Computer backend has been
flaky on hour-long EastAK computes (expiring SAS tokens, transient Azure
blob errors, scale-dependent stackstac hangs). AWS Open Data is
independent infrastructure with a different failure profile, and Earth
Search v1 has effectively identical STAC semantics to Planetary Computer
so the calling code is a one-import swap.

Sources:
  - Earth Search v1: https://element84.com/earth-search
  - STAC tutorial: https://stacspec.org/en/tutorials/access-sentinel-2-data-aws/
  - Registry of Open Data: https://registry.opendata.aws/sentinel-2-l2a-cogs/
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import pystac_client
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import box as shp_box

from ai_minerals.aoi import AOI, WORKING_CRS
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "sentinel2"
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_DATETIME = "2024-07-10/2024-08-25"
DEFAULT_MAX_CLOUD = 10

# Earth Search uses lowercase band names (blue/green/red/nir/swir16/swir22)
# as asset keys, differing from Planetary Computer's uppercase B02/B03/etc.
# Use the PC-style as the canonical names in the output.
BAND_ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}
BANDS = list(BAND_ASSETS.keys())


def _select_best_per_tile(items: list) -> list:
    by_tile: dict[str, object] = {}
    for item in sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 999)):
        # Earth Search exposes the MGRS tile as 's2:mgrs_tile' or in 'grid:code'
        tile = (
            item.properties.get("s2:mgrs_tile")
            or item.properties.get("grid:code", "").replace("MGRS-", "")
            or item.id.split("_")[1]  # e.g. S2A_T06VXQ_20240805T210029_L2A
        )
        if tile not in by_tile:
            by_tile[tile] = item
    return list(by_tile.values())


def _read_scene_tile(item, aoi: AOI, resolution: int) -> "xarray.DataArray | None":
    aoi_wgs = gpd.GeoSeries([aoi.polygon], crs=aoi.crs)
    bands_out = []
    native_crs = None
    clip_poly_native = None
    for band, asset_key in BAND_ASSETS.items():
        if asset_key not in item.assets:
            raise RuntimeError(f"item {item.id} lacks asset {asset_key!r}; available: {list(item.assets)}")
        asset = item.assets[asset_key]
        da = rioxarray.open_rasterio(asset.href, masked=True).squeeze()
        if native_crs is None:
            native_crs = da.rio.crs
            aoi_in_native = aoi_wgs.to_crs(native_crs).iloc[0]
            scene_poly = shp_box(*da.rio.bounds())
            if not aoi_in_native.intersects(scene_poly):
                return None
            clip_poly_native = aoi_in_native.intersection(scene_poly)
            if clip_poly_native.is_empty:
                return None
        w, s, e, n = clip_poly_native.bounds
        clipped = da.rio.clip_box(minx=w, miny=s, maxx=e, maxy=n)
        reprojected = clipped.rio.reproject(
            WORKING_CRS,
            resolution=resolution,
            resampling=Resampling.bilinear,
        )
        bands_out.append(reprojected)

    stacked = xr.concat(
        [b.expand_dims(band=[BANDS[i]]) for i, b in enumerate(bands_out)],
        dim="band",
    )
    stacked.rio.write_crs(WORKING_CRS, inplace=True)
    return stacked


def fetch(
    aoi: AOI,
    *,
    datetime: str = DEFAULT_DATETIME,
    max_cloud: int = DEFAULT_MAX_CLOUD,
    resolution: int = 120,
) -> Path:
    """Build a Sentinel-2 mosaic from Earth Search / AWS COGs (no auth)."""
    # No modifier — Earth Search is anonymous.
    client = pystac_client.Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=aoi.bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    print(f"[earth-search] {len(items)} candidates for AOI={aoi.name}, cloud<{max_cloud}%.", flush=True)
    if not items:
        raise RuntimeError("no low-cloud scenes found")

    selected = _select_best_per_tile(items)
    print(f"  keeping {len(selected)} scenes (one per MGRS tile)", flush=True)

    tiles_dir = dataset_dir(NAME) / "tiles_aws"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_pattern = f"s2_{aoi.name.lower()}_aws_"
    for p in tiles_dir.glob(f"{tile_pattern}*.tif"):
        p.unlink()

    written: list[Path] = []
    t0 = time.perf_counter()
    for i, item in enumerate(selected, 1):
        tile = (
            item.properties.get("s2:mgrs_tile")
            or item.properties.get("grid:code", "").replace("MGRS-", "")
            or f"scene{i}"
        )
        cc = item.properties.get("eo:cloud_cover", -1)
        t_start = time.perf_counter()
        try:
            da = _read_scene_tile(item, aoi, resolution)
            if da is None:
                print(f"  [{i}/{len(selected)}] {tile}: no AOI overlap, skipped", flush=True)
                continue
            tile_path = tiles_dir / f"{tile_pattern}{tile}.tif"
            da.rio.to_raster(tile_path, compress="deflate", tiled=True)
            written.append(tile_path)
            dt = time.perf_counter() - t_start
            print(
                f"  [{i}/{len(selected)}] {tile}: cloud={cc:.2f}%, "
                f"{tile_path.stat().st_size // 1024} KB in {dt:.1f}s",
                flush=True,
            )
        except Exception as exc:
            print(
                f"  [{i}/{len(selected)}] {tile}: FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )

    print(f"  total read: {time.perf_counter() - t0:.1f}s; wrote {len(written)} files", flush=True)

    # rasterio.merge rejects north-up TIFFs with negative pixel height.
    # Use rioxarray.merge.merge_arrays instead.
    from rioxarray.merge import merge_arrays as _rx_merge
    arrs = [rioxarray.open_rasterio(p, masked=True) for p in written]
    merged = _rx_merge(arrs, nodata=float("nan"))
    merged.attrs.pop("_FillValue", None)
    merged.encoding.pop("_FillValue", None)
    merged = merged.astype("float32")
    merged.rio.write_nodata(float("nan"), inplace=True)
    out_path = dataset_dir(NAME) / f"s2_mosaic_aws_{aoi.name.lower()}.tif"
    merged.rio.to_raster(
        out_path,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
        dtype="float32",
    )

    write_source_md(
        NAME,
        title="Sentinel-2 L2A mosaic (Element 84 Earth Search, AWS)",
        url=STAC_URL,
        license=(
            "Copernicus Sentinel data — free, full, and open. "
            "https://sentinels.copernicus.eu/web/sentinel/terms-conditions"
        ),
        notes=(
            f"AOI={aoi.name}, {len(written)} MGRS-tile scenes (one per tile, "
            f"lowest cloud), bands={BANDS}, resolution={resolution} m, "
            f"reprojected to {WORKING_CRS}. No auth required — data lives in "
            "s3://sentinel-s2-l2a-cogs (us-west-2). Intermediates at "
            "data/raw/sentinel2/tiles_aws/."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import EASTERN_ALASKA

    path = fetch(EASTERN_ALASKA)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)", flush=True)
