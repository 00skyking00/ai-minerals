"""Sentinel-2 mosaic via direct rioxarray reads — no stackstac, no dask.

Fallback for when `stackstac + dask + rio.to_raster` hangs on large AOIs
(reproducibly does on EastAK at exactly 40% completion regardless of scheduler
or temporal-reduction mode — see feedback_stackstac_deadlock memory).

Strategy — streaming per-tile writes to disk to keep peak memory bounded:

  1. Query STAC for scenes overlapping the AOI.
  2. Dedupe by MGRS tile, keep the lowest-cloud scene per tile.
  3. For each selected scene (~20 for EastAK):
     - Read all 6 bands with rioxarray (HTTP range reads over the AOI intersection)
     - Reproject to the target CRS + resolution
     - Stack into a (6, y, x) DataArray
     - **Write to disk as an intermediate per-tile GeoTIFF, free memory**
  4. At the end: rasterio.merge over the 20 per-tile TIFFs -> final mosaic.

Per-tile peak memory: ~6 bands × ~2000 × ~2000 × 4B = ~100 MB.
Total disk: ~20 tiles × ~15 MB = ~300 MB of intermediates + final ~200 MB.

No temporal reduction — each pixel takes its value from its MGRS tile's
lowest-cloud scene. Good enough for regional alteration indices.
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import box as shp_box

from ai_minerals.aoi import AOI, WORKING_CRS
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "sentinel2"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_DATETIME = "2024-07-10/2024-08-25"
DEFAULT_MAX_CLOUD = 10
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def _select_best_per_tile(items: list) -> list:
    by_tile: dict[str, object] = {}
    for item in sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 999)):
        tile = item.properties.get("s2:mgrs_tile") or item.id.split("_")[-2]
        if tile not in by_tile:
            by_tile[tile] = item
    return list(by_tile.values())


def _read_scene_tile(item, aoi: AOI, resolution: int) -> "xarray.DataArray | None":
    """Read all 6 bands for one scene over the AOI, stacked into (band, y, x).

    Returns None if the scene doesn't overlap the AOI.
    """
    # AOI polygon reprojected into each band's native CRS so we can clip before read.
    # All S2 L2A bands in one tile share a CRS (the UTM of that MGRS tile).
    aoi_wgs = gpd.GeoSeries([aoi.polygon], crs=aoi.crs)

    bands_out = []
    native_crs = None
    clip_poly_native = None
    for band in BANDS:
        asset = item.assets[band]
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
    """Build a Sentinel-2 mosaic by direct rioxarray reads + per-tile streaming."""
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = client.search(
        collections=[COLLECTION],
        bbox=aoi.bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    print(f"Sentinel-2: {len(items)} candidates for AOI={aoi.name}, cloud<{max_cloud}%.", flush=True)
    if not items:
        raise RuntimeError("no low-cloud scenes found; relax max_cloud or widen datetime.")

    selected = _select_best_per_tile(items)
    print(f"  keeping {len(selected)} scenes (one per MGRS tile)", flush=True)

    tiles_dir = dataset_dir(NAME) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous tile intermediates for this AOI.
    tile_pattern = f"s2_{aoi.name.lower()}_"
    for p in tiles_dir.glob(f"{tile_pattern}*.tif"):
        p.unlink()

    written: list[Path] = []
    t0 = time.perf_counter()
    for i, item in enumerate(selected, 1):
        tile = item.properties.get("s2:mgrs_tile", item.id[-11:-6])
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

    print(f"  all tiles: {time.perf_counter() - t0:.1f}s total; wrote {len(written)} files", flush=True)

    # Final mosaic via rioxarray.merge.merge_arrays. rasterio.merge refuses
    # north-up TIFFs with negative pixel height, which every tile here has.
    print(f"  merging {len(written)} tile GeoTIFFs → final mosaic...", flush=True)
    from rioxarray.merge import merge_arrays as _rx_merge
    arrs = [rioxarray.open_rasterio(p, masked=True) for p in written]
    merged = _rx_merge(arrs, nodata=float("nan"))
    merged.attrs.pop("_FillValue", None)
    merged.encoding.pop("_FillValue", None)
    merged = merged.astype("float32")
    merged.rio.write_nodata(float("nan"), inplace=True)
    out_path = dataset_dir(NAME) / f"s2_mosaic_{aoi.name.lower()}.tif"
    merged.rio.to_raster(
        out_path,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
        dtype="float32",
    )

    write_source_md(
        NAME,
        title="Sentinel-2 L2A mosaic (direct rioxarray, no stackstac)",
        url=f"{STAC_URL}/collections/{COLLECTION}",
        license=(
            "Copernicus Sentinel data — free, full, and open. "
            "https://sentinels.copernicus.eu/web/sentinel/terms-conditions"
        ),
        notes=(
            f"AOI={aoi.name}, {len(written)} MGRS-tile scenes (one per tile, "
            f"lowest cloud), bands={BANDS}, resolution={resolution} m, "
            f"reprojected to {WORKING_CRS}. No temporal reduction — each "
            "pixel takes the value from its tile's selected scene. "
            "Intermediates cached at data/raw/sentinel2/tiles/."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import EASTERN_ALASKA

    path = fetch(EASTERN_ALASKA)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)", flush=True)
