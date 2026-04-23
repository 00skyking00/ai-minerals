"""Sentinel-2 mosaic via direct rioxarray reads — no stackstac, no dask.

Fallback for when `stackstac + dask + rio.to_raster` hangs on large AOIs
(reproducibly does on EastAK at exactly 40% completion regardless of scheduler
or temporal-reduction mode — see feedback_stackstac_deadlock memory).

Strategy — streaming per-tile writes to disk to keep peak memory bounded:

  1. Query STAC for scenes overlapping the AOI in a summer window.
  2. For each MGRS tile, keep the top-N (default 3) lowest-cloud scenes
     with cloud<30%. Multiple-scenes-per-tile is load-bearing: any single
     low-cloud scene might be a partial acquisition that misses parts of
     the MGRS tile (we hit this on 06VWR — a 0.71%-cloud partial scene
     missed the Rainbow Ridge porphyry cluster).
  3. For each tile, read each selected scene's 6 bands with rioxarray,
     reproject to the target CRS+resolution, and take the per-pixel mean
     across scenes (NaN-skipping). A pixel only needs one valid scene to
     have data; this fills partial-acquisition gaps.
  4. Write each tile's mean to disk as an intermediate GeoTIFF, free memory.
  5. Merge per-tile GeoTIFFs into the final mosaic with
     rioxarray.merge.merge_arrays (rasterio.merge rejects north-up TIFFs).

Per-tile peak memory: ~6 bands × 3 scenes × ~2000 × ~2000 × 4B ≈ ~300 MB.
Total disk: ~20 tiles × ~15 MB = ~300 MB of intermediates + ~200 MB final.

Snow: the July-August window is Alaska's peak-summer minimum-snow
interval for middle-elevation surfaces (where porphyries outcrop). High
peaks in the Alaska Range (>~2500 m) stay glaciated year-round and will
have permanent-snow pixels regardless of window — those aren't
exploration-relevant. The resulting mosaic is *not* snow-masked; Day-3
feature engineering can add an NDSI-based snow mask if needed.
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

# v3: AWS Earth Search backend — no signing, URLs don't expire, so long-running
# mosaic jobs don't hit the 403-after-~60-minutes wall that Planetary Computer
# imposes with its SAS tokens. Asset keys are lowercase names on Earth Search.
NAME = "sentinel2"
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_DATETIME = "2024-07-10/2024-08-25"
DEFAULT_MAX_CLOUD = 30
DEFAULT_SCENES_PER_TILE = 3
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
# Earth Search asset keys differ from Planetary Computer's B-codes.
BAND_ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}


def _mgrs_tile_id(item) -> str:
    """Return a zero-padded MGRS tile identifier like '06VWP' for an item.

    Robust to both backends: Planetary Computer exposes `s2:mgrs_tile` directly;
    Earth Search provides `mgrs:utm_zone/latitude_band/grid_square` (zone without
    leading zero) or `grid:code` as `MGRS-6VWP`. Cached tile filenames use the
    zero-padded form (e.g. `06VWP`), so we normalize all paths to match.
    """
    props = item.properties
    zone = props.get("mgrs:utm_zone")
    band = props.get("mgrs:latitude_band")
    square = props.get("mgrs:grid_square")
    if zone is not None and band and square:
        return f"{int(zone):02d}{band}{square}"
    tile = props.get("s2:mgrs_tile")
    if tile:
        return str(tile).zfill(5)  # ensure zero-padded zone
    code = props.get("grid:code") or ""
    if code.startswith("MGRS-"):
        code = code[len("MGRS-"):]
        # Strip leading digits for zone and re-pad
        digits = ""
        for ch in code:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits and len(code) > len(digits):
            return f"{int(digits):02d}{code[len(digits):]}"
    # Fallback: rely on the underscored item id (unreliable across backends)
    return item.id.split("_")[-2]


def _select_top_n_per_tile(items: list, n: int) -> dict[str, list]:
    """Group items by MGRS tile, sort each group by cloud %, keep the top N."""
    by_tile: dict[str, list] = {}
    for item in sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 999)):
        tile = _mgrs_tile_id(item)
        by_tile.setdefault(tile, []).append(item)
    return {tile: scenes[:n] for tile, scenes in by_tile.items()}


def _read_scene_tile(item, aoi: AOI, resolution: int) -> "xarray.DataArray | None":
    """Read all 6 bands for one scene over the AOI, stacked into (band, y, x).

    Returns None if the scene doesn't overlap the AOI.
    """
    # v3: no signing — Earth Search S3 URLs are permanent, no 403-after-~1h.
    # AOI polygon reprojected into each band's native CRS so we can clip before read.
    # All S2 L2A bands in one tile share a CRS (the UTM of that MGRS tile).
    aoi_wgs = gpd.GeoSeries([aoi.polygon], crs=aoi.crs)

    bands_out = []
    native_crs = None
    clip_w = clip_s = clip_e = clip_n = None
    for band in BANDS:
        asset_key = BAND_ASSETS[band]
        if asset_key not in item.assets:
            raise RuntimeError(
                f"item {item.id} lacks asset {asset_key!r}; available: {list(item.assets)}"
            )
        asset = item.assets[asset_key]
        da = rioxarray.open_rasterio(asset.href, masked=True).squeeze()
        if native_crs is None:
            native_crs = da.rio.crs
            aoi_in_native = aoi_wgs.to_crs(native_crs).iloc[0]
            scene_bounds = da.rio.bounds()  # (minx, miny, maxx, maxy) in native UTM
            aoi_bounds = aoi_in_native.bounds
            # v2 fix: use the intersection of two rectangular BBOXes rather
            # than the bbox of the polygon intersection. The AOI polygon, after
            # reprojection from WGS84 to UTM, is a curved quadrilateral that
            # narrows at high latitudes — its intersection with the scene has
            # a tighter bbox than the scene itself, artificially clipping away
            # data at UTM zone boundaries (see Rainbow Ridge / 06VXR case).
            clip_w = max(aoi_bounds[0], scene_bounds[0])
            clip_s = max(aoi_bounds[1], scene_bounds[1])
            clip_e = min(aoi_bounds[2], scene_bounds[2])
            clip_n = min(aoi_bounds[3], scene_bounds[3])
            if clip_w >= clip_e or clip_s >= clip_n:
                return None
        clipped = da.rio.clip_box(minx=clip_w, miny=clip_s, maxx=clip_e, maxy=clip_n)
        reprojected = clipped.rio.reproject(
            WORKING_CRS,
            resolution=resolution,
            resampling=Resampling.bilinear,
        )
        bands_out.append(reprojected)

    # Reprojected bands can land on sub-pixel-offset grids because the 6
    # Sentinel-2 assets have different native resolutions (10 m: B02/B03/
    # B04/B08; 20 m: B11/B12). After independent reprojections to the 120 m
    # output grid, floating-point rounding produces sub-pixel coord drift.
    # join='exact' was tried first and rejected these tiles with
    # AlignmentError — use 'outer' which unions the coord axes (sub-pixel
    # diffs become NaN pads, functionally irrelevant at our scale).
    stacked = xr.concat(
        [b.expand_dims(band=[BANDS[i]]) for i, b in enumerate(bands_out)],
        dim="band",
        join="outer",
    )
    stacked.rio.write_crs(WORKING_CRS, inplace=True)
    return stacked


def fetch(
    aoi: AOI,
    *,
    datetime: str = DEFAULT_DATETIME,
    max_cloud: int = DEFAULT_MAX_CLOUD,
    resolution: int = 120,
    scenes_per_tile: int = DEFAULT_SCENES_PER_TILE,
    force: bool = False,
    min_tile_bytes: int = 50_000,
) -> Path:
    """Build a Sentinel-2 mosaic by direct rioxarray reads + per-tile streaming.

    By default, skips any per-tile intermediate that already exists on disk
    with size >= `min_tile_bytes`. This makes iterating on the mosaic cheap:
    a bug fix that only affects some tiles can be deployed and re-run, and
    only the affected tiles need to be re-fetched (delete them first, or
    pass force=True). The final mosaic is always re-merged.
    """
    # Anonymous STAC client — Earth Search is public, no modifier needed.
    client = pystac_client.Client.open(STAC_URL)
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

    tile_groups = _select_top_n_per_tile(items, scenes_per_tile)
    total_scene_reads = sum(len(v) for v in tile_groups.values())
    print(
        f"  keeping up to {scenes_per_tile} scenes per tile: "
        f"{len(tile_groups)} tiles, {total_scene_reads} scene-reads total",
        flush=True,
    )

    tiles_dir = dataset_dir(NAME) / "tiles_v2"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_pattern = f"s2_{aoi.name.lower()}_v2_"
    if force:
        for p in tiles_dir.glob(f"{tile_pattern}*.tif"):
            p.unlink()

    written: list[Path] = []
    t0 = time.perf_counter()
    for i, (tile, scenes) in enumerate(tile_groups.items(), 1):
        tile_path = tiles_dir / f"{tile_pattern}{tile}.tif"
        # Skip if an existing tile looks good on disk (idempotent re-run).
        if not force and tile_path.exists() and tile_path.stat().st_size >= min_tile_bytes:
            print(
                f"  [{i}/{len(tile_groups)}] {tile}: cached ({tile_path.stat().st_size // 1024} KB); skipping",
                flush=True,
            )
            written.append(tile_path)
            continue
        t_start = time.perf_counter()
        per_scene: list = []
        cc_list: list[float] = []
        for scene in scenes:
            cc = scene.properties.get("eo:cloud_cover", -1)
            try:
                da = _read_scene_tile(scene, aoi, resolution)
                if da is not None:
                    per_scene.append(da)
                    cc_list.append(cc)
            except Exception as exc:
                print(f"    scene {scene.id} failed: {type(exc).__name__}: {exc}", flush=True)

        if not per_scene:
            print(
                f"  [{i}/{len(tile_groups)}] {tile}: all {len(scenes)} scenes skipped",
                flush=True,
            )
            continue

        # Mean across scenes (NaN-skipping). xr.concat aligns the per-scene
        # grids by outer join, filling gaps with NaN; the mean then takes
        # whatever scene(s) happen to have data for each pixel.
        if len(per_scene) == 1:
            tile_mean = per_scene[0]
        else:
            stacked = xr.concat(
                [a.expand_dims(scene=[j]) for j, a in enumerate(per_scene)],
                dim="scene",
                join="outer",
            )
            tile_mean = stacked.mean(dim="scene", skipna=True)
        tile_mean.rio.write_crs(WORKING_CRS, inplace=True)

        tile_mean.rio.to_raster(tile_path, compress="deflate", tiled=True)
        written.append(tile_path)
        dt = time.perf_counter() - t_start
        cc_str = ", ".join(f"{c:.1f}" for c in cc_list)
        print(
            f"  [{i}/{len(tile_groups)}] {tile}: mean of {len(per_scene)} scene(s) "
            f"(cloud={cc_str}), {tile_path.stat().st_size // 1024} KB in {dt:.1f}s",
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
    out_path = dataset_dir(NAME) / f"s2_mosaic_v2_{aoi.name.lower()}.tif"
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
            f"AOI={aoi.name}, {len(written)} MGRS-tile means (top-{scenes_per_tile} "
            f"lowest-cloud scenes per tile, cloud<{max_cloud}%, datetime={datetime}), "
            f"bands={BANDS}, resolution={resolution} m, reprojected to {WORKING_CRS}. "
            "Each tile is a per-pixel mean across the selected scenes (NaN-skipping) "
            "— partial-acquisition gaps get filled by other scenes. "
            "Intermediates cached at data/raw/sentinel2/tiles/."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import EASTERN_ALASKA

    path = fetch(EASTERN_ALASKA)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)", flush=True)
