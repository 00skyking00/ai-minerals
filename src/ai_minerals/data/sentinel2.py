"""Sentinel-2 L2A composite via Microsoft Planetary Computer STAC.

Supports three aggregation modes with different cost/quality tradeoffs:

  mean   -- arithmetic mean over a small set of low-cloud scenes (default).
            Dask can stream; cheap. Tradeoff: a few residual cloud pixels.
  median -- true median across many scenes. Eliminates residual cloud/haze
            but is memory-heavy: dask must hold every scene in memory per
            output chunk, because median is not a streaming reduction.
  first  -- single best scene (lowest cloud). Fastest; noisy where the one
            scene happens to be obscured.

Usage (all optional args have sensible Tanacross defaults):

    # default: mean of 6 cleanest scenes at 60 m
    uv run python -m ai_minerals.data.sentinel2

    # full median of all low-cloud scenes at 120 m, with live progress
    uv run python -m ai_minerals.data.sentinel2 --mode median --resolution 120

    # single-threaded (easier to debug if things look stuck)
    uv run python -m ai_minerals.data.sentinel2 --mode median --scheduler synchronous

    # cap the scene count when experimenting
    uv run python -m ai_minerals.data.sentinel2 --mode median --scene-limit 10

Memory budget for median mode (per-chunk):
    chunksize^2 * scenes * bands * 4 bytes
  e.g. 512^2 * 30 * 6 * 4 B ~= 190 MB per chunk, * 24 threads ~= 4.5 GB.
The --chunk-size flag tightens this if you'd rather be slow than OOM.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import dask
import planetary_computer
import pystac_client
import stackstac
from dask.diagnostics import ProgressBar

from ai_minerals.aoi import AOI, WORKING_CRS
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "sentinel2"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_DATETIME = "2024-07-10/2024-08-25"
DEFAULT_MAX_CLOUD = 10
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Per-mode default scene count. None means "all scenes matching the search".
_DEFAULT_SCENE_LIMITS: dict[str, int | None] = {
    "mean": 6,
    "median": None,
    "first": 1,
}


def _search(
    aoi: AOI, datetime: str, max_cloud: int
) -> list:
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = client.search(
        collections=[COLLECTION],
        bbox=aoi.bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 999))
    return items


def fetch(
    aoi: AOI,
    *,
    mode: str = "mean",
    resolution: int = 60,
    scene_limit: int | None = None,
    datetime: str = DEFAULT_DATETIME,
    max_cloud: int = DEFAULT_MAX_CLOUD,
    chunksize: int = 512,
    scheduler: str | None = None,
    progress: bool = True,
) -> Path:
    """Compute a Sentinel-2 L2A composite over the AOI and write to a GeoTIFF.

    Parameters
    ----------
    mode : {"mean", "median", "first"}
        Temporal aggregation. See module docstring.
    resolution : int
        Output pixel size in meters.
    scene_limit : int or None
        Cap on the number of scenes used. Default depends on mode.
    datetime, max_cloud
        STAC search filters.
    chunksize : int
        Spatial chunk edge, in pixels. Smaller -> less peak memory, more overhead.
    scheduler : {"threads", "synchronous", "processes"}
        Dask scheduler. Use "synchronous" when you want deterministic
        single-threaded behavior for debugging.
    progress : bool
        Show a live dask ProgressBar during the to_raster write.
    """
    import rioxarray  # noqa: F401  -- registers rio accessor

    if mode not in _DEFAULT_SCENE_LIMITS:
        raise ValueError(f"mode must be one of {sorted(_DEFAULT_SCENE_LIMITS)}; got {mode!r}")

    # Why: stackstac + rio.to_raster + the default `threads` scheduler
    # deadlocks (all workers in futex_wait_queue) for median composites,
    # even on trivially small inputs. `synchronous` works. `threads` is
    # fine for mean/first which are streaming reductions.
    if scheduler is None:
        scheduler = "synchronous" if mode == "median" else "threads"

    items = _search(aoi, datetime, max_cloud)
    print(
        f"Sentinel-2: {len(items)} candidate items for AOI={aoi.name}, "
        f"datetime={datetime}, cloud<{max_cloud}%."
    )
    if not items:
        raise RuntimeError(
            "No Sentinel-2 items. Try relaxing --max-cloud or widening --datetime."
        )

    cap = scene_limit if scene_limit is not None else _DEFAULT_SCENE_LIMITS[mode]
    kept = items if cap is None else items[:cap]
    clouds = [round(it.properties.get("eo:cloud_cover", -1), 3) for it in kept]
    print(
        f"  Using {len(kept)} scene(s) in mode={mode!r}, resolution={resolution} m, "
        f"chunksize={chunksize}. cloud%: {clouds}"
    )

    stack = stackstac.stack(
        kept,
        assets=BANDS,
        epsg=int(WORKING_CRS.split(":")[1]),
        bounds_latlon=aoi.bbox,
        resolution=resolution,
        chunksize=chunksize,
    )
    print(f"  Raw stack shape (time, band, y, x): {stack.shape}")

    if mode == "mean":
        composite = stack.mean("time", skipna=True)
    elif mode == "median":
        composite = stack.median("time", skipna=True)
    elif mode == "first":
        composite = stack.isel(time=0)
    composite.name = f"sentinel2_l2a_{mode}"
    print(f"  Composite shape (band, y, x): {composite.shape}")

    out_path = dataset_dir(NAME) / f"s2_{mode}_{aoi.name.lower()}.tif"
    composite.rio.write_crs(WORKING_CRS, inplace=True)

    pb_ctx = ProgressBar() if progress else nullcontext()
    print(f"  Writing {out_path} with scheduler={scheduler!r}...")
    with dask.config.set(scheduler=scheduler), pb_ctx:
        composite.rio.to_raster(
            out_path,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

    write_source_md(
        NAME,
        title=f"Sentinel-2 L2A {mode} composite (Planetary Computer)",
        url=f"{STAC_URL}/collections/{COLLECTION}",
        license=(
            "Copernicus Sentinel data — free, full, and open. "
            "https://sentinels.copernicus.eu/web/sentinel/terms-conditions"
        ),
        notes=(
            f"AOI={aoi.name}, mode={mode}, scenes={len(kept)}, "
            f"datetime={datetime}, max_cloud={max_cloud}%, resolution={resolution} m, "
            f"bands={BANDS}, reprojected to {WORKING_CRS}."
        ),
    )
    return out_path


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="mean", choices=sorted(_DEFAULT_SCENE_LIMITS))
    parser.add_argument("--resolution", type=int, default=60, help="output pixel size in meters (default: 60)")
    parser.add_argument("--scene-limit", type=int, default=None, help="cap on scenes (default depends on --mode)")
    parser.add_argument("--datetime", default=DEFAULT_DATETIME, help=f"STAC datetime filter (default: {DEFAULT_DATETIME})")
    parser.add_argument("--max-cloud", type=int, default=DEFAULT_MAX_CLOUD, help="max cloud cover %% (default: 10)")
    parser.add_argument("--chunksize", type=int, default=512, help="dask chunk edge in pixels (default: 512)")
    parser.add_argument(
        "--scheduler", default=None,
        choices=["threads", "synchronous", "processes"],
        help=(
            "dask scheduler. Default is picked per mode: 'synchronous' for median "
            "(avoids a known dask/GDAL deadlock), 'threads' for mean/first."
        ),
    )
    parser.add_argument("--no-progress", action="store_true", help="suppress dask ProgressBar")
    args = parser.parse_args()

    from ai_minerals.aoi import TANACROSS

    path = fetch(
        TANACROSS,
        mode=args.mode,
        resolution=args.resolution,
        scene_limit=args.scene_limit,
        datetime=args.datetime,
        max_cloud=args.max_cloud,
        chunksize=args.chunksize,
        scheduler=args.scheduler,
        progress=not args.no_progress,
    )
    size = path.stat().st_size
    print(f"Wrote {path} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
