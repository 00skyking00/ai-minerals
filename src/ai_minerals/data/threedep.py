"""USGS 3DEP elevation — 1 m via OpenTopography (preferred), 10 m fallback.

Two access paths:

1. **1 m** via OpenTopography's USGS1m API. Requires an API key in
   `OPENTOPOGRAPHY_API_KEY`; chunked into ~0.5° subtiles to stay under the
   per-request size cap. Coverage is incomplete; cells without 1 m flown
   data return HTTP 400 / empty rasters and are recorded as gaps.

2. **10 m fallback** via the USGS National Map S3 bucket at
   `prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/`.
   Each tile is one degree on a side, named after its northeast corner
   (`USGS_13_n40w121.tif` covers lat 39-40, lon -121 to -120). Always
   succeeds; no key required.

Both paths mosaic into a single GeoTIFF at
`data/raw/3dep_lidar/3dep_1m_northern_sierra.tif` so downstream code reads
one fixed path. A sidecar `*_meta.json` records the actual resolution,
the tile list, and any coverage gaps.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import warnings
from pathlib import Path

import requests
# Load OPENTOPOGRAPHY_API_KEY (and any other project env vars) from the
# repo-root .env before the fetcher reads os.environ. Walks up the file tree
# from the cwd; works the same in interactive shells, VS Code terminals,
# subprocess invocations, and cron. python-dotenv is a base dep.
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "3dep_lidar"
OUT_BASENAME = "3dep_1m_northern_sierra"

OPENTOPO_API = "https://portal.opentopography.org/API/usgsdem"
OPENTOPO_DATASET = "USGS1m"
# OpenTopography's USGS1m enforces a 250 km² per-request area cap. At ~39°N
# one degree of latitude is ~111 km and one degree of longitude is ~86 km,
# so a 0.15° × 0.15° chunk is ~16.7 × 12.9 km ≈ 215 km² — comfortably under
# the cap. Larger chunks (e.g. 0.5°) return HTTP 400 "maximum area for
# USGS1m is 250 km²" and the fetcher falls through to 10 m, defeating the
# point of the OT+ subscription.
OPENTOPO_CHUNK_DEG = 0.15  # per-request bbox edge in degrees

TNM_BASE = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
)


def _opentopo_chunks(
    aoi: AOI, chunk: float = OPENTOPO_CHUNK_DEG
) -> list[tuple[float, float, float, float]]:
    """Tile the AOI into (west, south, east, north) sub-bboxes."""
    west, south, east, north = aoi.bbox
    nx = max(1, math.ceil((east - west) / chunk))
    ny = max(1, math.ceil((north - south) / chunk))
    dx = (east - west) / nx
    dy = (north - south) / ny
    out: list[tuple[float, float, float, float]] = []
    for j in range(ny):
        for i in range(nx):
            out.append(
                (
                    west + i * dx,
                    south + j * dy,
                    west + (i + 1) * dx,
                    south + (j + 1) * dy,
                )
            )
    return out


def _fetch_opentopo_chunk(
    api_key: str,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    timeout: int = 120,
    max_retries: int = 4,
) -> bool:
    """Fetch one OpenTopography USGS1m sub-tile with retry-and-backoff.

    Skips download if `out_path` already exists and is non-trivially sized
    (resumable: rerun the fetcher to continue a partial run).

    Returns True on success, False on a definitive HTTP error (non-200
    that isn't a transient timeout), or False after `max_retries`
    consecutive ReadTimeout / ConnectionError. The caller treats False
    as a coverage gap for that bbox.
    """
    if out_path.exists() and out_path.stat().st_size >= 1024:
        return True

    west, south, east, north = bbox
    params = {
        "datasetName": OPENTOPO_DATASET,
        "south": f"{south:.6f}",
        "north": f"{north:.6f}",
        "west": f"{west:.6f}",
        "east": f"{east:.6f}",
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    backoff = 2.0  # seconds; doubles per retry
    for attempt in range(max_retries):
        try:
            resp = requests.get(OPENTOPO_API, params=params, timeout=timeout, stream=True)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            if attempt == max_retries - 1:
                warnings.warn(
                    f"OpenTopography USGS1m bbox {bbox} timed out after "
                    f"{max_retries} attempts ({type(e).__name__}); treating as gap."
                )
                return False
            warnings.warn(
                f"OpenTopography USGS1m bbox {bbox} {type(e).__name__} "
                f"(attempt {attempt+1}/{max_retries}); retrying in {backoff:.1f}s"
            )
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code != 200:
            warnings.warn(
                f"OpenTopography USGS1m returned {resp.status_code} for bbox {bbox}: "
                f"{resp.text[:200]}"
            )
            return False
        out_path.write_bytes(resp.content)
        if out_path.stat().st_size < 1024:
            warnings.warn(f"OpenTopography chunk {bbox} returned <1 KiB; treating as empty.")
            out_path.unlink(missing_ok=True)
            return False
        return True
    return False


def _tnm_tile_names(
    aoi: AOI,
) -> list[str]:
    """Compute the list of TNM 1/3-arcsec tile names covering the AOI.

    Tiles are named by their northeast corner: `n40w121` covers lat 39-40,
    lon -121 to -120. So for an AOI we ceil() the north and west bounds and
    floor()+1 from there to capture all overlap.
    """
    west, south, east, north = aoi.bbox
    # NE-corner latitudes: smallest int >= ceil(south + epsilon) up to ceil(north).
    lat_start = math.floor(south) + 1
    lat_end = math.ceil(north)
    # NE-corner west longitudes (positive integer suffix); covers (-W, -W+1).
    # For lon range [west_neg, east_neg] with west_neg < east_neg < 0,
    # ne-corner positive longitudes run from ceil(-east_neg) up to ceil(-west_neg).
    lon_start = math.ceil(-east) + (0 if -east == math.ceil(-east) else 0)
    lon_end = math.ceil(-west)
    # Edge case: if east lies exactly on an integer, the tile to its east is
    # not actually overlapping.
    if -east == math.floor(-east):
        lon_start = int(-east) + 1
    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append(f"n{lat:02d}w{lon:03d}")
    return tiles


def _fetch_tnm_tile(tile: str, out_path: Path, timeout: int = 600) -> bool:
    """Download one TNM 1/3-arcsec tile. Returns True on success."""
    url = f"{TNM_BASE}/{tile}/USGS_13_{tile}.tif"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True
    with requests.get(url, stream=True, timeout=timeout) as resp:
        if resp.status_code != 200:
            warnings.warn(f"TNM tile {tile} returned {resp.status_code}: {url}")
            return False
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return out_path.stat().st_size > 1024


def _mosaic(tile_paths: list[Path], out_path: Path) -> None:
    """Merge per-tile GeoTIFFs into a single mosaic at out_path."""
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress="deflate",
            tiled=True,
            count=mosaic.shape[0],
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()


def _resolution_meters(path: Path) -> int:
    """Return the approximate pixel resolution in metres for the mosaic."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        # If the CRS is geographic, convert one pixel size to metres at the
        # mosaic centre latitude.
        if src.crs and src.crs.is_geographic:
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.bottom + src.bounds.top) / 2
            # Approx: 1 deg lat = 111_320 m; 1 deg lon = 111_320 * cos(lat).
            dx_m = abs(src.transform.a) * 111_320 * math.cos(math.radians(cy))
            dy_m = abs(src.transform.e) * 111_320
            return int(round((dx_m + dy_m) / 2))
        # Already projected; pixel size is metres.
        return int(round((abs(src.transform.a) + abs(src.transform.e)) / 2))


OPENTOPO_MAX_WORKERS = int(os.environ.get("OPENTOPO_MAX_WORKERS", "80"))

# TODO(1m-resume, 2026-06-01): OpenTopography OT+ has a hard rate limit of
# 200 API calls / 24 hours across USGS endpoints. The 2026-05-31 fetch burned
# ~150 calls (earlier serial attempts + the 80-way parallel run); the next
# 157 chunks returned HTTP 401 "API maximum rate limit reached". 81 of 238
# chunks landed on disk under data/raw/3dep_lidar/_tiles/. To resume:
#   1. Wait for the 24h reset (after ~midnight UTC on 2026-06-01).
#   2. Run: `OPENTOPO_MAX_WORKERS=12 .venv/bin/python -c \\
#         'from ai_minerals.aoi import NORTHERN_SIERRA; \\
#          from ai_minerals.data import threedep; \\
#          threedep.fetch(NORTHERN_SIERRA, resolution="1m")'`
#      (Lower concurrency to spread the 157 remaining requests over a longer
#      wall-clock window; staying well under the 200/24h cap.)
#   3. The mosaic step in fetch() will still OOM trying to build a single
#      ~234 GB float32 array for the full AOI at 1 m. Use a GDAL VRT
#      instead — see _mosaic() refactor TODO below. Or restrict to the
#      anchor-district band (38.8-39.5 N) which fits in memory.
#   4. Re-run paleochannel precompute on the new lidar_dem; then K.3
#      assemble; then K.5 training. The Phase 1 anchor gate should still
#      pass; the deliverable raster should improve on the deep-gravel
#      Tertiary branch (REM detects buried paleochannels better at 1 m).
# A scheduled task is set for tomorrow at 09:00 local to remind to kick
# this off — see `claude /schedule list` for the routine ID.


def _try_fetch_1m(aoi: AOI, work_dir: Path) -> tuple[list[Path], list[tuple], list[tuple]]:
    """Fetch 1 m chunks via OpenTopography concurrently. Returns (paths, ok_bboxes, gap_bboxes).

    Threads the per-chunk HTTP requests via ThreadPoolExecutor at
    `OPENTOPO_MAX_WORKERS` concurrency (default 80; override via env var).
    Each `_fetch_opentopo_chunk` call is independent (per-thread `requests`
    state) and idempotent (skips if tile is already on disk), so the pool
    is safe to scale up. Concurrent timeouts on no-data bboxes happen
    simultaneously instead of serially, which is the main win for our AOI
    where the lat-38 southern band is mostly out of USGS 3DEP 1 m coverage.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not api_key:
        warnings.warn(
            "OPENTOPOGRAPHY_API_KEY not set; cannot fetch 3DEP 1 m via OpenTopography."
        )
        return [], [], []
    chunks = _opentopo_chunks(aoi)
    paths: list[Path] = []
    ok: list[tuple] = []
    gaps: list[tuple] = []

    n_total = len(chunks)
    print(f"  fetching {n_total} chunks at {OPENTOPO_MAX_WORKERS}-way concurrency...")

    def _job(idx_bbox: tuple[int, tuple]) -> tuple[int, tuple, Path | None]:
        idx, bbox = idx_bbox
        chunk_path = work_dir / f"opentopo_chunk_{idx:03d}.tif"
        success = _fetch_opentopo_chunk(api_key, bbox, chunk_path)
        return idx, bbox, (chunk_path if success else None)

    completed = 0
    with ThreadPoolExecutor(max_workers=OPENTOPO_MAX_WORKERS) as pool:
        futures = [pool.submit(_job, (i, bbox)) for i, bbox in enumerate(chunks)]
        for fut in as_completed(futures):
            idx, bbox, chunk_path = fut.result()
            completed += 1
            if chunk_path is not None:
                paths.append(chunk_path)
                ok.append(bbox)
            else:
                gaps.append(bbox)
            if completed % 20 == 0 or completed == n_total:
                print(f"  progress: {completed}/{n_total} chunks "
                      f"({len(ok)} ok, {len(gaps)} gap)")
    return paths, ok, gaps


def _try_fetch_10m(aoi: AOI, work_dir: Path) -> tuple[list[Path], list[str], list[str]]:
    """Fetch 10 m tiles from the TNM S3 bucket. Returns (paths, ok_tiles, gap_tiles)."""
    tiles = _tnm_tile_names(aoi)
    paths: list[Path] = []
    ok: list[str] = []
    gaps: list[str] = []
    for tile in tiles:
        tile_path = work_dir / f"USGS_13_{tile}.tif"
        if _fetch_tnm_tile(tile, tile_path):
            paths.append(tile_path)
            ok.append(tile)
        else:
            gaps.append(tile)
    return paths, ok, gaps


def fetch(aoi: AOI, *, resolution: str = "1m") -> Path:
    """Fetch 3DEP elevation for the AOI. Mosaic into a single GeoTIFF.

    resolution="1m" tries OpenTopography first and falls back to 10 m TNM
    tiles when no API key is configured or all chunks fail. resolution="10m"
    skips the 1 m attempt entirely.

    Returns the mosaic path; downstream code reads the same file regardless
    of source resolution.
    """
    if resolution not in {"1m", "10m"}:
        raise ValueError(f"resolution must be '1m' or '10m', got {resolution!r}")

    out_dir = dataset_dir(NAME)
    out_path = out_dir / f"{OUT_BASENAME}.tif"
    meta_path = out_dir / f"{OUT_BASENAME}_meta.json"

    tmp_root = out_dir / "_tiles"
    tmp_root.mkdir(parents=True, exist_ok=True)

    actual_resolution = None
    source_path = None
    tile_list: list[str] = []
    gaps: list[str] = []

    if resolution == "1m":
        paths, ok, gap_bboxes = _try_fetch_1m(aoi, tmp_root)
        if paths:
            actual_resolution = "1m"
            source_path = "OpenTopography USGS1m"
            tile_list = [
                f"opentopo:{w:.4f},{s:.4f},{e:.4f},{n:.4f}" for (w, s, e, n) in ok
            ]
            gaps = [
                f"opentopo:{w:.4f},{s:.4f},{e:.4f},{n:.4f}"
                for (w, s, e, n) in gap_bboxes
            ]
            _mosaic(paths, out_path)
        else:
            warnings.warn(
                "1 m fetch produced no tiles; falling back to 10 m TNM tiles."
            )

    if source_path is None:
        paths, ok_tiles, gap_tiles = _try_fetch_10m(aoi, tmp_root)
        if not paths:
            raise RuntimeError(
                f"Could not fetch any 3DEP tiles for AOI {aoi.name} at either "
                "1 m or 10 m. Check connectivity and tile naming."
            )
        actual_resolution = "10m"
        source_path = "USGS National Map 1/3 arc-second"
        tile_list = ok_tiles
        gaps = gap_tiles
        _mosaic(paths, out_path)

    actual_res_m = _resolution_meters(out_path)

    meta = {
        "requested_resolution": resolution,
        "actual_resolution": actual_resolution,
        "actual_resolution_m": actual_res_m,
        "source": source_path,
        "aoi": {
            "name": aoi.name,
            "bbox": list(aoi.bbox),
        },
        "tiles": tile_list,
        "coverage_gaps": gaps,
        "mosaic_path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    write_source_md(
        NAME,
        title="USGS 3DEP elevation (1 m where flown, 10 m fallback)",
        url=(
            "https://portal.opentopography.org/raster?opentopoID=OTNED.082016.4269.1 "
            "(1 m); https://www.usgs.gov/3d-elevation-program (10 m)"
        ),
        license="US public domain (USGS 3DEP)",
        notes=(
            f"AOI={aoi.name} bbox={aoi.bbox}. Requested resolution={resolution}; "
            f"achieved={actual_resolution} via {source_path}. "
            f"Mosaic written to {out_path.name}; sidecar metadata in "
            f"{meta_path.name}. Tile list and coverage gaps recorded there. "
            f"Approximate native pixel size: {actual_res_m} m."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import AOI as _AOI

    northern_sierra = _AOI(
        name="NorthernSierra",
        min_lon=-121.55,
        min_lat=37.49,
        max_lon=-119.48,
        max_lat=40.01,
    )
    path = fetch(northern_sierra, resolution="1m")
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
