"""Native-resolution Alaska IFSAR DTM/DSM via the State DGGS ImageServer.

The State of Alaska DGGS elevation portal serves the statewide 5 m
Interferometric Synthetic Aperture Radar (IFSAR) elevation as two ArcGIS
ImageServers in EPSG:3338 (Alaska Albers):

  - `elevation/IFSAR_DTM` — bare-earth Digital Terrain Model (drainage, slope)
  - `elevation/IFSAR_DSM` — first-return Digital Surface Model (canopy/structures)

Both publish at a native 5 m pixel. `exportImage` caps a single request at
15000 px on a side (DTM) / 15000x4100 (DSM), so a district-sized pull is tiled
into cell-aligned blocks and stitched by windowed writes (one tile in memory at
a time, output streamed to disk) rather than an all-tiles-in-RAM mosaic.

This is the native-5 m fetcher. The 100 m grid-resampled pull used by the MPM
covariate stack lives in `scripts/nome_placer/mpm_fetch_district_terrain.py`;
that one asks the server to resample to the model grid. This module pulls the
genuine 5 m product and clips to a bbox, so downstream code can derive its own
slope/TPI at full resolution or downsample deliberately.

Candidate to graduate into `mintel` alongside the AGDB4/akmag fetchers
(ADR-015); kept self-contained with that move in mind.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.windows import Window

from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "ifsar_dggs"
WORKING_CRS = "EPSG:3338"
NODATA = np.float32(-9999.0)
# IFSAR returns ~-10000 over ocean/void (not flagged nodata); bilinear smears
# that plate into a sub-(-50 m) shoreline ramp. Real coastal land here never
# drops below ~-20 m, so anything <= -50 m is treated as off-coverage.
OCEAN_FLOOR_M = -50.0

_IMAGESERVER = (
    "https://geoportal.dggs.dnr.alaska.gov/arcgis/rest/services/"
    "elevation/IFSAR_{product}/ImageServer"
)

# exportImage size cap per service (px). DSM is the tighter one.
_MAX_PX = {"DTM": 15000, "DSM": 4100}


def _grid_size(bounds: tuple[float, float, float, float], cell: float) -> tuple[int, int]:
    left, bottom, right, top = bounds
    width = int(round((right - left) / cell))
    height = int(round((top - bottom) / cell))
    return width, height


def _tile_ranges(n: int, step: int) -> list[tuple[int, int]]:
    """Split [0, n) into contiguous (start, count) blocks of at most `step`."""
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        out.append((start, min(step, n - start)))
        start += step
    return out


def _export_tile(
    url: str,
    sub_bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    out_tif: Path,
    timeout: int = 300,
) -> None:
    """Export one cell-aligned tile to `out_tif`. Resumable: skips if present."""
    if out_tif.exists() and out_tif.stat().st_size > 1024:
        return
    left, bottom, right, top = sub_bounds
    params = {
        "bbox": f"{left},{bottom},{right},{top}",
        "bboxSR": "3338",
        "imageSR": "3338",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    r = requests.get(f"{url}/exportImage", params=params, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "tiff" not in ctype and "image" not in ctype:
        raise RuntimeError(f"ImageServer returned {ctype}: {r.text[:300]}")
    out_tif.write_bytes(r.content)
    if out_tif.stat().st_size <= 1024:
        out_tif.unlink(missing_ok=True)
        raise RuntimeError(f"Tile {sub_bounds} returned <1 KiB; aborting.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_native(
    bounds_3338: tuple[float, float, float, float],
    *,
    product: str = "DTM",
    cell: float = 5.0,
    out_path: Path,
    aoi_name: str = "nome_district",
    force: bool = False,
) -> dict:
    """Fetch the native 5 m IFSAR `product` over `bounds_3338`, clip, write GeoTIFF.

    `bounds_3338` is (left, bottom, right, top) in EPSG:3338. Returns a
    provenance dict (also written to `*_meta.json`) with the achieved
    resolution, tile manifest, coverage fraction, and a sha256 of the output.
    """
    product = product.upper()
    if product not in _MAX_PX:
        raise ValueError(f"product must be DTM or DSM, got {product!r}")
    url = _IMAGESERVER.format(product=product)
    # Stay under the service cap, and keep each tile's in-memory array small
    # (~256 MB at 8000x8000 F32) so a district pull is safe on an 8 GB box.
    max_px = min(_MAX_PX[product], 8000)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    if out_path.exists() and not force:
        print(f"{out_path} already present; skipping (force=True to refetch).")
        return json.loads(meta_path.read_text()) if meta_path.exists() else {}

    left, bottom, right, top = bounds_3338
    width, height = _grid_size(bounds_3338, cell)
    transform = Affine(cell, 0, left, 0, -cell, top)
    print(
        f"IFSAR {product}: native {cell} m grid {width}x{height} "
        f"over 3338 bounds {bounds_3338}"
    )

    tile_dir = out_path.parent / f"_tiles_{product.lower()}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    col_blocks = _tile_ranges(width, max_px)
    row_blocks = _tile_ranges(height, max_px)
    n_tiles = len(col_blocks) * len(row_blocks)
    print(f"  tiling into {len(col_blocks)}x{len(row_blocks)} = {n_tiles} block(s)")

    profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        width=width,
        height=height,
        crs=WORKING_CRS,
        transform=transform,
        nodata=NODATA,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )

    valid_cells = 0
    tiles_meta: list[dict] = []
    done = 0
    with rasterio.open(out_path, "w", **profile) as dst:
        for r0, rh in row_blocks:
            for c0, cw in col_blocks:
                sub = (
                    left + c0 * cell,
                    top - (r0 + rh) * cell,
                    left + (c0 + cw) * cell,
                    top - r0 * cell,
                )
                tile_tif = tile_dir / f"tile_r{r0}_c{c0}.tif"
                _export_tile(url, sub, cw, rh, tile_tif)
                with MemoryFile(tile_tif.read_bytes()) as mf, mf.open() as ds:
                    arr = ds.read(1).astype("float32")
                    src_nodata = ds.nodata
                if src_nodata is not None:
                    arr = np.where(arr == src_nodata, NODATA, arr)
                arr = np.where(arr <= OCEAN_FLOOR_M, NODATA, arr)
                valid_cells += int(np.count_nonzero(arr != NODATA))
                dst.write(arr, 1, window=Window(c0, r0, cw, rh))
                tiles_meta.append(
                    {"row_off": r0, "col_off": c0, "w": cw, "h": rh, "bbox_3338": list(sub)}
                )
                done += 1
                print(f"  tile {done}/{n_tiles} ({r0},{c0}) {cw}x{rh} written")

    total_cells = width * height
    coverage_frac = valid_cells / total_cells if total_cells else 0.0
    checksum = _sha256(out_path)
    meta = {
        "name": NAME,
        "product": product,
        "aoi_name": aoi_name,
        "service": url,
        "service_native_cell_m": 5,
        "requested_cell_m": cell,
        "crs": WORKING_CRS,
        "bounds_3338": list(bounds_3338),
        "size_px": [width, height],
        "tiles": tiles_meta,
        "coverage_fraction": round(coverage_frac, 4),
        "valid_cells": valid_cells,
        "total_cells": total_cells,
        "output": str(out_path),
        "sha256": checksum,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"Wrote {out_path} ({out_path.stat().st_size:,} B); "
        f"coverage {coverage_frac:.1%}; sha256 {checksum[:12]}…"
    )
    return meta


def write_provenance(metas: list[dict], retrieved: str) -> Path:
    """Record one SOURCE.md covering the products fetched in this run."""
    lines = []
    for m in metas:
        lines.append(
            f"- **{m['product']}** ({m['aoi_name']}): native {m['service_native_cell_m']} m, "
            f"grid {m['size_px'][0]}x{m['size_px'][1]} @ {m['requested_cell_m']} m, "
            f"coverage {m['coverage_fraction']:.1%}, sha256 `{m['sha256'][:16]}…`\n"
            f"  output `{m['output']}`"
        )
    return write_source_md(
        NAME,
        title="Alaska DGGS IFSAR 5 m DTM/DSM (native resolution)",
        url="https://elevation.alaska.gov/ "
        "(ImageServer: geoportal.dggs.dnr.alaska.gov/arcgis/rest/services/elevation/IFSAR_DTM)",
        license="Public domain (State of Alaska DGGS / USGS-IFSAR)",
        notes=(
            f"Retrieved {retrieved} via ArcGIS ImageServer exportImage, EPSG:3338, "
            f"native 5 m, F32, bilinear. Ocean/void (<= {OCEAN_FLOOR_M} m) masked to "
            f"{float(NODATA)}.\n\n" + "\n".join(lines)
        ),
    )
