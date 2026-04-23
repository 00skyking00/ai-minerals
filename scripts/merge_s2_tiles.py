"""Merge S2 mosaic tiles using rioxarray.merge.merge_arrays.

Workaround for `rasterio.merge` rejecting north-up TIFFs with negative
pixel height ("upside-down rasters") — unavoidable because TIFFs with
y-axis top-down are the standard.

    uv run python scripts/merge_s2_tiles.py [TILES_DIR] [OUT_PATH]
"""

from __future__ import annotations

import sys
from pathlib import Path

import rioxarray
from rioxarray.merge import merge_arrays


def main(tiles_dir: Path, out_path: Path) -> int:
    tile_paths = sorted(tiles_dir.glob("*.tif"))
    if not tile_paths:
        print(f"No *.tif in {tiles_dir}", file=sys.stderr)
        return 1
    print(f"Merging {len(tile_paths)} tiles from {tiles_dir}")
    arrs = []
    for p in tile_paths:
        da = rioxarray.open_rasterio(p, masked=True)
        arrs.append(da)
        print(f"  {p.name}: shape={da.shape} bounds={da.rio.bounds()}")

    merged = merge_arrays(arrs, nodata=float("nan"))
    print(f"Merged shape: {merged.shape}")
    print(f"Bounds: {merged.rio.bounds()}")

    # rioxarray/xarray double-set _FillValue. Strip both, cast to float32
    # (native NaN support), write NaN as nodata. Larger than uint16 but
    # downstream band-ratio calcs (iron-oxide, ferrous, clay) want float anyway.
    merged.attrs.pop("_FillValue", None)
    merged.encoding.pop("_FillValue", None)
    merged = merged.astype("float32")
    merged.rio.write_nodata(float("nan"), inplace=True)

    merged.rio.to_raster(
        out_path,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
        dtype="float32",
    )
    size = out_path.stat().st_size
    print(f"Wrote {out_path} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        tiles_dir = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
    else:
        from ai_minerals.data._common import DATA_RAW
        tiles_dir = DATA_RAW / "sentinel2" / "tiles"
        out_path = DATA_RAW / "sentinel2" / "s2_mosaic_eastak.tif"
    sys.exit(main(tiles_dir, out_path))
