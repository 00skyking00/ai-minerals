"""Write a per-cell value array to a GeoTIFF in the working CRS, plus a 4326 sibling.

Extracted from `scripts/motherlode/v2_postprocess_250m.py::write_geotiff` so the
motherlode lode pipeline, the northern-Sierra placer Phase 1 driver, and the
two-population fusion script can share one well-tested implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject


def write_geotiff_dual_crs(
    values: np.ndarray,
    df_xy: pd.DataFrame,
    *,
    resolution_m: float,
    src_crs: str,
    out_src: Path,
    out_4326: Path | None,
    dst_crs: str = "EPSG:4326",
) -> tuple[Path, Path | None]:
    """Build a (row, col)-indexed raster in src_crs (typically EPSG:3310),
    optionally reproject to dst_crs (typically EPSG:4326), write both.

    df_xy must carry `x` and `y` columns in src_crs meters that match the
    pixel-center positions implied by `resolution_m`. Cells in the bbox but
    not in df_xy (e.g. clipped out of the AOI polygon) are written as NaN.

    Returns (out_src_path, out_4326_path_or_None).
    """
    x_min = df_xy["x"].min() - resolution_m / 2
    x_max = df_xy["x"].max() + resolution_m / 2
    y_min = df_xy["y"].min() - resolution_m / 2
    y_max = df_xy["y"].max() + resolution_m / 2
    width = int(round((x_max - x_min) / resolution_m))
    height = int(round((y_max - y_min) / resolution_m))
    transform = Affine(resolution_m, 0.0, x_min, 0.0, -resolution_m, y_max)

    arr = np.full((height, width), np.nan, dtype=np.float32)
    cols_idx = ((df_xy["x"].values - x_min) / resolution_m).astype(int)
    rows_idx = ((y_max - df_xy["y"].values) / resolution_m).astype(int)
    cols_idx = np.clip(cols_idx, 0, width - 1)
    rows_idx = np.clip(rows_idx, 0, height - 1)
    arr[rows_idx, cols_idx] = np.asarray(values, dtype=np.float32)

    out_src.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_src, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_string(src_crs),
        transform=transform,
        nodata=float("nan"),
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)

    if out_4326 is None:
        return out_src, None

    out_4326.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_src) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": CRS.from_string(dst_crs),
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "compress": "deflate",
        })
        with rasterio.open(out_4326, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=CRS.from_string(dst_crs),
                resampling=Resampling.bilinear,
            )
    return out_src, out_4326
