"""Rasterize the 250m per-cell P(orogenic_gold) predictions to a GeoTIFF
in EPSG:4326 for consumption by the gldbg goldbug scorer.

Input:
    data/derived/motherlode/model_predictions_motherlode_250m.parquet
Output:
    data/derived/motherlode/prospectivity_motherlode_250m_3310.tif   (native)
    data/derived/motherlode/prospectivity_motherlode_250m_4326.tif   (gldbg)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject


DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
ML_DIR = DATA_DERIVED / "motherlode"
PRED_PARQUET = ML_DIR / "model_predictions_motherlode_250m.parquet"
TIF_NATIVE = ML_DIR / "prospectivity_motherlode_250m_3310.tif"
TIF_WGS84 = ML_DIR / "prospectivity_motherlode_250m_4326.tif"
RES_M = 250.0
SRC_CRS = "EPSG:3310"
DST_CRS = "EPSG:4326"


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(PRED_PARQUET)
    print(f"loaded {len(df):,} cells", flush=True)

    x_min = df["x"].min() - RES_M / 2
    x_max = df["x"].max() + RES_M / 2
    y_min = df["y"].min() - RES_M / 2
    y_max = df["y"].max() + RES_M / 2
    width = int(round((x_max - x_min) / RES_M))
    height = int(round((y_max - y_min) / RES_M))
    transform = Affine(RES_M, 0.0, x_min, 0.0, -RES_M, y_max)
    print(f"native grid: {height}r x {width}c at 250m in {SRC_CRS}", flush=True)

    arr = np.full((height, width), np.nan, dtype=np.float32)
    cols_idx = ((df["x"].values - x_min) / RES_M).astype(int)
    rows_idx = ((y_max - df["y"].values) / RES_M).astype(int)
    cols_idx = np.clip(cols_idx, 0, width - 1)
    rows_idx = np.clip(rows_idx, 0, height - 1)
    arr[rows_idx, cols_idx] = df["p_rf"].values.astype(np.float32)

    with rasterio.open(
        TIF_NATIVE, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_string(SRC_CRS),
        transform=transform,
        nodata=float("nan"),
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)
    print(f"wrote {TIF_NATIVE}  ({TIF_NATIVE.stat().st_size/1e6:.1f} MB)", flush=True)

    with rasterio.open(TIF_NATIVE) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": CRS.from_string(DST_CRS),
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "compress": "deflate",
        })
        with rasterio.open(TIF_WGS84, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=CRS.from_string(DST_CRS),
                resampling=Resampling.bilinear,
            )
    print(f"wrote {TIF_WGS84}  ({TIF_WGS84.stat().st_size/1e6:.1f} MB; "
          f"{dst_height}r x {dst_width}c in {DST_CRS})", flush=True)
    print(f"total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
