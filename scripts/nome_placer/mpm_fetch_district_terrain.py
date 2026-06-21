"""Fetch IFSAR DTM on the district grid + derive slope and TPI (ADR-013).

Source: Alaska DGGS IFSAR_DTM ImageServer (5 m IFSAR-derived DTM, EPSG:3338).
We pull a single exportImage at the district grid's exact 3338 bounds and
width/height so the DEM lands cell-aligned on the template, then derive slope
(Horn) and TPI (cell minus 3x3 mean) on the same grid.

DTM doubles as the on-land mask: cells where the ImageServer returns nodata are
offshore / out-of-coverage and become the MPM's background-exclusion mask.

Run: uv run python -m scripts.nome_placer.mpm_fetch_district_terrain
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile

RAW = Path("data/raw/nome_mpm")
TEMPLATE = Path("data/derived/nome_placer/covariates_mpm/_district_grid_template_3338.tif")
IMAGESERVER = (
    "https://geoportal.dggs.dnr.alaska.gov/arcgis/rest/services/"
    "elevation/IFSAR_DTM/ImageServer/exportImage"
)
NODATA = np.float32(-9999.0)


def fetch_dtm(left, bottom, right, top, width, height) -> np.ndarray:
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
    r = requests.get(IMAGESERVER, params=params, timeout=300)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype and "tiff" not in ctype:
        raise RuntimeError(f"ImageServer returned {ctype}: {r.text[:300]}")
    with MemoryFile(r.content) as mf, mf.open() as ds:
        arr = ds.read(1).astype("float32")
        src_nodata = ds.nodata
    if src_nodata is not None:
        arr = np.where(arr == src_nodata, NODATA, arr)
    # IFSAR returns -10000 over ocean / void (not flagged as nodata). Bilinear
    # resampling smears that plate into a thin sub-(-50 m) ramp at the shoreline.
    # Real land in this coastal district never drops below ~-20 m, so mask <= -50 m.
    arr = np.where(arr <= -50.0, NODATA, arr)
    return arr


def slope_horn(dem: np.ndarray, res: float, valid: np.ndarray) -> np.ndarray:
    z = np.where(valid, dem, np.nan)
    gy, gx = np.gradient(z, res, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return np.where(valid & np.isfinite(slope), slope, NODATA).astype("float32")


def tpi(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    z = np.where(valid, dem, 0.0).astype("float64")
    m = valid.astype("float64")
    mean = uniform_filter(z, size=3, mode="nearest")
    cnt = uniform_filter(m, size=3, mode="nearest")
    neigh = np.divide(mean, cnt, out=np.zeros_like(mean), where=cnt > 0)
    out = dem - neigh
    return np.where(valid, out, NODATA).astype("float32")


def write_band(path: Path, arr: np.ndarray, transform, crs, width, height, src: str) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", count=1,
        width=width, height=height, crs=crs, transform=transform,
        nodata=NODATA, compress="deflate", predictor=2, tiled=True,
    ) as o:
        o.write(arr, 1)
        o.update_tags(source=src)
    print(f"wrote {path}")


def main() -> None:
    with rasterio.open(TEMPLATE) as t:
        b = t.bounds
        transform, crs, width, height = t.transform, t.crs, t.width, t.height
        res = abs(transform.a)
    print(f"fetching IFSAR DTM at {width}x{height} bbox=({b.left},{b.bottom},{b.right},{b.top})")
    dem = fetch_dtm(b.left, b.bottom, b.right, b.top, width, height)
    valid = dem != NODATA
    print(f"DTM: valid_frac={valid.mean():.4f}, "
          f"elev min/mean/max={dem[valid].min():.1f}/{dem[valid].mean():.1f}/{dem[valid].max():.1f}")

    write_band(RAW / "ifsar_dem_district_3338.tif", np.where(valid, dem, NODATA),
               transform, crs, width, height, "AK DGGS IFSAR_DTM ImageServer (5m), district grid")
    write_band(RAW / "ifsar_slope_district_3338.tif", slope_horn(dem, res, valid),
               transform, crs, width, height, "Horn slope (deg) from IFSAR DTM, district grid")
    write_band(RAW / "ifsar_tpi_district_3338.tif", tpi(dem, valid),
               transform, crs, width, height, "TPI (cell - 3x3 mean) from IFSAR DTM, district grid")


if __name__ == "__main__":
    main()
