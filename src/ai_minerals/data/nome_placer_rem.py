"""Nome placer beach-line covariates from the 1 m bare-earth LiDAR -> EPSG:3338 grids.

Source: NOAA/USACE 2021 National Coastal Mapping Program bare-earth 1 m DEM
(dataset 10207, ``data/raw/nome_dem_1m_2021/``), NAD83(2011) / UTM zone 3N with
NAVD88 height in metres. A narrow coastal strip (~9.8 x 5.5 km) over the Nome
beach, far finer than the 5 m IfSAR the placer base model uses.

The Nome placer control is strandline / sea-stand geometry: gold concentrates
where stillstand beaches cross the glacial drift, plus on the modern abrasion
platform (genesis synthesis 2026-06-26; Nelson & Hopkins PP689; Tuck picks).
These bands target that control directly off the LiDAR:

  rem                     detrended elevation (raw minus a quadratic trend
                          surface): beach ridges and scarps the regional
                          slope-to-sea would otherwise swamp.
  dist_second             metres to the +38 ft (11.58 m NAVD88) Second Beach
                          strandline (raised; traced Hastings -> Bourbon Creek).
  dist_third              metres to the +70 ft (21.34 m) Third Beach strandline
                          (raised; the rich Little Creek line, bedrock ~+79 ft).
  dist_submerged_shallow  metres to the shallow submerged beach, -36 to -45 ft
                          (-11 to -14 m): the Submarine / first drowned stand
                          that the bare-earth grid still reaches.
  dist_abrasion_platform  metres to the modern wave-cut bench (near-0 m, low
                          slope) seaward of the beach.
  ew_ridge                E-W (shore-parallel) ridge enhancement: a positive
                          value marks an E-W-elongated high standing above its
                          cross-shore (N-S) surroundings. The grade correlogram
                          long-range is ~E-W, so the anisotropy is E-W, NOT the
                          lode NE/NW.

A companion ``strip_cover`` mask (1.0 where the LiDAR strip has data) lets the
marginal-AUC test restrict to cells the LiDAR actually covers, the same
clean-subset discipline the lode model uses with gems_extent.

NOT derivable from this strip (flagged, not faked): the Fourth Beach (+120 ft =
36.6 m) sits above the strip's +25.6 m ceiling, and the deeper submerged stands
(-55/-70/-80 ft) sit below its -13.7 m floor; both need the upland IfSAR or the
topobathy point cloud. The drift-on-beach intersection needs the surficial-drift
map (AOF 125 / RI 2024-6), a separate vector layer.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter

TILES_DIR = Path("data/raw/nome_dem_1m_2021")
SRC_HCRS = CRS.from_epsg(6332)  # NAD83(2011) / UTM zone 3N (horizontal part of the COMPD_CS)
TARGET_CRS = "EPSG:3338"
RES_M = 5.0  # resample the 1 m strip to 5 m: keeps berm detail, bounds the compute
NODATA = -9999.0
FT_TO_M = 0.3048

# Strandline targets as (band_name, centre_elev_m_NAVD88, half_band_m).
STRANDLINES = [
    ("dist_second", 38 * FT_TO_M, 1.5),            # +11.58 m, Second Beach
    ("dist_third", 70 * FT_TO_M, 1.5),             # +21.34 m, Third Beach
    ("dist_submerged_shallow", -40.5 * FT_TO_M, 2.0),  # -12.3 m, spans -36..-45 ft
]
REM_BANDS = ["rem", "dist_second", "dist_third", "dist_submerged_shallow",
             "dist_abrasion_platform", "ew_ridge"]


def _strip_dem_3338() -> tuple[np.ndarray, "rasterio.Affine", dict]:
    """Mosaic the 1 m tiles and resample to a 5 m EPSG:3338 grid over the strip.

    Returns (elev, transform, profile) with NaN outside valid data.
    """
    tiles = sorted(glob.glob(str(TILES_DIR / "*.tif")))
    if not tiles:
        raise FileNotFoundError(f"no DEM tiles under {TILES_DIR}")
    xs, ys = [], []
    for t in tiles:
        with rasterio.open(t) as ds:
            l, b, r, top = transform_bounds(ds.crs, TARGET_CRS, *ds.bounds)
            xs += [l, r]; ys += [b, top]
    minx = np.floor(min(xs) / RES_M) * RES_M
    maxy = np.ceil(max(ys) / RES_M) * RES_M
    width = int(np.ceil((max(xs) - minx) / RES_M))
    height = int(np.ceil((maxy - min(ys)) / RES_M))
    transform = from_origin(minx, maxy, RES_M, RES_M)
    dest = np.full((height, width), np.nan, np.float32)
    for t in tiles:
        with rasterio.open(t) as ds:
            src = ds.read(1)
            src_nod = ds.nodata
            tmp = np.full((height, width), np.nan, np.float32)
            reproject(
                source=src, destination=tmp,
                src_transform=ds.transform, src_crs=SRC_HCRS,
                dst_transform=transform, dst_crs=TARGET_CRS,
                src_nodata=src_nod, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        m = np.isfinite(tmp)
        dest[m] = tmp[m]
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "width": width, "height": height, "crs": TARGET_CRS,
               "transform": transform, "nodata": NODATA, "compress": "deflate"}
    return dest, transform, profile


def _detrend(elev: np.ndarray) -> np.ndarray:
    """Elevation minus a degree-2 polynomial trend surface (the REM)."""
    valid = np.isfinite(elev)
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]].astype(np.float64)
    xn = xx / elev.shape[1]
    yn = yy / elev.shape[0]
    A = np.column_stack([np.ones(valid.sum()), xn[valid], yn[valid],
                         xn[valid] ** 2, xn[valid] * yn[valid], yn[valid] ** 2])
    coef, *_ = np.linalg.lstsq(A, elev[valid].astype(np.float64), rcond=None)
    trend = (coef[0] + coef[1] * xn + coef[2] * yn
             + coef[3] * xn ** 2 + coef[4] * xn * yn + coef[5] * yn ** 2)
    return (elev - trend).astype(np.float32)


def _dist_to_band(elev: np.ndarray, centre: float, half: float) -> np.ndarray:
    """Metres to the nearest cell whose elevation is within +/-half of centre."""
    band = np.isfinite(elev) & (np.abs(elev - centre) <= half)
    if not band.any():
        return np.full(elev.shape, NODATA, np.float32)
    return (distance_transform_edt(~band) * RES_M).astype(np.float32)


def _abrasion_platform(elev: np.ndarray) -> np.ndarray:
    """Metres to the modern wave-cut bench: near-0 m elevation, low slope."""
    filled = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    gy, gx = np.gradient(filled, RES_M)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    bench = np.isfinite(elev) & (elev >= -2.0) & (elev <= 1.0) & (slope_deg <= 3.0)
    if not bench.any():
        return np.full(elev.shape, NODATA, np.float32)
    return (distance_transform_edt(~bench) * RES_M).astype(np.float32)


def _ew_ridge(elev: np.ndarray) -> np.ndarray:
    """E-W (shore-parallel) ridge enhancement; positive on E-W-elongated highs."""
    filled = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    along = gaussian_filter(filled, sigma=(2.0, 16.0))   # heavy along E-W (cols)
    across = gaussian_filter(filled, sigma=(16.0, 2.0))  # heavy across N-S (rows)
    out = (along - across).astype(np.float32)
    out[~np.isfinite(elev)] = NODATA
    return out


def _write(arr: np.ndarray, profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    return path


# Raised strandlines as (band_name, centre_elev_m_NAVD88) for the full-coverage
# IfSAR test. The Fourth Beach is included here (it is below the IfSAR ceiling,
# only above the narrow LiDAR strip's).
RAISED_STRANDLINES = [
    ("dist_second", 38 * FT_TO_M),
    ("dist_third", 70 * FT_TO_M),
    ("dist_fourth", 120 * FT_TO_M),
]
IFSAR_STRAND_BANDS = [b for b, _ in RAISED_STRANDLINES]


def build_strandline_on_dem(dem_path: Path, out_dir: Path, *, offset_m: float = 0.0,
                            half_m: float = 2.5, prefix: str = "ifsar") -> dict[str, Path]:
    """Raised-strandline proximity on an arbitrary full-coverage DEM.

    The narrow LiDAR strip reaches only 3 of 65 placer positives, too few to grade
    the strandline features. This builds the same proximity on the inbox IfSAR DEM
    (full coverage, all positives) so the marginal value over the POP base can be
    measured with power. ``offset_m`` is the DEM-minus-NAVD88 datum offset (the
    target elevation in the DEM frame is centre + offset). Bands keep the model
    grid; the clean auc is the whole-extent AUC because the DEM covers everything.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_path) as ds:
        elev = ds.read(1).astype(np.float32)
        nod = ds.nodata
        profile = ds.profile.copy()
        res = abs(ds.transform.a)
    elev = np.where((elev == nod) | ~np.isfinite(elev), np.nan, elev)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    paths: dict[str, Path] = {}
    for name, centre in RAISED_STRANDLINES:
        band = np.isfinite(elev) & (np.abs(elev - (centre + offset_m)) <= half_m)
        dist = (distance_transform_edt(~band) * res).astype(np.float32) if band.any() \
            else np.full(elev.shape, NODATA, np.float32)
        paths[name] = _write(dist, profile, out_dir / f"{prefix}_{name}.tif")
    return paths


def build_rem_bands(out_dir: Path) -> dict[str, Path]:
    """Build the beach-line covariate GeoTIFFs (5 m, EPSG:3338) over the LiDAR strip.

    Returns {band_name: path} for REM_BANDS plus the ``strip_cover`` mask.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    elev, _, profile = _strip_dem_3338()
    paths: dict[str, Path] = {}

    paths["rem"] = _write(_detrend(elev), profile, out_dir / "rem.tif")
    for name, centre, half in STRANDLINES:
        paths[name] = _write(_dist_to_band(elev, centre, half), profile, out_dir / f"{name}.tif")
    paths["dist_abrasion_platform"] = _write(_abrasion_platform(elev), profile,
                                             out_dir / "dist_abrasion_platform.tif")
    paths["ew_ridge"] = _write(_ew_ridge(elev), profile, out_dir / "ew_ridge.tif")

    cover = np.where(np.isfinite(elev), 1.0, 0.0).astype(np.float32)
    cprof = dict(profile); cprof["nodata"] = NODATA
    paths["strip_cover"] = _write(cover, cprof, out_dir / "strip_cover.tif")
    return paths
