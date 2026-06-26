"""1993 Nome DIGHEM-V airborne survey -> EPSG:3338 covariate grids.

Source: Alaska DGGS report GPR 2019-11 (DOI 10.14509/30189), the 1993 Nome
DIGHEM-V helicopter EM/magnetic survey gridded by the contractor at 25 m from
~400 m flight lines. This is a four-fold tighter line spacing than the ~1 km
statewide aeromagnetic composite (``akmag_*``) the lode model has been running
on, so it is the layer most likely to carry real structural signal.

The grids ship in ER Mapper format in NAD27 / UTM zone 3N. GDAL reads the .ers
label but cannot always resolve the "NAD27 / NUTM03" datum string without the
ER Mapper coordinate tables, so the source CRS is asserted as EPSG:26703 rather
than trusted from the header.

Bands produced (each reprojected and aligned to a caller-supplied model grid):

  mag_residual          residual total field (nT): total field minus a broad
                        Gaussian regional, so the model sees local susceptibility
                        contrast and not the ~55,000 nT main-field gradient.
  mag_1vd               first vertical derivative (nT/m), contractor product.
  mag_analytic_signal   analytic signal (nT/m), contractor product.
  mag_tilt              tilt angle (rad) = atan2(1VD, HGM); equalises anomaly
                        amplitude so weak shallow sources read as strongly as
                        large ones (Verduzco et al. 2004).
  res_log10_900hz       log10 apparent resistivity (ohm-m), 900 Hz coplanar
                        (deepest sensing, ~150 m).
  res_log10_7200hz      log10 apparent resistivity, 7200 Hz (intermediate).
  res_log10_56khz       log10 apparent resistivity, 56 kHz (shallowest, ~30 m).

Derivatives that benefit from the full survey footprint (residual, tilt) are
computed on the native 25 m grid before reprojection, so the boundary nodata of
the model grids never enters the FFT/gradient stencils.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

SRC_CRS = "EPSG:26703"  # NAD27 / UTM zone 3N (asserted; see module docstring)

# NADCON grid for the NAD27->NAD83 datum shift in Alaska. Without it PROJ falls
# back to a ballpark (zero-shift) transform that misregisters the Nome survey by
# ~160 m (6+ cells at 25 m), which would smear the magnetic anomalies off the
# occurrences and bias the marginal-AUC test toward a false negative. The grid is
# 0.5 m accuracy, ~0.5 MB, open license (PROJ CDN). PROJ_NETWORK=ON would fetch it
# automatically; we place it deterministically so an offline render is correct.
_NADCON_GRID = "us_noaa_alaska.tif"
_NADCON_URL = f"https://cdn.proj.org/{_NADCON_GRID}"
ERMAPPER_ZIP = Path(
    "data/raw/nome_dighem_1993/gpr2019_011_nome_1993_grids_ermapper.zip"
)
CACHE = Path("data/derived/nome_geophys/dighem1993/_ermapper")
NODATA = -999.0

# logical name -> ER Mapper grid basename inside the zip
_GRIDS = {
    "magtf": "nome_sim_magtf",
    "vd1": "nome_calculated1vd",
    "asig": "nome_analyticsignal",
    "res900": "nome_res900hz",
    "res7200": "nome_res7200hz",
    "res56k": "nome_res56khz",
}


def ensure_extracted(zip_path: Path = ERMAPPER_ZIP, cache: Path = CACHE) -> dict[str, Path]:
    """Extract the ER Mapper grids once and return {logical_name: .ers path}."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = {name: cache / f"{base}.ers" for name, base in _GRIDS.items()}
    if all(p.exists() for p in paths.values()):
        return paths
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            stem = Path(member).name
            for base in _GRIDS.values():
                if stem in (base, f"{base}.ers"):
                    target = cache / stem
                    target.write_bytes(zf.read(member))
    return paths


def ensure_nadcon_grid() -> Path | None:
    """Place the Alaska NAD27->NAD83 NADCON grid where rasterio's PROJ looks.

    Returns the grid path if present/installed, else None (caller should expect a
    ~160 m datum misregistration and the warp will warn). Idempotent.
    """
    proj_dir = Path(rasterio.__file__).parent / "proj_data"
    target = proj_dir / _NADCON_GRID
    if target.exists():
        return target
    try:
        proj_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_NADCON_URL, target)  # noqa: S310 (trusted PROJ CDN)
        return target
    except (OSError, urllib.error.URLError):
        os.environ.setdefault("PROJ_NETWORK", "ON")  # last-resort: let PROJ fetch at runtime
        return None


def _read_native(ers_path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """Read an .ers grid as float32 with its own nodata masked to NaN."""
    with rasterio.open(ers_path) as ds:
        arr = ds.read(1).astype(np.float32)
        nod = ds.nodata
        transform = ds.transform
    if nod is not None:
        arr = np.where(np.isclose(arr, nod, rtol=0, atol=abs(nod) * 1e-6 + 1e-3), np.nan, arr)
    return arr, transform


def _regional_residual(magtf: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Total field minus a Gaussian regional (high-pass). NaN-safe: the lowpass
    is computed on a mean-filled copy, then re-masked to the valid footprint."""
    valid = np.isfinite(magtf)
    filled = np.where(valid, magtf, np.nanmean(magtf))
    regional = ndimage.gaussian_filter(filled, sigma=sigma_cells, mode="nearest")
    out = magtf - regional
    out[~valid] = np.nan
    return out.astype(np.float32)


def _tilt_from(magtf: np.ndarray, vd1: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Tilt = atan2(1VD, HGM), HGM the horizontal gradient magnitude of the field."""
    valid = np.isfinite(magtf) & np.isfinite(vd1)
    g = np.where(np.isfinite(magtf), magtf, 0.0)
    gx = ndimage.sobel(g, axis=1) / (8.0 * dx)
    gy = ndimage.sobel(g, axis=0) / (8.0 * dy)
    hgm = np.sqrt(gx**2 + gy**2)
    out = np.arctan2(vd1, hgm + 1e-12)
    out[~valid] = np.nan
    return out.astype(np.float32)


def _warp_to(
    arr: np.ndarray, src_transform: rasterio.Affine, template_path: Path,
    *, resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, dict]:
    """Reproject a native-CRS array onto a template grid; return (array, profile)."""
    with rasterio.open(template_path) as tmpl:
        dst = np.full((tmpl.height, tmpl.width), np.nan, np.float32)
        reproject(
            source=arr, destination=dst,
            src_transform=src_transform, src_crs=SRC_CRS, src_nodata=np.nan,
            dst_transform=tmpl.transform, dst_crs=tmpl.crs, dst_nodata=np.nan,
            resampling=resampling,
        )
        profile = tmpl.profile.copy()
    out = np.where(np.isfinite(dst), dst, NODATA).astype(np.float32)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    return out, profile


def _write(arr: np.ndarray, profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def build_dighem_bands(
    template_path: Path, out_dir: Path, *, sigma_m: float = 4000.0,
) -> dict[str, Path]:
    """Reproject the DIGHEM bands onto ``template_path`` and write GeoTIFFs.

    sigma_m sets the regional-removal scale for ``mag_residual`` (default 4 km:
    removes the smooth main-field gradient, keeps anomalies shorter than a few km).
    Returns {band_name: geotiff_path}.
    """
    ensure_nadcon_grid()
    src = ensure_extracted()
    magtf, t = _read_native(src["magtf"])
    vd1, _ = _read_native(src["vd1"])
    asig, _ = _read_native(src["asig"])
    dx, dy = abs(t.a), abs(t.e)

    residual = _regional_residual(magtf, sigma_cells=sigma_m / dx)
    tilt = _tilt_from(magtf, vd1, dx, dy)

    native = {
        "mag_residual": (residual, t),
        "mag_1vd": (vd1, t),
        "mag_analytic_signal": (asig, t),
        "mag_tilt": (tilt, t),
    }
    for band, key in (("res_log10_900hz", "res900"),
                      ("res_log10_7200hz", "res7200"),
                      ("res_log10_56khz", "res56k")):
        res, rt = _read_native(src[key])
        native[band] = (np.log10(np.clip(res, 1e-3, None)).astype(np.float32), rt)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for band, (arr, src_t) in native.items():
        warped, profile = _warp_to(arr, src_t, template_path)
        paths[band] = _write(warped, profile, out_dir / f"dighem_{band}.tif")
    return paths


DIGHEM_BANDS = ["mag_residual", "mag_1vd", "mag_analytic_signal", "mag_tilt",
                "res_log10_900hz", "res_log10_7200hz", "res_log10_56khz"]
MAG_BANDS = ["mag_residual", "mag_1vd", "mag_analytic_signal", "mag_tilt"]
RES_BANDS = ["res_log10_900hz", "res_log10_7200hz", "res_log10_56khz"]
