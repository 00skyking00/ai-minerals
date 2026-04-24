"""GSC / NRCan national geophysics grids — aeromagnetic + gravity.

NRCan's `geophysical-data.canada.ca/portal` delivers these only through a
JS-only interactive order form with no programmatic entry point. This
module reads whatever the user manually downloaded from that portal and
normalizes it to a pair of reprojected + AOI-clipped GeoTIFFs that the
rest of the pipeline consumes.

**Manual-download protocol (documented for reproducibility):**

  1. Visit https://geophysical-data.canada.ca/portal
  2. In the search panel, Apply the region's lat/lon bbox and click Search.
  3. Add the following layers to the Layer Manager (scroll for them):
       - "Canada - 200m - Residual Magnetic Field - 2025" (or newer)
       - "Canada 2 km - GRAV - Isostatic Residual"
  4. For each: click the download icon → Download Properties dialog →
     choose **ER Mapper ERS** format (NOT Geosoft Grid — the portal mis-
     labels its proprietary HGD compressed format as "Geosoft Grid",
     which GDAL has no driver for). Keep native resolution. "Add To
     Download Manager".
  5. Save both ZIPs into `data/raw/gsc_geophysics/nrcan_manual/`.
  6. Run `python -m ai_minerals.data.gsc_geophysics --region <slug>` or
     let `build_feature_frame` pick them up automatically.

If the manual ZIPs are absent the fetcher falls back to NOAA EMAG2 v3
(2 arc-minute global aeromag, no gravity). See emag2_geophysics.py.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
import rioxarray  # noqa: F401
import xarray as xr

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md


NAME = "gsc_geophysics"
NRCAN_CRS = "EPSG:3978"  # NAD83 / Canada Atlas Lambert; ERS header omits it ("Projection=RAW")


def _write_nan_grid(out_path: Path, aoi: AOI, working_crs: str, *, field_name: str) -> Path:
    """Write a NaN-filled GeoTIFF matching the AOI extent; placeholder for
    data kinds we still can't get (e.g. gravity on a region without a
    manual download)."""
    import pyproj
    xf = pyproj.Transformer.from_crs(aoi.crs, working_crs, always_xy=True)
    minx, miny = xf.transform(aoi.min_lon, aoi.min_lat)
    maxx, maxy = xf.transform(aoi.max_lon, aoi.max_lat)
    res = 1000.0
    nx = max(int((maxx - minx) / res), 1)
    ny = max(int((maxy - miny) / res), 1)
    arr = np.full((ny, nx), np.nan, dtype=np.float32)
    transform = rasterio.transform.from_origin(minx, maxy, res, res)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", width=nx, height=ny, count=1,
        dtype="float32", crs=working_crs, transform=transform,
        nodata=np.nan, compress="deflate", tiled=True,
    ) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, field_name)
    return out_path


def _find_ers(manual_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    """Look for an extracted-or-extractable ERS file under `manual_dir`
    whose filename contains ANY of the given patterns (case-insensitive).

    Different NRCan products name their files differently:
      - national 200m magnetic:  "Canada - 200m - Residual Magnetic Field..."
      - regional 100m rtf:       "Alaska_Yukon_100m_MAG_rtf_v4.ERS"
      - regional 100m 1st-deriv: "Alaska_Yukon_100m_MAG_vd1_v4.ERS"
      - national 2km gravity:    "Canada 2 km - GRAV - Isostatic Residual..."
    so callers pass the discriminating substrings per layer.
    """
    manual_dir = manual_dir.resolve()
    pats = tuple(p.lower() for p in patterns)

    def _match(name: str) -> bool:
        lo = name.lower()
        return any(p in lo for p in pats)

    # Already extracted?
    for ers in manual_dir.rglob("*.ERS"):
        if _match(ers.name):
            return ers
    # Try extracting any ZIP that matches
    for zp in manual_dir.glob("*.zip"):
        if not _match(zp.name):
            continue
        dest = zp.parent / (zp.stem.replace(" ", "_") + "_ers")
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if not n.endswith("/"):
                    zf.extract(n, dest)
        for ers in dest.rglob("*.ERS"):
            if _match(ers.name):
                return ers
    return None


def _reproject_ers(ers_path: Path, aoi: AOI, working_crs: str, out_path: Path) -> Path:
    """Reproject an ERS raster into `working_crs`, clipped to the AOI bbox,
    write a GeoTIFF. NRCan ERS headers say `Projection=RAW` so we assign
    EPSG:3978 (documented in the bundled XML metadata)."""
    # Open with rasterio; override CRS since the header omits it.
    with rasterio.open(ers_path) as src:
        src_crs = src.crs if src.crs else NRCAN_CRS
        nodata = src.nodata if src.nodata is not None else -999999.0

        # Reproject AOI corners into the source CRS to get a clipping window.
        import pyproj
        west, south, east, north = aoi.bbox
        xf = pyproj.Transformer.from_crs(aoi.crs, src_crs, always_xy=True)
        xs, ys = xf.transform([west, east, east, west], [south, south, north, north])
        src_minx, src_maxx = min(xs), max(xs)
        src_miny, src_maxy = min(ys), max(ys)
        # Read full array; file sizes are small (~5 MB).
        arr = src.read(1)
        arr = np.where(arr == nodata, np.nan, arr).astype(np.float32)
        transform = src.transform

    # Wrap in xarray for rio.reproject.
    h, w = arr.shape
    x0, res_x = transform.c, transform.a
    y0, res_y = transform.f, transform.e  # res_y is negative
    xs = x0 + (np.arange(w) + 0.5) * res_x
    ys = y0 + (np.arange(h) + 0.5) * res_y
    da = xr.DataArray(arr, coords={"y": ys, "x": xs}, dims=("y", "x"))
    da.rio.write_crs(src_crs, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)

    # Clip in the source CRS, then reproject.
    clip_buf = max(abs(res_x), abs(res_y)) * 2
    da_clip = da.rio.clip_box(
        minx=src_minx - clip_buf, maxx=src_maxx + clip_buf,
        miny=src_miny - clip_buf, maxy=src_maxy + clip_buf,
    )
    da_rp = da_clip.rio.reproject(working_crs, resampling=1)  # bilinear
    da_rp.rio.to_raster(out_path, compress="deflate", tiled=True)
    return out_path


def fetch(aoi: AOI, working_crs: str = "EPSG:3005", *, force: bool = False) -> tuple[Path, Path]:
    """Produce AOI-clipped magnetic + gravity GeoTIFFs for the region.

    Order of precedence:
      1. NRCan 200 m magnetic + 2 km gravity (manual downloads, ERS format)
      2. EMAG2 v3 global 2-arc-minute magnetic (for regions without manual data)
      3. NaN-filled placeholder (if nothing else works for the AOI)
    """
    out_dir = dataset_dir(NAME)
    manual_dir = out_dir / "nrcan_manual"
    manual_dir.mkdir(parents=True, exist_ok=True)

    mag_path = out_dir / f"magnetic_{aoi.name.lower()}.tif"
    mag_1vd_path = out_dir / f"magnetic_1vd_{aoi.name.lower()}.tif"
    grav_path = out_dir / f"gravity_{aoi.name.lower()}.tif"

    # Magnetic RTF: prefer NRCan manual. Discriminators cover both the
    # national 200 m ("Residual Magnetic Field") and regional 100 m AK-Yukon
    # RTF ("mag_rtf").
    mag_ers = _find_ers(manual_dir, ("residual_magnetic", "mag_rtf"))
    mag_source = "placeholder"
    if mag_ers is not None:
        print(f"[magnetic-RTF] using NRCan: {mag_ers.name}")
        _reproject_ers(mag_ers, aoi, working_crs, mag_path)
        mag_source = "NRCan"
    else:
        # Fall back to EMAG2
        try:
            from ai_minerals.data.emag2_geophysics import fetch as emag2_fetch
            print("[magnetic-RTF] no NRCan ERS found; falling back to EMAG2 v3")
            emag2_fetch(aoi, working_crs=working_crs, force=force)
            mag_source = "EMAG2_v3"
        except Exception as e:
            print(f"[magnetic-RTF] EMAG2 fallback failed: {e}; using NaN placeholder")
            _write_nan_grid(mag_path, aoi, working_crs, field_name="residual_magnetic_nT")

    # Magnetic 1VD (1st vertical derivative): regional AK-Yukon only; no
    # fallback — pipeline treats this as an optional feature.
    mag_1vd_ers = _find_ers(manual_dir, ("mag_vd1", "vertical_derivative", "1vd"))
    mag_1vd_source = "absent"
    if mag_1vd_ers is not None:
        print(f"[magnetic-1VD] using NRCan: {mag_1vd_ers.name}")
        try:
            _reproject_ers(mag_1vd_ers, aoi, working_crs, mag_1vd_path)
            mag_1vd_source = "NRCan"
        except Exception as e:
            print(f"[magnetic-1VD]   reproject failed ({type(e).__name__}); skipping")
            mag_1vd_source = "absent"
            if mag_1vd_path.exists():
                mag_1vd_path.unlink()

    # Gravity: only NRCan 2 km available; no free global Bouguer replacement.
    # Note the NRCan grid is Canada-only — ERS-reproject will fail NoDataInBounds
    # for Alaska/USA AOIs; catch and fall back to NaN placeholder.
    grav_ers = _find_ers(manual_dir, ("grav",))
    grav_source = "placeholder"
    if grav_ers is not None:
        print(f"[gravity] using NRCan 2 km: {grav_ers.name}")
        try:
            _reproject_ers(grav_ers, aoi, working_crs, grav_path)
            grav_source = "NRCan_2km_IsostaticResidual"
        except Exception as e:
            # Typical: NoDataInBounds when AOI is outside Canada.
            print(f"[gravity]   AOI outside data coverage ({type(e).__name__}); NaN placeholder")
            _write_nan_grid(grav_path, aoi, working_crs, field_name="bouguer_gravity_mGal")
    else:
        print("[gravity] no NRCan ERS found; writing NaN placeholder (v2.1 TODO)")
        _write_nan_grid(grav_path, aoi, working_crs, field_name="bouguer_gravity_mGal")

    write_source_md(
        NAME,
        title=f"GSC / NRCan geophysics — mag-RTF={mag_source}, mag-1VD={mag_1vd_source}, grav={grav_source}",
        url="https://geophysical-data.canada.ca/portal",
        license="Open Government Licence - Canada (when NRCan data is present).",
        notes=(
            f"magnetic-RTF source:       {mag_source}\n"
            f"magnetic-1VD source:       {mag_1vd_source}\n"
            f"gravity source:            {grav_source}\n\n"
            "NRCan's portal delivers 'Geosoft Grid' as a proprietary compressed\n"
            "HGD variant that GDAL cannot read. Re-download in ER Mapper ERS\n"
            "format (the other option) to get a GDAL-readable file. See the\n"
            "module docstring for the manual-download protocol."
        ),
    )
    return mag_path, grav_path


if __name__ == "__main__":
    import argparse
    from ai_minerals.regions.eastak import EASTAK
    from ai_minerals.regions.bcgt import BCGT
    from ai_minerals.regions.motherlode import MOTHERLODE
    regions_by_slug = {r.slug: r for r in (EASTAK, BCGT, MOTHERLODE)}
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="bcgt")
    args = p.parse_args()
    r = regions_by_slug[args.region]
    fetch(r.aoi, working_crs=r.working_crs)
