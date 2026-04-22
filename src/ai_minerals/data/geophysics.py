"""USGS national-scale geophysics grids (magnetic + gravity) for US/Canada.

National-scale GeoTIFFs are published on ScienceBase as companion datasets to the
tri-national Critical Minerals Mapping Initiative. Alaska is included in both
the magnetic and gravity composites. For Tanacross-scale work these are coarse
but sufficient as regional features; for prospect-scale refinement the
2015 DGGS Tanacross aeromagnetic survey (GPR 2015-6) is higher resolution but
not needed in v1.

Sources:
- Magnetic: ScienceBase 619a9a3ad34eb622f692f961 -> GeophysicsMag_USCanada.zip (~127 MB)
- Gravity:  ScienceBase 619a9f02d34eb622f692f96c -> GeophysicsGravity_USCanada.zip (~16.5 MB)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from ai_minerals.aoi import AOI, WORKING_CRS
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "geophysics"

MAG_ITEM = "619a9a3ad34eb622f692f961"
MAG_FILE = "GeophysicsMag_USCanada.zip"

GRAV_ITEM = "619a9f02d34eb622f692f96c"
GRAV_FILE = "GeophysicsGravity_USCanada.zip"


def _sb_url(item: str, filename: str) -> str:
    return f"https://www.sciencebase.gov/catalog/file/get/{item}?name={filename}"


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"  {dest.name} already present ({dest.stat().st_size:,} B); skipping.")
        return dest
    print(f"  Downloading {url}")
    with requests.get(url, stream=True, timeout=1200) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"    {written / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:4.1f}%)", end="\r")
        print()
    return dest


def _extract(zip_path: Path, *, force: bool = False) -> Path:
    extract_dir = zip_path.parent / zip_path.stem
    if not extract_dir.exists() or force:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        print(f"  Extracted to {extract_dir}")
    return extract_dir


def fetch(*, force: bool = False) -> dict[str, Path]:
    """Download + extract magnetic and gravity national GeoTIFF archives.

    Returns a dict mapping product name -> extract directory.
    """
    out_dir = dataset_dir(NAME)
    results: dict[str, Path] = {}

    for label, item, fname in [
        ("magnetic", MAG_ITEM, MAG_FILE),
        ("gravity", GRAV_ITEM, GRAV_FILE),
    ]:
        print(f"--- {label} ---")
        zip_path = out_dir / fname
        _download(_sb_url(item, fname), zip_path, force=force)
        results[label] = _extract(zip_path, force=force)

    write_source_md(
        NAME,
        title="USGS national-scale geophysics (magnetic + gravity) GeoTIFFs",
        url=(
            f"https://www.sciencebase.gov/catalog/item/{MAG_ITEM} and "
            f"https://www.sciencebase.gov/catalog/item/{GRAV_ITEM}"
        ),
        license="US public domain (USGS)",
        notes=(
            f"Magnetic: {MAG_FILE}, gravity: {GRAV_FILE}. Continental-scale "
            "composites. Alaska is included in both. Reproject + clip to the "
            "AOI in the notebook. Radiometric grids are not bundled here; "
            "defer to https://mrdata.usgs.gov/radiometric/ as a stretch goal."
        ),
    )
    return results


def clip_to_aoi(tif_path: Path, aoi: AOI, out_path: Path) -> Path:
    """Reproject + clip a national GeoTIFF to the AOI and save as a new TIFF."""
    import rioxarray  # noqa: F401

    src = rioxarray.open_rasterio(tif_path, masked=True)
    # Reproject to working CRS, then clip by bbox polygon in working CRS.
    if str(src.rio.crs) != WORKING_CRS:
        src = src.rio.reproject(WORKING_CRS)
    # Reproject AOI polygon to working CRS.
    import geopandas as gpd
    from shapely.geometry import box as shp_box
    poly_wgs = gpd.GeoDataFrame({"geometry": [aoi.polygon]}, crs=aoi.crs).to_crs(WORKING_CRS)
    clipped = src.rio.clip(poly_wgs.geometry, from_disk=True)
    clipped.rio.to_raster(out_path, compress="deflate", tiled=True)
    print(f"  clipped -> {out_path}")
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import TANACROSS

    results = fetch()
    print("\nClipping to AOI...")
    for label, extract_dir in results.items():
        tifs = list(extract_dir.rglob("*.tif"))
        if not tifs:
            print(f"  {label}: no .tif found in {extract_dir}")
            continue
        src = tifs[0]
        out = dataset_dir(NAME) / f"{label}_{TANACROSS.name.lower()}.tif"
        clip_to_aoi(src, TANACROSS, out)
