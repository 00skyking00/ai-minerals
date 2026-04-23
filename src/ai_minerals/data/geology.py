"""USGS SIM 3340 geologic map of Alaska — bulk FileGDB download + clip.

Alaska is *not* in the conterminous SGMC; it has a dedicated state-scale map.
The file geodatabase (367 MB) is ~half the size of the shapefile bundle
(651 MB) and loads fine via GDAL's OpenFileGDB driver.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "geology_ak"

GDB_ZIP_URL = "https://pubs.usgs.gov/sim/3340/sim3340_gdb.zip"
CATALOG_URL = "https://pubs.usgs.gov/publication/sim3340"


def fetch(*, force: bool = False) -> Path:
    """Download and extract the SIM 3340 File Geodatabase."""
    out_dir = dataset_dir(NAME)
    zip_path = out_dir / "sim3340_gdb.zip"

    if zip_path.exists() and not force:
        print(
            f"SIM3340 gdb zip already present at {zip_path} "
            f"({zip_path.stat().st_size:,} B); skipping download."
        )
    else:
        print(f"Downloading SIM 3340 Alaska geology geodatabase (~367 MB) from {GDB_ZIP_URL}")
        with requests.get(GDB_ZIP_URL, stream=True, timeout=1200) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            written = 0
            with zip_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(f"  {written / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:4.1f}%)", end="\r")
            print()

    extract_dir = out_dir / "sim3340"
    if not extract_dir.exists() or force:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    write_source_md(
        NAME,
        title="USGS SIM 3340 — Geologic Map of Alaska (File Geodatabase)",
        url=CATALOG_URL,
        license="US public domain (USGS)",
        notes=(
            f"Source: {GDB_ZIP_URL}. Contains polygon lithology + linework at "
            "nominal 1:500,000 and 1:1,584,000 scales. Clip to AOI via GeoPandas."
        ),
    )
    return zip_path


def _find_gdb(extract_dir: Path) -> Path:
    candidates = [p for p in extract_dir.rglob("*.gdb") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No .gdb found in {extract_dir}")
    return candidates[0]


def list_layers() -> list[str]:
    """Return the layer names available in the downloaded geodatabase."""
    import fiona

    gdb = _find_gdb(dataset_dir(NAME) / "sim3340")
    return list(fiona.listlayers(str(gdb)))


def clip_units_to_aoi(aoi: AOI, *, layer: str | None = None) -> "geopandas.GeoDataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Load the geologic-units polygon layer and clip to AOI.

    Defaults to the detailed AKStategeol_poly (nominal 1:500k), which is what
    we want for a quadrangle-scale analysis. Pass layer='AKStategeolpoly_generalized'
    for the 1:1.584M generalized version.
    """
    import geopandas as gpd

    gdb = _find_gdb(dataset_dir(NAME) / "sim3340")

    if layer is None:
        layer = "AKStategeol_poly"

    print(f"Loading layer {layer!r} from {gdb}")
    gdf = gpd.read_file(str(gdb), layer=layer)
    print(f"Layer CRS: {gdf.crs}; AOI CRS: {aoi.crs}")

    # Build an AOI GeoSeries with an explicit CRS so .clip() reprojects correctly.
    aoi_series = gpd.GeoSeries([aoi.polygon], crs=aoi.crs)
    if gdf.crs != aoi_series.crs:
        aoi_series = aoi_series.to_crs(gdf.crs)

    gdf_aoi = gdf.clip(aoi_series.iloc[0])
    print(f"Geology: {len(gdf_aoi):,} polygons in {aoi.name} (of {len(gdf):,} total).")
    return gdf_aoi


if __name__ == "__main__":
    from ai_minerals.aoi import EASTERN_ALASKA

    fetch()
    layers = list_layers()
    print(f"GDB layers ({len(layers)}):")
    for n in layers:
        print(f"  {n}")
    gdf = clip_units_to_aoi(EASTERN_ALASKA)
    out_path = dataset_dir(NAME) / f"geology_{EASTERN_ALASKA.name.lower()}.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    print(f"Wrote {out_path}")
