"""USGS Hydraulic Mine Pits of California (Orlando 2016) — fetcher.

Downloads the shapefile bundle attached to ScienceBase item
571976c2e4b071321fe22947 (DOI 10.5066/F7J38QMD), unzips it, clips to the
AOI, reprojects to WGS84, and writes a single GeoPackage. The bundle is
~210 kB so we always re-download when `force=True`; otherwise the cached
GeoPackage short-circuits.

The dataset is 167 polygons across northern California compiled from
TOMS (California Dept of Conservation), Yeend 1974, and on-screen
digitizing on 2015 imagery. Source CRS is USA_Contiguous_Albers_Equal_
Area_Conic (ESRI:102003 / equivalent EPSG:5070).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "hydraulic_pits"
SB_ITEM_ID = "571976c2e4b071321fe22947"
SB_LANDING = f"https://www.sciencebase.gov/catalog/item/{SB_ITEM_ID}"
SB_DOWNLOAD = f"https://www.sciencebase.gov/catalog/file/get/{SB_ITEM_ID}"
DOI_URL = "https://doi.org/10.5066/F7J38QMD"

OUT_FILENAME = "hydraulic_mine_pits_ca.gpkg"
SHAPEFILE_STEM = "Hydraulic_Mine_Pits_of_California"


def _download_and_extract(out_dir: Path) -> Path:
    """Download the ScienceBase bundle and unzip into out_dir. Returns the .shp path."""
    extract_root = out_dir / "shp"
    shp_path = extract_root / SHAPEFILE_STEM / f"{SHAPEFILE_STEM}.shp"
    if shp_path.exists():
        return shp_path

    extract_root.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SB_DOWNLOAD, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(extract_root)

    if not shp_path.exists():
        # ScienceBase occasionally renames inner folders; fall back to glob.
        hits = list(extract_root.rglob(f"{SHAPEFILE_STEM}.shp"))
        if not hits:
            raise RuntimeError(
                f"Hydraulic Mine Pits shapefile not found after extracting "
                f"{SB_DOWNLOAD} into {extract_root}; contents: "
                f"{[p.name for p in extract_root.rglob('*')]}"
            )
        shp_path = hits[0]
    return shp_path


def fetch(aoi: AOI, *, force: bool = False) -> Path:
    """Download Orlando 2016 hydraulic mine pit polygons, clip to AOI, write GPKG.

    Output path matches the Region's `raw_paths["hydraulic_pits"]` contract:
    `data/raw/hydraulic_pits/hydraulic_mine_pits_ca.gpkg` (WGS84).
    """
    out_dir = dataset_dir(NAME)
    out_path = out_dir / OUT_FILENAME

    if out_path.exists() and not force:
        return out_path

    shp_path = _download_and_extract(out_dir)
    gdf = gpd.read_file(shp_path)

    # Clip to AOI in source CRS, reproject to WGS84.
    aoi_in_src = (
        gpd.GeoSeries([box(*aoi.bbox)], crs="EPSG:4326")
        .to_crs(gdf.crs)
        .iloc[0]
    )
    clipped = gdf[gdf.intersects(aoi_in_src)].copy()
    # Preserve stable per-row IDs before reprojection.
    clipped["pit_id"] = clipped.index.astype("int64")
    clipped = clipped.to_crs("EPSG:4326")

    clipped.to_file(out_path, driver="GPKG")

    write_source_md(
        NAME,
        title="Hydraulic Mine Pits of California (Orlando 2016)",
        url=DOI_URL,
        license="US public domain (USGS)",
        notes=(
            f"ScienceBase item {SB_ITEM_ID} (landing: {SB_LANDING}). "
            f"Downloaded the attached shapefile bundle from {SB_DOWNLOAD}, "
            f"clipped to AOI={aoi.name} bbox={aoi.bbox}, and reprojected "
            f"from USA_Contiguous_Albers_Equal_Area_Conic to EPSG:4326. "
            f"Source carries 167 polygons across northern California compiled "
            f"from TOMS (CA Dept of Conservation), Yeend 1974, and on-screen "
            f"digitizing on 2015 imagery. Columns: Pit_Name, DataSource, "
            f"AreaAcres, pit_id (stable per-row int), geometry. After AOI "
            f"clip: {len(clipped)} polygons."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER

    path = fetch(NORTHERN_SIERRA_PLACER.aoi)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
