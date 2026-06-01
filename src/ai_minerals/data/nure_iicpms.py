"""USGS NURE archived stream-sediment reanalysis (four-acid ICP-MS, 51 elements).

ScienceBase release for DOI 10.5066/F7765DHF: ~60,000 archived NURE-HSSR
stream-sediment splits re-analyzed by the USGS lab using four-acid digestion
and ICP-MS. Covers the conterminous western United States. One row per
sample; one column per element. Lat/long are encoded in NAD27 in the source
CSV.

Fetcher contract:
- Downloads `Reanalyzed_NURE-HSSRv8.csv` from ScienceBase (~25 MB).
- Parses, reprojects sample coordinates NAD27 -> WGS84, writes a single
  national GeoPackage at `data/raw/nure_iicpms/nure_western_us.gpkg`.
- All element columns are preserved as-is; the source already uses
  `<El>_<unit>` (e.g. `Ag_ppm`, `Al_pct`, `Au_sq_ppm`). Au in this release
  is reported in `ppm` (semi-quantitative), NOT ppb as in older NURE
  bulletins; we keep the source units and let the adapter normalize.
- Below-detection-limit values are encoded as negative numbers in the
  source; we leave them as-is in the GPKG and let the adapter mask to NaN.

The adapter (`adapters/geochem/nure_iicpms.py`) clips to AOI at load time.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "nure_iicpms"

# ScienceBase item for DOI 10.5066/F7765DHF (resolved 2026-05).
# TODO if this 404s: candidate parent IDs to re-search are the USGS
# Geochemistry and Geochronology Science Center release page, or query
# `https://www.sciencebase.gov/catalog/items?q=Reanalyzed+NURE-HSSRv8&format=json`.
ITEM_ID = "5a0b3136e4b09af898cb6f56"
CSV_FILENAME = "Reanalyzed_NURE-HSSRv8.csv"
CSV_URL = f"https://www.sciencebase.gov/catalog/file/get/{ITEM_ID}?name={CSV_FILENAME}"
CATALOG_URL = f"https://www.sciencebase.gov/catalog/item/{ITEM_ID}"

# Source CRS for Lat_NAD27 / Long_NAD27 columns.
NAD27 = "EPSG:4267"

LAT_COL = "Lat_NAD27"
LON_COL = "Long_NAD27"


def fetch(aoi: AOI, *, force: bool = False) -> Path:
    """Download the NURE-ICP-MS national CSV and convert to GeoPackage.

    The dataset spans the entire western US (~60k samples); we deliberately
    keep the GPKG national (the file size is dominated by the wide element
    table, not the row count). The adapter clips to AOI at load time.
    The `aoi` argument is unused in the fetch but kept for orchestrator
    signature consistency.
    """
    del aoi  # signature parity with other fetchers; clip happens in adapter
    out_dir = dataset_dir(NAME)
    csv_path = out_dir / CSV_FILENAME
    gpkg_path = out_dir / "nure_western_us.gpkg"

    if gpkg_path.exists() and not force:
        print(
            f"NURE-ICPMS GPKG already present at {gpkg_path} "
            f"({gpkg_path.stat().st_size:,} B); skipping."
        )
        return gpkg_path

    if not csv_path.exists() or force:
        print(f"Downloading NURE-ICPMS CSV (~25 MB) from {CSV_URL}")
        with requests.get(CSV_URL, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            written = 0
            with csv_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(
                            f"  {written / 1e6:6.1f} / {total / 1e6:.1f} MB "
                            f"({pct:4.1f}%)",
                            end="\r",
                        )
            print()

    print(f"Parsing {csv_path.name}...")
    df = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
    print(f"  {len(df):,} rows x {len(df.columns)} cols")

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise RuntimeError(
            f"Expected {LAT_COL!r} and {LON_COL!r} columns in NURE CSV; "
            f"got first 40 columns: {list(df.columns)[:40]}"
        )

    # Drop rows with no coords; you can't do much with a stream sediment
    # whose location is unknown.
    df = df.dropna(subset=[LAT_COL, LON_COL]).copy()
    print(f"  {len(df):,} rows after dropping null coords")

    geom = gpd.points_from_xy(df[LON_COL], df[LAT_COL], crs=NAD27)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=NAD27).to_crs("EPSG:4326")

    print(f"  writing {gpkg_path}...")
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"  wrote {gpkg_path} ({gpkg_path.stat().st_size:,} bytes)")

    write_source_md(
        NAME,
        title=(
            "USGS NURE archived stream-sediment reanalysis "
            "(four-acid ICP-MS, 51 elements), western US"
        ),
        url=CATALOG_URL,
        license="US public domain (USGS)",
        notes=(
            f"ScienceBase item {ITEM_ID}, DOI 10.5066/F7765DHF. Source CSV "
            f"`{CSV_FILENAME}` (~25 MB) downloaded and converted to a single "
            f"national GeoPackage `{gpkg_path.name}` with WGS84 point geometry "
            f"built from `{LAT_COL}` / `{LON_COL}` (source datum NAD27, "
            f"reprojected to EPSG:4326). All ~51 element columns preserved "
            f"with their source `<El>_<unit>` names: most metals in `ppm`, "
            f"major oxides in `pct`, Au reported as `Au_sq_ppm` (ppm, "
            f"semi-quantitative). Below-detection-limit values are encoded "
            f"as negative numbers in the source; the adapter masks them to "
            f"NaN at load time."
        ),
    )
    return gpkg_path


if __name__ == "__main__":
    from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER

    path = fetch(NORTHERN_SIERRA_PLACER.aoi)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
