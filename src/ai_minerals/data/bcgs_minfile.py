"""BC MINFILE — the British Columbia mineral-occurrence database.

Flat CSV export via the BC Data Catalogue. Columns include MINFILE_NO,
MINERAL_DEPOSIT_TYPE_CODE (BC profile codes like L03/H04/K01), LATITUDE,
LONGITUDE, COMMODITIES_LIST. Analogous to ARDF for Alaska.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "bcgs_minfile"
CSV_URL = (
    "https://catalogue.data.gov.bc.ca/dataset/"
    "92206d94-bc64-4111-a295-cd14eb5a501c/resource/"
    "120d5ee6-bff5-4cbe-b106-e419c790c395/download/minfile_mineral.csv"
)
LANDING_URL = "https://catalogue.data.gov.bc.ca/dataset/minfile-mineral-occurrence-database"


def fetch(*, force: bool = False) -> Path:
    """Download the full MINFILE CSV (~8.8 MB)."""
    out_dir = dataset_dir(NAME)
    csv_path = out_dir / "minfile_mineral.csv"
    if csv_path.exists() and not force:
        print(f"MINFILE CSV present at {csv_path} ({csv_path.stat().st_size:,} B); skipping.")
    else:
        print(f"Downloading MINFILE from {CSV_URL}")
        with requests.get(CSV_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with csv_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"Wrote {csv_path} ({csv_path.stat().st_size:,} bytes)")

    write_source_md(
        NAME,
        title="BC MINFILE — mineral occurrence database",
        url=LANDING_URL,
        license="Open Government Licence - British Columbia",
        notes=(
            "Flat CSV export via the BC Data Catalogue CKAN resource. "
            "Filter to AOI using LATITUDE/LONGITUDE columns. BC deposit-"
            "profile codes live in MINERAL_DEPOSIT_TYPE_CODE "
            "(e.g. L03=porphyry Cu-Au, H04=epithermal)."
        ),
    )
    return csv_path


def clip_to_aoi(aoi: AOI) -> Path:
    """Filter the provincial MINFILE CSV to the AOI, save as GeoJSON."""
    csv_path = dataset_dir(NAME) / "minfile_mineral.csv"
    if not csv_path.exists():
        fetch()

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"MINFILE provincial: {len(df):,} records")

    # BC Geographic Warehouse CSV carries codes as fixed-width strings — strip
    # trailing whitespace so downstream code-matching works.
    for c in (
        "DEPOSIT_TYPE_CODE1", "DEPOSIT_TYPE_CODE2",
        "STATUS_CODE",
        "COMMODITY_CODE1", "COMMODITY_CODE2", "COMMODITY_CODE3",
    ):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace("nan", "")

    west, south, east, north = aoi.bbox
    mask = (
        df["DECIMAL_LATITUDE"].between(south, north)
        & df["DECIMAL_LONGITUDE"].between(west, east)
    )
    sub = df[mask].copy()
    print(f"MINFILE in {aoi.name}: {len(sub):,} records")

    gdf = gpd.GeoDataFrame(
        sub,
        geometry=gpd.points_from_xy(sub["DECIMAL_LONGITUDE"], sub["DECIMAL_LATITUDE"]),
        crs="EPSG:4326",
    )
    out_path = dataset_dir(NAME) / f"minfile_{aoi.name.lower()}.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    from ai_minerals.regions.bcgt import BCGT
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="bcgt")
    args = p.parse_args()
    fetch()
    clip_to_aoi(BCGT.aoi)
