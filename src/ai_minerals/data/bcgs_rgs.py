"""BC Regional Geochemical Survey (RGS) — provincial stream/soil geochem.

GeoFile 2020-08 publication: ~65k samples, ~5M determinations, Excel
workbooks. Analogous to USGS AGDB4 for Alaska. Column names follow
`<ELEMENT>_<METHOD>_<UNIT>` (e.g. CU_AAS_PPM, AU_INAA_PPB); multiple
methods per element means we pick a priority order.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "bcgs_rgs"
ZIP_URL = (
    "https://cmscontent.nrs.gov.bc.ca/geoscience/PublicationCatalogue/"
    "GeoFile/BCGS_GF2020-08.zip"
)
LANDING_URL = (
    "https://www2.gov.bc.ca/gov/content/industry/mineral-exploration-mining/"
    "british-columbia-geological-survey/geology/regional-geochemical-survey"
)


def fetch(*, force: bool = False) -> Path:
    """Download + extract the RGS 2020 data zip."""
    out_dir = dataset_dir(NAME)
    zip_path = out_dir / "BCGS_GF2020-08.zip"

    if zip_path.exists() and list(out_dir.glob("*.xlsx")) and not force:
        print(f"RGS data already extracted at {out_dir}; skipping.")
        return next(out_dir.glob("*data*.xlsx"), next(out_dir.glob("*.xlsx")))

    if not zip_path.exists() or force:
        print(f"Downloading BC RGS from {ZIP_URL}")
        with requests.get(ZIP_URL, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"  {zip_path.stat().st_size:,} bytes zipped")

    print("Extracting xlsx files...")
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            if n.endswith((".xlsx", ".csv")):
                target = out_dir / Path(n).name
                with zf.open(n) as src, target.open("wb") as dst:
                    dst.write(src.read())
                print(f"  extracted {target.name} ({target.stat().st_size:,} B)")

    write_source_md(
        NAME,
        title="BC Regional Geochemical Survey (BCGS GeoFile 2020-08)",
        url=LANDING_URL,
        license="Open Government Licence - British Columbia",
        notes=(
            "GeoFile 2020-08 RGS compilation. ~65k samples, ~5M determinations. "
            "Stream sediment, moss-mat, lake sediment, water samples. "
            "Column naming: <ELEMENT>_<METHOD>_<UNIT>; filter by MEDIA_TYPE."
        ),
    )
    return next(out_dir.glob("*data*.xlsx"), next(out_dir.glob("*.xlsx")))


def clip_to_aoi(aoi: AOI) -> Path:
    """Load the RGS data xlsx, filter to AOI bbox, save as parquet."""
    out_dir = dataset_dir(NAME)
    data_xlsx = next(out_dir.glob("*data*.xlsx"), None) or next(out_dir.glob("*.xlsx"), None)
    if data_xlsx is None:
        fetch()
        data_xlsx = next(out_dir.glob("*data*.xlsx"), None) or next(out_dir.glob("*.xlsx"))

    print(f"Reading {data_xlsx} — this takes a minute for a 65k-row sheet")
    df = pd.read_excel(data_xlsx, engine="openpyxl")
    print(f"  shape: {df.shape}  cols sample: {list(df.columns)[:12]}")

    # Normalize column names to uppercase (BCGS mixes cases)
    df.columns = [c.strip().upper() for c in df.columns]

    lat_col = "LATITUDE" if "LATITUDE" in df.columns else "LAT"
    lon_col = "LONGITUDE" if "LONGITUDE" in df.columns else "LONG"

    west, south, east, north = aoi.bbox
    mask = df[lat_col].between(south, north) & df[lon_col].between(west, east)
    sub = df[mask].copy()
    print(f"RGS in {aoi.name}: {len(sub):,} samples")

    out_path = out_dir / f"rgs_{aoi.name.lower()}.parquet"
    sub.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    from ai_minerals.regions.bcgt import BCGT
    fetch()
    clip_to_aoi(BCGT.aoi)
