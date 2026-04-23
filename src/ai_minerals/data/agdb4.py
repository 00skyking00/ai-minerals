"""USGS Alaska Geochemical Database v4.0 (AGDB4).

The archive is ~212 MB zipped. File layout (tab-delimited .txt):
- AGDB4_text/Geol_DeDuped.txt  -- unique samples with coords and site metadata
- AGDB4_text/Geol_AllSpls.txt  -- all samples (may contain duplicates across reanalyses)
- AGDB4_text/BV_*.txt          -- best-value data by element range (Ag-Br, C-Gd, etc.)
- AGDB4_text/BV_WholeRock_Majors.txt -- best-value major oxides
- AGDB4_text/Chem_*.txt        -- raw analytical values (use BV_* instead for modeling)
- AGDB4_text/AnalyticMethod*.txt, Agency_Biblio.txt, LabName.txt, Parameter*.txt -- ref tables
- AGDB4_text/DataDictionary.txt  -- field definitions

We download once and filter locally. For Day 2 we extract the archive and
write a bbox-filtered sample locations table (Parquet). Day 3 joins the BV
element files onto sample IDs.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "agdb4"

ITEM_ID = "6500b2bed34ed30c2057f99b"
CSV_ZIP_URL = (
    f"https://www.sciencebase.gov/catalog/file/get/{ITEM_ID}?name=AGDB4_text.zip"
)
CATALOG_URL = f"https://www.sciencebase.gov/catalog/item/{ITEM_ID}"

GEOL_TABLE = "AGDB4_text/Geol_DeDuped.txt"


def fetch(*, force: bool = False) -> Path:
    """Download the AGDB4 CSV archive (tab-delimited .txt files inside)."""
    out_dir = dataset_dir(NAME)
    zip_path = out_dir / "AGDB4_text.zip"

    if zip_path.exists() and not force:
        print(f"AGDB4 zip already present at {zip_path} ({zip_path.stat().st_size:,} B); skipping.")
        return zip_path

    print(f"Downloading AGDB4 text archive (~212 MB) from {CSV_ZIP_URL}")
    with requests.get(CSV_ZIP_URL, stream=True, timeout=1200) as resp:
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

    write_source_md(
        NAME,
        title="USGS Alaska Geochemical Database v4.0 (AGDB4) — tab-delimited text archive",
        url=CATALOG_URL,
        license="US public domain (USGS)",
        notes=(
            f"ScienceBase item {ITEM_ID}. AGDB4_text.zip contains tab-delimited .txt "
            "files (not .csv). Use Geol_DeDuped.txt for unique sample locations + "
            "BV_*.txt for best-value element data."
        ),
    )
    return zip_path


def list_entries() -> list[str]:
    zip_path = dataset_dir(NAME) / "AGDB4_text.zip"
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def load_geol(*, deduped: bool = True) -> "pandas.DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Read the geology / sample-location table from the archive."""
    import pandas as pd

    zip_path = dataset_dir(NAME) / "AGDB4_text.zip"
    table = "AGDB4_text/Geol_DeDuped.txt" if deduped else "AGDB4_text/Geol_AllSpls.txt"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(table) as f:
            raw = f.read()
    # AGDB4 "text" files are comma-delimited CSV with quoted strings
    # despite the .txt extension.
    return pd.read_csv(io.BytesIO(raw), sep=",", low_memory=False, encoding="latin-1")


def load_bbox(aoi: AOI) -> "pandas.DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Return unique samples within the AOI bounding box."""
    df = load_geol(deduped=True)
    west, south, east, north = aoi.bbox

    # Find lat/lon columns — AGDB4 convention is LATITUDE / LONGITUDE in decimal degrees.
    lat_col = next(
        (c for c in df.columns if c.upper() in {"LATITUDE", "LAT", "LATITUDE_DD", "LAT_DD"}),
        None,
    )
    lon_col = next(
        (
            c for c in df.columns
            if c.upper() in {"LONGITUDE", "LON", "LNG", "LONGITUDE_DD", "LONG_DD"}
        ),
        None,
    )
    if lat_col is None or lon_col is None:
        print(f"Columns in Geol_DeDuped: {list(df.columns)[:30]}")
        raise RuntimeError(
            "Could not locate lat/lon columns. Sample columns printed above."
        )

    mask = (
        (df[lon_col] >= west)
        & (df[lon_col] <= east)
        & (df[lat_col] >= south)
        & (df[lat_col] <= north)
    )
    sub = df[mask].copy()
    print(
        f"AGDB4: {len(sub):,} unique samples in AOI {aoi.name} "
        f"(of {len(df):,} total; lat={lat_col!r}, lon={lon_col!r})."
    )
    return sub


if __name__ == "__main__":
    from ai_minerals.aoi import EASTERN_ALASKA

    zip_path = fetch()
    print(f"Archive: {zip_path}")
    entries = list_entries()
    print(f"Archive has {len(entries)} entries.")
    df = load_bbox(EASTERN_ALASKA)
    out_parquet = dataset_dir(NAME) / f"agdb4_samples_{EASTERN_ALASKA.name.lower()}.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote clipped sample locations: {out_parquet}")
    print(f"Columns: {list(df.columns)[:25]}")
