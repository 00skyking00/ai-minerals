"""USGS AGDB4 (Alaska Geochemical Database v4) → canonical geochem samples.

Input:
  - `samples_path`: parquet of AGDB4 sample locations (produced by
    `data/agdb4.py::fetch`).
  - `bv_zip`: AGDB4 'best-value' text bundle (AGDB4_text.zip), which holds
    per-element ppm values in multiple BV_*.txt tables joined on DDPD_ID.

Output is a canonical-schema GeoDataFrame with one row per sample and a
`<el>_ppm` column per pathfinder element. Below-detection sentinels
(negative values in the BV tables) are masked to NaN.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geochem


# (BV_*.txt filename, ppm column name) per element. AGDB4 best-value
# schema; elements partition across four tables by element group.
PATHFINDER_FILES = {
    "Ag": ("BV_Ag_Br.txt", "Ag_ppm"),
    "As": ("BV_Ag_Br.txt", "As_ppm"),
    "Au": ("BV_Ag_Br.txt", "Au_ppm"),
    "Bi": ("BV_Ag_Br.txt", "Bi_ppm"),
    "Cu": ("BV_C_Gd.txt",  "Cu_ppm"),
    "Mo": ("BV_Ge_Os.txt", "Mo_ppm"),
    "Pb": ("BV_P_Te.txt",  "Pb_ppm"),
    "Sb": ("BV_P_Te.txt",  "Sb_ppm"),
    "Te": ("BV_P_Te.txt",  "Te_ppm"),
    "Zn": ("BV_Th_Zr.txt", "Zn_ppm"),
}


def _load_bv_element(bv_zip: Path, bv_file: str, column: str) -> pd.DataFrame:
    """Read just [DDPD_ID, column] from one BV_*.txt, streaming via usecols.

    AGDB4 encodes below-detection-limit values as negatives (e.g. -50 → "<50").
    Mask as NaN rather than treat the sentinel as a real value.
    """
    with zipfile.ZipFile(bv_zip) as zf:
        with zf.open(f"AGDB4_text/{bv_file}") as f:
            raw = f.read()
    df = pd.read_csv(
        io.BytesIO(raw), sep=",", usecols=["DDPD_ID", column],
        low_memory=False, encoding="latin-1",
    )
    df.loc[df[column] < 0, column] = np.nan
    return df


def load(
    path: Path,
    aoi: AOI,
    *,
    bv_zip: Path | None = None,
    elements: Iterable[str] = tuple(PATHFINDER_FILES),
) -> gpd.GeoDataFrame:
    """Load AGDB4 samples + best-value element columns.

    `path` is the AOI-clipped samples parquet (LONGITUDE/LATITUDE + DDPD_ID).
    `bv_zip` points at `AGDB4_text.zip` in the agdb4 dataset dir; defaults
    to `<path>.parent / "AGDB4_text.zip"`.
    """
    samples = pd.read_parquet(path)
    if bv_zip is None:
        bv_zip = path.parent / "AGDB4_text.zip"

    joined = samples.copy()
    for el in elements:
        bv_file, col = PATHFINDER_FILES[el]
        assay = _load_bv_element(bv_zip, bv_file, col)
        joined = joined.merge(assay, on="DDPD_ID", how="left")
        joined = joined.rename(columns={col: f"{el}_ppm"})

    out = gpd.GeoDataFrame(
        joined,
        geometry=gpd.points_from_xy(joined["LONGITUDE"], joined["LATITUDE"]),
        crs="EPSG:4326",
    )
    out["sample_id"] = out["DDPD_ID"].astype(str)
    out["source"] = "AGDB4"
    # AGDB4 doesn't carry a per-sample date or standardized sample-type field
    # at this schema level; leave those optional columns absent.
    return validate_geochem(out)
