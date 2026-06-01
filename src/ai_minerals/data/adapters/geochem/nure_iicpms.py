"""USGS NURE archived ICP-MS reanalysis -> canonical geochem samples.

Reads the national GeoPackage written by `data/nure_iicpms.py::fetch` and
returns AOI-clipped, schema-validated geochem points.

The source CSV uses `<El>_<unit>` column naming verbatim (e.g. `Ag_ppm`,
`Au_sq_ppm`, `Al_pct`). This adapter:
  - clips to AOI bbox
  - renames `Lab_ID` -> `sample_id`, `Primary_Class` -> `sample_type`,
    `Collection_Date` -> `sample_date`
  - masks below-detection-limit values (encoded as negative numbers in the
    source) to NaN across all element columns
  - normalizes element column names: `Au_sq_ppm` -> `Au_ppm`, `Ag_ppm`
    stays `Ag_ppm`, etc., so downstream feature code doesn't need to know
    which lab method produced the value
  - if `elements` is given, drops any `<el>_ppm` / `<el>_ppb` columns not
    in the list (major-oxide `_pct` columns are also dropped under the
    same filter)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geochem


# Au in this NURE release is reported as `Au_sq_ppm` (semi-quantitative
# ppm). Downstream pathfinder code expects `Au_ppm` or `Au_ppb`. We rename
# to `Au_ppm` (keeping the source unit) so the rest of the column scheme
# stays uniform.
AU_RENAME = {"Au_sq_ppm": "Au_ppm"}

# Element-column pattern. We accept "<symbol>_ppm", "<symbol>_ppb",
# "<symbol>_pct", and the `Au_sq_ppm` variant. Anything else (Lab_ID,
# Lat_NAD27, Sample_Source, ...) is metadata and is left untouched until
# the explicit metadata-rename step.
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)(?:_sq)?_(ppm|ppb|pct)$")


def _element_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if _ELEMENT_RE.match(c)]


def load(
    path: Path,
    aoi: AOI,
    *,
    elements: Iterable[str] | None = None,
) -> gpd.GeoDataFrame:
    """Read the NURE-ICPMS national GPKG as canonical geochem samples.

    Output schema:
      - geometry (WGS84, EPSG:4326)
      - sample_id (Lab_ID)
      - source ('NURE_ICPMS')
      - sample_type (from Primary_Class), sample_date (from Collection_Date)
      - <el>_ppm / <el>_ppb columns for each element kept

    If `elements` is given, drop any element columns not in that list
    (matched on the bare symbol, e.g. `elements=["Au", "As"]` keeps
    `Au_ppm` and `As_ppm`).
    """
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    west, south, east, north = aoi.bbox
    minx, miny, maxx, maxy = west, south, east, north
    gdf = gdf.cx[minx:maxx, miny:maxy].copy()

    # Mask below-detection-limit sentinels (negative values) to NaN
    # across all element columns.
    el_cols = _element_columns(gdf.columns)
    for c in el_cols:
        # Use to_numeric for safety: occasionally these CSV columns come in
        # as object dtype if a stray string slipped through.
        col = pd.to_numeric(gdf[c], errors="coerce")
        col = col.where(col >= 0, np.nan)
        gdf[c] = col

    # Rename Au_sq_ppm -> Au_ppm so downstream code can reference a stable
    # column name across NURE / NGDB / AGDB4.
    rename_map = {k: v for k, v in AU_RENAME.items() if k in gdf.columns}
    if rename_map:
        gdf = gdf.rename(columns=rename_map)

    # Metadata renames. Field presence checked because the source schema
    # has drifted between revision dates.
    meta_rename = {}
    if "Lab_ID" in gdf.columns:
        meta_rename["Lab_ID"] = "sample_id"
    elif "LAB_ID" in gdf.columns:
        meta_rename["LAB_ID"] = "sample_id"
    if "Primary_Class" in gdf.columns:
        meta_rename["Primary_Class"] = "sample_type"
    if "Collection_Date" in gdf.columns:
        meta_rename["Collection_Date"] = "sample_date"
    if meta_rename:
        gdf = gdf.rename(columns=meta_rename)

    if "sample_id" not in gdf.columns:
        raise ValueError(
            f"NURE adapter could not find a Lab_ID column; got "
            f"{list(gdf.columns)[:30]}"
        )

    gdf["sample_id"] = gdf["sample_id"].astype(str)
    gdf["source"] = "NURE_ICPMS"

    if elements is not None:
        keep_symbols = {el for el in elements}
        # Drop element columns whose bare symbol isn't requested.
        current_el_cols = _element_columns(gdf.columns)
        drop = []
        for c in current_el_cols:
            m = _ELEMENT_RE.match(c)
            if m and m.group(1) not in keep_symbols:
                drop.append(c)
        if drop:
            gdf = gdf.drop(columns=drop)

    return validate_geochem(gdf)
