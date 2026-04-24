"""USGS ARDF (Alaska Resource Data File) → canonical occurrences."""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_occurrences


_CODE_PATTERN = re.compile(r"\b(\d{1,2}[a-z]?)\b", re.IGNORECASE)


def _parse_codes(model_code: str | None) -> tuple[str, ...]:
    """Extract Cox-&-Singer codes from the ARDF model_code field.

    ARDF stores codes as free text like '21a' or '17' or '21a, 20c'; a few
    older records hold 'Porphyry Cu (21a)'. Extract every code-looking
    substring, prefix with 'usgs:'.
    """
    if model_code is None or (isinstance(model_code, float) and pd.isna(model_code)):
        return ()
    matches = _CODE_PATTERN.findall(str(model_code))
    return tuple(f"usgs:{m.lower()}" for m in matches)


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read an ARDF GeoPackage (already AOI-clipped by the fetch module)
    and emit canonical occurrence records.

    Mapping:
      `comm_main`   → `commodity` (free-text element list)
      `model_code`  → `deposit_codes` (tuple of `usgs:<code>`)
      `rept_date`   → `year`
      `ardf_num`    → `raw_record_id`
      `source`      = "ARDF"
    """
    gdf = gpd.read_file(path)

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "commodity": gdf["comm_main"].astype("string"),
            "deposit_codes": gdf["model_code"].apply(_parse_codes),
            "year": pd.to_datetime(gdf.get("rept_date"), errors="coerce").dt.year,
            "source": "ARDF",
            "raw_record_id": gdf["ardf_num"].astype(str),
        },
        crs=gdf.crs,
    )
    return validate_occurrences(out)
