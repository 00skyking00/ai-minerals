"""BC MINFILE → canonical occurrences.

MINFILE's deposit-classification field is BC-specific profile codes
(e.g. `L03` porphyry Cu-Au, `H04` epithermal Au-Ag, `K01` skarn Cu-Au),
distinct from USGS Cox-&-Singer. Prefixed as `bc:<code>` in the canonical
schema.

Schema (from BCGS/BCGW MINFILE_MINERAL CSV export):
  MINERAL_FILE_NUMBER, MINFILE_NAME1, DECIMAL_LATITUDE, DECIMAL_LONGITUDE,
  DEPOSIT_TYPE_CODE1, DEPOSIT_TYPE_CODE2, COMMODITY_CODE1..8,
  COMMODITY_DESCRIPTION1..8, DEPOSIT_CLASS_CODE1, STATUS_CODE, ...

Adapter reads the AOI-clipped GeoPackage produced by
`ai_minerals.data.bcgs_minfile.clip_to_aoi`.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_occurrences


def _parse_deposit_codes(code1: str | None, code2: str | None) -> tuple[str, ...]:
    """Collect DEPOSIT_TYPE_CODE1 + CODE2 into a prefixed tuple."""
    codes = []
    for c in (code1, code2):
        if c is None:
            continue
        s = str(c).strip()
        if s and s != "nan" and s != "*":
            codes.append(f"bc:{s.lower()}")
    return tuple(codes)


def _commodity_str(row: pd.Series) -> str:
    """Join COMMODITY_DESCRIPTION1..8 into one comma-separated string."""
    parts = []
    for i in range(1, 9):
        v = row.get(f"COMMODITY_DESCRIPTION{i}")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s and s != "nan":
            parts.append(s)
    return ", ".join(parts)


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read an AOI-clipped BC MINFILE GeoPackage and emit canonical occurrences."""
    gdf = gpd.read_file(path)

    # Codes: BCGW writes them space-padded; strip defensively even if the
    # fetcher already did so.
    for c in ("DEPOSIT_TYPE_CODE1", "DEPOSIT_TYPE_CODE2"):
        if c in gdf.columns:
            gdf[c] = gdf[c].astype("string").str.strip()

    codes_tuples = [
        _parse_deposit_codes(a, b)
        for a, b in zip(
            gdf.get("DEPOSIT_TYPE_CODE1", pd.Series([None] * len(gdf))),
            gdf.get("DEPOSIT_TYPE_CODE2", pd.Series([None] * len(gdf))),
        )
    ]

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "commodity": gdf.apply(_commodity_str, axis=1).astype("string"),
            "deposit_codes": codes_tuples,
            "year": None,  # MINFILE CSV doesn't carry discovery-year directly
            "source": "BC_MINFILE",
            "raw_record_id": gdf["MINERAL_FILE_NUMBER"].astype(str),
        },
        crs=gdf.crs,
    )
    return validate_occurrences(out)
