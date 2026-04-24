"""USGS MRDS → canonical occurrences.

MRDS's `model_type` is free-text (e.g. 'Porphyry Cu-Au-Mo-Ag') rather than
Cox-&-Singer codes, so `deposit_codes` is empty here. MRDS feeds only the
any-mineral-occurrence exclusion mask; deposit-code-based filtering runs
against ARDF.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_occurrences


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    commodity = gdf["commodity"].astype("string") if "commodity" in gdf.columns else pd.Series([""] * len(gdf), dtype="string")
    record_id = gdf["id"].astype(str) if "id" in gdf.columns else gdf.index.astype(str)
    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "commodity": commodity,
            "deposit_codes": [() for _ in range(len(gdf))],
            "year": None,
            "source": "MRDS",
            "raw_record_id": record_id,
        },
        crs=gdf.crs,
    )
    return validate_occurrences(out)
