"""BC Digital Geology (BCGS) → canonical geology polygons.

Bedrock polygons from BCGS's provincial compilation (Bedrock_ll83_poly
layer). Maps:
  rock_class  → lith_group  (one of {intrusive, volcanic, sedimentary,
                             metamorphic, surficial, other})
  rock_code   → lith_class  (2-letter lithology code; enumerated integer-
                             hashed for one-hot compatibility)
  age_min_ma  → age_ma      (minimum age in millions of years)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geology_poly


# BC `rock_class` strings → canonical coarse buckets.
_ROCK_CLASS_MAP = {
    "intrusive rocks":                   "intrusive",
    "ultramafic rocks":                  "intrusive",
    "volcanic rocks":                    "volcanic",
    "sedimentary rocks":                 "sedimentary",
    "sedimentary and volcanic rocks":    "volcanic",
    "volcanic and sedimentary rocks":    "volcanic",
    "metamorphic rocks":                 "metamorphic",
}


def _classify_group(rock_class: str | None) -> str:
    if rock_class is None or (isinstance(rock_class, float) and pd.isna(rock_class)):
        return "other"
    return _ROCK_CLASS_MAP.get(str(rock_class).strip().lower(), "other")


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    # `rock_code` is a 2-letter alphabetic code; encode as a stable int so
    # downstream one-hot works the same as EastAK's USGS `CLASS` integers.
    rock_code = gdf["rock_code"].astype("string").str.strip().fillna("")
    code_to_int = {code: i for i, code in enumerate(sorted(rock_code.unique()))}
    lith_class = rock_code.map(code_to_int).astype("int64")

    lith_group = gdf["rock_class"].apply(_classify_group)

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "lith_class": lith_class,
            "lith_group": lith_group,
            "age_ma": gdf.get("age_min_ma"),
            "source": "BCGS_DigitalGeology",
        },
        crs=gdf.crs,
    )
    return validate_geology_poly(out)
