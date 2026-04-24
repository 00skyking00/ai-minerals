"""USGS SGMC (State Geologic Map Compilation) → canonical geology polygons."""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geology_poly


# Coarse lith_group classifier — keyword-match against STATE_UNITNAME /
# AGE_RANGE to produce one of {intrusive, volcanic, sedimentary, metamorphic,
# surficial, other}. Patterns ordered by specificity; first match wins.
_LITH_RULES = [
    ("surficial",   re.compile(r"surficial|alluvi|glacia|till|moraine|quaternary", re.I)),
    ("intrusive",   re.compile(r"granit|granodior|diorit|gabbro|plutonic|intrus|syenite|porphyr|monzon", re.I)),
    ("volcanic",    re.compile(r"volcan|basalt|andesite|rhyolite|tuff|dacite|tephra|lava", re.I)),
    ("metamorphic", re.compile(r"schist|gneiss|amphib|metam|migmatit|phyllit|slate|quartzit|marble", re.I)),
    ("sedimentary", re.compile(r"sedim|sandstone|mudstone|shale|limestone|conglomer|dolostone|siltst|chert", re.I)),
]


def _classify_group(unit_name: str | None, age: str | None) -> str:
    text = f"{unit_name or ''}  {age or ''}"
    for group, pat in _LITH_RULES:
        if pat.search(text):
            return group
    return "other"


def load(path: Path, aoi: AOI, *, class_column: str = "CLASS") -> gpd.GeoDataFrame:
    """Read SGMC GeoPackage and emit canonical geology polygons.

    `CLASS` (int) → `lith_class` (jurisdiction-local code, preserves v1
    one-hot behavior). `STATE_UNITNAME` + `AGE_RANGE` → `lith_group`
    (coarse bucket for cross-region consistency).
    """
    gdf = gpd.read_file(path)

    if class_column not in gdf.columns:
        raise ValueError(f"Expected {class_column!r} in SGMC schema: {list(gdf.columns)}")

    unit_name = gdf.get("STATE_UNITNAME", pd.Series([None] * len(gdf)))
    age_range = gdf.get("AGE_RANGE", pd.Series([None] * len(gdf)))
    lith_group = [_classify_group(u, a) for u, a in zip(unit_name, age_range)]

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "lith_class": gdf[class_column].astype("int64"),
            "lith_group": lith_group,
            "age_ma": None,  # SGMC carries string age ranges, not Ma; leave null
            "source": "SGMC",
        },
        crs=gdf.crs,
    )
    return validate_geology_poly(out)
