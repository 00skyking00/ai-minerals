"""USGS SGMC (State Geologic Map Compilation) → canonical geology polygons.

Reads the per-region clipped GeoPackages written by `data/sgmc.py`. Maps
SGMC's controlled-vocabulary `GENERALIZED_LITH` field directly onto the
canonical `lith_group` categorical, instead of regex-matching against
unit names. SGMC built `GENERALIZED_LITH` specifically as a cross-region
lithology classifier, so we use it.

This adapter previously assumed the schema of the Alaska sim3340 GDB
(`STATE_UNITNAME`, `AGE_RANGE`, `CLASS`), which does not match the
actual SGMC schema (`UNIT_NAME`, `AGE_MIN`, `AGE_MAX`, `GENERALIZED_LITH`,
`MAJOR1-3`, `MINOR1-5`, `UNIT_LINK`). The rewrite uses real SGMC field
names and the controlled-vocabulary classifier.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geology_poly


# SGMC GENERALIZED_LITH controlled vocabulary -> our canonical lith_group.
# Vocabulary verified against SGMC v1.1 (Horton et al. 2017) by exhaustive
# enumeration of distinct values nationwide.
#
# Categories prioritize what matters for orogenic-Au prospectivity:
# - "greenstone" surfaces metamorphic-volcanic units (the Calaveras
#   Complex / Foothills Terrane / Mariposa Slate hosts of the Mother
#   Lode).
# - "ultramafic" captures serpentinite, the other shear-zone-associated
#   host.
# - "metasediment" captures schist + sedimentary-clastic metamorphic
#   units.
# - "melange" surfaces accreted oceanic-crust melange units.
# - "intrusive" captures granitoids (Sierra Nevada batholith).
_GENLITH_TO_GROUP: dict[str, str] = {
    # Orogenic-Au-prone hosts
    "Metamorphic, volcanic": "greenstone",
    "Metamorphic, serpentinite": "ultramafic",
    "Metamorphic, schist": "metasediment",
    "Metamorphic, sedimentary": "metasediment",
    "Metamorphic, sedimentary clastic": "metasediment",
    "Metamorphic, amphibolite": "metasediment",
    "Metamorphic, gneiss": "metamorphic",
    "Metamorphic, carbonate": "metamorphic",
    "Metamorphic, granulite": "metamorphic",
    "Metamorphic, igneous": "metamorphic",
    "Metamorphic, intrusive": "metamorphic",
    "Metamorphic, undifferentiated": "metamorphic",
    "Metamorphic, other": "metamorphic",
    "Melange": "melange",
    # Granitoids
    "Igneous, intrusive": "intrusive",
    "Igneous, undifferentiated": "intrusive",
    # Volcanics (unmetamorphosed)
    "Igneous, volcanic": "volcanic",
    # Sediments
    "Sedimentary, clastic": "sedimentary",
    "Sedimentary, carbonate": "sedimentary",
    "Sedimentary, chemical": "sedimentary",
    "Sedimentary, evaporite": "sedimentary",
    "Sedimentary, iron formation, undifferentiated": "sedimentary",
    "Sedimentary, undifferentiated": "sedimentary",
    "Unconsolidated, undifferentiated": "surficial",
    "Unconsolidated and Sedimentary, undifferentiated": "surficial",
    # Mixed / undifferentiated buckets
    "Igneous and Metamorphic, undifferentiated": "metamorphic",
    "Igneous and Sedimentary, undifferentiated": "sedimentary",
    "Metamorphic and Sedimentary, undifferentiated": "metasediment",
    # Tectonites
    "Tectonite, undifferentiated": "tectonite",
    # Non-rock
    "Water": "water",
    "Ice": "water",
    "Dam": "water",
    "Unknown": "other",
}


def _classify_group(generalized_lith: str | None) -> str:
    if not generalized_lith:
        return "other"
    return _GENLITH_TO_GROUP.get(generalized_lith.strip(), "other")


# Coarse age-bin extractor. SGMC AGE_MIN/AGE_MAX are strings like
# "Phanerozoic - Paleozoic - Carboniferous - Pennsylvanian"; we extract
# the broad era for downstream feature engineering.
_AGE_KEYWORDS: list[tuple[str, str]] = [
    ("Cenozoic", "cenozoic"),
    ("Quaternary", "cenozoic"),
    ("Tertiary", "cenozoic"),
    ("Mesozoic", "mesozoic"),
    ("Cretaceous", "mesozoic"),
    ("Jurassic", "mesozoic"),
    ("Triassic", "mesozoic"),
    ("Paleozoic", "paleozoic"),
    ("Proterozoic", "proterozoic"),
    ("Archean", "archean"),
    ("preCambrian", "precambrian"),
]


def _classify_age(age_min: str | None, age_max: str | None) -> str:
    text = f"{age_min or ''}  {age_max or ''}"
    for kw, era in _AGE_KEYWORDS:
        if kw in text:
            return era
    return "unknown"


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read a per-region SGMC geology GeoPackage and emit canonical polygons.

    Output schema:
    - geometry (already in working CRS, reprojected by `data/sgmc.py`)
    - lith_class (int, jurisdiction-local code via factorize on UNIT_LINK)
    - lith_group (str, canonical category from `_GENLITH_TO_GROUP`)
    - age_era (str, broad era from AGE_MIN/AGE_MAX)
    - source ('SGMC')
    """
    gdf = gpd.read_file(path)

    if "GENERALIZED_LITH" not in gdf.columns:
        raise ValueError(
            f"Expected 'GENERALIZED_LITH' in SGMC schema; got {list(gdf.columns)}"
        )

    lith_group = [_classify_group(v) for v in gdf["GENERALIZED_LITH"]]
    age_era = [
        _classify_age(amin, amax)
        for amin, amax in zip(
            gdf.get("AGE_MIN", pd.Series([None] * len(gdf))),
            gdf.get("AGE_MAX", pd.Series([None] * len(gdf))),
        )
    ]
    unit_link = gdf.get("UNIT_LINK", pd.Series(["?"] * len(gdf), dtype="string"))
    codes, _ = pd.factorize(unit_link.astype("string").fillna("?"))

    # v3.1: also factorize the SGMC MAJOR1/2/3 fields, which carry
    # finer-grained lithology than GENERALIZED_LITH (e.g., "Slate",
    # "Granodiorite", "Tonalite", "Schist"). Each gets its own
    # per-region integer encoding so downstream one-hot encoding can
    # pick up specific named units that GENERALIZED_LITH lumps together.
    def _factorize(series_name: str) -> "pd.Series":
        s = gdf.get(series_name, pd.Series([""] * len(gdf), dtype="string"))
        s = s.astype("string").fillna("").str.strip().str.lower()
        codes_, _ = pd.factorize(s.where(s != "", other="(none)"))
        return pd.Series(codes_.astype("int64"), index=gdf.index)

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "lith_class": codes.astype("int64"),
            "lith_group": lith_group,
            "age_era": age_era,
            "major1_class": _factorize("MAJOR1"),
            "major2_class": _factorize("MAJOR2"),
            "major3_class": _factorize("MAJOR3"),
            "source": "SGMC",
        },
        crs=gdf.crs,
    )
    return validate_geology_poly(out)
