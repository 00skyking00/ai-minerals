"""ADGGS Cape Nome surficial geology -> canonical polygons.

Reads the merged GeoPackage written by ``data/adggs_surficial.py``
(AOF 125 Tolstoi Point - Cape Nome + PDF 94-39 Nome Mining District).
Both are GeMS-schema publications; the merged file carries the original
``MapUnit`` code (a short ADGGS unit label such as ``Qb`` for beach
gravel or ``Qif`` for Iron Creek drift) alongside the geometry.

All ADGGS surficial polygons are surficial by definition, so
``lith_group`` is uniformly ``"surficial"``. ``lith_class`` factorizes
the ``MapUnit`` code so downstream one-hot encoding can pick up
individual drift / alluvium / beach units. The original ``MapUnit``
string is preserved in ``mapunit`` for human review and for feature
engineering in ``features/coastal.py`` (where Iron Creek vs Nome River
drift codes get mapped onto the QM / TB population masks).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geology_poly


_MAPUNIT_FIELD_CANDIDATES = ("MapUnit", "mapunit", "MAPUNIT", "Map_Unit", "Symbol", "Label")


def _find_mapunit_column(gdf: gpd.GeoDataFrame) -> str:
    for name in _MAPUNIT_FIELD_CANDIDATES:
        if name in gdf.columns:
            return name
    # Fall back to the first non-geometry string column. Better than
    # crashing on an off-schema variant; downstream factorize handles it.
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        if gdf[col].dtype == object:
            return col
    raise ValueError(
        f"No MapUnit column found in ADGGS gpkg; columns={list(gdf.columns)}"
    )


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read the merged ADGGS Cape Nome surficial GeoPackage; emit canonical polys.

    Output schema:
    - ``geometry`` (native ADGGS CRS; reprojection to working CRS happens
      downstream)
    - ``lith_class`` (int, factorize on the ADGGS MapUnit code)
    - ``lith_group`` ('surficial')
    - ``source`` ('ADGGS_AOF125' / 'ADGGS_PDF94_39' from the ``pub_key``
      column the fetcher tagged)
    - ``mapunit`` (str, the original ADGGS code such as ``Qb`` or ``Qif``)
    """
    gdf = gpd.read_file(path)
    mapunit_col = _find_mapunit_column(gdf)

    mapunit = gdf[mapunit_col].astype("string").fillna("(none)")
    codes, _ = pd.factorize(mapunit)

    if "pub_key" in gdf.columns:
        source_series = "ADGGS_" + gdf["pub_key"].astype("string").str.upper()
    else:
        source_series = pd.Series(["ADGGS"] * len(gdf), index=gdf.index)

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "lith_class": codes.astype("int64"),
            "lith_group": "surficial",
            "source": source_series,
            "mapunit": mapunit,
        },
        crs=gdf.crs,
    )
    out.attrs["aoi_name"] = aoi.name
    return validate_geology_poly(out)
