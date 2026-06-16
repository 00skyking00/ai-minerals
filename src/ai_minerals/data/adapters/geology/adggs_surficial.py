"""ADGGS Nome surficial geology -> canonical polygons.

Two source paths:

1. DGGS PDF 94-39 "Digital geologic map of the Nome mining district"
   (Kaufman, Hopkins, and others). The cache landed via bearcub on
   2026-06-15 at ``data/raw/bearcub_nome/nome_surface_geology.gpkg``,
   already reprojected to EPSG:4326 and joined with ``surface_class``
   labels (``glacial_drift_outwash``, ``marine_terrace_beach``,
   ``bedrock_schist``, ``placer_tailings``). This is the canonical
   layer for the Cape Nome AOI.
2. DGGS AOF 125 "Surficial geology of the Tolstoi Point - Cape Nome
   area" lives at ``data/raw/adggs/`` and is reserved for an east-of-
   Cape-Nome AOI expansion. AOF 125 carries no glacial-drift units
   (lumps all till into "undifferentiated Quaternary"); the published
   GeoPackage's ``MapUnitPolys`` is empty (only ice-wedge polygons
   are populated). Earlier code in this repo treated AOF 125's ``Qif``
   as "Iron Creek drift" -- that is wrong. In AOF 125 ``Qif`` is
   intertidal flat. Drift stratigraphy comes from PDF 94-39 only.

All ADGGS surficial polygons are surficial by definition, so
``lith_group`` is uniformly ``"surficial"``. ``lith_class`` factorizes
the ``MapUnit`` code so downstream one-hot encoding can pick up
individual drift / alluvium / beach units. The original ``MapUnit``
string is preserved in ``mapunit`` for human review. When the input
carries a coarse ``surface_class`` column (PDF 94-39 does, AOF 125
does not), the loader passes it through so downstream features can
key off ``surface_class == "glacial_drift_outwash"`` (the QM in-drift
mask) without re-deriving from individual ``Qd*`` codes.
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
    """Read an ADGGS Nome surficial GeoPackage; emit canonical polys.

    Output schema:
    - ``geometry`` (native CRS; reprojection to working CRS happens
      downstream)
    - ``lith_class`` (int, factorize on the MapUnit code)
    - ``lith_group`` ('surficial')
    - ``source`` ('ADGGS_PDF94_39' for the bearcub-published Nome cache,
      'ADGGS_AOF125' for the Tolstoi Point pub, otherwise 'ADGGS')
    - ``mapunit`` (str, the original code such as ``Qd2`` Nome River
      drift or ``Qmt`` Pelukian marine terrace)
    - ``surface_class`` (str, coarse class when present in the input;
      e.g. ``glacial_drift_outwash``, ``bedrock_schist``,
      ``marine_terrace_beach``, ``placer_tailings``). Absent inputs get
      the empty string so downstream code can mask cleanly.
    """
    gdf = gpd.read_file(path)
    mapunit_col = _find_mapunit_column(gdf)

    mapunit = gdf[mapunit_col].astype("string").fillna("(none)")
    codes, _ = pd.factorize(mapunit)

    if "pub_key" in gdf.columns:
        source_series = "ADGGS_" + gdf["pub_key"].astype("string").str.upper()
    elif "nome_surface_geology" in str(path):
        source_series = pd.Series(["ADGGS_PDF94_39"] * len(gdf), index=gdf.index)
    else:
        source_series = pd.Series(["ADGGS"] * len(gdf), index=gdf.index)

    if "surface_class" in gdf.columns:
        surface_class = gdf["surface_class"].astype("string").fillna("")
    else:
        surface_class = pd.Series([""] * len(gdf), index=gdf.index, dtype="string")

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "lith_class": codes.astype("int64"),
            "lith_group": "surficial",
            "source": source_series,
            "mapunit": mapunit,
            "surface_class": surface_class,
        },
        crs=gdf.crs,
    )
    out.attrs["aoi_name"] = aoi.name
    return validate_geology_poly(out)
