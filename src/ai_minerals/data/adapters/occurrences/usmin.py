"""USMIN placer/gravel point features → canonical occurrences.

USMIN's `FTR_TYPE` is a free-text feature class extracted from historical
USGS topo sheets. The fetcher (`ai_minerals.data.usmin`) already filters
the per-state shapefile to placer/gravel-style classes; this adapter
just normalizes the surviving rows to the canonical occurrence schema.

Every USMIN placer/gravel point gets both Cox-&-Singer placer codes
(`usgs:39a` Quaternary modern-channel + `usgs:39b` Tertiary deep-gravel)
because the source topo-sheet symbol does not distinguish modern from
buried-paleochannel placers. Downstream code that wants only one
population should re-filter on geomorphic context (e.g., a
hydraulic-pit-proximity threshold for the Tertiary subset) rather than
relying on USMIN to make that call.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_occurrences


SOURCE_TAG = "USMIN"
DEPOSIT_CODES: tuple[str, ...] = ("usgs:39a", "usgs:39b")


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Load USMIN placer/gravel points as canonical occurrences.

    Output schema (per `validate_occurrences`):
      geometry, commodity ("au gold"), deposit_codes (("usgs:39a", "usgs:39b")),
      source ("USMIN"), raw_record_id (from `gda_id` / `GDA_ID`, else row index).
    """
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() not in {"EPSG:4326", "WGS 84"}:
        gdf = gdf.to_crs("EPSG:4326")

    if "usmin_gda_id" in gdf.columns:
        record_id = gdf["usmin_gda_id"].astype("string")
    elif "GDA_ID" in gdf.columns:
        record_id = gdf["GDA_ID"].astype("string")
    else:
        record_id = gdf.index.astype(str)

    ftr_type = gdf["FTR_TYPE"] if "FTR_TYPE" in gdf.columns else None
    ftr_name = gdf["FTR_NAME"] if "FTR_NAME" in gdf.columns else None
    topo_name = gdf["TOPO_NAME"] if "TOPO_NAME" in gdf.columns else None

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "commodity": "au gold",
            "deposit_codes": [DEPOSIT_CODES] * len(gdf),
            "year": None,
            "source": SOURCE_TAG,
            "raw_record_id": record_id,
            "ftr_type": ftr_type,
            "ftr_name": ftr_name,
            "topo_name": topo_name,
        },
        crs="EPSG:4326",
    )
    return validate_occurrences(out)
