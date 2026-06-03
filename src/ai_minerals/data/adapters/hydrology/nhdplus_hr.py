"""NHDPlus HR clipped GeoPackage → canonical hydrology network.

The fetcher in `data/nhdplus_hr.py` already joins NHDFlowline geometries
to NHDPlusFlowlineVAA, clips to the buffered AOI, and writes a
GeoPackage with canonical column names. This adapter is a thin loader +
schema validator + WGS84 reprojection guard.

The returned GeoDataFrame matches the `HYDROLOGY_NET` schema in
`schemas.py` and is what downstream placer-model code (downstream-walk
from positives, drainage-area features) consumes.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_hydrology_network


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read the NHDPlus HR flowline GeoPackage as canonical hydrology lines.

    Output schema:
    - geometry (LineString, EPSG:4326)
    - comid (int64, NHDPlusID join key)
    - arbolate_sum (float64, cumulative upstream channel length per NHDPlus HR)
    - stream_order (int64, Strahler order)
    - fcode (int64, NHD feature classification)
    - hydroseq (int64, deterministic downstream-walk key)
    - slope (float64, per-flowline longitudinal slope, dimensionless, NaN-preserving)
    - source ('NHDPlus_HR')
    """
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    gdf["source"] = "NHDPlus_HR"

    # The fetcher writes lowercase canonical names; defend against any
    # older snapshots that might still carry the raw NHD CamelCase.
    rename = {
        "NHDPlusID": "comid",
        "COMID": "comid",
        "ArbolateSum": "arbolate_sum",
        "StreamOrde": "stream_order",
        "FCode": "fcode",
        "Hydroseq": "hydroseq",
        "Slope": "slope",
    }
    to_rename = {k: v for k, v in rename.items() if k in gdf.columns and v not in gdf.columns}
    if to_rename:
        gdf = gdf.rename(columns=to_rename)

    return validate_hydrology_network(gdf)
