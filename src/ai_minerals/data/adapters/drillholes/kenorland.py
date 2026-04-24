"""Kenorland Minerals Tanacross manual stub CSV → canonical drill-hole collars.

The CSV is manually curated from Kenorland's public disclosures; see
`data/raw/kenorland/SOURCE.md` for precision caveats. One named hole
(`23ETD062`) plus four approximate project-polygon centroids.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_drillholes


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    intersected = df["outcome"].eq("positive")
    out = gpd.GeoDataFrame(
        {
            "geometry": gpd.points_from_xy(df["lon"], df["lat"]),
            "hole_id": df["hole_id"].astype(str),
            "drill_date": pd.to_datetime(df["drill_year"].astype(str).str[:4], errors="coerce"),
            "total_depth_m": df.get("interval_m"),
            "intersected": intersected,
            "max_cu_pct": df.get("cu_pct"),
            "max_mo_pct": df.get("mo_pct"),
            "max_au_gpt": df.get("au_gpt"),
            "source": "Kenorland_manual",
        },
        crs="EPSG:4326",
    )
    return validate_drillholes(out)
