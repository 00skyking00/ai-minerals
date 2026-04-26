"""Bear Cub Murray drill-log archive → canonical drill-hole collars.

24 historical placer-Au drill-hole headers from the family-held Murray
archive at the Bear Cub claim (MS 1178, Cape Nome mining district, Alaska),
drilled 1925-1955 across three form types (Hammon Field Log, Hammon
Prospect Drilling Log, Drill Report for Frozen Ground Only, Alaska Gold
Company). Per-interval gold yields are not in the public CSV — only
header-level data (collar coordinates, depths, dates, form type, OCR
confidence). See `data/raw/bear_cub/SOURCE.md` for transcription protocol
and georeferencing methodology.

Local-grid coordinates are translated to WGS84 by the upstream
georeferencing step (anchor: hole 7754 → MS 1178 BR corner;
cardinal-aligned grid; ~±100 ft absolute accuracy).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_drillholes


_FT_PER_M = 3.28084


def load(path: Path, aoi: AOI | None = None) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "lat_wgs84" not in df.columns or "lon_wgs84" not in df.columns:
        raise ValueError(
            f"{path} missing lat_wgs84/lon_wgs84 columns — "
            "run the georeferencing step first."
        )
    out = gpd.GeoDataFrame(
        {
            "geometry": gpd.points_from_xy(df["lon_wgs84"], df["lat_wgs84"]),
            "hole_id": df["hole_id"].astype(str),
            "drill_date": pd.to_datetime(df["date_drilled"], errors="coerce"),
            "total_depth_m": df["total_depth_ft"] / _FT_PER_M,
            "bedrock_depth_m": df["bedrock_depth_ft"] / _FT_PER_M,
            "elevation_m": df["elevation_ft"] / _FT_PER_M,
            "easting_local_ft": df["easting_local_ft"],
            "northing_local_ft": df["northing_local_ft"],
            "form_type": df["form_type"],
            "ocr_confidence": df["ocr_confidence"],
            "district": df["district"],
            "source": "bear_cub_murray_archive",
        },
        crs="EPSG:4326",
    )
    if aoi is not None:
        out = out[out.geometry.within(aoi.polygon)].copy()
    return validate_drillholes(out)
