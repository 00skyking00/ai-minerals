"""BC Regional Geochemical Survey (GeoFile 2020-08) → canonical geochem samples.

The RGS data has multiple analytical methods per element
(`<EL>_<METHOD>_<UNIT>`). This adapter picks, per element, the highest-
priority method that has a value in each sample row, and normalizes
everything to ppm (dividing ppb by 1000 where needed). Output column
names match AGDB4 convention: `<El>_ppm`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geochem


# Per element, ordered list of (column_name, unit_scale_to_ppm).
# Priority: most reliable method first. INA and FA are sensitive low-level
# methods; ICP and AAS are bulk methods.
_ELEMENT_PRIORITY: dict[str, list[tuple[str, float]]] = {
    "Ag": [("AG_FA_PPM", 1.0), ("AG_INA_PPM", 1.0), ("AG_ICP_PPB", 1e-3), ("AG_AAS_PPM", 1.0)],
    "As": [("AS_INA_PPM", 1.0), ("AS_ICP_PPM", 1.0), ("AS_AAS_PPM", 1.0)],
    "Au": [("AU_FA_PPB", 1e-3), ("AU_INA_PPB", 1e-3), ("AU_ICP_PPB", 1e-3)],
    "Ba": [("BA_INA_PPM", 1.0), ("BA_ICP_PPM", 1.0), ("BA_AAS_PPM", 1.0)],
    "Bi": [("BI_ICP_PPM", 1.0), ("BI_AAS_PPM", 1.0)],
    "Cu": [("CU_ICP_PPM", 1.0), ("CU_AAS_PPM", 1.0)],
    "Hg": [("HG_INA_PPM", 1.0), ("HG_AAS_PPB", 1e-3), ("HG_ICP_PPB", 1e-3)],
    "Mo": [("MO_INA_PPM", 1.0), ("MO_ICP_PPM", 1.0), ("MO_AAS_PPM", 1.0)],
    "Pb": [("PB_ICP_PPM", 1.0), ("PB_AAS_PPM", 1.0)],
    "Sb": [("SB_INA_PPM", 1.0), ("SB_ICP_PPM", 1.0), ("SB_AAS_PPM", 1.0)],
    "Te": [("TE_ICP_PPM", 1.0)],
    "Tl": [("TL_ICP_PPM", 1.0)],
    "Zn": [("ZN_INA_PPM", 1.0), ("ZN_ICP_PPM", 1.0), ("ZN_AAS_PPM", 1.0)],
}


def _best_value(row: pd.Series, candidates: list[tuple[str, float]]) -> float:
    """First non-NaN column from a priority list, converted to ppm."""
    for col, scale in candidates:
        if col not in row.index:
            continue
        v = row[col]
        if pd.notna(v) and v != "" and v > 0:
            return float(v) * scale
    return np.nan


def load(
    path: Path,
    aoi: AOI,
    *,
    elements: Iterable[str] = tuple(_ELEMENT_PRIORITY),
    **_extra,
) -> gpd.GeoDataFrame:
    """Read the AOI-clipped RGS parquet and emit canonical geochem samples.

    `_extra` absorbs any region-specific kwargs (e.g. `bv_zip` from AGDB4)
    that don't apply here.
    """
    df = pd.read_parquet(path)
    df.columns = [c.strip().upper() for c in df.columns]

    # Numeric coercion on all element-method columns (Excel can yield strings).
    for el in elements:
        for col, _ in _ELEMENT_PRIORITY.get(el, []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    out_data: dict[str, pd.Series] = {}
    for el in elements:
        candidates = _ELEMENT_PRIORITY.get(el, [])
        out_data[f"{el}_ppm"] = df.apply(lambda r: _best_value(r, candidates), axis=1)

    lat_col = "LATITUDE" if "LATITUDE" in df.columns else "LAT"
    lon_col = "LONGITUDE" if "LONGITUDE" in df.columns else "LONG"

    sample_id = df["MASTERID"].astype(str) if "MASTERID" in df.columns else df.index.astype(str)
    sample_type = df["TYPE2"].astype("string") if "TYPE2" in df.columns else pd.Series([None] * len(df), dtype="string")
    sample_date = pd.to_datetime(df.get("DATE"), errors="coerce")

    out = gpd.GeoDataFrame(
        {
            "geometry": gpd.points_from_xy(df[lon_col], df[lat_col]),
            "sample_id": sample_id,
            "sample_type": sample_type,
            "sample_date": sample_date,
            "source": "BCGS_RGS_2020",
            **out_data,
        },
        crs="EPSG:4326",
    )
    return validate_geochem(out)
