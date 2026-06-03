"""CGS Jennings 2010 Fault Activity Map → canonical fault lines split by activity.

The Jennings classification distinguishes pre-Quaternary (Mesozoic / older,
mostly inactive) faults from Quaternary-active faults (Holocene, Late
Quaternary, undifferentiated Quaternary). For placer prospectivity the two
classes reflect different physical controls:

- pre-Quaternary structural fabric controls lode emplacement and therefore
  paystreak supply to Tertiary deep-gravel placers.
- Quaternary-active faults tilt and warp modern drainages, controlling where
  Holocene channel placers concentrate.

The adapter normalizes whatever class column the FeatureServer ships with
(field names have shifted across CGS releases — common values include
`fault_activity_class`, `rcomp_clas`, `activity`, `RECENCY`) into a single
canonical column `activity_class` with two values: `pre_quaternary` and
`quaternary_active`.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_fault_lines


# Candidate column names the CGS Jennings publication has used over time.
_CLASS_COL_CANDIDATES: tuple[str, ...] = (
    "fault_activity_class",
    "rcomp_clas",
    "RECENCY",
    "recency",
    "activity",
    "ACTIVITY",
    "FaultActivityClass",
)

# Substrings that mark a fault as pre-Quaternary. Anything with "pre-" or
# "pre " or "pre_" prefixed to Quaternary / Pleistocene falls here.
_PRE_QUATERNARY_MARKERS: tuple[str, ...] = (
    "pre-",
    "pre ",
    "pre_",
    "prequaternary",
    "prepleistocene",
)


def _normalize_activity(raw: object) -> str | None:
    """Map a raw class value to {'pre_quaternary', 'quaternary_active'} or None."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s == "nan":
        return None
    for marker in _PRE_QUATERNARY_MARKERS:
        if marker in s:
            return "pre_quaternary"
    # Holocene / Late Quaternary / Quaternary / undifferentiated all map to active.
    if "holocene" in s or "quaternary" in s or "pleistocene" in s:
        return "quaternary_active"
    return None


def _pick_class_column(gdf: gpd.GeoDataFrame) -> str | None:
    for candidate in _CLASS_COL_CANDIDATES:
        if candidate in gdf.columns:
            return candidate
    return None


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read the CGS Jennings 2010 GeoPackage and emit canonical split fault lines.

    Output schema:
    - geometry (line geometry in source CRS — typically WGS84 from the fetcher)
    - activity_class (str, in {'pre_quaternary', 'quaternary_active'})
    - fault_type (str, the raw class string for traceability)
    - source ('CGS_JENNINGS_2010')
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        out = gpd.GeoDataFrame(
            {
                "geometry": [],
                "activity_class": pd.Series([], dtype="string"),
                "fault_type": pd.Series([], dtype="string"),
                "source": pd.Series([], dtype="string"),
            },
            crs=gdf.crs or "EPSG:4326",
        )
        return validate_fault_lines(out)

    class_col = _pick_class_column(gdf)
    if class_col is None:
        # No recognized class column — treat every fault as unclassified.
        # We still return a valid GeoDataFrame so the caller can decide how
        # to handle it; activity_class is filled with NaN.
        raw = pd.Series([None] * len(gdf))
    else:
        raw = gdf[class_col]

    activity = [_normalize_activity(v) for v in raw]
    raw_str = raw.astype("string").fillna("")

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "activity_class": pd.Series(activity, dtype="string"),
            "fault_type": raw_str,
            "source": "CGS_JENNINGS_2010",
        },
        crs=gdf.crs,
    )
    return validate_fault_lines(out)
