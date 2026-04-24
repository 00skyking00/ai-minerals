"""BC GeoFile 2025-11 drillhole database → canonical drill-hole collars.

The phase-B blind-test set. The fetch layer (`data/bcgs_drillholes.py`)
pre-pivots the relational `determin` table into per-hole element maxima
(Cu/Mo/Au/Ag/Pb/Zn/As/Sb), normalized to ppm (or ppb for Au). This
adapter reads that pre-pivoted GeoPackage and flags each hole as
`intersected=True` if any per-element max clears its threshold:

  - Cu ≥ 0.2 %  (2,000 ppm)
  - Mo ≥ 0.03 % (300 ppm)
  - Au ≥ 0.5 g/t (500 ppb)
  - Ag ≥ 10 g/t (10 ppm)

Per-hole maxima are carried as `max_cu_pct` / `max_mo_pct` / `max_au_gpt`
/ `max_ag_gpt` in user-friendly units.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_drillholes


CU_INTERSECT_PPM = 2000.0     # 0.2 %
MO_INTERSECT_PPM = 300.0      # 0.03 %
AU_INTERSECT_PPB = 500.0      # 0.5 g/t
AG_INTERSECT_PPM = 10.0       # 10 g/t


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read the pre-pivoted BC drill-hole GeoPackage and emit canonical records."""
    gdf = gpd.read_file(path)

    cu = gdf["max_cu_ppm"].fillna(0)
    mo = gdf["max_mo_ppm"].fillna(0)
    au = gdf["max_au_ppb"].fillna(0)
    ag = gdf["max_ag_ppm"].fillna(0)

    intersected = (
        (cu >= CU_INTERSECT_PPM)
        | (mo >= MO_INTERSECT_PPM)
        | (au >= AU_INTERSECT_PPB)
        | (ag >= AG_INTERSECT_PPM)
    )

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "hole_id": gdf["hole_id"].astype(str),
            "drill_date": pd.to_datetime(gdf.get("drill_date"), errors="coerce"),
            "total_depth_m": gdf.get("total_depth_m"),
            "intersected": intersected,
            "max_cu_pct": cu / 10_000.0,     # ppm → %
            "max_mo_pct": mo / 10_000.0,     # ppm → %
            "max_au_gpt": au / 1_000.0,      # ppb → g/t
            "max_ag_gpt": ag,                # ppm = g/t (same mass units)
            "source": "BCGS_GF2025-11",
        },
        crs=gdf.crs,
    )
    return validate_drillholes(out)
