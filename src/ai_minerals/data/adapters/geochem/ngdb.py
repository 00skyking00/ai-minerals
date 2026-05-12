"""USGS NGDB (National Geochemical Database) → canonical geochem samples.

Reads per-region clipped GeoPackages written by `data/ngdb.py`. The fetcher
already pivoted bestvalue.csv to wide format with `<el>_ppm` columns and
masked below-detection-limit values to NaN, so this adapter is a thin
schema-validator + canonical-column rename.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_geochem


def load(
    path: Path,
    aoi: AOI,
    *,
    elements: Iterable[str] | None = None,
) -> gpd.GeoDataFrame:
    """Read a per-region NGDB sediment GeoPackage as canonical geochem samples.

    Output schema:
    - geometry (in working CRS, set by fetcher)
    - sample_id (alias for lab_id)
    - source ('NGDB_sediment')
    - sample_type (from primary_class / sample_source)
    - sample_date (from date_collect, optional)
    - <el>_ppm columns for each requested pathfinder element

    If `elements` is given, drop any pathfinder columns not in the list.
    The fetcher always emits the full pathfinder set; this filter just
    trims to the region's pathfinder list at adapter time.
    """
    gdf = gpd.read_file(path)

    if "lab_id" not in gdf.columns:
        raise ValueError(f"Expected 'lab_id' in NGDB schema; got {list(gdf.columns)}")

    gdf = gdf.rename(columns={
        "lab_id": "sample_id",
        "primary_class": "sample_type",
        "date_collect": "sample_date",
    })
    gdf["source"] = "NGDB_sediment"

    if elements is not None:
        keep_ppm = {f"{el}_ppm" for el in elements}
        ppm_cols = [c for c in gdf.columns if c.endswith("_ppm")]
        drop = [c for c in ppm_cols if c not in keep_ppm]
        if drop:
            gdf = gdf.drop(columns=drop)

    return validate_geochem(gdf)
