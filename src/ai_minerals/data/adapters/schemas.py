"""Canonical schemas for the internal data layer.

Every source-specific adapter normalizes raw files to one of these schemas.
Downstream code (features, labels, models) consumes only canonical data —
no `if region == ...` branches.

Schemas are expressed as column-name tuples; validators raise on missing
required columns. Extra columns are allowed (and in practice carry source-
specific debug/reference fields).
"""

from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import pandas as pd


# --- Occurrences ---
# ARDF/MRDS-like mineral-occurrence records. Deposit codes are
# jurisdiction-prefixed to avoid collision between code systems
# (USGS Cox-&-Singer "21a" vs BC MINFILE profile "L03").
OCCURRENCE_REQUIRED = ("geometry", "commodity", "deposit_codes", "source", "raw_record_id")
OCCURRENCE_OPTIONAL = ("year",)


# --- Geochem samples ---
# Best-value compiled rock/soil/stream-sediment geochemistry. Pathfinder
# element columns are suffixed `_ppm` (or `_ppb` for Au in some schemas).
# Negative values (below-detection-limit sentinel) are masked to NaN by
# the adapter.
GEOCHEM_REQUIRED = ("geometry", "sample_id", "source")
GEOCHEM_OPTIONAL = ("sample_type", "sample_date")


# --- Geology polygons ---
# `lith_class` is the jurisdiction-local integer code (preserves fidelity
# within a region for one-hot encoding). `lith_group` is a coarse bucket
# that's consistent across jurisdictions: {intrusive, volcanic, sedimentary,
# metamorphic, surficial, other}.
GEOLOGY_POLY_REQUIRED = ("geometry", "lith_class", "lith_group", "source")
GEOLOGY_POLY_OPTIONAL = ("age_ma",)


# --- Fault lines ---
FAULT_LINE_REQUIRED = ("geometry", "source")
FAULT_LINE_OPTIONAL = ("fault_type",)


# --- Drill-hole collars ---
# One row per hole. `intersected` marks whether the hole hit mineralization
# above a per-deposit-type threshold (adapter-assigned; adapters document
# thresholds in their docstrings).
DRILLHOLE_REQUIRED = ("geometry", "hole_id", "source")
DRILLHOLE_OPTIONAL = ("drill_date", "total_depth_m", "intersected",
                     "max_cu_pct", "max_mo_pct", "max_au_gpt", "max_ag_gpt")


# Geophysics is an xarray DataArray (not a GeoDataFrame), so no column
# validator here. Adapter contract: returns 2-D DataArray with NaN in
# nodata cells; attrs include `units`, `field_name`, `source`.


def _validate(df: pd.DataFrame, required: Iterable[str], schema_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{schema_name} schema violation: missing columns {missing}. "
            f"Got: {list(df.columns)}"
        )


def validate_occurrences(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _validate(df, OCCURRENCE_REQUIRED, "Occurrence")
    return df


def validate_geochem(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _validate(df, GEOCHEM_REQUIRED, "Geochem")
    return df


def validate_geology_poly(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _validate(df, GEOLOGY_POLY_REQUIRED, "GeologyPoly")
    return df


def validate_fault_lines(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _validate(df, FAULT_LINE_REQUIRED, "FaultLine")
    return df


def validate_drillholes(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _validate(df, DRILLHOLE_REQUIRED, "DrillHole")
    return df
