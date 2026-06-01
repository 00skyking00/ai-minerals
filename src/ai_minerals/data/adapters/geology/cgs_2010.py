"""CGS 2010 Geologic Map of California → canonical geology polygons + faults.

The CGS source carries rock-type as a two-letter (or longer) abbreviation in
`PTYPE` (occasionally `PTYPE2`). The first character is the broad age letter
(Q=Quaternary, T=Tertiary, K=Cretaceous, J=Jurassic, ...); the second
character(s) encode lithology (gr=granite, b=basalt, s=sandstone, ...).

For placer-Au prospectivity over the northern Sierra, the critical signal is
the Quaternary alluvial cover (Qa, Qal, Qg, Qoa, ...) that hosts the modern
and Tertiary-channel placer deposits. We expose `is_quaternary_alluvium` as a
direct boolean column so the assemble step does not have to redo the mapping.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import (
    validate_fault_lines,
    validate_geology_poly,
)


# Quaternary-alluvium pattern. CGS uses "Q" + lowercase lithology code
# for surficial Quaternary units; alluvial / gravel / fan / lake / wash
# subcodes all start with "Qa" / "Qg" / "Qf" / "Ql" / "Qw" / "Qoa" etc.
# We err on the side of inclusiveness: any PTYPE matching the regex
# below is flagged True. Tightening to specific alluvial subcodes
# can happen downstream if a unit-frequency check shows over-inclusion.
_Q_ALLUV_RE = re.compile(r"^Q(a|al|g|f|l|w|oa|s|ls|p|e|c|pf|d|m|t|ya|oal)?$", re.IGNORECASE)


def _is_q_alluvium(ptype: str | None) -> bool:
    if not ptype:
        return False
    return bool(_Q_ALLUV_RE.match(ptype.strip()))


# Second-character → broad lithology group for the CGS PTYPE convention.
# Strips the leading age letter then keys on the residual code's first
# character(s). Mappings are coarse but cover the dominant northern-
# Sierra units (granitoids, metasediments, volcanics, alluvium).
_LITH_GROUP_PREFIX: list[tuple[re.Pattern[str], str]] = [
    # Surficial Quaternary
    (re.compile(r"^Q"), "surficial"),
    # Intrusive plutonic (gr, gd, qm, di, gb, um for ultramafic, etc.)
    (re.compile(r"^[A-Z](gr|gd|qm|tn|di|gb|um|sy|mz|ad|to|pl|alk)"), "intrusive"),
    # Volcanic (v, b, a, r, da, rh, an, pyr, tu)
    (re.compile(r"^[A-Z](v|b|a|r|da|rh|an|py|tu|ig|ba)"), "volcanic"),
    # Metamorphic (m, sch, gn, qz, mb, ph, sl)
    (re.compile(r"^[A-Z](m|sch|gn|qz|mb|ph|sl|am)"), "metamorphic"),
    # Sedimentary (s, c, sh, ss, ls, cg, mu, ms)
    (re.compile(r"^[A-Z](s|c|sh|ss|ls|cg|mu|ms|dl|ch)"), "sedimentary"),
]


def _classify_group(ptype: str | None) -> str:
    if not ptype:
        return "other"
    code = ptype.strip()
    if not code:
        return "other"
    for pat, group in _LITH_GROUP_PREFIX:
        if pat.match(code):
            return group
    return "other"


def _pick_ptype(row_ptype: object, row_ptype2: object) -> str | None:
    """Prefer PTYPE; fall back to PTYPE2 if PTYPE is missing/blank."""
    for v in (row_ptype, row_ptype2):
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return None


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read the CGS 2010 geology GeoPackage and emit canonical polygons.

    Output schema:
    - geometry (in source CRS — typically WGS84 from the fetcher)
    - lith_class (int, factorized PTYPE)
    - lith_group (str, coarse bucket)
    - is_quaternary_alluvium (bool, flag for Qa/Qal/Qg-class units)
    - source ('CGS_2010')
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        out = gpd.GeoDataFrame(
            {
                "geometry": [],
                "lith_class": pd.Series([], dtype="int64"),
                "lith_group": pd.Series([], dtype="string"),
                "is_quaternary_alluvium": pd.Series([], dtype="bool"),
                "source": pd.Series([], dtype="string"),
            },
            crs=gdf.crs or "EPSG:4326",
        )
        return validate_geology_poly(out)

    ptype_col = gdf["PTYPE"] if "PTYPE" in gdf.columns else pd.Series([None] * len(gdf))
    ptype2_col = gdf["PTYPE2"] if "PTYPE2" in gdf.columns else pd.Series([None] * len(gdf))
    ptype = [_pick_ptype(a, b) for a, b in zip(ptype_col, ptype2_col)]

    lith_group = [_classify_group(p) for p in ptype]
    is_qal = [_is_q_alluvium(p) for p in ptype]

    ptype_series = pd.Series(ptype, dtype="string").fillna("(none)")
    codes, _ = pd.factorize(ptype_series)

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "lith_class": codes.astype("int64"),
            "lith_group": lith_group,
            "is_quaternary_alluvium": is_qal,
            "ptype": ptype_series,
            "source": "CGS_2010",
        },
        crs=gdf.crs,
    )
    return validate_geology_poly(out)


def load_faults(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Read the CGS 2010 fault/structure GeoPackage and emit canonical fault lines.

    Output schema:
    - geometry (line geometry in source CRS)
    - fault_type (str, from LTYPE / FTYPE if present)
    - source ('CGS_2010')
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        out = gpd.GeoDataFrame(
            {
                "geometry": [],
                "fault_type": pd.Series([], dtype="string"),
                "source": pd.Series([], dtype="string"),
            },
            crs=gdf.crs or "EPSG:4326",
        )
        return validate_fault_lines(out)

    # CGS structure-line attribute name varies between releases: LTYPE, FTYPE,
    # or NAME. Pick the first one present so the adapter is tolerant.
    fault_type_col: pd.Series
    for candidate in ("LTYPE", "FTYPE", "NAME", "TYPE"):
        if candidate in gdf.columns:
            fault_type_col = gdf[candidate].astype("string")
            break
    else:
        fault_type_col = pd.Series(["unknown"] * len(gdf), dtype="string")

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "fault_type": fault_type_col,
            "source": "CGS_2010",
        },
        crs=gdf.crs,
    )
    return validate_fault_lines(out)
