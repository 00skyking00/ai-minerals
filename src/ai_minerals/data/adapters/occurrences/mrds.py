"""USGS MRDS → canonical occurrences.

MRDS's `model_type` is free-text (e.g. 'Porphyry Cu-Au-Mo-Ag') rather than
Cox-&-Singer codes, so `deposit_codes` cannot be derived directly from
the source. For regions where MRDS is the primary occurrence source
(Mother Lode, Klamath, etc.), we synthesize coarse deposit-class codes
from the commodity field so that `deposit_positives` works without
modification. The mapping is intentionally conservative:

  Au-bearing record → ("usgs:36a", "usgs:36b")  (orogenic + low-sulfidation
                                                  Au; covers most CA lode-Au)
  Cu-bearing record → ("usgs:21a", "usgs:21b")  (porphyry Cu)
  Pb/Zn-bearing     → ("usgs:32a",)             (MVT Pb-Zn)

This is a known simplification: some MRDS Au records are placer or
epithermal, but for the orogenic-Au-dominated AOIs we target the noise
is small. When MRDS is used only for the any-mineral-occurrence
exclusion mask (BCGT, EastAK), this synthesis is invisible because
`deposit_positives` is not called against MRDS in those flows.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_occurrences


# Coarse commodity-to-Cox-Singer synthesis. Order matters because some
# records have multiple commodities (e.g. "au, ag, cu") and we want to
# pick the dominant one for label assignment.
_COMMOD_TO_CODES: list[tuple[str, tuple[str, ...]]] = [
    ("au",   ("usgs:36a", "usgs:36b")),
    ("gold", ("usgs:36a", "usgs:36b")),
    ("cu",   ("usgs:21a", "usgs:21b")),
    ("copper", ("usgs:21a", "usgs:21b")),
    ("pb",   ("usgs:32a",)),
    ("zn",   ("usgs:32a",)),
    ("lead", ("usgs:32a",)),
    ("zinc", ("usgs:32a",)),
]


# v3.1: Au records that should NOT be classified as orogenic Au — placer
# alluvial deposits, stream-sediment Au, etc. We use a regex on dep_type
# free text plus an exact match on oper_type. Records flagged as
# placer-style are dropped from the orogenic-Au synthesis (deposit_codes
# stays empty), so the binary `is_orogenic_gold` label excludes them.
import re as _re
_PLACER_DEP_TYPE_RE = _re.compile(
    r"placer|alluvial|stream.?placer|paleo.?placer|black.?sand|residual|eluvial",
    _re.I,
)


def _is_placer_au(dep_type: str | None, oper_type: str | None) -> bool:
    if oper_type and "placer" in oper_type.lower():
        return True
    if dep_type and _PLACER_DEP_TYPE_RE.search(dep_type):
        return True
    return False


def _synthesize_codes(
    commodity_str: str | None,
    dep_type: str | None = None,
    oper_type: str | None = None,
) -> tuple[str, ...]:
    if not commodity_str:
        return ()
    s = commodity_str.lower()
    for needle, codes in _COMMOD_TO_CODES:
        if needle in s:
            # v3.1: Au records flagged as placer/alluvial don't get
            # orogenic-Au codes. They still carry their `commodity`
            # string, so they stay in the any-mineral-occurrence
            # exclusion mask, just not in the binary label set.
            if needle in ("au", "gold") and _is_placer_au(dep_type, oper_type):
                return ()
            return codes
    return ()


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    commodity = gdf["commodity"].astype("string") if "commodity" in gdf.columns else pd.Series([""] * len(gdf), dtype="string")
    record_id = gdf["id"].astype(str) if "id" in gdf.columns else gdf.index.astype(str)
    # v3.1: pull dep_type + oper_type for the placer-vs-orogenic Au
    # refinement. Falls back to None where the source schema doesn't
    # carry these (older MRDS exports).
    dep_type = gdf.get("dep_type", pd.Series([None] * len(gdf)))
    oper_type = gdf.get("oper_type", pd.Series([None] * len(gdf)))
    deposit_codes = [
        _synthesize_codes(c, dt, ot)
        for c, dt, ot in zip(commodity, dep_type, oper_type)
    ]

    out = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "commodity": commodity,
            "deposit_codes": deposit_codes,
            "year": None,
            "source": "MRDS",
            "raw_record_id": record_id,
        },
        crs=gdf.crs,
    )
    return validate_occurrences(out)
