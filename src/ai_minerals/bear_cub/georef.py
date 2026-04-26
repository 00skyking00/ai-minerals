"""Local-grid (feet) ↔ WGS84 conversion for the Bear Cub Murray drill data.

The Murray subset uses a cardinal-aligned local grid in feet (E_local axis ≈
true east, N_local ≈ true north). Anchor: hole 7754 → MS 1178 BR corner
(BLM patent). See `data/raw/bear_cub/SOURCE.md` for the derivation +
sanity-check overlay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Anchor: hole 7754 (E_local=77,696 ft, N_local=22,702 ft) → MS 1178 BR corner.
ANCHOR_E_LOCAL_FT = 77696.0
ANCHOR_N_LOCAL_FT = 22702.0
ANCHOR_LAT = 64.531171
ANCHOR_LON = -165.332170

FT_PER_DEG_LAT = 364400.0
FT_PER_DEG_LON_AT_ANCHOR = FT_PER_DEG_LAT * math.cos(math.radians(ANCHOR_LAT))


# Bear Cub MS 1178 patent corners (BLM Master Title Plat, WGS84).
MS_1178_CORNERS = {
    "TR": (64.532488, -165.337952),
    "TL": (64.531784, -165.341551),
    "BR": (64.531171, -165.332170),
    "BL": (64.530095, -165.335329),
}


@dataclass(frozen=True)
class LocalGridAnchor:
    e_local_ft: float
    n_local_ft: float
    lat: float
    lon: float


DEFAULT_ANCHOR = LocalGridAnchor(
    e_local_ft=ANCHOR_E_LOCAL_FT,
    n_local_ft=ANCHOR_N_LOCAL_FT,
    lat=ANCHOR_LAT,
    lon=ANCHOR_LON,
)


def local_to_wgs84(
    e_local_ft: float, n_local_ft: float, anchor: LocalGridAnchor = DEFAULT_ANCHOR
) -> tuple[float, float]:
    lat = anchor.lat + (n_local_ft - anchor.n_local_ft) / FT_PER_DEG_LAT
    lon = anchor.lon + (e_local_ft - anchor.e_local_ft) / FT_PER_DEG_LON_AT_ANCHOR
    return lat, lon


def wgs84_to_local(
    lat: float, lon: float, anchor: LocalGridAnchor = DEFAULT_ANCHOR
) -> tuple[float, float]:
    e = anchor.e_local_ft + (lon - anchor.lon) * FT_PER_DEG_LON_AT_ANCHOR
    n = anchor.n_local_ft + (lat - anchor.lat) * FT_PER_DEG_LAT
    return e, n
