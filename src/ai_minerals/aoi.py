"""Area-of-interest definitions shared across fetch and modeling code."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon, box


WORKING_CRS = "EPSG:3338"
"""NAD83 / Alaska Albers Equal Area Conic — standard for Alaska-wide analysis."""

WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class AOI:
    name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    crs: str = WGS84

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in degrees (WGS84)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def polygon(self) -> Polygon:
        return box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def bbox_str(self, sep: str = ",") -> str:
        return sep.join(f"{v:.6f}" for v in self.bbox)


TANACROSS = AOI(
    name="Tanacross",
    min_lon=-143.0,
    min_lat=63.0,
    max_lon=-141.0,
    max_lat=64.0,
)
"""USGS 1:250,000-scale Tanacross quadrangle (TC), east-central Alaska.

Standard 1° × 2° Alaska quadrangle. Kept for backward-compat and as a smaller
test AOI; the v1 modeling AOI is EASTERN_ALASKA (below).
"""


EASTERN_ALASKA = AOI(
    name="EastAK",
    min_lon=-147.0,
    min_lat=62.0,
    max_lon=-141.0,
    max_lat=64.0,
)
"""Three-quadrangle AOI covering the eastern-Alaska porphyry belt.

  - Tanacross (TC):    63-64°N, 143-141°W   — Yukon-Tanana upland
  - Mt Hayes (MH):     63-64°N, 147-144°W   — Wrangellia / Delta Range
  - Nabesna (NB):      62-63°N, 144-141°W   — Wrangellia / eastern Alaska Range

Spans the Wrangellia–Yukon-Tanana tectonic boundary, the regional porphyry-belt
framing KoBold's Skolai project sits within. ~62 porphyry Cu-Mo positives across
~67,000 km² vs 15 in Tanacross alone. Chosen as v1 AOI for lower model variance
and a stronger interior-Alaska-belt narrative.
"""


NOME_PLACER = AOI(
    name="NomePlacer",
    min_lon=-165.50,
    min_lat=64.45,
    max_lon=-165.20,
    max_lat=64.62,
)
"""Nome Placer Fields AOI — Cape Nome mining district, Seward Peninsula, AK.

Bounds the lowland coastal placer benches (Bear Cub area, near 64.531°N
165.337°W) and the Anvil-area uplands further inland. Family-held claim
properties span both zones: 1 lowland near Bear Cub, 1 lowland adjacent
(non-Murray-drilled), 2 upland Anvil-area.
"""


NORTHERN_SIERRA = AOI(
    name="NorthernSierra",
    min_lon=-121.55,
    min_lat=37.49,
    max_lon=-119.48,
    max_lat=40.01,
)
"""Northern Sierra deep-gravel placer AOI — California.

Matches the existing motherlode lode-raster extent (37.49–40.01°N,
121.55–119.48°W) so the placer 250 m grid registers cell-for-cell with
the lode raster. The classic deep-gravel districts (Malakoff Diggins,
Dutch Flat, North San Juan, You Bet, Forest Hill, Iowa Hill,
Michigan Bluff) all sit between 39.0°N and 39.4°N inside this bbox.
"""


CALAVERAS_PLACER = AOI(
    name="CalaverasPlacer",
    min_lon=-120.70,
    min_lat=38.00,
    max_lon=-120.30,
    max_lat=38.40,
)
"""Calaveras placer AOI — central Sierra, California.

Covers the Mokelumne Hill / San Andreas / Murphys / Angels Camp /
Carson Hill districts. The Calaveras-area placers lean more on modern
drainage than the northern-Sierra Tertiary deep-gravel pattern, so the
two-population framing exercised in the northern-Sierra build should
expect a sparser Tertiary positive set here.

Sits entirely inside the existing motherlode lode-raster extent, which
means the motherlode geophysics grids (magnetic_motherlode.tif,
gravity_motherlode.tif) cover this AOI without re-fetching.
"""
