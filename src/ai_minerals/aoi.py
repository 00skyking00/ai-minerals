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
"""USGS 1:250,000-scale Tanacross quadrangle, east-central Alaska.

The standard 1° × 2° Alaska quadrangle convention. Verified against the official
USGS Alaska quadrangle shapefile in the notebook during Day 2 setup — refine if
the shapefile disagrees.
"""
