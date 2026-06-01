"""Anchor districts for the Calaveras placer transfer test.

Eight historic placer-mining towns in Calaveras County, California.
Used by the Sierra → Calaveras coarse-transfer driver
(`scripts/northern_sierra_placer_calaveras_transfer.py`) to score
known producer districts under the Sierra-trained model and check
whether deciles transfer.

Centroids are approximate (decimal degrees, WGS84) — at 250 m grid
resolution sub-arcsecond precision isn't meaningful. The towns are
historic district centroids, not pit centroids; a reviewer wanting
tighter targeting should sample the nearest hydraulic-pit polygon
from `data/raw/hydraulic_pits/hydraulic_mine_pits_ca.gpkg` where one
exists.
"""

from __future__ import annotations


ANCHOR_DISTRICTS: dict[str, tuple[float, float]] = {
    # (longitude, latitude), WGS84
    "Mokelumne Hill":  (-120.706, 38.301),
    "San Andreas":     (-120.681, 38.196),
    "Murphys":         (-120.461, 38.138),
    "Angels Camp":     (-120.541, 38.072),
    "Jenny Lind":      (-120.864, 38.083),
    "West Point":      (-120.530, 38.391),
    "Vallecito":       (-120.473, 38.094),
    "Carson Hill":     (-120.555, 38.041),
}
