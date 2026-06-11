"""Lindgren PP73 secondary placer diggings for the northern-Sierra blind set.

Hand-geocoded centroids for Tertiary deep-gravel and Quaternary modern-channel
diggings named in Lindgren 1911 (USGS Professional Paper 73, DOI 10.3133/pp73)
that are NOT among the 7 anchor districts in `_northern_sierra_anchors.py`.

Use case: held-out blind validation in
`scripts/northern_sierra_placer/lindgren_blind_set.py`. None of these
centroids are used to train any classifier; the placer fetch + Phase 2
training/calibration paths never see this fixture.

Centroid sources: USGS GNIS populated-place + Lindgren PP73 plates 1, 2,
and 6. Precision is ~0.005-0.01 degrees (a few hundred meters), which
is well below the 250 m grid resolution we evaluate against.

All centroids must sit inside the model AOI bbox
(`NORTHERN_SIERRA_PLACER.aoi`: 37.49-40.01 N, -121.55 to -119.48 W);
`assert_in_aoi()` at import time catches drift.
"""

from __future__ import annotations


# Tertiary deep-gravel diggings (paleochannel / hydraulic-mining sites
# named in PP73, distinct from the 7 anchor districts).
_TERTIARY: dict[str, tuple[float, float]] = {
    # (longitude, latitude), WGS84
    "Smartsville":      (-121.302, 39.214),
    "Camptonville":     (-121.045, 39.452),
    "Alleghany":        (-120.840, 39.469),
    "Damascus":         (-120.690, 39.146),
    "Lake City":        (-120.984, 39.420),
    "Sucker Flat":      (-121.298, 39.225),
    "French Corral":    (-121.180, 39.305),
    "Sweetland":        (-121.123, 39.395),
    "Cherokee":         (-121.546, 39.622),
    "Indian Hill":      (-120.880, 39.395),
    "Brandy City":      (-120.971, 39.523),
    "Snake Creek":      (-120.795, 39.220),
    "Quartz Mountain":  (-120.860, 39.330),
}


# Quaternary modern-channel diggings (active-stream placers in PP73).
# "Downieville (Quaternary)" is the Quaternary modern-channel placer
# along the North Yuba at Downieville; the Tertiary deep-gravel mines
# of Sierra County are listed above (Alleghany etc.).
_QUATERNARY: dict[str, tuple[float, float]] = {
    "Downieville (Quaternary)": (-120.825, 39.560),
    "Goodyears Bar":            (-120.890, 39.531),
    "Sierra City":              (-120.629, 39.566),
    "Washington (CA)":          (-120.802, 39.357),
    "Auburn (CA)":              (-121.077, 38.897),
}


LINDGREN_SECONDARY_DIGGINGS: dict[str, tuple[float, float]] = {
    **_TERTIARY,
    **_QUATERNARY,
}
"""All Lindgren secondary diggings. Key: place name; value: (lon, lat) WGS84."""


LINDGREN_TERTIARY_NAMES: frozenset[str] = frozenset(_TERTIARY)
LINDGREN_QUATERNARY_NAMES: frozenset[str] = frozenset(_QUATERNARY)


def assert_in_aoi(
    aoi_bbox: tuple[float, float, float, float] = (-121.55, 37.49, -119.48, 40.01),
) -> None:
    """Raise if any Lindgren centroid sits outside the AOI."""
    west, south, east, north = aoi_bbox
    out = [
        (name, lon, lat)
        for name, (lon, lat) in LINDGREN_SECONDARY_DIGGINGS.items()
        if not (west <= lon <= east and south <= lat <= north)
    ]
    if out:
        raise ValueError(
            f"Lindgren secondary centroids outside AOI bbox {aoi_bbox}: {out}"
        )


# Fail fast at import time if anyone edits the fixture into an
# inconsistent state.
assert_in_aoi()
