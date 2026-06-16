"""Smoke tests for the bearcub Hammon-local <-> WGS84 importer.

Checks the wrapper loads bearcub's published module and the round-trip
holds. If the bearcub file moves or is removed these tests fail clearly.
"""

from __future__ import annotations

import pytest

from ai_minerals.data import bearcub_georef as bc


def test_round_trip_bear_cub_collar():
    """A known Bear Cub collar round-trips to under 0.5 ft."""
    e0, n0 = 77144.0, 22818.5
    lat, lon = bc.local_to_wgs84(e0, n0)
    e1, n1 = bc.wgs84_to_local(lat, lon)
    assert abs(e1 - e0) < 0.5
    assert abs(n1 - n0) < 0.5


def test_collar_lands_in_expected_wgs84_neighbourhood():
    """Bear Cub collar (e=77144, n=22818) sits near MS 1178 at WGS84."""
    lat, lon = bc.local_to_wgs84(77144.0, 22818.5)
    # Bear Cub MS 1178 sits at approximately lat 64.52, lon -165.33
    # The published file says envelope is lat 64.50-64.56, lon -165.36-(-165.31).
    assert 64.50 < lat < 64.56
    assert -165.36 < lon < -165.31


def test_envelope_check():
    """In-envelope returns True for Bear Cub, False for Molasses (north)."""
    # Bear Cub collar: in envelope
    assert bc.in_validated_envelope(77144.0, 22818.5)
    # 5 km north of Bear Cub: out of envelope. 1 km ~ 3281 ft so 5 km ~ 16400 ft
    out_n = 22818.5 + 16400.0
    assert not bc.in_validated_envelope(77144.0, out_n)


def test_extent_dict_has_required_keys():
    keys = bc.VALIDATED_EXTENT_LOCAL_FT.keys()
    assert {"easting_min", "easting_max", "northing_min", "northing_max"} <= set(keys)
