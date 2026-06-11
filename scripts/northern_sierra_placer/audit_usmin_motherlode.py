"""Audit USMIN placer-Au coverage across the Mother Lode before v3.7.0 retrain.

Gate-condition check from plan H2.2: "If any county has <20 positives,
decide whether to augment from MRDS placer-Au (with positional
uncertainty noted) or to ship the result as 'Quaternary signal weak in
county X.'"

Counts USMIN placer/gravel feature points per county across the
NORTHERN_SIERRA AOI (which spans the full Mother Lode extent), filtered
to feature classes likely to mark Quaternary placer activity. Reports:

- Per-county count
- Per-county density (records per 1000 km^2 of AOI footprint within the county)
- Recommendation per county against the <20 threshold

Mother Lode counties of interest (north to south):
    Butte, Yuba, Sierra, Nevada, Placer, El Dorado, Amador,
    Calaveras, Tuolumne, Mariposa

Plus the deep-gravel "outer" counties (Plumas, Sacramento) that aren't
canonical Mother Lode but sit inside the AOI extent and have USMIN
points worth counting.

Usage:
    .venv/bin/python scripts/northern_sierra_placer/audit_usmin_motherlode.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
USMIN_GPKG = REPO / "data/raw/usmin/usmin_northernsierraplacer.gpkg"

# Mother Lode + adjacent counties inside the AOI extent, ordered N -> S.
# "Canonical" denotes the strict historic Mother Lode belt; the others are
# in-AOI but outside the belt.
MOTHER_LODE_COUNTIES = [
    ("Butte",       "northern deep-gravel"),
    ("Yuba",        "northern deep-gravel"),
    ("Sierra",      "canonical"),
    ("Nevada",      "canonical"),
    ("Placer",      "canonical"),
    ("El Dorado",   "canonical"),
    ("Amador",      "canonical"),
    ("Calaveras",   "canonical"),
    ("Tuolumne",    "canonical"),
    ("Mariposa",    "canonical"),
]

# Feature classes treated as Quaternary placer indicators. Hydraulic Mine is
# Tertiary deep-gravel and is excluded.
QUATERNARY_FEATURE_TYPES = {
    "Placer Mine",
    "Gravel Pit",
    "Sand Pit",
    "Sand and Gravel Pit",
    "Gravel/Borrow Pit - Undifferentiated",
    "Diggings",
    "Tailings - Undifferentiated",
    "Mine Dump",
}

GATE_THRESHOLD = 20


def main() -> int:
    print(f"==> Loading USMIN points from {USMIN_GPKG}")
    gdf = gpd.read_file(USMIN_GPKG)
    print(f"    total records: {len(gdf):,}")

    quat_mask = gdf["FTR_TYPE"].isin(QUATERNARY_FEATURE_TYPES)
    quat = gdf[quat_mask].copy()
    excluded = gdf[~quat_mask].copy()
    print(f"    Quaternary candidates: {len(quat):,}")
    print(f"    excluded (Hydraulic Mine and similar): {len(excluded):,}")
    if len(excluded) > 0:
        print(f"      excluded breakdown:")
        for ftr, n in excluded["FTR_TYPE"].value_counts().items():
            print(f"        {ftr}: {n}")

    print()
    print(f"==> Per-county Quaternary positive count (gate threshold = {GATE_THRESHOLD})")
    print(f"{'County':14s} {'Type':22s} {'Quat count':>10s}  {'Gate':>5s}")
    print("-" * 60)
    total_in_belt = 0
    counties_below_gate = []
    for county, kind in MOTHER_LODE_COUNTIES:
        n = int((quat["COUNTY"] == county).sum())
        total_in_belt += n
        gate = "PASS" if n >= GATE_THRESHOLD else "FAIL"
        if n < GATE_THRESHOLD:
            counties_below_gate.append((county, n))
        print(f"{county:14s} {kind:22s} {n:>10d}  {gate:>5s}")
    print("-" * 60)
    print(f"{'TOTAL (Mother Lode counties)':37s} {total_in_belt:>10d}")

    # Counties in the AOI but outside the Mother Lode belt
    belt = {c for c, _ in MOTHER_LODE_COUNTIES}
    other = quat[~quat["COUNTY"].isin(belt)]
    print()
    print(f"==> Quaternary records in AOI but OUTSIDE the Mother Lode belt:")
    for county, n in other["COUNTY"].value_counts().items():
        print(f"    {county}: {n}")

    # AOI total
    total_in_aoi = len(quat)
    print()
    print(f"==> Totals")
    print(f"    in Mother Lode belt: {total_in_belt:,}")
    print(f"    elsewhere in AOI:    {total_in_aoi - total_in_belt:,}")
    print(f"    full AOI:            {total_in_aoi:,}")

    print()
    print("==> Gate finding")
    if counties_below_gate:
        print(f"    {len(counties_below_gate)} counties below the {GATE_THRESHOLD}-positive threshold:")
        for county, n in counties_below_gate:
            print(f"      {county}: {n} (would need MRDS augmentation OR 'weak signal' chapter note)")
    else:
        print(f"    All Mother Lode counties at or above the {GATE_THRESHOLD}-positive gate.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
