"""Phase 5: Klamath/Trinity belt data fetches for cross-region transfer test.

Runs all the per-region clip + reproject steps for the KLAMATH region
config. Reuses the same source GDBs/CSVs that Mother Lode used (SGMC
GeoDatabase already extracted, NGDB national CSV already extracted,
USGS gravity GXFs cached, USGS magnetic GXF cached, EMAG2 global cached,
DEM tiles cached as needed, Sentinel-2 STAC fetched fresh).

The MRDS shapefile fetch combines CA (FIPS 06) + OR (FIPS 41) shapefile
bundles already present in `data/raw/mrds/`.

After this script, run motherlode_phase5_transfer.py to score Klamath
cells with the Mother Lode-trained RF and compute capture curves on
known Klamath orogenic-Au districts.
"""

from __future__ import annotations

from pathlib import Path
import time

from ai_minerals.regions.klamath import KLAMATH
from ai_minerals.data import (
    sgmc, ngdb, usgs_gravity, usgs_magnetic, dem, sentinel2,
)
from ai_minerals.data import mrds_shapefile

CA_DIR = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds-fUS06-1")
OR_DIR = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds-fUS41")


def main() -> None:
    aoi = KLAMATH.aoi
    crs = KLAMATH.working_crs
    print(f"Klamath AOI: {aoi}, CRS={crs}\n")

    print("=" * 60)
    print("MRDS shapefile (CA + OR)")
    print("=" * 60)
    t0 = time.perf_counter()
    mrds_path = mrds_shapefile.fetch(aoi, working_crs=crs, state_dirs=[CA_DIR, OR_DIR])
    print(f"-> {mrds_path}  ({time.perf_counter()-t0:.1f}s)\n")

    print("=" * 60)
    print("SGMC (CA + OR)")
    print("=" * 60)
    t0 = time.perf_counter()
    sgmc.fetch(aoi, working_crs=crs, states=["CA", "OR"])
    print(f"  ({time.perf_counter()-t0:.1f}s)\n")

    print("=" * 60)
    print("NGDB sediment (CA + OR)")
    print("=" * 60)
    t0 = time.perf_counter()
    ngdb.fetch(aoi, working_crs=crs, states=["CA", "OR"])
    print(f"  ({time.perf_counter()-t0:.1f}s)\n")

    print("=" * 60)
    print("USGS gravity (Bouguer + isostatic)")
    print("=" * 60)
    t0 = time.perf_counter()
    usgs_gravity.fetch(aoi, working_crs=crs)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n")

    print("=" * 60)
    print("USGS magnetic")
    print("=" * 60)
    t0 = time.perf_counter()
    usgs_magnetic.fetch(aoi, working_crs=crs)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n")

    print("=" * 60)
    print("Copernicus DEM")
    print("=" * 60)
    t0 = time.perf_counter()
    dem.fetch(aoi)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n")

    print("=" * 60)
    print("Sentinel-2 mosaic (long pole)")
    print("=" * 60)
    t0 = time.perf_counter()
    sentinel2.fetch(aoi, working_crs=crs)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n")


if __name__ == "__main__":
    main()
