"""Path 3 Stage A: fetch all data sources for Arizona porphyry-Cu AOI.

Mirrors `scripts/klamath_fetch_all.py`. Reuses the same source data
files (national SGMC GeoDatabase, national NGDB CSV, USGS gravity GXFs,
USGS magnetic GXF) plus the AZ MRDS shapefile bundle just downloaded.

After this runs, follow with:
  scripts/arizona_features.py     (build feature frame)
  scripts/arizona_decompose.py    (ILR-PCA + GLCM, parallel to Path 5 Stage A)
  scripts/arizona_oof_comparison.py (DevNet + RF OOF)
"""

from __future__ import annotations

import time
from pathlib import Path

from ai_minerals.regions.arizona import ARIZONA
from ai_minerals.data import sgmc, ngdb, usgs_gravity, usgs_magnetic, dem, sentinel2
from ai_minerals.data import mrds_shapefile

AZ_DIR = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds-fUS04-1")


def main() -> None:
    aoi = ARIZONA.aoi
    crs = ARIZONA.working_crs
    print(f"Arizona AOI: {aoi}, CRS={crs}\n", flush=True)

    print("=" * 60); print("MRDS shapefile (AZ)"); print("=" * 60)
    t0 = time.perf_counter()
    mrds_path = mrds_shapefile.fetch(aoi, working_crs=crs, state_dirs=[AZ_DIR])
    print(f"-> {mrds_path}  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("SGMC (AZ)"); print("=" * 60)
    t0 = time.perf_counter()
    sgmc.fetch(aoi, working_crs=crs, states=["AZ"])
    print(f"  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("NGDB sediment (AZ)"); print("=" * 60)
    t0 = time.perf_counter()
    ngdb.fetch(aoi, working_crs=crs, states=["AZ"])
    print(f"  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("USGS gravity (Bouguer + isostatic)"); print("=" * 60)
    t0 = time.perf_counter()
    usgs_gravity.fetch(aoi, working_crs=crs)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("USGS magnetic + derivatives"); print("=" * 60)
    t0 = time.perf_counter()
    usgs_magnetic.fetch(aoi, working_crs=crs)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("DEM"); print("=" * 60)
    t0 = time.perf_counter()
    dem.fetch(aoi, working_crs=crs)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("Sentinel-2 mean composite"); print("=" * 60)
    t0 = time.perf_counter()
    sentinel2.fetch(aoi, working_crs=crs)
    print(f"  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("done"); print("=" * 60)


if __name__ == "__main__":
    main()
