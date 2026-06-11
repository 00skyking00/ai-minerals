"""Bear Lodge WY (carbonatite REE) — fetch all data sources.

Mirrors `scripts/arizona/fetch_all.py`. Bear Lodge sits in NE Wyoming
with the eastern edge of the AOI crossing into South Dakota, so we
pull MRDS shapefile bundles for both states (FIPS 56 = WY, FIPS 46 = SD).

The remaining data sources (SGMC geology, NGDB stream sediments, USGS
gravity + magnetic, DEM) are national grids that we clip to the AOI.

Sentinel-2 is skipped per the Microsoft Planetary Computer outage
documented during the Arizona / S2 work; can be added once the
Element-84 STAC rewrite lands.

After this runs, follow with:
  scripts/cross_region/bear_lodge_features.py
  scripts/cross_region/bear_lodge_decompose_features.py
  scripts/cross_region/bear_lodge_devnet.py
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from ai_minerals.regions.bear_lodge import BEAR_LODGE
from ai_minerals.data import sgmc, ngdb, usgs_gravity, usgs_magnetic, dem
from ai_minerals.data import mrds_shapefile

MRDS_ROOT = Path("/home/sky/src/learning/ai-minerals/data/raw/mrds")
# FIPS state codes used by mrdata.usgs.gov/mrds/output/mrds-fUS<NN>.zip
STATES = {
    "WY": "56",
    "SD": "46",
}


def _ensure_mrds_state_bundle(fips: str, state_name: str) -> Path:
    """Download + unzip the MRDS shapefile bundle for one state.

    Returns the path to the unzipped directory. Mirrors the layout
    Sky used for Arizona (mrds-fUS04-1).
    """
    zip_name = f"mrds-fUS{fips}.zip"
    zip_path = MRDS_ROOT / zip_name
    # The published bundles unzip to mrds-fUS<NN>-1/.
    unzip_dir = MRDS_ROOT / f"mrds-fUS{fips}-1"

    if unzip_dir.exists() and any(unzip_dir.glob("*.shp")):
        print(f"  MRDS {state_name} already extracted at {unzip_dir}", flush=True)
        return unzip_dir

    if not zip_path.exists():
        url = f"https://mrdata.usgs.gov/mrds/output/{zip_name}"
        print(f"  downloading {url} ...", flush=True)
        t0 = time.perf_counter()
        urlretrieve(url, zip_path)
        print(f"    -> {zip_path}  ({zip_path.stat().st_size:,} B, "
              f"{time.perf_counter()-t0:.1f}s)", flush=True)

    print(f"  extracting {zip_path.name} -> {unzip_dir} ...", flush=True)
    unzip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unzip_dir)
    return unzip_dir


def main() -> None:
    aoi = BEAR_LODGE.aoi
    crs = BEAR_LODGE.working_crs
    print(f"Bear Lodge AOI: {aoi}, CRS={crs}\n", flush=True)

    print("=" * 60); print("MRDS shapefile bundles (WY + SD)"); print("=" * 60)
    state_dirs: list[Path] = []
    for state_name, fips in STATES.items():
        state_dirs.append(_ensure_mrds_state_bundle(fips, state_name))

    t0 = time.perf_counter()
    mrds_path = mrds_shapefile.fetch(aoi, working_crs=crs, state_dirs=state_dirs)
    print(f"-> {mrds_path}  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("SGMC (WY + SD)"); print("=" * 60)
    t0 = time.perf_counter()
    sgmc.fetch(aoi, working_crs=crs, states=list(STATES.keys()))
    print(f"  ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    print("=" * 60); print("NGDB sediment (WY + SD)"); print("=" * 60)
    t0 = time.perf_counter()
    ngdb.fetch(aoi, working_crs=crs, states=list(STATES.keys()))
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

    print("=" * 60); print("done (Sentinel-2 SKIPPED, MPC outage)")
    print("=" * 60)


if __name__ == "__main__":
    main()
