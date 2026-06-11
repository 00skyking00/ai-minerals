"""Pre-fetch the OSM river network for the paleochannel REM, into osmnx's cache.

Why this exists: with `rem_source="osm"`, riverrem queries OpenStreetMap (via the
Overpass API) for the river centerlines it detrends against. This footprint is
large enough that osmnx splits it into ~46 Overpass sub-queries, and Overpass
rate-limits hard — it refuses connections partway through, so a single run often
caches only a partial network and the REM falls back to flow-derived.

osmnx caches each sub-query response as it arrives (`./.osm_cache`), so the fix is
simply to retry: each attempt re-fetches only the sub-queries still missing, and
the cache fills in incrementally. Run this from a machine with working network
(e.g. your own WSL terminal, not a sandboxed one) until it reports success. Once
the cache is complete, `rem_source="osm"` cache-hits everything and runs offline
and reproducibly — so commit `./.osm_cache` if you want that reproducibility.

This drives riverrem's OWN `REMMaker.get_river_centerline()` so the Overpass
query strings (and therefore the cache keys) match exactly what the REM run will
look up. Reimplementing the bbox would risk an axis-order mismatch and a cache
miss.

Usage (from the repo root, so ./.osm_cache lands where riverrem looks):
    .venv/bin/python scripts/northern_sierra_placer/fetch_osm_rivers.py
    # optional: try a mirror if overpass-api.de keeps refusing
    OVERPASS_URL=https://overpass.openstreetmap.fr/api/interpreter \
        .venv/bin/python scripts/northern_sierra_placer/fetch_osm_rivers.py

Then build the OSM-REM raster:
    REM_SOURCE=osm scripts/run_capped.sh --mem 12G --swap 0 -- \
        .venv/bin/python scripts/northern_sierra_placer/precompute_paleochannel.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER

REGION = NORTHERN_SIERRA_PLACER
# Same DEM the REM run queries against. The precompute downsamples to 30 m first,
# and the OSM bbox is the DEM's geographic extent, so the decimated copy has the
# identical bbox (and cache key) as the native file — use whichever exists.
DEM_NATIVE = REGION.raw_paths["lidar_dem"]
DEM_30M = DEM_NATIVE.with_suffix(".ds30m.tif")

MAX_ATTEMPTS = 40
BACKOFF_S = 30.0  # Overpass rate-limits; wait between attempts


def _install_shims() -> None:
    """Same gdal/osmnx-2.x shims the REM path uses, so REMMaker imports + queries."""
    import sys as _sys
    from osgeo import gdal as g, ogr as o, osr as s
    _sys.modules.setdefault("gdal", g)
    _sys.modules.setdefault("ogr", o)
    _sys.modules.setdefault("osr", s)
    import osmnx as ox
    if not hasattr(ox, "geometries_from_bbox") and hasattr(ox, "features_from_bbox"):
        def _geometries_from_bbox(*bbox, tags=None, **_ignored):
            north, south, east, west = bbox
            return ox.features_from_bbox((west, south, east, north), tags)
        ox.geometries_from_bbox = _geometries_from_bbox
    ox.settings.use_cache = True
    ox.settings.cache_folder = "./.osm_cache"
    # Be polite to Overpass and survive its rate-limiting.
    ox.settings.overpass_rate_limit = True
    ox.settings.requests_timeout = 180
    url = os.environ.get("OVERPASS_URL")
    if url:
        ox.settings.overpass_url = url
        print(f"  using Overpass endpoint: {url}")


def main() -> int:
    dem = DEM_30M if DEM_30M.exists() else DEM_NATIVE
    if not dem.exists():
        print(f"ERROR: DEM not found ({DEM_30M} or {DEM_NATIVE}). Run the precompute "
              f"once first so the 30 m DEM exists, or fetch the native DEM.", file=sys.stderr)
        return 2

    _install_shims()
    from riverrem.REMMaker import REMMaker

    cache = Path("./.osm_cache")
    print(f"==> Warming OSM river cache for {dem.name}")
    print(f"    cache dir: {cache.resolve()}")

    maker = REMMaker(dem=str(dem), out_dir=str(dem.parent / "_rem_cache"))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        n_cached = len(list(cache.glob("*.json"))) if cache.exists() else 0
        print(f"  attempt {attempt}/{MAX_ATTEMPTS}  (cached responses so far: {n_cached})", flush=True)
        try:
            maker.get_river_centerline()
        except Exception as exc:
            # ConnectionError / Overpass refusal: the sub-queries that DID succeed
            # are now cached, so the next attempt fetches only the rest.
            name = type(exc).__name__
            if "No rivers found" in str(exc):
                print(f"  ERROR: {exc}", file=sys.stderr)
                return 1
            print(f"    incomplete ({name}: {str(exc)[:120]}); retrying in {BACKOFF_S:.0f}s", flush=True)
            time.sleep(BACKOFF_S)
            continue
        n_cached = len(list(cache.glob("*.json")))
        print(f"==> Complete: river network downloaded ({len(maker.rivers)} segments, "
              f"{n_cached} cached sub-queries).")
        print("    Now build the OSM-REM raster:")
        print("      REM_SOURCE=osm scripts/run_capped.sh --mem 12G --swap 0 -- \\")
        print("        .venv/bin/python scripts/northern_sierra_placer/precompute_paleochannel.py")
        print("    Commit ./.osm_cache if you want this reproducible offline.")
        return 0

    print(f"ERROR: still incomplete after {MAX_ATTEMPTS} attempts. Overpass may be "
          f"down/blocking; try OVERPASS_URL=<mirror> or rerun later. The cache keeps "
          f"what it fetched, so progress is not lost.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
