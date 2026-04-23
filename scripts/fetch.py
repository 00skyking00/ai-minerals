"""Orchestrator for Day 2 data acquisition.

Runs each dataset fetcher in order of increasing download size. Pass
--only name1,name2 to run a subset.

Usage:
    uv run python scripts/fetch.py                   # all fetchers
    uv run python scripts/fetch.py --only mrds,ardf  # light ones only
    uv run python scripts/fetch.py --skip sentinel2  # everything except S2
"""

from __future__ import annotations

import argparse
import time

from ai_minerals.aoi import EASTERN_ALASKA
from ai_minerals.data import agdb4, ardf, dem, geology, geophysics, kenorland, mrds, sentinel2

AOI = EASTERN_ALASKA


def _run_mrds():
    return mrds.fetch(AOI)


def _run_ardf():
    # Pull records for all three adjacent quadrangles (TC + MH + NB),
    # then geographically clip to the AOI polygon.
    ardf.fetch()
    return ardf.load_quadrangles(["TC", "MH", "NB"], aoi=AOI)


def _run_agdb4():
    agdb4.fetch()
    return agdb4.load_bbox(AOI)


def _run_geology():
    geology.fetch()
    return geology.clip_units_to_aoi(AOI)


def _run_kenorland():
    return kenorland.fetch()


def _run_dem():
    return dem.fetch(AOI)


def _run_geophysics():
    return geophysics.fetch()


def _run_s2():
    return sentinel2.fetch(AOI)


# Ordered from smallest / fastest to largest / slowest.
FETCHERS = {
    "mrds": _run_mrds,
    "ardf": _run_ardf,
    "kenorland": _run_kenorland,
    "dem": _run_dem,
    "agdb4": _run_agdb4,
    "geology_ak": _run_geology,
    "geophysics": _run_geophysics,
    "sentinel2": _run_s2,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated subset of fetcher names")
    parser.add_argument("--skip", help="comma-separated fetcher names to skip")
    args = parser.parse_args()

    names = list(FETCHERS)
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
    if args.skip:
        skip = {n.strip() for n in args.skip.split(",")}
        names = [n for n in names if n not in skip]

    errors: list[tuple[str, Exception]] = []
    for name in names:
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        try:
            result = FETCHERS[name]()
            dt = time.perf_counter() - t0
            print(f"{name}: ok ({dt:.1f}s) -> {result}")
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"{name}: FAILED ({dt:.1f}s): {type(e).__name__}: {e}")
            errors.append((name, e))

    if errors:
        print(f"\n{len(errors)} fetcher(s) failed:")
        for name, e in errors:
            print(f"  - {name}: {type(e).__name__}: {e}")
        return 1
    print("\nAll fetchers completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
