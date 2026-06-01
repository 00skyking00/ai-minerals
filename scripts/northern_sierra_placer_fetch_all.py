"""Fetch all raw data for the northern-Sierra placer model.

Runs five fetchers in sequence:

  1. MRDS occurrences (lode + placer; reuses motherlode pull if present)
  2. Hydraulic Mine Pits of California (167 polygons, ScienceBase DOI 10.5066/F7J38QMD)
  3. CGS 2010 Geologic Map (surficial + structure) — FeatureServer paged query
  4. NHDPlus High Resolution flowlines (HUC4 1802 + 1804, ~4-6 GB raw)
  5. NURE western-US ICP-MS reanalysis (ScienceBase DOI 10.5066/F7765DHF)
  6. 3DEP elevation (1 m via OpenTopography if OPENTOPOGRAPHY_API_KEY is
     set; 10 m fallback from USGS National Map otherwise)

Why serial: NHDPlus HR is the heaviest (multi-GB; ~4-6 GB peak resident)
and 3DEP via OpenTopography has a per-day quota — parallelising both
risks blowing through limits. Serial keeps the bandwidth predictable.

Usage:
    .venv/bin/python scripts/northern_sierra_placer_fetch_all.py
    .venv/bin/python scripts/northern_sierra_placer_fetch_all.py --only nhdplus_hr
    .venv/bin/python scripts/northern_sierra_placer_fetch_all.py --skip threedep,nhdplus_hr
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections.abc import Callable

from ai_minerals.aoi import AOI, NORTHERN_SIERRA
from ai_minerals.data import (
    cgs_2010,
    hydraulic_pits,
    mrds,
    nhdplus_hr,
    nure_iicpms,
    threedep,
)


SourceFetcher = Callable[[AOI], object]


SOURCES: dict[str, SourceFetcher] = {
    "mrds":           mrds.fetch,
    "hydraulic_pits": hydraulic_pits.fetch,
    "cgs_2010":       cgs_2010.fetch,
    "nhdplus_hr":     nhdplus_hr.fetch,
    "nure_iicpms":    nure_iicpms.fetch,
    "threedep":       threedep.fetch,
}


def _parse_csv_list(s: str | None) -> set[str]:
    if not s:
        return set()
    return {t.strip() for t in s.split(",") if t.strip()}


def _run_one(name: str, fetcher: SourceFetcher, aoi: AOI) -> tuple[str, str, float]:
    start = time.monotonic()
    try:
        result = fetcher(aoi)
    except Exception:
        elapsed = time.monotonic() - start
        return name, f"FAIL ({elapsed:.1f}s): {traceback.format_exc().splitlines()[-1]}", elapsed
    elapsed = time.monotonic() - start
    return name, f"OK ({elapsed:.1f}s) -> {result}", elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated source names to run (default: all). "
             f"Choices: {sorted(SOURCES)}",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated source names to skip.",
    )
    args = parser.parse_args(argv)

    only = _parse_csv_list(args.only)
    skip = _parse_csv_list(args.skip)
    unknown = (only | skip) - set(SOURCES)
    if unknown:
        print(f"ERROR: unknown source(s): {sorted(unknown)}. Known: {sorted(SOURCES)}",
              file=sys.stderr)
        return 2

    selected = [
        (name, fn) for name, fn in SOURCES.items()
        if (not only or name in only) and name not in skip
    ]
    if not selected:
        print("Nothing to fetch (after --only / --skip filtering).", file=sys.stderr)
        return 2

    print(f"Fetching {len(selected)} source(s) for AOI={NORTHERN_SIERRA.name} "
          f"bbox={NORTHERN_SIERRA.bbox}")
    print()

    results: list[tuple[str, str, float]] = []
    for name, fn in selected:
        print(f"==> {name}")
        result = _run_one(name, fn, NORTHERN_SIERRA)
        print(f"    {result[1]}")
        results.append(result)

    failures = [r for r in results if r[1].startswith("FAIL")]
    print()
    print(f"Summary: {len(results) - len(failures)} OK / {len(failures)} failed "
          f"/ {sum(r[2] for r in results):.1f}s total")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
