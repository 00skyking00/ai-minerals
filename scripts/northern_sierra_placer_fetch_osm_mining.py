"""Fetch OSM mining-tag features inside the northern-Sierra placer AOI.

Phase D.4 of the v3 placer pipeline. The four Overpass tags

    historic=mine
    man_made=mineshaft
    man_made=adit
    landuse=quarry

are sparse but high-precision; downstream we treat them as supplementary
positive labels for the Quaternary PU classifier (the OSM mining-tag
distribution concentrates on modern-channel placer workings and small
prospects, not on the buried Tertiary deep-gravel pits, which are already
covered by the Hydraulic Mine Pits of California polygons).

Data source: OpenStreetMap, accessed via osmnx + Overpass. License is
ODbL 1.0; downstream products that incorporate this layer must carry
the OpenStreetMap attribution (see model card). This script writes a
LICENSE/attribution note into the sibling SOURCE.md so the requirement
travels with the data.

The cache directory `./.osm_cache` is shared with
`scripts/northern_sierra_placer_fetch_osm_rivers.py`; entries are
content-addressed by query, so the two fetchers do not collide.

Usage (from the repo root, so `./.osm_cache` lands where osmnx looks):
    .venv/bin/python scripts/northern_sierra_placer_fetch_osm_mining.py
    # optional: try a mirror if overpass-api.de keeps refusing
    OVERPASS_URL=https://overpass.openstreetmap.fr/api/interpreter \
        .venv/bin/python scripts/northern_sierra_placer_fetch_osm_mining.py

Output:
    data/raw/osm_mining/osm_mining_<region>.gpkg
    data/raw/osm_mining/SOURCE.md
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER

REGION = NORTHERN_SIERRA_PLACER

MAX_ATTEMPTS = 20
BACKOFF_S = 30.0  # Overpass rate-limits; wait between attempts

TAGS: dict[str, list[str]] = {
    "historic": ["mine"],
    "man_made": ["mineshaft", "adit"],
    "landuse": ["quarry"],
}


def _install_osmnx() -> "object":
    """Configure osmnx to use the local cache and rate-limited Overpass."""
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = "./.osm_cache"
    ox.settings.overpass_rate_limit = True
    ox.settings.requests_timeout = 180
    url = os.environ.get("OVERPASS_URL")
    if url:
        ox.settings.overpass_url = url
        print(f"  using Overpass endpoint: {url}")
    return ox


def _source_md(out_path: Path, n_features: int) -> str:
    return f"""# OpenStreetMap mining-tag features ({REGION.slug})

- **Dataset key:** `osm_mining`
- **Source:** OpenStreetMap via Overpass API (queried through `osmnx`)
- **Retrieved:** {time.strftime('%Y-%m-%d')}
- **License:** Open Database License (ODbL) 1.0
- **Attribution:** "(c) OpenStreetMap contributors" must appear in any
  downstream product that incorporates this layer; see
  <https://www.openstreetmap.org/copyright>.

## Query

Overpass tags fetched within the placer AOI bbox
(lon {REGION.aoi.min_lon}, lat {REGION.aoi.min_lat},
lon {REGION.aoi.max_lon}, lat {REGION.aoi.max_lat}):

```
historic=mine
man_made=mineshaft
man_made=adit
landuse=quarry
```

## Notes

{n_features} features written to `{out_path.name}`. Use as supplementary
positive labels for the Quaternary PU classifier
(`scripts/northern_sierra_placer_assemble_250m.py --augment-osm-mining`);
not used as a feature. Points and polygon centroids are snapped to the
working 250 m grid by the assembler.
"""


def main() -> int:
    ox = _install_osmnx()

    out_dir = DATA_RAW / "osm_mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"osm_mining_{REGION.data_prefix}.gpkg"

    west, south, east, north = REGION.aoi.bbox
    print(f"==> Fetching OSM mining tags for {REGION.slug}")
    print(f"    bbox (W,S,E,N) = ({west}, {south}, {east}, {north})")
    print(f"    cache dir: {Path('./.osm_cache').resolve()}")
    print(f"    output:    {out_path}")

    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        n_cached = len(list(Path("./.osm_cache").glob("*.json"))) \
            if Path("./.osm_cache").exists() else 0
        print(f"  attempt {attempt}/{MAX_ATTEMPTS}  "
              f"(cached responses so far: {n_cached})", flush=True)
        try:
            # osmnx 2.x signature: features_from_bbox(bbox=(W,S,E,N), tags=...)
            gdf = ox.features_from_bbox((west, south, east, north), TAGS)
        except Exception as exc:
            last_exc = exc
            name = type(exc).__name__
            print(f"    incomplete ({name}: {str(exc)[:160]}); "
                  f"retrying in {BACKOFF_S:.0f}s", flush=True)
            time.sleep(BACKOFF_S)
            continue

        n = 0 if gdf is None else len(gdf)
        print(f"==> Got {n} features from Overpass.")
        if n == 0:
            print("    (Overpass returned an empty set; not writing a file.)",
                  file=sys.stderr)
            return 1

        # Drop columns whose dtype confuses Fiona (osmnx returns lists for
        # multi-valued tags). Keep geometry + the four query tags as scalars.
        keep = ["geometry"] + [t for t in TAGS.keys() if t in gdf.columns]
        gdf = gdf[keep].copy()
        for col in gdf.columns:
            if col == "geometry":
                continue
            gdf[col] = gdf[col].astype("string")

        gdf.to_file(out_path, driver="GPKG")
        (out_dir / "SOURCE.md").write_text(_source_md(out_path, n))
        print(f"    wrote {out_path}")
        print(f"    wrote {out_dir / 'SOURCE.md'}")
        return 0

    print(f"ERROR: still failing after {MAX_ATTEMPTS} attempts; "
          f"last error: {last_exc!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
