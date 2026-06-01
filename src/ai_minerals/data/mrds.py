"""USGS Mineral Resources Data System (MRDS) — fetch occurrences by bbox."""

from __future__ import annotations

from pathlib import Path

import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "mrds"
# /mrds/search-bbox.php was chronically dead (Read timeout). The /services/mrds
# endpoint accepts the same AOI as a CSV-style geometry string and returns
# GeoJSON in ~1 s. Documented at https://mrdata.usgs.gov/services/help/.
SERVICES_API = "https://mrdata.usgs.gov/services/mrds"
BBOX_API = "https://mrdata.usgs.gov/mrds/search-bbox.php"  # legacy fallback


def fetch(aoi: AOI) -> Path:
    """Fetch MRDS records within aoi; write GeoJSON response to data/raw/mrds/.

    Uses /services/mrds (active as of 2026-05). Falls back to the legacy
    /mrds/search-bbox.php on non-200 responses.
    """
    west, south, east, north = aoi.bbox
    geom = f"{west},{south},{east},{north}"
    out_path = dataset_dir(NAME) / f"mrds_{aoi.name.lower()}.geojson"

    primary_url = f"{SERVICES_API}?geometry={geom}&f=geojson"
    resp = requests.get(primary_url, timeout=120)
    if resp.status_code != 200:
        # Fall back to the legacy endpoint
        legacy = requests.get(
            BBOX_API,
            params={"xmin": west, "ymin": south, "xmax": east, "ymax": north, "f": "json"},
            timeout=60,
        )
        legacy.raise_for_status()
        resp = legacy
        used_url = (f"{BBOX_API}?xmin={west}&ymin={south}&xmax={east}&"
                    f"ymax={north}&f=json")
    else:
        used_url = primary_url
    out_path.write_bytes(resp.content)

    write_source_md(
        NAME,
        title="USGS Mineral Resources Data System (MRDS)",
        url=used_url,
        license="US public domain (USGS)",
        notes=(
            f"bbox query for AOI={aoi.name}: west={west}, south={south}, "
            f"east={east}, north={north}. Output is a GeoJSON FeatureCollection."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import EASTERN_ALASKA

    path = fetch(EASTERN_ALASKA)
    size = path.stat().st_size
    print(f"Wrote {path} ({size:,} bytes)")
