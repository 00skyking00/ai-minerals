"""USGS Mineral Resources Data System (MRDS) — fetch occurrences by bbox."""

from __future__ import annotations

from pathlib import Path

import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "mrds"
BBOX_API = "https://mrdata.usgs.gov/mrds/search-bbox.php"


def fetch(aoi: AOI) -> Path:
    """Fetch MRDS records within aoi; write JSON response to data/raw/mrds/."""
    west, south, east, north = aoi.bbox
    params = {
        "xmin": west,
        "ymin": south,
        "xmax": east,
        "ymax": north,
        "f": "json",
    }
    out_path = dataset_dir(NAME) / f"mrds_{aoi.name.lower()}.geojson"
    resp = requests.get(BBOX_API, params=params, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)

    write_source_md(
        NAME,
        title="USGS Mineral Resources Data System (MRDS)",
        url=f"{BBOX_API}?xmin={west}&ymin={south}&xmax={east}&ymax={north}&f=json",
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
