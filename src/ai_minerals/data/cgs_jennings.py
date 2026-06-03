"""CGS Fault Activity Map of California (Jennings 2010) — paged FeatureServer fetch.

California Geological Survey publishes the 2010 statewide fault-activity map
as an ArcGIS FeatureServer. The single layer carries each mapped fault as a
line geometry with an activity classification (Holocene, Late Quaternary,
Quaternary, pre-Quaternary, pre-Pleistocene). We page through the layer with
`resultOffset`/`resultRecordCount` and clip server-side to the AOI bbox via
`geometry`+`esriSpatialRelIntersects`, then materialize the result as a
per-AOI GeoPackage.

The Jennings classification is what lets v3 split distance-to-fault into two
features: pre-Quaternary faults reflect the Mesozoic lode-control structural
fabric (paystreak controls for Tertiary deep-gravel placers), while
Quaternary-active faults reflect modern drainage-tilt controls relevant to
Holocene channel placers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "cgs_jennings"

# CGS-hosted FeatureServer for the Jennings 2010 fault-activity map.
# Confirmed live 2026-06-02; license CGS open data (public).
SERVICE_BASE = (
    "https://gis.conservation.ca.gov/server/rest/services/CGS/"
    "FaultActivityMapCA/FeatureServer"
)
FAULTS_LAYER = 0
PAGE_SIZE = 2000
LANDING_URL = (
    "https://www.conservation.ca.gov/cgs/Pages/Information/"
    "fault-activity-map.aspx"
)


def _page_layer(layer_id: int, aoi: AOI) -> list[dict]:
    """Page through one FeatureServer layer, return GeoJSON features."""
    west, south, east, north = aoi.bbox
    base_params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "geometry": json.dumps(
            {
                "xmin": west,
                "ymin": south,
                "xmax": east,
                "ymax": north,
                "spatialReference": {"wkid": 4326},
            }
        ),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "f": "geojson",
        "resultRecordCount": PAGE_SIZE,
    }

    url = f"{SERVICE_BASE}/{layer_id}/query"
    offset = 0
    all_features: list[dict] = []
    while True:
        params = dict(base_params)
        params["resultOffset"] = offset
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(
                f"CGS Jennings REST error on layer {layer_id}: {payload['error']}"
            )
        feats = payload.get("features", []) or []
        all_features.extend(feats)
        if not payload.get("exceededTransferLimit") and len(feats) < PAGE_SIZE:
            break
        offset += len(feats) if feats else PAGE_SIZE
        time.sleep(0.1)
    return all_features


def _write_layer(features: list[dict], out_path: Path) -> None:
    if not features:
        gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326").to_file(
            out_path, driver="GPKG"
        )
        return
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf.to_file(out_path, driver="GPKG")


def fetch_cgs_jennings(aoi: AOI, *, force: bool = False) -> Path:
    """Page CGS Jennings 2010 fault lines within aoi to a single GeoPackage.

    Returns the path to the written GPKG.
    """
    out_dir = dataset_dir(NAME)
    fault_path = out_dir / f"cgs_jennings_{aoi.name.lower()}.gpkg"

    if not force and fault_path.exists():
        print(
            f"CGS Jennings artifact already present "
            f"({fault_path.stat().st_size:,} B); skipping fetch."
        )
    else:
        print(f"Fetching CGS Jennings 2010 fault lines for AOI={aoi.name} ...")
        features = _page_layer(FAULTS_LAYER, aoi)
        print(f"  faults: {len(features)} features")
        _write_layer(features, fault_path)
        print(f"  wrote {fault_path} ({fault_path.stat().st_size:,} bytes)")

    write_source_md(
        NAME,
        title="CGS Fault Activity Map of California (Jennings 2010)",
        url=LANDING_URL,
        license="California Geological Survey open data (public)",
        notes=(
            f"Paged FeatureServer query against {SERVICE_BASE} layer "
            f"{FAULTS_LAYER}, clipped server-side to AOI={aoi.name} bbox "
            f"via esriSpatialRelIntersects. The activity-class column "
            f"(typically `fault_activity_class` or `rcomp_clas`) carries "
            f"the Holocene / Late Quaternary / Quaternary / pre-Quaternary "
            f"/ pre-Pleistocene classification; the adapter splits this into "
            f"pre-Quaternary vs Quaternary-active subsets for the v3 "
            f"two-fault distance features."
        ),
    )
    return fault_path


# Backwards-compatible alias following the cgs_2010 fetcher's `fetch` name.
fetch = fetch_cgs_jennings


if __name__ == "__main__":
    from ai_minerals.aoi import AOI as _AOI

    NORTHERN_SIERRA = _AOI(
        name="NorthernSierra",
        min_lon=-121.55,
        min_lat=37.49,
        max_lon=-119.48,
        max_lat=40.01,
    )
    fetch_cgs_jennings(NORTHERN_SIERRA)
