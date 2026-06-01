"""CGS Geologic Map of California (2010, 1:750,000) — paged FeatureServer fetch.

California Geological Survey publishes the 2010 statewide geologic map as an
ArcGIS REST service. Layer 0 is the geology polygons (PTYPE = rock-type code),
layer 1 is the structural lines (faults, contacts, folds). We page through both
with `resultOffset`/`resultRecordCount` and clip server-side to the AOI bbox via
`geometry`+`esriSpatialRelIntersects`, then materialize each layer to a
per-AOI GeoPackage.

Endpoint discovery: the CGS GIS Service catalog lists the layer under
`CGS/CGS_Geologic_Map_of_California`. If the canonical hostname changes, swap
the constant at the top — the request shape stays the same.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "cgs_2010"

# The CGS-hosted FeatureServer (`gis.conservation.ca.gov/server/...`) was
# decommissioned. The 2010 map's "Generalized Rock Types" layer is now
# published by CGS via the CNRA Open Data Hub at the AGOL org below.
# Confirmed live 2026-05-31; license CC-BY (CGS attribution).
SERVICE_BASE = (
    "https://services2.arcgis.com/zr3KAIbsRSUyARHG/arcgis/rest/services/"
    "GMC_Geology/FeatureServer"
)
GEOLOGY_LAYER = 0
# The CGS 2010 FeatureServer publishes only the rock-types polygon layer.
# Faults shipped as a separate publication (CGS Quaternary Fault and Fold
# Database / USGS Qfaults) and are not available on this service. Set to
# None so fetch() skips the second layer cleanly and the placer Region
# points geology_arcs at the SGMC structure layer instead.
FAULTS_LAYER: int | None = None
PAGE_SIZE = 2000
LANDING_URL = "https://www.conservation.ca.gov/cgs/maps-data/gmc"


def _page_layer(layer_id: int, aoi: AOI) -> list[dict]:
    """Page through one FeatureServer/MapServer layer, return GeoJSON features."""
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
            raise RuntimeError(f"CGS REST error on layer {layer_id}: {payload['error']}")
        feats = payload.get("features", []) or []
        all_features.extend(feats)
        # `exceededTransferLimit` is the ArcGIS contract for "more rows available".
        if not payload.get("exceededTransferLimit") and len(feats) < PAGE_SIZE:
            break
        offset += len(feats) if feats else PAGE_SIZE
        # Small polite delay between pages.
        time.sleep(0.1)
    return all_features


def _write_layer(features: list[dict], out_path: Path) -> None:
    if not features:
        # Write an empty file with WGS84 so the adapter can still open it.
        gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326").to_file(out_path, driver="GPKG")
        return
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf.to_file(out_path, driver="GPKG")


def fetch(aoi: AOI, *, force: bool = False) -> Path:
    """Page CGS 2010 geology + fault layers within aoi to two GeoPackages.

    Returns the directory containing both outputs (geology + faults). The
    adapter reads each layer by filename, so a single returned Path is fine.
    """
    out_dir = dataset_dir(NAME)
    geo_path = out_dir / f"cgs_geology_{aoi.name.lower()}.gpkg"
    fault_path = out_dir / f"cgs_faults_{aoi.name.lower()}.gpkg"

    have_geo = geo_path.exists()
    have_faults = fault_path.exists() if FAULTS_LAYER is not None else True
    if not force and have_geo and have_faults:
        print(
            f"CGS 2010 geology artifact already present "
            f"({geo_path.stat().st_size:,} B); skipping fetch."
        )
    else:
        print(f"Fetching CGS 2010 geology polygons for AOI={aoi.name} ...")
        geo_features = _page_layer(GEOLOGY_LAYER, aoi)
        print(f"  geology: {len(geo_features)} features")
        _write_layer(geo_features, geo_path)
        print(f"  wrote {geo_path} ({geo_path.stat().st_size:,} bytes)")

        if FAULTS_LAYER is not None:
            print(f"Fetching CGS 2010 fault lines for AOI={aoi.name} ...")
            fault_features = _page_layer(FAULTS_LAYER, aoi)
            print(f"  faults: {len(fault_features)} features")
            _write_layer(fault_features, fault_path)
            print(f"  wrote {fault_path} ({fault_path.stat().st_size:,} bytes)")
        else:
            print("  faults layer not published by CGS 2010 FeatureServer; "
                  "use SGMC structure layer for faults instead.")

    write_source_md(
        NAME,
        title="CGS Geologic Map of California (2010), 1:750,000",
        url=LANDING_URL,
        license="California Geological Survey open data (public)",
        notes=(
            f"Paged FeatureServer/MapServer query against {SERVICE_BASE} "
            f"layer {GEOLOGY_LAYER} (geology polygons) and layer "
            f"{FAULTS_LAYER} (fault/structure lines), clipped server-side "
            f"to AOI={aoi.name} bbox via esriSpatialRelIntersects. "
            f"PTYPE/PTYPE2 carry rock-type abbreviations; Quaternary "
            f"alluvial units appear with codes starting 'Q' (Qa, Qal, Qg)."
        ),
    )
    return geo_path


if __name__ == "__main__":
    from ai_minerals.aoi import AOI as _AOI

    NORTHERN_SIERRA = _AOI(
        name="NorthernSierra",
        min_lon=-121.55,
        min_lat=37.49,
        max_lon=-119.48,
        max_lat=40.01,
    )
    fetch(NORTHERN_SIERRA)
