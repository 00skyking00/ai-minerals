"""USGS Hydraulic Mine Pits of California (Orlando 2016) — adapter.

Two public entry points:

- `load(path, aoi)` — one row per pit, geometry = polygon centroid,
  emitted as a canonical occurrence with deposit code 39b (Tertiary
  deep-gravel placer Au) so the hydraulic pits seed the placer-Tertiary
  positive set alongside MRDS placer points.
- `load_polygons(path, aoi)` — raw polygons in WGS84 with their source
  metadata (pit name, source attribution, acreage). Used by
  `features/placer_geology.py::hydraulic_pit_proximity_m` to rasterize
  a distance-to-nearest-pit feature, and by the labels pipeline if a
  caller wants polygon seeds rather than centroids.

The raw shapefile already lives in WGS84 after the fetcher writes the
GeoPackage, so reprojection here is a no-op unless the caller passes a
different file.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_occurrences


SOURCE_TAG = "HYDRAULIC_PITS"
# USGS Cox-&-Singer 39b: Tertiary placer Au (deep gravel / paleoplacer).
DEPOSIT_CODES: tuple[str, ...] = ("usgs:39b",)


def _read_wgs84(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; expected EPSG:4326 from fetcher")
    if str(gdf.crs).upper() not in {"EPSG:4326", "WGS 84"}:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def load(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Emit one canonical-occurrence row per hydraulic pit (centroid geometry)."""
    polys = _read_wgs84(path)

    # Prefer the stable pit_id written by the fetcher; fall back to the
    # row index for callers that pass the raw shapefile directly.
    if "pit_id" in polys.columns:
        record_id = polys["pit_id"].astype(str)
    else:
        record_id = polys.index.astype(str)

    pit_name = polys["Pit_Name"] if "Pit_Name" in polys.columns else None
    area_acres = polys["AreaAcres"] if "AreaAcres" in polys.columns else None
    data_source = polys["DataSource"] if "DataSource" in polys.columns else None

    # Compute centroids in California Albers (EPSG:3310) to silence the
    # geographic-CRS warning; reproject back to WGS84 for the canonical
    # occurrence schema. For 167 polygons up to a few hundred acres each
    # the difference vs a naive WGS84 centroid is sub-meter, but doing it
    # right keeps logs clean.
    centroids = polys.geometry.to_crs("EPSG:3310").centroid.to_crs("EPSG:4326")

    out = gpd.GeoDataFrame(
        {
            "geometry": centroids,
            "commodity": "au gold",
            "deposit_codes": [DEPOSIT_CODES] * len(polys),
            "year": None,
            "source": SOURCE_TAG,
            "raw_record_id": record_id,
            "pit_name": pit_name,
            "area_acres": area_acres,
            "data_source": data_source,
        },
        crs="EPSG:4326",
    )
    return validate_occurrences(out)


def load_polygons(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Return the raw pit polygons in WGS84 with their source metadata.

    Geometry stays as polygons (not centroids). Used downstream as a
    proximity-raster input and as polygonal positive seeds.
    """
    polys = _read_wgs84(path)

    if "pit_id" in polys.columns:
        record_id = polys["pit_id"].astype("int64")
    else:
        record_id = polys.index.astype("int64")

    keep_cols = {
        "geometry": polys.geometry,
        "pit_id": record_id,
        "pit_name": polys["Pit_Name"] if "Pit_Name" in polys.columns else None,
        "data_source": polys["DataSource"] if "DataSource" in polys.columns else None,
        "area_acres": polys["AreaAcres"] if "AreaAcres" in polys.columns else None,
        "source": SOURCE_TAG,
    }
    return gpd.GeoDataFrame(keep_cols, crs="EPSG:4326")
