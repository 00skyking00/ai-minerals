"""Export a datum-correct, wider-bbox ARDF extract for fossick to re-ingest.

Coordinator handoff 2026-06-21 (re-export ARDF for fossick, WGS84 + wider bbox).

The fix: fossick's earlier `nome_ardf_all.geojson` carried the raw NAD27
(EPSG:4267) shapefile coordinates as bare geometry, which fossick's KG export
then emitted as WGS84 WKT -> a fixed ~155 m SE offset. This script reprojects
the ARDF points NAD27 -> WGS84 *before* writing, so every coordinate fossick
sees (geometry and the latitude/longitude properties) is already WGS84.

Two changes vs the prior 139-record Cape Nome cut:
  1. WGS84 geometry + lat/lon, reprojected from the source EPSG:4267 shapefile.
  2. Wider district-lode bbox lon[-166.0, -164.0] lat[64.3, 64.8], which pulls
     in the Solomon (SO) lode occurrences east of Cape Nome (e.g. Big Hurrah)
     that the old bbox silently dropped.

Output carries the full descriptive ARDF record per point (geol_desc, comments,
location, status, deposit model, etc.) so fossick's `_raw_text()` keeps the
record fully searchable. The raw NAD27 lat/lon columns are replaced by their
WGS84 values; the geometry is authoritative.

Run:  .venv/bin/python scripts/nome_placer/export_ardf_fossick_wgs84.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

# Source: full ARDF shapefile, EPSG:4267 (NAD27). See data/raw/ardf/ardf/ardf.prj.
SRC = Path("data/raw/ardf/ardf/ardf.shp")
OUT_DIR = Path("handoff/outbox/2026-06-21-ardf-nome-wgs84-wider")

# District-lode grid extent in WGS84 (coordinator-granted bbox).
LON_MIN, LON_MAX = -166.0, -164.0
LAT_MIN, LAT_MAX = 64.3, 64.8

# Descriptive fields fossick's builder + _raw_text() consume. ardf_num/mrds_num
# anchor identity; the rest flatten into the searchable raw record.
KEEP = [
    "ardf_num", "mrds_num", "site", "location",
    "quad_250", "quad_63360",
    "comm_main", "comm_other", "ore", "gangue",
    "site_type", "status", "production",
    "dep_model", "model_code", "alteration",
    "geol_desc", "work_expl", "comments",
    "expand_ref", "rept_names", "rept_date", "age", "rept_aff",
    "prod_notes", "reserves", "prime_ref",
]


def main() -> int:
    gdf = gpd.read_file(SRC)
    if gdf.crs is None or gdf.crs.to_epsg() != 4267:
        raise RuntimeError(f"Expected EPSG:4267 source, got {gdf.crs}")

    wgs = gdf.to_crs("EPSG:4326")
    x, y = wgs.geometry.x, wgs.geometry.y
    mask = (x >= LON_MIN) & (x <= LON_MAX) & (y >= LAT_MIN) & (y <= LAT_MAX)
    sub = wgs[mask].copy()

    # Reprojected coordinates as columns, matching the geometry. These overwrite
    # the source NAD27 latitude/longitude so nothing downstream reads the old datum.
    sub["longitude"] = sub.geometry.x.round(6)
    sub["latitude"] = sub.geometry.y.round(6)
    sub["datum"] = "WGS84"
    sub["source_datum"] = "NAD27 (EPSG:4267) reprojected to WGS84 (EPSG:4326)"

    cols = [c for c in KEEP if c in sub.columns] + ["latitude", "longitude", "datum", "source_datum"]
    out = sub[cols + ["geometry"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geojson_path = OUT_DIR / "nome_ardf_all_wgs84.geojson"
    csv_path = OUT_DIR / "nome_ardf_all_wgs84.csv"
    out.to_file(geojson_path, driver="GeoJSON")
    out.drop(columns="geometry").to_csv(csv_path, index=False)

    print(f"records: {len(out)}")
    print(f"  quads: {sub['quad_250'].value_counts().to_dict()}")
    print(f"  lon [{sub.longitude.min()}, {sub.longitude.max()}] "
          f"lat [{sub.latitude.min()}, {sub.latitude.max()}]")
    print(f"wrote {geojson_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
