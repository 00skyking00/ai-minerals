"""Export Nome paleo-shoreline contours for the bearcub loopback artifact.

Produces one GeoJSON polyline file per documented stand at
``data/derived/nome_placer/beach_contours/<stand>_contour_3338.geojson``
in EPSG:3338 working CRS plus an EPSG:4326 sibling for goldbug.

bearcub asked for this artifact in
``handoff/inbox/2026-06-14-from-bearcub-phase0-close-reply.md``: the
single-claim GP recommender's negative leave-one-out R^2 is missing
a beach-line covariate, and these polylines provide it. The contract
is "GeoJSON polylines per stand" without any further schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

from ai_minerals.aoi import NOME_PLACER
from ai_minerals.data.adapters.elevation.ifsar_alaska import load
from ai_minerals.features.coastal import (
    STAND_ELEVATIONS_FT,
    paleo_shoreline_contours_at_stand,
    stand_elevation_m,
)
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION


OUT_DIR = Path("data/derived/nome_placer/beach_contours")
BEARCUB_DROP_BASENAME = "nome_beach_contours"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    da_full = load(NOME_PLACER_REGION.raw_paths["dem"], NOME_PLACER)
    dem = da_full.rio.clip_box(*NOME_PLACER.bbox).rio.reproject("EPSG:3338")

    manifest = {
        "source_dem": str(NOME_PLACER_REGION.raw_paths["dem"]),
        "aoi": NOME_PLACER.name,
        "aoi_bbox": list(NOME_PLACER.bbox),
        "working_crs": "EPSG:3338",
        "stands": {},
    }

    for stand in STAND_ELEVATIONS_FT:
        contours = paleo_shoreline_contours_at_stand(dem, stand)
        elev_m = stand_elevation_m(stand)
        if contours.empty:
            print(f"  {stand}: 0 polylines at {elev_m:+.2f} m -- skipping")
            manifest["stands"][stand] = {
                "elevation_ft": STAND_ELEVATIONS_FT[stand],
                "elevation_m": elev_m,
                "polyline_count": 0,
                "total_length_m": 0.0,
            }
            continue

        out_3338 = OUT_DIR / f"{stand}_contour_3338.geojson"
        out_4326 = OUT_DIR / f"{stand}_contour_4326.geojson"
        contours.to_file(out_3338, driver="GeoJSON")
        contours.to_crs("EPSG:4326").to_file(out_4326, driver="GeoJSON")
        total_km = float(contours.geometry.length.sum()) / 1000
        manifest["stands"][stand] = {
            "elevation_ft": STAND_ELEVATIONS_FT[stand],
            "elevation_m": elev_m,
            "polyline_count": int(len(contours)),
            "total_length_m": float(contours.geometry.length.sum()),
            "geojson_3338": str(out_3338),
            "geojson_4326": str(out_4326),
        }
        print(f"  {stand:18s}: {len(contours):3d} polylines  {total_km:6.1f} km  ({elev_m:+7.2f} m)")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest -> {manifest_path}")
    print(f"Stands exported -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
