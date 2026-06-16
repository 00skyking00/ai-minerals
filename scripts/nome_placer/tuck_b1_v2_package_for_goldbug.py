"""Package the Map B1 v1.5 v2 GeoTIFF for goldbug delivery.

Computes summary metrics from Sky's 10 QGIS GCPs (the same set used by
gdalwarp -tps to produce the warped raster), emits a sheet manifest
JSON, and copies both into the goldbug inbox.

What's in the v1.5 v2 package:
- tuck1942_map_b1.tif: the gdalwarp -tps result, EPSG:4326
- tuck1942_map_b1_v1p5_v2_manifest.json: bounds, GCP count, fit notes
- The other 7 sheets stay at v1.5 v1 quality (already in goldbug's tree)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio


WARPED_TIF = Path("data/derived/tuck1942/v1p5_v2_refined/tuck1942_map_b1.tif")
GCPS_FILE = Path("data/derived/tuck1942/v1p5_v2_refined_gcps/tuck1942_map_b1.tif.points")
ROTATION_SIDECAR = Path("data/derived/tuck1942/georef_source_rotated/tuck1942_map_b1_rotation.json")
GOLDBUG_INBOX = Path("/home/sky/src/learning/gldbg/handoff/inbox/2026-06-15-from-ai-minerals-tuck-b1-v1p5-v2")
MANIFEST_OUT = WARPED_TIF.parent / "tuck1942_map_b1_v1p5_v2_manifest.json"


def parse_gcps(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or "mapX" in ln or not ln.strip():
            continue
        parts = ln.split(",")
        if int(float(parts[4])) != 1:
            continue
        rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def main() -> None:
    if not WARPED_TIF.exists():
        raise SystemExit(f"warped TIF not found: {WARPED_TIF}")

    sidecar = json.loads(ROTATION_SIDECAR.read_text())
    gcps = parse_gcps(GCPS_FILE)

    # Per-GCP geographic spread (lon/lat extent + centroid)
    lons = np.array([g[0] for g in gcps])
    lats = np.array([g[1] for g in gcps])
    centroid_lon = float(lons.mean())
    centroid_lat = float(lats.mean())
    lon_extent_m = float((lons.max() - lons.min()) * 111_320 * np.cos(np.radians(centroid_lat)))
    lat_extent_m = float((lats.max() - lats.min()) * 111_320)

    with rasterio.open(WARPED_TIF) as src:
        bounds = src.bounds
        width, height = src.width, src.height
        crs = str(src.crs)

    # gdalinfo for the embedded TPS coefficients
    info = subprocess.run(
        ["gdalinfo", str(WARPED_TIF)], capture_output=True, text=True, check=True,
    ).stdout

    manifest = {
        "schema_version": "1.0",
        "sheet": "tuck1942_map_b1",
        "phase": "1.5 v2 (Sky's 10-GCP QGIS-pinned TPS warp via gdalwarp)",
        "tif_relative_url": "tuck1942_map_b1.tif",
        "width_px": width,
        "height_px": height,
        "crs": crs,
        "bounds_4326": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "rotation_undone_deg_ccw": -sidecar["rotation_deg_ccw"],
        "rotation_undone_note": (
            "Source PDF page was rotated; the unreferenced rotated TIFF "
            "had its rotation removed by the TPS warp through GCPs in "
            "true lat/lon coordinates."
        ),
        "georef_method": "gdalwarp -tps (Thin Plate Spline)",
        "gcps": {
            "count": len(gcps),
            "source_clicked_by": "Sky in QGIS 4.0 Georeferencer (manual)",
            "spread": {
                "lon_extent_m": round(lon_extent_m, 1),
                "lat_extent_m": round(lat_extent_m, 1),
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
            },
            "concentration_note": (
                "GCPs concentrated in the Bear Cub claim block (MS 1178 + "
                "adjacent claims). TPS fits the GCPs exactly within the "
                "block. Toward the sheet perimeter (Nome city, north edge, "
                "east coastline) extrapolation drift can reach a few hundred "
                "metres."
            ),
        },
        "validation": {
            "in_sample_rms_m": 0.0,
            "in_sample_rms_note": "TPS interpolates GCPs exactly.",
            "underlying_affine_residual_m": {
                "from_bootstrap_5_gcps": "2-38 m, median ~10 m",
                "note": "5 bootstrap GCPs (Snake R mouth, Anvil confluence, "
                        "north edge, 2 Bear Cub area) before refinement.",
            },
        },
        "georef_quality_note": (
            "v1.5 v2 supersedes v1.5 v1 for Map B1 only. Other 7 sheets "
            "(Nome District Contours/General, Map A, B, C, D, E) remain at "
            "v1.5 v1 (rough-affine, +/- 200-500 m). Revisit when needed."
        ),
        "next_step": (
            "If full-sheet survey-grade accuracy is needed for the goldbug "
            "viewer, Sky adds 3-4 perimeter GCPs (Snake R mouth, coast east "
            "of Cape Nome, north edge) and re-runs scripts/nome_placer/"
            "tuck_run_tps_warp.py."
        ),
    }

    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {MANIFEST_OUT}")

    GOLDBUG_INBOX.mkdir(parents=True, exist_ok=True)
    for src in (WARPED_TIF, MANIFEST_OUT):
        dst = GOLDBUG_INBOX / src.name
        shutil.copy(src, dst)
        print(f"  delivered: {dst} ({dst.stat().st_size:,} B)")

    print(f"\nDelivered to goldbug: {GOLDBUG_INBOX}")
    print(f"GCP count: {len(gcps)}")
    print(f"Bounds: {bounds}")


if __name__ == "__main__":
    main()
