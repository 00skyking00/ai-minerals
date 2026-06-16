"""Run gdal_translate + gdalwarp -tps directly from a QGIS .points file.

Bypasses QGIS Georeferencer when the GUI throws "transform not solvable"
errors. Outputs a north-up EPSG:4326 GeoTIFF + a summary of per-GCP
residuals so we can see what gdal computes (versus QGIS).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


import sys


def paths_for_sheet(sheet_key: str) -> dict:
    """Per-sheet input/output paths. Defaults to Map B1 when unset."""
    return {
        "input_tif": Path(f"data/derived/tuck1942/georef_source_rotated/{sheet_key}.tif"),
        "points_file": Path(f"data/derived/tuck1942/v1p5_v2_refined_gcps/{sheet_key}.tif.points"),
        "rotation_sidecar": Path(f"data/derived/tuck1942/georef_source_rotated/{sheet_key}_rotation.json"),
        "intermediate_vrt": Path(f"/tmp/{sheet_key}_with_gcps.vrt"),
        "output_tif": Path(f"data/derived/tuck1942/v1p5_v3_refined/{sheet_key}.tif"),
    }


# Default sheet is Map B1 (the original anchor). Override via:
#     uv run python -m scripts.nome_placer.tuck_run_tps_warp tuck1942_map_b
SHEET_KEY = sys.argv[1] if len(sys.argv) > 1 else "tuck1942_map_b1"
_paths = paths_for_sheet(SHEET_KEY)
INPUT_TIF = _paths["input_tif"]
POINTS_FILE = _paths["points_file"]
ROTATION_SIDECAR = _paths["rotation_sidecar"]
INTERMEDIATE_VRT = _paths["intermediate_vrt"]
OUTPUT_TIF = _paths["output_tif"]


def parse_points(path: Path) -> list[tuple[float, float, float, float]]:
    """Return list of (sourceX, sourceY, mapX, mapY) for enabled rows."""
    rows = []
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or "mapX" in ln or not ln.strip():
            continue
        parts = ln.split(",")
        mapX, mapY, sX, sY, enable = parts[:5]
        if int(float(enable)) != 1:
            continue
        rows.append((float(sX), float(sY), float(mapX), float(mapY)))
    return rows


def main() -> None:
    sidecar = json.loads(ROTATION_SIDECAR.read_text())
    rotated_w, rotated_h = sidecar["rotated_size"]
    print(f"Rotated TIFF size: {rotated_w} x {rotated_h}")

    gcps = parse_points(POINTS_FILE)
    print(f"Enabled GCPs: {len(gcps)}")
    for sX, sY, mX, mY in gcps:
        pixel_row = rotated_h - sY
        print(
            f"  pixel ({sX:.1f}, {pixel_row:.1f})  -> map ({mX:.5f}, {mY:.5f})"
        )

    # Build gdal_translate -gcp ... call. GCP source coords are PIXEL coords
    # (col, row) — convert sY (world Y in y-flipped frame) to pixel row.
    gcp_args: list[str] = []
    for sX, sY, mX, mY in gcps:
        pixel_row = rotated_h - sY
        gcp_args += ["-gcp", f"{sX}", f"{pixel_row}", f"{mX}", f"{mY}"]

    print(f"\nStep 1: gdal_translate VRT with GCPs -> {INTERMEDIATE_VRT}")
    subprocess.run(
        ["gdal_translate", "-of", "VRT", "-a_srs", "", *gcp_args,
         str(INPUT_TIF), str(INTERMEDIATE_VRT)],
        check=True,
    )

    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nStep 2: gdalwarp -tps -> {OUTPUT_TIF}")
    # -dstalpha: add an alpha band so the rotated-corner fill ships as
    # transparent instead of baked black RGB(0,0,0). Without this, the
    # tile pipeline sees a solid black rectangle over the basemap
    # (goldbug 2026-06-15 black-fill report).
    result = subprocess.run(
        ["gdalwarp", "-tps", "-r", "cubic",
         "-t_srs", "EPSG:4326",
         "-dstalpha",
         "-co", "COMPRESS=LZW", "-co", "TILED=YES",
         "-overwrite",
         str(INTERMEDIATE_VRT), str(OUTPUT_TIF)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise SystemExit(result.returncode)
    print(f"\nWarp succeeded -> {OUTPUT_TIF} ({OUTPUT_TIF.stat().st_size:,} B)")

    # Quick sanity: read the output GeoTIFF and report bounds
    import rasterio
    with rasterio.open(OUTPUT_TIF) as src:
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Size: {src.width} x {src.height}")


if __name__ == "__main__":
    main()
