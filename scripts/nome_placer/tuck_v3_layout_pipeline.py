"""v1.5 v3: layout-based georeference for Maps A, B, C, D, E.

Sky's 2026-06-16 hint: stop trying to use the bearcub control points
for the per-sheet bootstrap (they all sit on Map B/B1 only). Instead,
treat the Tuck 1942 atlas as a tiled layout where each sheet covers
a fixed sub-area of the Cape Nome district:

    +---------+---------+
    |    D    |    E    |    (D over A; E over B + C)
    +---------+----+----+
    |    A    | B  | C  |    (B = "Band B1"; same bounds as B1)
    +---------+----+----+
                   ^
                   v1.5 v2 Map B1 (Sky's QGIS-pinned 10 GCPs)

B1 is the anchor (already warped via TPS). All other sheets get
4-corner GCPs based on layout offsets from B1's warped bounds.

Limitation: this assumes each Tuck plate covers its layout cell
exactly. The cartographer's choice of crop on each plate might not
match perfectly, so Bear Cub on Map B might be off from Bear Cub on
Map B1 by ~10s of metres. The 4-corner-GCP approach gives bulk
alignment; pixel-exact registration would need either QGIS clicks on
each sheet or image-matching against B1.

Nome District Contours + Nome District General are NOT covered here.
Their geographic scope wasn't in Sky's 2026-06-16 layout note. They
stay at v1.5 v2 quality (whole-district overviews) until Sky says.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import rasterio


B1_WARPED = Path("data/derived/tuck1942/v1p5_v2_refined/tuck1942_map_b1.tif")
ROTATED_SOURCE_DIR = Path("data/derived/tuck1942/georef_source_rotated")
REFINED_DIR = Path("data/derived/tuck1942/v1p5_v3_refined")
GOLDBUG_INBOX = Path(
    "/home/sky/src/learning/gldbg/handoff/inbox/2026-06-16-from-ai-minerals-tuck-v1p5-v3-layout"
)


def get_b1_bounds() -> tuple[float, float, float, float]:
    """Read B1's warped lat/lon bounds; these anchor the layout grid."""
    with rasterio.open(B1_WARPED) as src:
        return tuple(src.bounds)  # left, bottom, right, top


def get_sheet_rotated_dims(sheet_key: str) -> tuple[int, int]:
    sidecar = json.loads(
        (ROTATED_SOURCE_DIR / f"{sheet_key}_rotation.json").read_text()
    )
    return tuple(sidecar["rotated_size"])  # (W, H)


def warp_sheet_to_bounds(
    sheet_key: str,
    bounds_4326: tuple[float, float, float, float],
    out_tif: Path,
) -> None:
    """gdalwarp -tps a rotated sheet to a target lat/lon bbox via 4-corner GCPs."""
    src_tif = ROTATED_SOURCE_DIR / f"{sheet_key}.tif"
    W, H = get_sheet_rotated_dims(sheet_key)
    min_lon, min_lat, max_lon, max_lat = bounds_4326

    # 4 corner GCPs: pixel coords (col, row) -> world (lon, lat)
    # pixel (0, 0) is top-left of the rotated raster (north + west corner)
    gcps = [
        (0,    0,   min_lon, max_lat),  # top-left  (NW)
        (W,    0,   max_lon, max_lat),  # top-right (NE)
        (0,    H,   min_lon, min_lat),  # bottom-left  (SW)
        (W,    H,   max_lon, min_lat),  # bottom-right (SE)
    ]

    vrt = Path(f"/tmp/{sheet_key}_layout_gcps.vrt")
    gcp_args = []
    for col, row, lon, lat in gcps:
        gcp_args += ["-gcp", str(col), str(row), str(lon), str(lat)]

    subprocess.run(
        ["gdal_translate", "-of", "VRT", "-a_srs", "", *gcp_args,
         str(src_tif), str(vrt)],
        check=True, capture_output=True,
    )

    result = subprocess.run(
        ["gdalwarp", "-tps", "-r", "cubic",
         "-t_srs", "EPSG:4326", "-dstalpha",
         "-co", "COMPRESS=LZW", "-co", "TILED=YES",
         "-overwrite",
         str(vrt), str(out_tif)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  {sheet_key} STDERR: {result.stderr}")
        raise SystemExit(result.returncode)
    print(f"  {sheet_key}: warped to {bounds_4326} -> {out_tif.name} ({out_tif.stat().st_size:,} B)")


def main() -> None:
    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    GOLDBUG_INBOX.mkdir(parents=True, exist_ok=True)

    # Anchor: B1's warped bounds
    b1 = get_b1_bounds()
    b1_min_lon, b1_min_lat, b1_max_lon, b1_max_lat = b1
    lon_span = b1_max_lon - b1_min_lon
    lat_span = b1_max_lat - b1_min_lat
    print(f"B1 anchor: lon [{b1_min_lon:.5f}, {b1_max_lon:.5f}]  lat [{b1_min_lat:.5f}, {b1_max_lat:.5f}]")
    print(f"  sheet span: {lon_span:.5f} deg lon  x  {lat_span:.5f} deg lat")

    # Layout (each tuple is min_lon, min_lat, max_lon, max_lat):
    #     +---------+----------------+
    #     |    D    |       E        |
    #     +---------+---+----+-------+
    #     |    A    | B  |   C       |  (B = same as B1)
    #     +---------+----+-----------+
    layout = {
        "tuck1942_map_a": (b1_min_lon - lon_span, b1_min_lat,
                            b1_min_lon, b1_max_lat),
        "tuck1942_map_b": (b1_min_lon, b1_min_lat,
                            b1_max_lon, b1_max_lat),
        "tuck1942_map_c": (b1_max_lon, b1_min_lat,
                            b1_max_lon + lon_span, b1_max_lat),
        "tuck1942_map_d": (b1_min_lon - lon_span, b1_max_lat,
                            b1_min_lon, b1_max_lat + lat_span),
        "tuck1942_map_e": (b1_min_lon, b1_max_lat,
                            b1_max_lon + lon_span, b1_max_lat + lat_span),
    }
    print("\nLayout (per Sky's 2026-06-16 sheet placement note):")
    for k, b in layout.items():
        print(f"  {k:30s}  lon [{b[0]:.5f}, {b[2]:.5f}]  lat [{b[1]:.5f}, {b[3]:.5f}]")

    print(f"\nWarping {len(layout)} sheets to layout bounds ...")
    for sheet, bounds in layout.items():
        warp_sheet_to_bounds(sheet, bounds, REFINED_DIR / f"{sheet}.tif")

    # Map B1 itself: same bounds, but the canonical TIFF is the existing
    # TPS-warped v1.5 v2 file. Copy that into v1p5_v3_refined/ unchanged
    # for delivery completeness.
    shutil.copy(B1_WARPED, REFINED_DIR / "tuck1942_map_b1.tif")
    print(f"  tuck1942_map_b1.tif copied from v1.5 v2 (unchanged anchor)")

    # Deliver
    print(f"\nDelivering to {GOLDBUG_INBOX}")
    for sheet in list(layout.keys()) + ["tuck1942_map_b1"]:
        tif = REFINED_DIR / f"{sheet}.tif"
        shutil.copy(tif, GOLDBUG_INBOX / tif.name)

    manifest = {
        "schema_version": "1.0",
        "phase": "1.5 v3 (layout-based: A/B/C/D/E from Sky's 2026-06-16 hint)",
        "anchor": "tuck1942_map_b1 v1.5 v2 (Sky's QGIS-pinned 10 GCPs)",
        "layout_bounds_4326": {
            k: list(b) for k, b in layout.items()
        },
        "layout_bounds_b1": list(b1),
        "method": "4-corner GCP gdalwarp -tps -dstalpha per sheet",
        "limitation": (
            "Bulk alignment by layout cell. Pixel-exact Bear Cub overlap "
            "between B and B1 not guaranteed; cartographer's per-plate "
            "crop choice might offset Bear Cub by tens of metres. Use "
            "opacity-slide on the goldbug viewer to verify; if any "
            "specific sheet needs pixel-exact alignment, QGIS GCP-pick "
            "on that sheet."
        ),
        "out_of_scope": [
            "tuck1942_nome_district_contours",
            "tuck1942_nome_district_general",
        ],
        "out_of_scope_note": (
            "Whole-district overviews; their geographic scope wasn't in "
            "Sky's layout note. Currently still at v1.5 v2 (bulk pipeline). "
            "Sky's call to redo when their use case clarifies."
        ),
    }
    (GOLDBUG_INBOX / "tuck1942_overlay_metadata_v1p5_v3.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"  manifest written")
    print("\nDone. Sky verifies in goldbug viewer via opacity-slide.")


if __name__ == "__main__":
    main()
