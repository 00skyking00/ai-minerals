"""Project bearcub's Map B1 dredge-cleanup polygons to WGS84.

bearcub delivered 670 polygons in NORMALIZED [0,1] coordinates of the
Map B1 source image (top-left origin, x = col/width, y = row/height).
We apply the same TPS warp Sky pinned in QGIS this morning
(10 GCPs in v1p5_v2_refined_gcps/tuck1942_map_b1.tif.points) to project
each polygon vertex to (lon, lat).

Output: data/derived/bearcub_nome/tuck_b1_dredge_cleanup_4326.geojson
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator


SRC_GEOJSON = Path("data/raw/bearcub_nome/tuck_b1_dredge_cleanup_pixelspace.geojson")
GCPS_FILE = Path("data/derived/tuck1942/v1p5_v2_refined_gcps/tuck1942_map_b1.tif.points")
ROTATION_SIDECAR = Path("data/derived/tuck1942/georef_source_rotated/tuck1942_map_b1_rotation.json")
OUT_GEOJSON = Path("data/derived/bearcub_nome/tuck_b1_dredge_cleanup_4326.geojson")


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


def fit_tps(
    src_xy: np.ndarray, dst_xy: np.ndarray,
) -> tuple[RBFInterpolator, RBFInterpolator]:
    """Return (lon_interp, lat_interp) RBF/TPS interpolators."""
    lon_interp = RBFInterpolator(src_xy, dst_xy[:, 0], kernel="thin_plate_spline")
    lat_interp = RBFInterpolator(src_xy, dst_xy[:, 1], kernel="thin_plate_spline")
    return lon_interp, lat_interp


def main() -> None:
    sidecar = json.loads(ROTATION_SIDECAR.read_text())
    orig_w, orig_h = sidecar["original_size"]
    rotated_w, rotated_h = sidecar["rotated_size"]
    print(f"Original size: {orig_w} x {orig_h}")
    print(f"Rotated size: {rotated_w} x {rotated_h}")

    # Build TPS in rotated pixel space (matching how the GCPs were picked).
    gcps = parse_gcps(GCPS_FILE)
    print(f"GCPs: {len(gcps)}")
    src_pts = []
    dst_pts = []
    for mapX, mapY, sX, sY in gcps:
        # .points sourceY is world Y in y-flipped frame: world Y = rotated_h - row.
        col_r = sX
        row_r = rotated_h - sY
        src_pts.append([col_r, row_r])
        dst_pts.append([mapX, mapY])
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    lon_interp, lat_interp = fit_tps(src_pts, dst_pts)

    # Read source polygons (normalized [0, 1] coords on bearcub render).
    src = json.loads(SRC_GEOJSON.read_text())
    features = src["features"]
    print(f"Polygons: {len(features)}")

    out_features = []
    n_vertices_total = 0
    for feat in features:
        geom_type = feat["geometry"]["type"]
        coords = feat["geometry"]["coordinates"]

        def transform_ring(ring: list[list[float]]) -> list[list[float]]:
            nonlocal n_vertices_total
            arr = np.asarray(ring, dtype=np.float64)  # (N, 2) normalized [0, 1]
            n_vertices_total += len(arr)
            # Step 1: scale normalized -> my original-Map-B1 pixel space at 300 DPI
            col_o = arr[:, 0] * orig_w
            row_o = arr[:, 1] * orig_h
            # Step 2: forward-rotate to my rotated TIFF frame
            # PIL.rotate(-90, expand=True): original (col, row) -> rotated (orig_h-1-row, col)
            col_r = (orig_h - 1) - row_o
            row_r = col_o
            # Step 3: TPS to (lon, lat)
            pix_xy = np.column_stack([col_r, row_r])
            lon = lon_interp(pix_xy)
            lat = lat_interp(pix_xy)
            return [[float(lo), float(la)] for lo, la in zip(lon, lat)]

        if geom_type == "Polygon":
            new_coords = [transform_ring(ring) for ring in coords]
        elif geom_type == "MultiPolygon":
            new_coords = [[transform_ring(ring) for ring in poly] for poly in coords]
        else:
            raise ValueError(f"Unsupported geom type: {geom_type}")

        new_feat = {
            "type": "Feature",
            "properties": dict(feat["properties"]),
            "geometry": {"type": geom_type, "coordinates": new_coords},
        }
        out_features.append(new_feat)

    out = {
        "type": "FeatureCollection",
        "name": "tuck_b1_dredge_cleanup_4326",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "properties": {
            "source_geojson": str(SRC_GEOJSON),
            "source_polygons": len(features),
            "source_vertices": int(n_vertices_total),
            "note": (
                "Polygons projected from bearcub's normalized-pixel "
                "tuck_b1_dredge_cleanup_pixelspace.geojson via the "
                "Map B1 v1.5 v2 10-GCP TPS warp Sky pinned in QGIS "
                "2026-06-15. Inside the Bear Cub block: tight. Toward "
                "sheet perimeter (Snake R mouth, north edge, eastern "
                "Cape Nome): TPS extrapolates with drift potentially "
                "in the hundreds of metres."
            ),
            "georef_source": str(GCPS_FILE),
            "georef_gcps": len(gcps),
        },
        "features": out_features,
    }

    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_GEOJSON.write_text(json.dumps(out))
    print(f"\nProjected vertices: {n_vertices_total:,}")
    print(f"Wrote: {OUT_GEOJSON} ({OUT_GEOJSON.stat().st_size:,} B)")

    # Quick sanity: bounding box of all projected vertices
    all_lon = []
    all_lat = []
    for feat in out_features:
        coords = feat["geometry"]["coordinates"]
        if feat["geometry"]["type"] == "Polygon":
            for ring in coords:
                for x, y in ring:
                    all_lon.append(x)
                    all_lat.append(y)
        else:
            for poly in coords:
                for ring in poly:
                    for x, y in ring:
                        all_lon.append(x)
                        all_lat.append(y)
    print(f"Projected lon range: {min(all_lon):.5f} to {max(all_lon):.5f}")
    print(f"Projected lat range: {min(all_lat):.5f} to {max(all_lat):.5f}")


if __name__ == "__main__":
    main()
