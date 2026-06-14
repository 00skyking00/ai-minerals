"""Tuck 1942 Nome District bedrock-contour line extractor.

The "Nome District Map (Contours).pdf" from the Tuck 1942 Map Folio
(Sheet 2) shows the bedrock surface beneath the Cape Nome coastal plain
as red-dashed iso-elevation contour lines, overlaid with production
polygons, beachline gold zones, and the claim grid. The bedrock-contour
lines are what we want as a district-scale bedrock-topo input;
production polygons would replicate Map B1.

This adapter colour-thresholds the rendered map for red pixels and
vectorizes the resulting mask as Shapely geometry. The output is
**pixel-space line geometry**; converting it to a bedrock-elevation
raster requires (a) elevation labels associated with each contour
line, which on the original map sit as faint hand-written labels on
each segment, and (b) interpolation between contours via a TIN or
spline.

What v1 ships:

- Red-mask vectorization as MultiPolygon (rasterio.features.shapes
  treats the mask as a binary raster and produces polygons; the
  centerline of each polygon is the contour line).
- Rough-affine georeference using the same GCP set as Map B1 (the
  two maps cover the same Cape Nome area).
- Documented elevation-label gap: the contour elevations are NOT
  extracted here; they need OCR or manual annotation.

What v1 does NOT ship (queued for Phase 2):

- Per-contour elevation labels (needs OCR or manual annotation).
- Bedrock-elevation raster (needs labels + TIN/spline interpolation).
- Skeletonization to centerline polylines (the polygon centerlines
  are usable as-is for proximity features; centerline polylines would
  be neater for direct elevation interpolation).

Outputs:

- ``data/derived/tuck1942/bedrock_contours_pixel.geojson``
- ``data/derived/tuck1942/bedrock_contours_4326.geojson``
- ``data/derived/tuck1942/bedrock_contours_metadata.json``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, mapping

# Same rough-affine GCPs as Map B1 -- both maps cover the same area.
from ai_minerals.data.tuck1942_map_b1 import (
    DEFAULT_RENDER_DPI,
    PIXEL_GCPS_AT_200DPI,
    _color_threshold_pink,  # imported only as a reference; not used here
    _fit_rough_affine,
    _mask_to_polygons,
    _render_map_b1 as _render_pdf,
)

SOURCE_PDF_NAME = "Nome District Map (Contours).pdf"

# Red pixel HSV threshold. Tuck's contour ink reads as a saturated
# red-orange; tuned to capture the dashed segments without picking up
# the pink dredge polygons (those are hue ~340 with lower saturation).
RED_HUE_RANGES_DEG: tuple[tuple[float, float], ...] = (
    (0.0, 25.0),     # red-orange
    (340.0, 360.0),  # deep red wraparound
)
RED_SAT_MIN = 0.40
RED_VAL_MIN = 0.40
RED_MIN_PIXEL_AREA = 5  # contour segments are thin; small area floor


@dataclass(frozen=True)
class BedrockContoursExtraction:
    n_segments: int
    total_pixel_area: float
    rough_affine_4326: tuple[float, ...]
    segments_pixel: MultiPolygon
    segments_4326: MultiPolygon


def _color_threshold_red(png_path: Path) -> np.ndarray:
    """Return a boolean mask where True = red contour-ink pixel."""
    from PIL import Image

    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / np.where(maxc > 0, maxc, 1.0), 0)
    rc = (maxc - R) / np.where(maxc - minc > 0, maxc - minc, 1.0)
    gc = (maxc - G) / np.where(maxc - minc > 0, maxc - minc, 1.0)
    bc = (maxc - B) / np.where(maxc - minc > 0, maxc - minc, 1.0)
    h = np.zeros_like(R)
    h = np.where(R == maxc, bc - gc, h)
    h = np.where(G == maxc, 2.0 + rc - bc, h)
    h = np.where(B == maxc, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h_deg = h * 360.0

    hue_ok = np.zeros_like(R, dtype=bool)
    for lo, hi in RED_HUE_RANGES_DEG:
        if lo < hi:
            hue_ok |= (h_deg >= lo) & (h_deg <= hi)
        else:
            hue_ok |= (h_deg >= lo) | (h_deg <= hi)

    mask = hue_ok & (s >= RED_SAT_MIN) & (v >= RED_VAL_MIN)
    return mask.astype(bool)


def extract(
    pdf_path: Path,
    cache_dir: Path,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    force: bool = False,
) -> BedrockContoursExtraction:
    """Render the Nome District (Contours) map to PNG, threshold red,
    vectorize, rough-georeference."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    png_path = cache_dir / f"bedrock_contours_dpi{dpi}.png"
    if force or not png_path.exists():
        _render_pdf(pdf_path, png_path, dpi=dpi)

    mask = _color_threshold_red(png_path)
    polygons = _mask_to_polygons(mask, min_pixel_area=RED_MIN_PIXEL_AREA)
    pixel_mp = MultiPolygon(polygons) if polygons else MultiPolygon()
    total_pixel_area = float(sum(p.area for p in polygons))

    affine = _fit_rough_affine(PIXEL_GCPS_AT_200DPI)
    polys_4326 = [affine_transform(p, affine) for p in polygons]
    mp_4326 = MultiPolygon(polys_4326) if polys_4326 else MultiPolygon()

    return BedrockContoursExtraction(
        n_segments=len(polygons),
        total_pixel_area=total_pixel_area,
        rough_affine_4326=affine,
        segments_pixel=pixel_mp,
        segments_4326=mp_4326,
    )


def save_extraction(result: BedrockContoursExtraction, out_dir: Path) -> dict[str, Path]:
    """Write pixel + 4326 GeoJSONs + metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pixel_path = out_dir / "bedrock_contours_pixel.geojson"
    wgs84_path = out_dir / "bedrock_contours_4326.geojson"
    meta_path = out_dir / "bedrock_contours_metadata.json"

    def _gjson(mp: MultiPolygon, crs: str | None) -> dict:
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": mapping(mp)}
            ],
            **({"crs": {"type": "name", "properties": {"name": crs}}} if crs else {}),
        }

    pixel_path.write_text(json.dumps(_gjson(result.segments_pixel, None), indent=2))
    wgs84_path.write_text(json.dumps(
        _gjson(result.segments_4326, "urn:ogc:def:crs:EPSG::4326"), indent=2,
    ))
    meta_path.write_text(json.dumps({
        "n_segments": result.n_segments,
        "total_pixel_area": result.total_pixel_area,
        "rough_affine_4326": list(result.rough_affine_4326),
        "rough_affine_note": (
            "v1 rough-affine inherited from Map B1 because the two maps "
            "cover the same Cape Nome area. Positional error +/- 200-500 m."
        ),
        "elevation_label_gap": (
            "Contour elevations are NOT extracted here. Each contour line "
            "on the original map carries a hand-written elevation label "
            "that requires OCR or manual annotation. Without labels, the "
            "vectorized lines are usable as a 'this cell is near a "
            "documented bedrock contour' proximity feature but cannot "
            "form a bedrock-elevation raster. The bedrock-elevation "
            "interpolation step is queued for Phase 2."
        ),
        "use_cases_v1": [
            "Proximity feature: distance from each grid cell to nearest "
            "bedrock contour line. Useful as a 'this cell is near a "
            "documented bedrock feature' signal.",
            "Visualization overlay alongside the bearcub Bear-Cub-claim "
            "GP bedrock-topo surface; cross-check at the overlap.",
        ],
        "use_cases_v2": [
            "OCR or manually annotate elevations on each contour line.",
            "Interpolate elevations to a TIN / spline surface.",
            "Reproject to EPSG:3338 25 m raster as bedrock_topo_district.tif.",
            "Merge with bearcub's Bear-Cub-claim GP surface (use bearcub "
            "GP variance to downweight extrapolation outside the drill "
            "envelope; Tuck contours dominate at district scale).",
        ],
    }, indent=2))

    return {"pixel": pixel_path, "wgs84": wgs84_path, "meta": meta_path}
