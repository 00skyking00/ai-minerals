"""Tuck 1942 Map B1 dredge-cleanup polygon vectorizer.

Map B1 is "Gold Distribution, Dredge Cleanups, and Bedrock Contours of
Map B Area" from the Metcalfe and Tuck 1942 Map Folio (Sheet 7). The
pink-shaded polygons are the historical dredge cleanup areas across
the Cape Nome / Snake River / Nome River lowland.

This adapter colour-thresholds the rendered Map B1 PNG to extract the
pink polygons as pixel-space geometry, then applies a rough-affine
transform to publish them in EPSG:4326 for downstream feature use.

**The rough-affine transform is a v1 approximation, not a survey-grade
georeference.** Map B1's coordinate grid is in Nome District local
survey feet (origin not in the standard CRS catalog). Production
georeferencing needs either published Nome survey control points
referenced to NAD27 / NAD83, or interactive georeferencing in QGIS
against modern landmarks (Snake River mouth, Nome city centre, the
Cape Nome promontory). The v1 rough-affine derived here has roughly
+/- 200-500 m positional error, which is fine for the Phase 1
auxiliary-label use (broad "this cell was dredged historically" flag)
but NOT for sub-cell-scale feature alignment. Refinement is queued as
a Phase 1.5 sub-task.

Output (when fetched):

- ``data/derived/tuck1942/map_b1_dredge_cleanups_pixel.geojson``: pixel-
  space MultiPolygon, no CRS. Authoritative raw extraction.
- ``data/derived/tuck1942/map_b1_dredge_cleanups_4326.geojson``: rough-
  georeferenced MultiPolygon in EPSG:4326, +/- 200-500 m error.
- ``data/derived/tuck1942/map_b1_metadata.json``: rendering params, GCPs
  used, area-of-coverage flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, mapping

# Source map PDF (Sky placed in /tmp/tuck/ on 2026-06-14; we cache a
# rendered PNG under data/raw/tuck1942/ for repeatability).
SOURCE_PDF_NAME = "Map B1.pdf"
DEFAULT_RENDER_DPI = 200

# Rough-affine GCPs: pixel (col, row) in the 200-DPI render mapped to
# (lon, lat) WGS84 of recognised landmarks in the Cape Nome area. These
# are approximate; the operator can refine in QGIS later. Anchors chosen
# from clearly labelled features on Map B1:
#
#   - Snake River mouth at the Bering Sea shoreline
#   - Nome city centre / port
#   - Cape Nome promontory / Submarine Beach offshore label
#   - McDonald Creek confluence with the Nome River (NW corner area)
#
# Pixel coordinates below were estimated by visual inspection of the
# B1_hires.png crops in /tmp/tuck_maps/ at 200 DPI. They are TENTATIVE
# until a future georeferencing pass with QGIS produces survey-grade
# values.
PIXEL_GCPS_AT_200DPI: list[tuple[tuple[float, float], tuple[float, float], str]] = [
    # (pixel_col, pixel_row), (lon, lat), description
    ((2700, 5700), (-165.405, 64.501), "Nome city centre / Snake River mouth"),
    ((1500, 5400), (-165.450, 64.520), "Lower Snake River / Bourbon Creek confluence"),
    ((4500, 6500), (-165.330, 64.475), "Cape Nome promontory area"),
    ((1200, 3800), (-165.460, 64.555), "McDonald Creek headwaters area"),
]

# Pink colour threshold (HSV space gives best discrimination against
# the cream-coloured paper background and black ink). Tuned visually.
PINK_HUE_RANGE_DEG = (320.0, 360.0)  # magenta-to-pink hue
PINK_SAT_MIN = 0.10                  # exclude faded paper colour
PINK_VAL_MIN = 0.60                  # exclude shadows


@dataclass(frozen=True)
class MapB1Extraction:
    """Result of one Map B1 vectorization pass."""
    n_polygons: int
    total_pixel_area: float
    rough_affine_4326: tuple[float, ...]  # (a, b, d, e, xoff, yoff) per shapely convention
    polygons_pixel: MultiPolygon
    polygons_4326: MultiPolygon


def _render_map_b1(pdf_path: Path, out_png: Path, dpi: int = DEFAULT_RENDER_DPI) -> None:
    import fitz

    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=dpi)
    pix.save(str(out_png))
    doc.close()


def _color_threshold_pink(png_path: Path) -> np.ndarray:
    """Return a boolean mask where True = pink dredge-polygon pixel."""
    from PIL import Image
    import colorsys

    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / np.where(maxc > 0, maxc, 1.0), 0)
    # Hue: compute via numpy without per-pixel python loops.
    rc = (maxc - R) / np.where(maxc - minc > 0, maxc - minc, 1.0)
    gc = (maxc - G) / np.where(maxc - minc > 0, maxc - minc, 1.0)
    bc = (maxc - B) / np.where(maxc - minc > 0, maxc - minc, 1.0)
    h = np.zeros_like(R)
    h = np.where(R == maxc, bc - gc, h)
    h = np.where(G == maxc, 2.0 + rc - bc, h)
    h = np.where(B == maxc, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0  # in [0, 1)
    h_deg = h * 360.0

    lo, hi = PINK_HUE_RANGE_DEG
    if lo < hi:
        hue_ok = (h_deg >= lo) & (h_deg <= hi)
    else:  # wrap around 0 deg
        hue_ok = (h_deg >= lo) | (h_deg <= hi)

    mask = hue_ok & (s >= PINK_SAT_MIN) & (v >= PINK_VAL_MIN)
    return mask.astype(bool)


def _mask_to_polygons(mask: np.ndarray, min_pixel_area: int = 50) -> list[Polygon]:
    """Extract Shapely polygons from a boolean mask via rasterio.features."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    # rasterio.features.shapes wants an integer raster with a transform;
    # we use identity (pixel coords) for the pixel-space extraction.
    geoms = []
    for geom_dict, value in shapes(
        mask.astype("uint8"), mask=mask, connectivity=8
    ):
        poly = shape(geom_dict)
        if poly.area >= min_pixel_area:
            geoms.append(poly)
    return geoms


def _fit_rough_affine(
    gcps: list[tuple[tuple[float, float], tuple[float, float], str]],
) -> tuple[float, float, float, float, float, float]:
    """Fit a 2-D affine `[x' = a*x + b*y + xoff;  y' = d*x + e*y + yoff]`
    that maps pixel (col, row) -> (lon, lat) using least squares on the
    GCP set. Returns (a, b, d, e, xoff, yoff) in shapely's
    affine-transform convention.
    """
    if len(gcps) < 3:
        raise ValueError(f"Need >= 3 GCPs for a 2-D affine; got {len(gcps)}")

    pix = np.array([(c, r) for (c, r), _, _ in gcps], dtype=np.float64)
    geo = np.array([(lon, lat) for _, (lon, lat), _ in gcps], dtype=np.float64)

    # Set up: for each GCP, two equations:
    #   lon = a*c + b*r + xoff
    #   lat = d*c + e*r + yoff
    A = np.zeros((2 * len(gcps), 6), dtype=np.float64)
    Y = np.zeros((2 * len(gcps),), dtype=np.float64)
    for i, ((c, r), (lon, lat), _) in enumerate(gcps):
        A[2 * i] = [c, r, 0, 0, 1, 0]
        Y[2 * i] = lon
        A[2 * i + 1] = [0, 0, c, r, 0, 1]
        Y[2 * i + 1] = lat
    params, *_ = np.linalg.lstsq(A, Y, rcond=None)
    a, b, d, e, xoff, yoff = params.tolist()
    return (a, b, d, e, xoff, yoff)


def extract(
    pdf_path: Path,
    cache_dir: Path,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    force: bool = False,
) -> MapB1Extraction:
    """Render Map B1 to PNG, threshold pink, vectorize, rough-georeference."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    png_path = cache_dir / f"map_b1_dpi{dpi}.png"
    if force or not png_path.exists():
        _render_map_b1(pdf_path, png_path, dpi=dpi)

    mask = _color_threshold_pink(png_path)
    polygons = _mask_to_polygons(mask)

    pixel_mp = MultiPolygon(polygons) if polygons else MultiPolygon()
    total_pixel_area = float(sum(p.area for p in polygons))

    affine = _fit_rough_affine(PIXEL_GCPS_AT_200DPI)
    # shapely.affinity.affine_transform takes (a, b, d, e, xoff, yoff)
    # mapping (x, y) -> (a*x + b*y + xoff, d*x + e*y + yoff)
    polys_4326 = [affine_transform(p, affine) for p in polygons]
    mp_4326 = MultiPolygon(polys_4326) if polys_4326 else MultiPolygon()

    return MapB1Extraction(
        n_polygons=len(polygons),
        total_pixel_area=total_pixel_area,
        rough_affine_4326=affine,
        polygons_pixel=pixel_mp,
        polygons_4326=mp_4326,
    )


def save_extraction(result: MapB1Extraction, out_dir: Path) -> dict[str, Path]:
    """Write pixel + 4326 GeoJSONs + metadata JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pixel_path = out_dir / "map_b1_dredge_cleanups_pixel.geojson"
    wgs84_path = out_dir / "map_b1_dredge_cleanups_4326.geojson"
    meta_path = out_dir / "map_b1_metadata.json"

    def _gjson(mp: MultiPolygon, crs: str | None) -> dict:
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": mapping(mp)}
            ],
            **({"crs": {"type": "name", "properties": {"name": crs}}} if crs else {}),
        }

    pixel_path.write_text(json.dumps(_gjson(result.polygons_pixel, None), indent=2))
    wgs84_path.write_text(json.dumps(
        _gjson(result.polygons_4326, "urn:ogc:def:crs:EPSG::4326"), indent=2,
    ))
    meta_path.write_text(json.dumps({
        "n_polygons": result.n_polygons,
        "total_pixel_area": result.total_pixel_area,
        "rough_affine_4326": list(result.rough_affine_4326),
        "rough_affine_note": (
            "v1 rough-affine derived from visually estimated GCPs in the "
            "Cape Nome area; positional error +/- 200-500 m. Refine via "
            "QGIS georeferencing against published Nome District survey "
            "control points for production use."
        ),
        "gcps_pixel_to_4326": [
            {"pixel": [c, r], "lonlat": [lon, lat], "description": desc}
            for (c, r), (lon, lat), desc in PIXEL_GCPS_AT_200DPI
        ],
    }, indent=2))

    return {"pixel": pixel_path, "wgs84": wgs84_path, "meta": meta_path}
