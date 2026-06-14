"""Export Tuck 1942 plan-view maps as web-optimised PNGs + GeoTIFFs for goldbug.

Per goldbug 2026-06-14 historical-overlay handoff ack:

- Ship transparent-background PNGs so the Folium HTML doesn't blow
  up with base64-embedded full-resolution rasters.
- Also ship GeoTIFFs for the record.
- Include a ``tuck1942_overlay_metadata.json`` sidecar with per-sheet
  title / source / purpose / georef-quality caveat / bounds.

Inputs: the source PDFs in ``/tmp/tuck/``. Outputs land at
``data/derived/tuck1942/overlays/`` and copied to the goldbug inbox.

Georeferencing for v1 uses the same rough-affine fitted in
``data/tuck1942_map_b1.py`` (4 visually estimated GCPs in the Cape
Nome area). Documented +/- 200-500 m positional error in the
metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ai_minerals.data.tuck1942_map_b1 import (
    DEFAULT_RENDER_DPI,
    PIXEL_GCPS_AT_200DPI,
    _fit_rough_affine,
    _render_map_b1 as _render_pdf,
)


SHEETS = {
    "tuck1942_map_b1": {
        "pdf_name": "Map B1.pdf",
        "title": "Tuck 1942 Map B1: Gold Distribution + Dredge Cleanups + Bedrock Contours",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 7",
        "purpose": "Pink-shaded polygons = historical dredge cleanup areas (proxy for economically dredgable Au)",
    },
    "tuck1942_nome_district_contours": {
        "pdf_name": "Nome District Map (Contours).pdf",
        "title": "Tuck 1942 Nome District Map: Bedrock Contours + Production Areas",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 2",
        "purpose": "Red-dashed bedrock contours under the coastal plain; production polygons; beachline gold zones",
    },
    "tuck1942_nome_district_general": {
        "pdf_name": "Nome district Map.pdf",
        "title": "Tuck 1942 Nome District Map: Sheet 1 General",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 1",
        "purpose": "Companion district plan view",
    },
    "tuck1942_map_a": {
        "pdf_name": "Map A.pdf",
        "title": "Tuck 1942 Map A: Submarine Beach + Cape Nome SE subarea",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 3",
        "purpose": "Plan view of the Cape Nome SE subarea including Submarine Beach and Penny River valley",
    },
    "tuck1942_map_b": {
        "pdf_name": "Map B.pdf",
        "title": "Tuck 1942 Map B: Snake River / Center Creek / Dry Creek subarea",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 5",
        "purpose": "Plan view of the central Cape Nome subarea; workings + claims",
    },
    "tuck1942_map_c": {
        "pdf_name": "Map C.pdf",
        "title": "Tuck 1942 Map C: Third Beach + Nome River subarea",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 8",
        "purpose": "Plan view of the Third Beach and Nome River area",
    },
    "tuck1942_map_d": {
        "pdf_name": "Map D.pdf",
        "title": "Tuck 1942 Map D: Anvil Creek + Dexter Creek subarea",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 9",
        "purpose": "Plan view of the Anvil + Dexter upland workings including the head-of-Dexter-Creek high-bench gravel (Molasses / Honey setting)",
    },
    "tuck1942_map_e": {
        "pdf_name": "Map E.pdf",
        "title": "Tuck 1942 Map E: Western Cape Nome subarea",
        "source": "Metcalfe and Tuck 1942, Map Folio Sheet 11",
        "purpose": "Plan view of the western Cape Nome area",
    },
}


WEB_MAX_DIMENSION = 2400  # px; goldbug viewer doesn't need full IfSAR-scale rasters


def _make_transparent_bg(rgb_img: Image.Image, threshold: int = 235) -> Image.Image:
    """Convert white / cream paper background to transparent.

    Pixels whose min RGB channel value exceeds ``threshold`` are treated
    as paper background and set to fully transparent. Ink (black /
    coloured / dark) survives.
    """
    arr = np.asarray(rgb_img.convert("RGB"))
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    minc = np.minimum(np.minimum(R, G), B)
    alpha = np.where(minc > threshold, 0, 255).astype(np.uint8)
    out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    out[..., :3] = arr
    out[..., 3] = alpha
    return Image.fromarray(out, mode="RGBA")


def _resize_for_web(img: Image.Image, max_dim: int = WEB_MAX_DIMENSION) -> Image.Image:
    W, H = img.size
    if max(W, H) <= max_dim:
        return img
    scale = max_dim / max(W, H)
    new_size = (int(W * scale), int(H * scale))
    return img.resize(new_size, Image.LANCZOS)


def _pixel_bounds_to_4326(W: int, H: int, affine: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Apply the rough-affine to the four image corners and return
    (min_lon, min_lat, max_lon, max_lat). The map is north-up so:
      (col=0, row=0)  -> NW corner -> (min_lon, max_lat)
      (col=W, row=0)  -> NE corner -> (max_lon, max_lat)
      (col=0, row=H)  -> SW corner -> (min_lon, min_lat)
      (col=W, row=H)  -> SE corner -> (max_lon, min_lat)
    """
    a, b, d, e, xoff, yoff = affine
    def apply(c, r):
        return (a * c + b * r + xoff, d * c + e * r + yoff)
    lons = []
    lats = []
    for c, r in [(0, 0), (W, 0), (0, H), (W, H)]:
        x, y = apply(c, r)
        lons.append(x); lats.append(y)
    return (min(lons), min(lats), max(lons), max(lats))


def _save_as_tif(rgb_img: Image.Image, out_path: Path) -> None:
    """Save the rendered PDF as a GeoTIFF (for the record; not georef'd
    here since v1 uses a rough-affine only). LZW-compressed."""
    rgb_img.save(out_path, format="TIFF", compression="tiff_lzw")


def main(out_dir: Path, tuck_dir: Path = Path("/tmp/tuck"), goldbug_inbox: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = _fit_rough_affine(PIXEL_GCPS_AT_200DPI)
    manifest = {"schema_version": "1.0", "sheets": {}}

    for key, sheet in SHEETS.items():
        pdf_path = tuck_dir / sheet["pdf_name"]
        if not pdf_path.exists():
            print(f"  {key}: source PDF not at {pdf_path}; skipping")
            continue
        png_full_path = out_dir / f"{key}_full.png"
        png_web_path = out_dir / f"{key}_web.png"
        tif_path = out_dir / f"{key}.tif"

        # Render to a temp PNG at DPI, then post-process.
        _render_pdf(pdf_path, png_full_path, dpi=DEFAULT_RENDER_DPI)
        img = Image.open(png_full_path).convert("RGB")
        W, H = img.size

        _save_as_tif(img, tif_path)

        # Transparent background + web resize
        rgba = _make_transparent_bg(img)
        web = _resize_for_web(rgba, max_dim=WEB_MAX_DIMENSION)
        web.save(png_web_path, format="PNG", optimize=True)

        bounds = _pixel_bounds_to_4326(W, H, affine)
        sheet_meta = {
            **{k: v for k, v in sheet.items() if k != "pdf_name"},
            "png_relative_url": png_web_path.name,
            "tif_relative_url": tif_path.name,
            "bounds_4326": list(bounds),
            "render_dpi": DEFAULT_RENDER_DPI,
            "georef_quality": (
                "v1 rough-affine fitted from 4 visually estimated GCPs in "
                "the Cape Nome area; +/- 200-500 m positional error. "
                "Refine via QGIS survey-grade pass in Phase 1.5."
            ),
        }
        manifest["sheets"][key] = sheet_meta
        print(f"  {key}: {png_web_path.stat().st_size:,} B web PNG, bounds={bounds}")

    manifest_path = out_dir / "tuck1942_overlay_metadata.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest -> {manifest_path}")

    if goldbug_inbox is not None and goldbug_inbox.parent.exists():
        goldbug_inbox.mkdir(parents=True, exist_ok=True)
        import shutil
        for fpath in out_dir.glob("*"):
            if fpath.name.endswith(("_full.png",)):
                continue  # don't ship the full-resolution PNG, just TIF + web PNG
            shutil.copy(fpath, goldbug_inbox / fpath.name)
        print(f"Delivered -> {goldbug_inbox}")


if __name__ == "__main__":
    main(
        out_dir=Path("data/derived/tuck1942/overlays"),
        goldbug_inbox=Path(
            "/home/sky/src/learning/gldbg/handoff/inbox/"
            "2026-06-14-from-ai-minerals-tuck1942-overlay-package"
        ),
    )
