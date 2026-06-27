"""Georeference USGS MF-247 (Hummel 1962, Nome C-1 quad, 1:63,360) to EPSG:3338.

The scanned sheet has no spatial reference. We register it on its four graticule
neatline corners, read off the quad definition 64 deg 30' - 64 deg 45' N x
165 deg 00' - 165 deg 30' W. The corner pixel positions were found by fitting the
four neatline edges (each a faint, slightly tilted line) and intersecting them;
they were visually confirmed to sit on the neatline corners next to the printed
graticule labels. The 1962 sheet is NAD27, so GCPs are tagged EPSG:4267 and the
warp to EPSG:3338 (NAD83) carries the ~150 m datum shift.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.mf247_georeference
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import fitz
import numpy as np
import rasterio
from PIL import Image
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

Image.MAX_IMAGE_PIXELS = None

PDF = Path("research/nome_debate_library/Hummel_1962_MF247_NomeC1_geologic.pdf")
OUT_DIR = Path("data/derived/nome_placer/mf247")
SCAN_TIF = OUT_DIR / "mf247_scan_gcp.tif"        # cropped scan + GCPs (NAD27)
GEOREF_TIF = OUT_DIR / "mf247_nomeC1_3338.tif"   # warped to EPSG:3338 (deliverable raster)

# Neatline corner pixels in the full embedded scan (12704 x 9660), from edge-fit +
# visual confirmation against the printed graticule labels.
CORNERS_PX = {           # (col, row)  ->  (lon, lat) NAD27
    "NW": (858.3, 652.4),    # 165 deg 30' W, 64 deg 45' N
    "NE": (6724.9, 647.5),   # 165 deg 00' W, 64 deg 45' N
    "SW": (833.3, 7522.7),   # 165 deg 30' W, 64 deg 30' N
    "SE": (6769.7, 7525.7),  # 165 deg 00' W, 64 deg 30' N
}
CORNERS_LL = {
    "NW": (-165.5, 64.75), "NE": (-165.0, 64.75),
    "SW": (-165.5, 64.50), "SE": (-165.0, 64.50),
}
# crop the map region out of the full sheet (small margin outside the neatline)
CROP = (790, 600, 6810, 7560)   # col0, row0, col1, row1


def _extract_scan() -> np.ndarray:
    doc = fitz.open(PDF)
    xref = doc.load_page(0).get_images(full=True)[0][0]
    img = Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("RGB")
    return np.asarray(img)


def affine_rms_m() -> float:
    """Least-squares affine pixel->3338 residual over the four corners (meters)."""
    px = np.array([CORNERS_PX[k] for k in CORNERS_PX])
    ll = np.array([CORNERS_LL[k] for k in CORNERS_PX])
    xy = np.array([_ll_to_3338(lon, lat) for lon, lat in ll])
    A = np.column_stack([px[:, 0], px[:, 1], np.ones(len(px))])
    res = []
    for j in range(2):
        coef, *_ = np.linalg.lstsq(A, xy[:, j], rcond=None)
        res.append(xy[:, j] - A @ coef)
    res = np.array(res).T
    return float(np.sqrt((res ** 2).sum(axis=1).mean()))


def _ll_to_3338(lon: float, lat: float) -> tuple[float, float]:
    out = subprocess.run(["gdaltransform", "-s_srs", "EPSG:4267", "-t_srs", "EPSG:3338"],
                         input=f"{lon} {lat}\n", capture_output=True, text=True, check=True)
    x, y, *_ = out.stdout.split()
    return float(x), float(y)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arr = _extract_scan()
    c0, r0, c1, r1 = CROP
    crop = arr[r0:r1, c0:c1]

    gcps = [GroundControlPoint(row=CORNERS_PX[k][1] - r0, col=CORNERS_PX[k][0] - c0,
                               x=CORNERS_LL[k][0], y=CORNERS_LL[k][1], id=k)
            for k in CORNERS_PX]
    profile = {"driver": "GTiff", "height": crop.shape[0], "width": crop.shape[1],
               "count": 3, "dtype": "uint8", "compress": "deflate"}
    with rasterio.open(SCAN_TIF, "w", gcps=gcps, crs=CRS.from_epsg(4267), **profile) as dst:
        for b in range(3):
            dst.write(crop[:, :, b], b + 1)

    # warp to 3338 at 5 m (matches the IfSAR DEM); TPS fits the 4 corners exactly.
    subprocess.run(["gdalwarp", "-overwrite", "-tps", "-r", "bilinear",
                    "-t_srs", "EPSG:3338", "-tr", "5", "5", "-dstnodata", "255",
                    "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
                    str(SCAN_TIF), str(GEOREF_TIF)], check=True, capture_output=True)

    rms = affine_rms_m()
    with rasterio.open(GEOREF_TIF) as ds:
        print(f"wrote {GEOREF_TIF}  size={ds.width}x{ds.height}  res={ds.res}  bounds={ds.bounds}")
    print(f"affine 4-corner registration RMS = {rms:.1f} m "
          f"(map scale 1:63360; TPS warp fits corners exactly, RMS reflects scan keystone)")


if __name__ == "__main__":
    main()
