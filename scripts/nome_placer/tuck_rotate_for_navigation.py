"""Rotate Tuck Map B1 by 135 degrees CCW so N points up for QGIS navigation.

The source PDF has the Cape Nome geography rotated so true-north on the
compass rose points to the lower-left (~7:30 clock position). PIL.rotate
by +135 degrees CCW brings N to the top.

Output retains the y-flipped identity transform (so QGIS displays row 0
at the top) and a sidecar JSON capturing the rotation parameters so
this script's caller can convert GCP pixel coordinates between rotated
and original frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import rasterio
from PIL import Image
from rasterio.transform import Affine


# Disable PIL's pixel-count safeguard for large historical scans.
Image.MAX_IMAGE_PIXELS = None


ROTATION_DEG = -90.0  # CCW (negative = CW). Sky's compass: N at 270 deg
                       # (pointing left) in the source PDF; rotating the
                       # image 90 deg CW brings N to the top.


def rotate_one(in_path: Path, out_path: Path, sidecar_path: Path) -> None:
    img = Image.open(in_path)
    orig_w, orig_h = img.size

    rotated = img.rotate(
        ROTATION_DEG,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor="white",
    )
    new_w, new_h = rotated.size
    rotated.save(out_path, compression="tiff_lzw")

    # Re-write via rasterio to add the y-flipped identity transform so
    # QGIS displays the rotated raster with row 0 at the top.
    with rasterio.open(out_path) as src:
        profile = src.profile.copy()
        arr = src.read()
    profile.update(
        crs=None,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(new_h)),
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)

    sidecar_path.write_text(json.dumps({
        "source_tif": str(in_path),
        "rotation_deg_ccw": ROTATION_DEG,
        "original_size": [orig_w, orig_h],
        "rotated_size": [new_w, new_h],
        "note": (
            "To convert (rotated_col, rotated_row) -> (orig_col, orig_row): "
            "translate by (-new_w/2, -new_h/2), rotate by -ROTATION_DEG, "
            "translate by (+orig_w/2, +orig_h/2)."
        ),
    }, indent=2))
    print(f"  {in_path.name} -> {out_path}  ({out_path.stat().st_size:,} B)")
    print(f"    rotation params -> {sidecar_path}")


def main() -> None:
    in_path = Path("data/derived/tuck1942/georef_source/tuck1942_map_b1.tif")
    out_dir = Path("data/derived/tuck1942/georef_source_rotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / in_path.name
    sidecar_path = out_dir / (in_path.stem + "_rotation.json")
    rotate_one(in_path, out_path, sidecar_path)


if __name__ == "__main__":
    main()
