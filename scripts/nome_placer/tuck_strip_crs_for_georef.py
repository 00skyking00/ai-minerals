"""Strip the embedded CRS from Tuck overlay TIFFs for QGIS Georeferencer use.

The v1.5 v1 TIFFs in overlays_v1p5/ carry the rough-affine EPSG:4326 transform
for goldbug consumption. QGIS Georeferencer expects an unreferenced raster
(pixel-space only): when it sees an embedded CRS it displays the raster at
its embedded coordinates while reading .points sourceX/sourceY as pixel
coordinates, so the GCP crosses appear in white space far from the raster.

This script writes a clean unreferenced copy of each sheet to a sibling
directory so the v1.5 v1 deliverable to goldbug stays intact.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.transform import Affine


# Use a y-flipped identity transform so QGIS renders row 0 at the top
# (Affine.identity() has positive e, which puts row 0 at world y=0;
# QGIS displays with y-axis up, so that flips the image vertically and
# the text reads backwards).
def _yflip_identity(height: int) -> Affine:
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(height))


SHEET_NAMES = [
    "tuck1942_map_b1",
    "tuck1942_nome_district_contours",
    "tuck1942_nome_district_general",
    "tuck1942_map_a",
    "tuck1942_map_b",
    "tuck1942_map_c",
    "tuck1942_map_d",
    "tuck1942_map_e",
]


def strip_one(in_path: Path, out_path: Path) -> None:
    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        profile.update(crs=None, transform=_yflip_identity(src.height))
        arr = src.read()
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)
    print(f"  {in_path.name} -> {out_path}  ({out_path.stat().st_size:,} B)")


def main() -> None:
    in_dir = Path("data/derived/tuck1942/overlays_v1p5")
    out_dir = Path("data/derived/tuck1942/georef_source")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in SHEET_NAMES:
        in_path = in_dir / f"{name}.tif"
        out_path = out_dir / f"{name}.tif"
        if not in_path.exists():
            print(f"  SKIP (missing): {in_path}")
            continue
        strip_one(in_path, out_path)

    print(f"\nUnreferenced TIFFs ready at {out_dir}")


if __name__ == "__main__":
    main()
