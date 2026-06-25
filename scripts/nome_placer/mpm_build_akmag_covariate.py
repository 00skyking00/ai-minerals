"""Build akmag aeromagnetic covariate on the Nome target grid.

Source: USGS OFR 97-520 Alaska Composite aeromagnetic grid (akc.gxf), 1 km,
NAD27 / custom Albers (std parallels 55/65, central meridian -151, lat origin 55,
Clarke 1866). Parsed from GXF ASCII, reprojected to the EPSG:3338 target grid.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

RAW = Path("data/raw/nome_mpm")
OUT = Path("data/derived/nome_placer/covariates_mpm")
TEMPLATE = Path(
    "data/derived/nome_placer/prospectivity_v1p5/"
    "nome_placer_prospectivity_v1p5_v3p1_3338.tif"
)
GXF = RAW / "akc.gxf"

# Native CRS of the akc grid (OFR 97-520 readme): NAD27 Albers, Clarke 1866.
SRC_CRS = CRS.from_proj4(
    "+proj=aea +lat_1=55 +lat_2=65 +lat_0=55 +lon_0=-151 "
    "+x_0=0 +y_0=0 +datum=NAD27 +units=m +no_defs"
)


def parse_gxf(path: Path) -> tuple[np.ndarray, dict]:
    """Parse a GeoSoft GXF ASCII grid; return (z float32 with NaN nodata, hdr)."""
    text = path.read_text()
    # Split into labeled objects; #GRID is last and holds the numbers.
    grid_idx = text.index("#GRID")
    header_txt = text[:grid_idx]
    grid_txt = text[grid_idx + len("#GRID"):]

    def kw(name: str) -> str | None:
        m = re.search(rf"#{name}\s*\n\s*(.+)", header_txt)
        return m.group(1).strip() if m else None

    ncols = int(kw("POINTS"))
    nrows = int(kw("ROWS"))
    dx = float(kw("PTSEPARATION"))
    dy = float(kw("RWSEPARATION"))
    x0 = float(kw("XORIGIN"))  # bottom-left corner X (BCS)
    y0 = float(kw("YORIGIN"))  # bottom-left corner Y (BCS)
    sense = int(kw("SENSE") or "1")
    dummy = float(kw("DUMMY"))
    tr = kw("TRANSFORM")
    scale, offset = (1.0, 0.0)
    if tr:
        parts = tr.split()
        scale, offset = float(parts[0]), float(parts[1])

    vals = np.fromstring(grid_txt.replace("\n", " "), sep=" ", dtype=np.float64)
    assert vals.size == nrows * ncols, f"GXF size {vals.size} != {nrows}*{ncols}"
    g = vals.reshape(nrows, ncols)

    z = np.where(g == dummy, np.nan, g * scale + offset).astype(np.float32)

    # SENSE=1: first point bottom-left, rows run left->right, going upward.
    # Rasterio rasters are top-row first, so flip vertically.
    if sense == 1:
        z = z[::-1, :]
    else:
        raise ValueError(f"GXF SENSE={sense} not handled")

    hdr = dict(ncols=ncols, nrows=nrows, dx=dx, dy=dy, x0=x0, y0=y0,
               scale=scale, offset=offset, dummy=dummy)
    return z, hdr


def main() -> None:
    z, hdr = parse_gxf(GXF)
    # XORIGIN/YORIGIN = bottom-left cell *center* (GXF point grid). Convert to
    # the top-left pixel-corner origin rasterio expects.
    nrows, ncols = z.shape
    dx, dy = hdr["dx"], hdr["dy"]
    west = hdr["x0"] - dx / 2.0
    north = hdr["y0"] + (nrows - 1) * dy + dy / 2.0
    src_transform = Affine(dx, 0, west, 0, -dy, north)

    finite = np.isfinite(z)
    print(f"GXF parsed: {nrows}x{ncols}, native nT "
          f"min/mean/max = {np.nanmin(z):.2f}/{np.nanmean(z):.2f}/{np.nanmax(z):.2f}, "
          f"finite={finite.mean()*100:.1f}%")

    with rasterio.open(TEMPLATE) as t:
        dst_crs = t.crs
        dst_transform = t.transform
        dst_w, dst_h = t.width, t.height

    NODATA = np.float32(-99999.0)
    dst = np.full((dst_h, dst_w), NODATA, dtype=np.float32)
    src = np.where(finite, z, NODATA)

    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=SRC_CRS,
        src_nodata=NODATA,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=NODATA,
        resampling=Resampling.bilinear,
    )

    aoi = dst[dst != NODATA]
    nodata_frac = float((dst == NODATA).mean())
    print(f"AOI after warp: nodata_frac={nodata_frac:.4f}, "
          f"nT min/mean/max = {aoi.min():.2f}/{aoi.mean():.2f}/{aoi.max():.2f}")

    out_path = OUT / "akmag_nome_3338.tif"
    profile = dict(
        driver="GTiff", dtype="float32", count=1,
        width=dst_w, height=dst_h, crs=dst_crs, transform=dst_transform,
        nodata=NODATA, compress="deflate", predictor=2, tiled=True,
    )
    with rasterio.open(out_path, "w", **profile) as o:
        o.write(dst, 1)
        o.update_tags(
            source="USGS OFR 97-520 akc.gxf (Alaska Composite aeromagnetic, 1km)",
            native_crs="NAD27 Albers lat1=55 lat2=65 lat0=55 lon0=-151 Clarke1866",
            resample="bilinear",
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
