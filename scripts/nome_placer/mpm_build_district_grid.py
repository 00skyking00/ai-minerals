"""Build the DISTRICT-WIDE MPM grid template + all lode covariates (ADR-013).

The lode-gold MPM v0 (mpm_lode_presence_cv.py) ran on the coastal placer-core
AOI (697x846 @ 25 m), which holds only 33 of the district's ~104 lode sites and
EXCLUDES the Big Hurrah anchor (-164.25, 64.65) and the eastern hinterland. This
script rebuilds every covariate on a district-wide 100 m grid spanning the full
Nome lode district so the presence/background CV can run at the correct scope.

District grid: WGS84 bbox lon -166.0..-164.0, lat 64.3..64.8 -> EPSG:3338, 100 m.
The 4 bbox corners are transformed to 3338, the bounding box is taken, snapped to
100 m, giving width x height. The template carries the transform/shape for every
covariate (all suffixed `_district` so the placer-core rasters are not clobbered).

Covariates rebuilt here (re-using raw data already on disk, all of which spans the
district):
  1. akmag      -> akmag_district_3338.tif         (USGS OFR 97-520 akc.gxf, 1 km)
  2. geology    -> geology_unit_code_district_3338.tif + legend
                   (OFR 2009-1254 Nome quad nm_ddp + Solomon quad so_ddp, merged)
  3. faults     -> dist_to_fault_district_3338.tif  (nm_dda + so_dda ARC_CODE 4/30/60)
  4. geochem    -> geochem_ss_idw_log_district_3338.tif (AGDB4 stream-sed, 6 bands)

IFSAR terrain (DEM/slope/TPI) is fetched separately by
mpm_fetch_district_terrain.py because it needs an HTTP ImageServer call.

Run: uv run python -m scripts.nome_placer.mpm_build_district_grid
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

RAW = Path("data/raw/nome_mpm")
OUT = Path("data/derived/nome_placer/covariates_mpm")
TEMPLATE = OUT / "_district_grid_template_3338.tif"

# --- district bbox (WGS84) ------------------------------------------------
BBOX_WGS84 = dict(lon_min=-166.0, lon_max=-164.0, lat_min=64.3, lat_max=64.8)
TARGET_CRS = "EPSG:3338"
RES = 100.0

# --- geology -------------------------------------------------------------
GEOL_QUADS = [  # (polygon shp, arc shp) for each OFR 2009-1254 quad in the bbox
    (RAW / "geol/nmgeol_dd/nm_ddp.shp", RAW / "geol/nmgeol_dd/nm_dda.shp"),
    (RAW / "geol/sogeol_dd/so_ddp.shp", RAW / "geol/sogeol_dd/so_dda.shp"),
]
FAULT_CODES = {4, 30, 60}  # normal-certain, sense-uncertain, concealed (per OFR meta)
FAULT_PAD_M = 8000.0

# --- akmag ---------------------------------------------------------------
GXF = RAW / "akc.gxf"
AKMAG_SRC_CRS = CRS.from_proj4(
    "+proj=aea +lat_1=55 +lat_2=65 +lat_0=55 +lon_0=-151 "
    "+x_0=0 +y_0=0 +datum=NAD27 +units=m +no_defs"
)
AKMAG_NODATA = np.float32(-99999.0)

# --- geochem -------------------------------------------------------------
AGDB4 = RAW / "agdb4_nome.csv"
ELEMENTS = ["Au_ppm", "As_ppm", "Sb_ppm", "Bi_ppm", "Hg_ppm", "W_ppm"]
STREAM_SED = {"stream", "stream sediment", "stream/river", "sediment"}
GEOCHEM_K = 12
GEOCHEM_POWER = 2.0
GEOCHEM_NODATA = np.float32(-9999.0)


# ========================================================================
# grid template
# ========================================================================
def build_template() -> tuple[Affine, int, int, CRS]:
    tr = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    lons = [BBOX_WGS84["lon_min"], BBOX_WGS84["lon_max"],
            BBOX_WGS84["lon_min"], BBOX_WGS84["lon_max"]]
    lats = [BBOX_WGS84["lat_min"], BBOX_WGS84["lat_min"],
            BBOX_WGS84["lat_max"], BBOX_WGS84["lat_max"]]
    xs, ys = tr.transform(lons, lats)
    xs, ys = np.asarray(xs), np.asarray(ys)
    left = math.floor(xs.min() / RES) * RES
    right = math.ceil(xs.max() / RES) * RES
    bottom = math.floor(ys.min() / RES) * RES
    top = math.ceil(ys.max() / RES) * RES
    width = int(round((right - left) / RES))
    height = int(round((top - bottom) / RES))
    transform = Affine(RES, 0, left, 0, -RES, top)
    crs = CRS.from_epsg(3338)
    print(f"district grid: {width}x{height} @ {RES:.0f} m {crs}")
    print(f"  bounds L={left} B={bottom} R={right} T={top}")

    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        TEMPLATE, "w", driver="GTiff", dtype="int8", count=1,
        width=width, height=height, crs=crs, transform=transform,
        nodata=0, compress="deflate", tiled=True,
    ) as o:
        o.write(np.ones((height, width), dtype="int8"), 1)
        o.update_tags(
            description="district MPM grid template (1=on-grid)",
            wgs84_bbox=json.dumps(BBOX_WGS84),
        )
    print(f"wrote {TEMPLATE}")
    return transform, width, height, crs


# ========================================================================
# akmag
# ========================================================================
def parse_gxf(path: Path) -> tuple[np.ndarray, dict]:
    text = path.read_text()
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
    x0 = float(kw("XORIGIN"))
    y0 = float(kw("YORIGIN"))
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
    if sense == 1:
        z = z[::-1, :]
    else:
        raise ValueError(f"GXF SENSE={sense} not handled")
    hdr = dict(nrows=nrows, ncols=ncols, dx=dx, dy=dy, x0=x0, y0=y0)
    return z, hdr


def build_akmag(transform: Affine, width: int, height: int, crs: CRS) -> None:
    z, hdr = parse_gxf(GXF)
    nrows, ncols = z.shape
    dx, dy = hdr["dx"], hdr["dy"]
    west = hdr["x0"] - dx / 2.0
    north = hdr["y0"] + (nrows - 1) * dy + dy / 2.0
    src_transform = Affine(dx, 0, west, 0, -dy, north)
    finite = np.isfinite(z)

    dst = np.full((height, width), AKMAG_NODATA, dtype=np.float32)
    src = np.where(finite, z, AKMAG_NODATA)
    reproject(
        source=src, destination=dst,
        src_transform=src_transform, src_crs=AKMAG_SRC_CRS, src_nodata=AKMAG_NODATA,
        dst_transform=transform, dst_crs=crs, dst_nodata=AKMAG_NODATA,
        resampling=Resampling.bilinear,
    )
    nodata_frac = float((dst == AKMAG_NODATA).mean())
    aoi = dst[dst != AKMAG_NODATA]
    print(f"akmag: nodata_frac={nodata_frac:.4f}, "
          f"nT min/mean/max={aoi.min():.1f}/{aoi.mean():.1f}/{aoi.max():.1f}")
    path = OUT / "akmag_district_3338.tif"
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", count=1,
        width=width, height=height, crs=crs, transform=transform,
        nodata=AKMAG_NODATA, compress="deflate", predictor=2, tiled=True,
    ) as o:
        o.write(dst, 1)
        o.update_tags(source="USGS OFR 97-520 akc.gxf (Alaska Composite aeromag 1km)",
                      resample="bilinear")
    print(f"wrote {path}")


# ========================================================================
# geology + faults (merge Nome + Solomon quads)
# ========================================================================
def build_geology(transform: Affine, width: int, height: int, crs: CRS,
                  bounds: tuple[float, float, float, float]) -> None:
    left, bottom, right, top = bounds
    polys = pd.concat(
        [gpd.read_file(p).to_crs(TARGET_CRS)[["LABEL", "geometry"]]
         for p, _ in GEOL_QUADS],
        ignore_index=True,
    )
    polys = gpd.GeoDataFrame(polys, crs=TARGET_CRS)
    aoi_polys = polys.cx[left:right, bottom:top]
    labels_in_aoi = sorted(aoi_polys["LABEL"].dropna().unique().tolist())
    print(f"geology: {len(polys)} polys (Nome+Solomon), {len(aoi_polys)} in district; "
          f"{len(labels_in_aoi)} units: {labels_in_aoi}")

    code_map = {lab: i + 1 for i, lab in enumerate(labels_in_aoi)}
    NODATA_U = 0
    shapes = [
        (geom, code_map[lab])
        for geom, lab in zip(polys.geometry, polys["LABEL"])
        if lab in code_map and geom is not None and not geom.is_empty
    ]
    unit = rasterize(shapes, out_shape=(height, width), transform=transform,
                     fill=NODATA_U, dtype="int32", all_touched=False)
    gap = int((unit == NODATA_U).sum())
    gap_frac = gap / (height * width)
    print(f"geology: {gap} cells with NO coverage (gap_frac={gap_frac:.4f})")

    unit_path = OUT / "geology_unit_code_district_3338.tif"
    with rasterio.open(
        unit_path, "w", driver="GTiff", dtype="int32", count=1,
        width=width, height=height, crs=crs, transform=transform,
        nodata=NODATA_U, compress="deflate", tiled=True,
    ) as o:
        o.write(unit.astype("int32"), 1)
        o.update_tags(source="USGS OFR 2009-1254 nm_ddp+so_ddp (Nome+Solomon quads)",
                      gap_frac=f"{gap_frac:.4f}")
    legend = {str(c): lab for lab, c in code_map.items()}
    legend["0"] = "<unmapped/nodata>"
    legend_path = OUT / "geology_unit_code_district_legend.json"
    legend_path.write_text(json.dumps(legend, indent=2))
    print(f"wrote {unit_path} + {legend_path.name}")

    # --- distance to fault (merge arcs from both quads) ------------------
    arcs = pd.concat(
        [gpd.read_file(a).to_crs(TARGET_CRS)[["ARC_CODE", "geometry"]]
         for _, a in GEOL_QUADS],
        ignore_index=True,
    )
    arcs = gpd.GeoDataFrame(arcs, crs=TARGET_CRS)
    faults = arcs[arcs["ARC_CODE"].isin(FAULT_CODES)].copy()
    n_aoi = len(faults.cx[left:right, bottom:top])
    print(f"faults: {len(faults)} arcs (codes {sorted(FAULT_CODES)}), {n_aoi} in district")

    fault_shapes = [(g, 1) for g in faults.geometry if g is not None and not g.is_empty]
    px = abs(transform.a)
    pad = int(round(FAULT_PAD_M / px))
    big_transform = Affine(transform.a, transform.b, transform.c - pad * px,
                           transform.d, transform.e, transform.f + pad * px)
    big_h, big_w = height + 2 * pad, width + 2 * pad
    fault_mask = rasterize(fault_shapes, out_shape=(big_h, big_w),
                           transform=big_transform, fill=0, dtype="uint8",
                           all_touched=True)
    dist_big = distance_transform_edt(fault_mask == 0, sampling=px)
    dist = dist_big[pad:pad + height, pad:pad + width].astype("float32")
    print(f"dist_to_fault (m): min/mean/max={dist.min():.1f}/{dist.mean():.1f}/{dist.max():.1f}")

    dist_path = OUT / "dist_to_fault_district_3338.tif"
    with rasterio.open(
        dist_path, "w", driver="GTiff", dtype="float32", count=1,
        width=width, height=height, crs=crs, transform=transform,
        nodata=np.float32(-1.0), compress="deflate", predictor=2, tiled=True,
    ) as o:
        o.write(dist, 1)
        o.update_tags(source="USGS OFR 2009-1254 nm_dda+so_dda ARC_CODE in {4,30,60}",
                      method=f"distance_transform_edt, {px:.0f} m sampling, pad={FAULT_PAD_M:.0f}m")
    print(f"wrote {dist_path}")


# ========================================================================
# geochem (stream-sed log-IDW)
# ========================================================================
def clean_element(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    v = v.mask(v < 0, v.abs() / 2.0)
    v = v.where(v > 0)
    return np.log10(v)


def idw(sample_xy: np.ndarray, sample_val: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    tree = cKDTree(sample_xy)
    dist, idx = tree.query(grid_xy, k=min(GEOCHEM_K, len(sample_xy)))
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.power(np.maximum(dist, 1e-6), GEOCHEM_POWER)
    exact = dist < 1e-6
    w = np.where(exact.any(axis=1, keepdims=True), np.where(exact, 1.0, 0.0), w)
    return (w * sample_val[idx]).sum(axis=1) / w.sum(axis=1)


def build_geochem(transform: Affine, width: int, height: int, crs: CRS) -> None:
    df = pd.read_csv(AGDB4, low_memory=False)
    ss = df[df["SAMPLE_SOURCE"].astype(str).str.strip().str.lower().isin(STREAM_SED)].copy()
    print(f"geochem: stream-sed samples {len(ss)} of {len(df)}")
    tr = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    sx, sy = tr.transform(ss["LONGITUDE"].values, ss["LATITUDE"].values)
    sample_xy_all = np.column_stack([sx, sy])

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    gx, gy = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    grid_xy = np.column_stack([gx, gy])

    bands = np.full((len(ELEMENTS), height, width), GEOCHEM_NODATA, dtype="float32")
    descriptions = []
    for i, el in enumerate(ELEMENTS):
        logv = clean_element(ss[el])
        m = logv.notna().values
        surf = idw(sample_xy_all[m], logv.values[m], grid_xy).reshape(height, width)
        bands[i] = surf.astype("float32")
        descriptions.append(f"log10_{el}_ss_idw")
        print(f"  {el:8s} n={int(m.sum()):5d}  "
              f"log10 min/med/max={surf.min():.3f}/{np.median(surf):.3f}/{surf.max():.3f}")

    path = OUT / "geochem_ss_idw_log_district_3338.tif"
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", count=len(ELEMENTS),
        width=width, height=height, crs=crs, transform=transform,
        nodata=GEOCHEM_NODATA, compress="deflate", tiled=True,
    ) as o:
        o.write(bands)
        for i, d in enumerate(descriptions, 1):
            o.set_band_description(i, d)
        o.update_tags(source="AGDB4 stream-sed log-IDW (k=12, p=2)")
    print(f"wrote {path} ({len(ELEMENTS)} bands)")


def main() -> None:
    transform, width, height, crs = build_template()
    with rasterio.open(TEMPLATE) as t:
        b = t.bounds
        bounds = (b.left, b.bottom, b.right, b.top)
    build_akmag(transform, width, height, crs)
    build_geology(transform, width, height, crs, bounds)
    build_geochem(transform, width, height, crs)
    print("\ndistrict covariates built (akmag, geology, faults, geochem).")


if __name__ == "__main__":
    main()
