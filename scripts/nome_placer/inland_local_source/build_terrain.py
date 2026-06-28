"""Round 5 step 2: DEM terrain + hydrology for the inland-alluvial test.

From the 25 m IfSAR DEM (EPSG:3338): depression-fill (Whitebox), D8 pointer +
flow accumulation (Whitebox), a DEM-derived stream network, local relief, and
a confined-upland-valley mask with a foothill dead-zone. The confined-valley
mask is where the modern channel is judged to coincide with the ancestral one
(narrow incised valleys), so the down-channel distance feature can be trusted;
the flat coastal plain and the wide valley/fan reaches are excluded.

The Whitebox D8 pointer convention is verified empirically (following it must
go downhill on the filled DEM) before it is decoded into (drow, dcol) receiver
offsets used by the down-channel walk in build_distance_features.py.

Writes filled DEM, flow-accum, receiver-rows/cols, streams, relief, and the
confined/zone masks to data/derived/nome_placer/inland_local_source/.

Run: uv run python -m scripts.nome_placer.inland_local_source.build_terrain
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import maximum_filter, minimum_filter
from whitebox_workflows import WbEnvironment

DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
TYPED = Path("data/derived/nome_placer/inland_local_source/placers_typed.geojson")
OUTDIR = Path("data/derived/nome_placer/inland_local_source")

# WhiteboxTools D8 pointer scheme:  64 128  1 / 32  0  2 / 16  8  4
PTR_OFFSETS = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
               16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}

STREAM_THRESH_CELLS = 200      # ~0.125 km^2 contributing area at 25 m
RELIEF_WIN_CELLS = 10          # radius -> 21x21 window ~525 m
CONFINE_RELIEF_M = 45.0        # >= this local relief -> confined upland valley
COASTAL_RELIEF_M = 25.0        # <= this -> coastal plain (marine-control zone)
                               # between the two -> foothill dead-zone


def _profile(src: rasterio.DatasetReader, dtype: str, nodata) -> dict:
    p = src.profile.copy()
    p.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    return p


def main() -> None:
    with rasterio.open(DEM) as src:
        dem = src.read(1).astype(np.float64)
        nod = src.nodata
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
    cellsize = abs(transform.a)
    valid = np.isfinite(dem) & (dem != nod)
    dem_m = np.where(valid, dem, np.nan)

    # 1) Depression-fill with Whitebox (writes/reads via tempfiles).
    with tempfile.TemporaryDirectory(prefix="wbw_terrain_") as tmp:
        tmp = Path(tmp)
        fill_in = tmp / "dem.tif"
        wbnod = -32768.0
        arr = np.where(valid, dem, wbnod).astype(np.float32)
        pin = profile.copy()
        pin.update(dtype="float32", count=1, nodata=wbnod, compress="lzw")
        with rasterio.open(fill_in, "w", **pin) as dst:
            dst.write(arr, 1)
        wbe = WbEnvironment()
        wbe.working_directory = str(tmp)
        dem_ras = wbe.read_raster(str(fill_in))
        filled_ras = wbe.hydrology.fill_depressions(dem=dem_ras)
        ptr_ras = wbe.hydrology.d8_pointer(dem=filled_ras)
        acc_ras = wbe.hydrology.d8_flow_accum(input=filled_ras, out_type="cells")
        wbe.write_raster(filled_ras, str(tmp / "filled.tif"))
        wbe.write_raster(ptr_ras, str(tmp / "ptr.tif"))
        wbe.write_raster(acc_ras, str(tmp / "acc.tif"))
        with rasterio.open(tmp / "filled.tif") as s:
            filled = s.read(1).astype(np.float64)
            fnod = s.nodata
        with rasterio.open(tmp / "ptr.tif") as s:
            ptr = s.read(1).astype(np.int32)
        with rasterio.open(tmp / "acc.tif") as s:
            acc = s.read(1).astype(np.float64)
            anod = s.nodata
    filled = np.where(filled == fnod, np.nan, filled)
    acc = np.where(acc == anod, np.nan, acc)

    # 2) Decode pointer -> receiver row/col, then VERIFY it goes downhill.
    H, W = dem.shape
    rr = np.full((H, W), -1, dtype=np.int32)
    rc = np.full((H, W), -1, dtype=np.int32)
    for code, (dr, dc) in PTR_OFFSETS.items():
        m = ptr == code
        ys, xs = np.where(m)
        ny, nx = ys + dr, xs + dc
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        rr[ys[ok], xs[ok]] = ny[ok]
        rc[ys[ok], xs[ok]] = nx[ok]

    has_recv = (rr >= 0) & np.isfinite(filled)
    ys, xs = np.where(has_recv)
    z_here = filled[ys, xs]
    z_recv = filled[rr[ys, xs], rc[ys, xs]]
    downhill = np.mean(z_recv <= z_here + 1e-6)
    print(f"pointer check: {downhill:.4f} of cells flow to a <= neighbor "
          f"(should be ~1.0); n={len(ys)}")
    assert downhill > 0.98, "D8 pointer convention mismatch -- fix PTR_OFFSETS"

    # 3) Stream network from accumulation.
    streams = (np.nan_to_num(acc, nan=0.0) >= STREAM_THRESH_CELLS) & valid
    print(f"stream cells: {int(streams.sum())} "
          f"({100*streams.sum()/valid.sum():.1f}% of valid land)")

    # 4) Local relief -> confined / coastal / dead-zone zonation.
    z = np.where(valid, dem, np.nan)
    zmax = maximum_filter(np.nan_to_num(z, nan=-1e9), size=2 * RELIEF_WIN_CELLS + 1)
    zmin = minimum_filter(np.nan_to_num(z, nan=1e9), size=2 * RELIEF_WIN_CELLS + 1)
    relief = np.where(valid, zmax - zmin, np.nan)

    zone = np.zeros((H, W), dtype=np.uint8)        # 0 = nodata/sea
    zone[valid & (relief <= COASTAL_RELIEF_M)] = 1  # coastal plain
    zone[valid & (relief > COASTAL_RELIEF_M) & (relief < CONFINE_RELIEF_M)] = 2  # dead-zone
    zone[valid & (relief >= CONFINE_RELIEF_M)] = 3  # upland
    confined = streams & (zone == 3)

    # 5) Diagnostics: relief at alluvial vs marine placer points.
    typed = gpd.read_file(TYPED)
    for label in ("alluvial-stream", "marine-beach"):
        sub = typed[typed.geol_type == label]
        if len(sub) == 0:
            continue
        rows, cols = rasterio.transform.rowcol(
            transform, sub.geometry.x.to_numpy(), sub.geometry.y.to_numpy())
        rows = np.clip(rows, 0, H - 1); cols = np.clip(cols, 0, W - 1)
        rv = relief[rows, cols]
        zv = zone[rows, cols]
        nz = {int(k): int(v) for k, v in zip(*np.unique(zv, return_counts=True))}
        print(f"{label}: n={len(sub)} relief med={np.nanmedian(rv):.0f}m "
              f"zone-counts(0sea/1coast/2dead/3upland)={nz}")

    # 6) Write rasters.
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DEM) as src:
        def wr(name, data, dtype, nodata):
            with rasterio.open(OUTDIR / name, "w", **_profile(src, dtype, nodata)) as d:
                d.write(np.where(np.isfinite(data), data, nodata).astype(dtype)
                        if dtype.startswith("float") else data.astype(dtype), 1)
        wr("filled_dem.tif", filled, "float32", -32768.0)
        wr("flow_accum.tif", acc, "float32", -1.0)
        wr("relief.tif", relief, "float32", -1.0)
        wr("recv_row.tif", rr, "int32", -1)
        wr("recv_col.tif", rc, "int32", -1)
        wr("streams.tif", streams.astype(np.uint8), "uint8", 255)
        wr("confined_valley.tif", confined.astype(np.uint8), "uint8", 255)
        wr("zone.tif", zone, "uint8", 255)
    meta = {
        "cellsize_m": cellsize, "stream_thresh_cells": STREAM_THRESH_CELLS,
        "relief_win_cells": RELIEF_WIN_CELLS, "confine_relief_m": CONFINE_RELIEF_M,
        "coastal_relief_m": COASTAL_RELIEF_M,
        "n_stream_cells": int(streams.sum()), "n_confined_cells": int(confined.sum()),
        "pointer_downhill_frac": round(float(downhill), 4),
    }
    (OUTDIR / "terrain_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote terrain rasters + terrain_meta.json")


if __name__ == "__main__":
    main()
