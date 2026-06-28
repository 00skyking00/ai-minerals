"""H2 redesign step 2: DEM terrain + hydrology over the RI 2024-7 map area.

Adapts the round-5 inland_local_source/build_terrain.py to the native 5 m IFSAR
DTM of the Big Hurrah-Council-Bluff area (the round-5 note flagged that the 25 m
grid missed the narrowest gulches; this is the 5 m fix). Depression-fill +
D8 pointer + flow accumulation (Whitebox), a DEM-derived stream network, local
relief, and the confined-upland-valley mask that gotcha 2 requires (the modern
DEM only routes the ancestral channel inside confined V-valleys).

Run at a 10 m working grid (the native 5 m DTM downsampled 2x on read): the
whitebox_workflows hydrology runs in-process, so its Rust rasters plus the
numpy arrays do not fit a 136 Mcell (5 m) grid under the RAM cap. 10 m holds the
whole map in ~34 Mcell and is still 2.5x finer than the round-5 25 m grid that
missed the gulches; the native 5 m DTM is kept on disk for any finer follow-up.

Resolution-dependent thresholds at the 10 m working grid:
  - stream init 500 cells = 0.05 km^2 contributing area (matches round 5's area)
  - relief window radius 26 cells ~= the same 525 m window as round 5
Arrays are float32 (not float64).

Run (heavy; wrap in run_capped):
  scripts/run_capped.sh --mem 12G --swap 0 -- \
    .venv/bin/python -m scripts.nome_placer.h2_confined_reach.build_terrain
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from scipy.ndimage import maximum_filter, minimum_filter
from whitebox_workflows import WbEnvironment

DEM = Path("data/raw/ifsar_dggs/ifsar_dtm_5m_bighurrah_council_bluff_3338.tif")
OUTDIR = Path("data/derived/nome_placer/h2_confined_reach")

DOWNSAMPLE = 2                 # 5 m native -> 10 m working grid (memory)
# WhiteboxTools D8 pointer scheme:  64 128  1 / 32  0  2 / 16  8  4
PTR_OFFSETS = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
               16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}

STREAM_THRESH_CELLS = 500      # ~0.05 km^2 contributing area at 10 m
RELIEF_WIN_CELLS = 26          # radius -> 53x53 window ~525 m at 10 m
CONFINE_RELIEF_M = 45.0        # >= this local relief -> confined upland valley
COASTAL_RELIEF_M = 25.0        # <= this -> coastal plain; between -> dead-zone


def _profile(transform, crs, H, W, dtype: str, nodata) -> dict:
    return dict(driver="GTiff", dtype=dtype, count=1, width=W, height=H,
                crs=crs, transform=transform, nodata=nodata, compress="lzw",
                tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")


def main() -> None:
    # Downsample the native 5 m DTM to a 10 m working grid on read.
    with rasterio.open(DEM) as src:
        nod = src.nodata
        crs = src.crs
        H, W = src.height // DOWNSAMPLE, src.width // DOWNSAMPLE
        dem = src.read(1, out_shape=(H, W), resampling=Resampling.average).astype(np.float32)
        t = src.transform
        transform = Affine(t.a * DOWNSAMPLE, 0, t.c, 0, t.e * DOWNSAMPLE, t.f)
    cellsize = abs(transform.a)
    valid = np.isfinite(dem) & (dem != nod)
    print(f"working grid {W}x{H} @ {cellsize:.0f} m (native 5 m downsampled {DOWNSAMPLE}x)")

    # 1) Depression-fill with Whitebox (writes/reads via tempfiles).
    with tempfile.TemporaryDirectory(prefix="wbw_h2_terrain_") as tmp:
        tmp = Path(tmp)
        fill_in = tmp / "dem.tif"
        wbnod = -32768.0
        arr = np.where(valid, dem, wbnod).astype(np.float32)
        with rasterio.open(fill_in, "w", **_profile(transform, crs, H, W, "float32", wbnod)) as dst:
            dst.write(arr, 1)
        del arr
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
            filled = s.read(1).astype(np.float32)
            fnod = s.nodata
        with rasterio.open(tmp / "ptr.tif") as s:
            ptr = s.read(1).astype(np.int32)
        with rasterio.open(tmp / "acc.tif") as s:
            acc = s.read(1).astype(np.float32)
            anod = s.nodata
    filled = np.where(filled == fnod, np.nan, filled)
    acc = np.where(acc == anod, np.nan, acc)

    # 2) Decode pointer -> receiver row/col, then VERIFY it goes downhill.
    rr = np.full((H, W), -1, dtype=np.int32)
    rc = np.full((H, W), -1, dtype=np.int32)
    for code, (dr, dc) in PTR_OFFSETS.items():
        m = ptr == code
        ys, xs = np.where(m)
        ny, nx = ys + dr, xs + dc
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        rr[ys[ok], xs[ok]] = ny[ok]
        rc[ys[ok], xs[ok]] = nx[ok]
    del ptr

    has_recv = (rr >= 0) & np.isfinite(filled)
    ys, xs = np.where(has_recv)
    z_here = filled[ys, xs]
    z_recv = filled[rr[ys, xs], rc[ys, xs]]
    downhill = float(np.mean(z_recv <= z_here + 1e-6))
    print(f"pointer check: {downhill:.4f} of cells flow to a <= neighbor "
          f"(should be ~1.0); n={len(ys)}")
    assert downhill > 0.98, "D8 pointer convention mismatch -- fix PTR_OFFSETS"

    # 3) Stream network from accumulation.
    streams = (np.nan_to_num(acc, nan=0.0) >= STREAM_THRESH_CELLS) & valid
    print(f"stream cells: {int(streams.sum())} "
          f"({100*streams.sum()/valid.sum():.2f}% of valid land)")

    # 4) Local relief -> confined / coastal / dead-zone zonation.
    zmax = maximum_filter(np.where(valid, dem, -1e9), size=2 * RELIEF_WIN_CELLS + 1)
    zmin = minimum_filter(np.where(valid, dem, 1e9), size=2 * RELIEF_WIN_CELLS + 1)
    relief = np.where(valid, zmax - zmin, np.nan).astype(np.float32)
    del zmax, zmin

    zone = np.zeros((H, W), dtype=np.uint8)         # 0 = nodata/sea
    zone[valid & (relief <= COASTAL_RELIEF_M)] = 1   # coastal plain
    zone[valid & (relief > COASTAL_RELIEF_M) & (relief < CONFINE_RELIEF_M)] = 2  # dead-zone
    zone[valid & (relief >= CONFINE_RELIEF_M)] = 3   # upland
    confined = streams & (zone == 3)
    print(f"confined-upland stream cells: {int(confined.sum())}")

    # 5) Write rasters.
    OUTDIR.mkdir(parents=True, exist_ok=True)

    def wr(name, data, dtype, nodata):
        with rasterio.open(OUTDIR / name, "w",
                           **_profile(transform, crs, H, W, dtype, nodata)) as d:
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
        "dem": str(DEM), "cellsize_m": cellsize,
        "stream_thresh_cells": STREAM_THRESH_CELLS,
        "stream_thresh_km2": round(STREAM_THRESH_CELLS * cellsize ** 2 / 1e6, 4),
        "relief_win_cells": RELIEF_WIN_CELLS,
        "relief_win_m": (2 * RELIEF_WIN_CELLS + 1) * cellsize,
        "confine_relief_m": CONFINE_RELIEF_M, "coastal_relief_m": COASTAL_RELIEF_M,
        "n_stream_cells": int(streams.sum()), "n_confined_cells": int(confined.sum()),
        "pointer_downhill_frac": round(downhill, 4),
    }
    (OUTDIR / "terrain_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote terrain rasters + terrain_meta.json")


if __name__ == "__main__":
    main()
