"""Round 5 step 3-4: distance-to-lode feature rasters.

PRIMARY (down-channel): walk the DEM-derived D8 receiver network downstream
from each 36a lode seed, accumulating along-channel distance, and tag every
downstream cell with the distance to its NEAREST upstream lode. Clipped to the
confined-upland-valley mask (where modern channel == ancestral channel). This
is the local-source gradient feature; it is the DEM adaptation of
hydrology.py:distance_downstream_from_lode (whose NHD/hydroseq walk does not
apply to a DEM network), and it carries the same placer-leakage guard: lode
seeds must not themselves be placer occurrences.

SECONDARY (straight-line): per-cell Euclidean distance to the nearest lode
point (the comparison baseline for the down-channel feature; mirrors
hydrology.py:distance_to_lode_m).

The schist-limestone contact distance already exists at
bedrock_contact/dist_to_contact.tif and is reused as-is.

Writes down_channel_dist_to_lode.tif and straight_line_dist_to_lode.tif.

Run: uv run python -m scripts.nome_placer.inland_local_source.build_distance_features
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.spatial import cKDTree

LODE = Path("data/derived/nome_placer/ardf_nome_lode_au_sources.geojson")
OUTDIR = Path("data/derived/nome_placer/inland_local_source")
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")

SNAP_TOL_M = 300.0       # max distance a lode may be from a stream cell to seed
MAX_DOWNCHANNEL_M = 30_000.0
MAX_STRAIGHT_M = 25_000.0


def assert_no_placer_lodes(lode: gpd.GeoDataFrame) -> None:
    """Refuse placer-flagged seeds (label leak). 36a is lode; 39* is placer."""
    mc = lode["model_code"].astype(str).str.lower()
    comm = lode["comm_main"].astype(str).str.lower()
    placer_like = mc.str.contains("39") | comm.str.contains("placer")
    if placer_like.any():
        bad = lode.loc[placer_like, ["ardf_num", "model_code"]].to_dict("records")
        raise AssertionError(f"placer-flagged lode seeds would leak the label: {bad}")


def main() -> None:
    with rasterio.open(DEM) as src:
        transform = src.transform
        crs = src.crs
        H, W = src.height, src.width
        profile = src.profile.copy()
    cellsize = abs(transform.a)
    diag = cellsize * np.sqrt(2.0)

    rr = rasterio.open(OUTDIR / "recv_row.tif").read(1)
    rc = rasterio.open(OUTDIR / "recv_col.tif").read(1)
    streams = rasterio.open(OUTDIR / "streams.tif").read(1) == 1
    confined = rasterio.open(OUTDIR / "confined_valley.tif").read(1) == 1

    # Lode seeds: drop placer-flagged, clip to grid, keep 36a-style Au lodes.
    lode = gpd.read_file(LODE).to_crs(crs)
    keep = ~(lode["model_code"].astype(str).str.contains("39")
             | lode["comm_main"].astype(str).str.lower().str.contains("placer"))
    lode = lode[keep].copy()
    assert_no_placer_lodes(lode)
    lx = lode.geometry.x.to_numpy(); ly = lode.geometry.y.to_numpy()
    lrows, lcols = rasterio.transform.rowcol(transform, lx, ly)
    lrows = np.asarray(lrows); lcols = np.asarray(lcols)
    inb = (lrows >= 0) & (lrows < H) & (lcols >= 0) & (lcols < W)
    lrows, lcols = lrows[inb], lcols[inb]
    n_lode_in = int(inb.sum())

    # Snap each in-grid lode to the nearest STREAM cell within SNAP_TOL_M.
    sy, sx = np.where(streams)
    stream_xy = np.column_stack([sx, sy]).astype(float)
    tree = cKDTree(stream_xy)
    seeds = []
    snap_cells = SNAP_TOL_M / cellsize
    for r, c in zip(lrows, lcols):
        d, idx = tree.query([c, r], k=1)
        if d <= snap_cells:
            seeds.append((int(sy[idx]), int(sx[idx])))
    seeds = list(set(seeds))
    print(f"lode seeds: {n_lode_in} in-grid, {len(seeds)} snapped to streams "
          f"(<= {SNAP_TOL_M:.0f} m)")

    # Down-channel walk: from each seed follow receivers, accumulate distance,
    # keep the per-cell minimum (= distance from nearest upstream lode).
    dch = np.full((H, W), np.inf, dtype=np.float64)
    for (sr, sc) in seeds:
        r, c, dist = sr, sc, 0.0
        steps = 0
        while True:
            if dist < dch[r, c]:
                dch[r, c] = dist
            nr, nc = rr[r, c], rc[r, c]
            if nr < 0 or nc < 0:
                break                       # outlet / edge
            step = diag if (nr != r and nc != c) else cellsize
            dist += step
            if dist > MAX_DOWNCHANNEL_M:
                break
            steps += 1
            if steps > H * W:               # cycle backstop (should not trigger)
                break
            r, c = nr, nc
    dch[~np.isfinite(dch)] = np.nan
    # Clip the feature to confined upland valleys (modern == ancestral channel).
    dch_clipped = np.where(confined, dch, np.nan)
    n_valid = int(np.isfinite(dch_clipped).sum())
    print(f"down-channel cells (confined & seeded): {n_valid}; "
          f"median {np.nanmedian(dch_clipped):.0f} m, max {np.nanmax(dch_clipped):.0f} m"
          if n_valid else "down-channel cells: 0")

    # Straight-line distance to nearest lode point (full grid).
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    cell_xy = np.column_stack([np.asarray(cx), np.asarray(cy)])
    lode_xy = np.column_stack([lx[inb], ly[inb]])
    ltree = cKDTree(lode_xy)
    sd, _ = ltree.query(cell_xy, k=1, distance_upper_bound=MAX_STRAIGHT_M)
    sd = np.where(np.isinf(sd), np.nan, sd).reshape(H, W)

    # Write rasters aligned to the DEM grid.
    def wr(name, data):
        p = profile.copy(); p.update(dtype="float32", count=1, nodata=-1.0, compress="lzw")
        with rasterio.open(OUTDIR / name, "w", **p) as d:
            d.write(np.where(np.isfinite(data), data, -1.0).astype(np.float32), 1)
    wr("down_channel_dist_to_lode.tif", dch_clipped)
    wr("straight_line_dist_to_lode.tif", sd)

    meta = {
        "n_lode_total": int(len(lode)), "n_lode_in_grid": n_lode_in,
        "n_seeds_snapped": len(seeds), "snap_tol_m": SNAP_TOL_M,
        "n_downchannel_valid_cells": n_valid,
        "max_downchannel_m": MAX_DOWNCHANNEL_M, "max_straight_m": MAX_STRAIGHT_M,
    }
    (OUTDIR / "distance_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote down_channel + straight_line rasters + distance_meta.json")


if __name__ == "__main__":
    main()
