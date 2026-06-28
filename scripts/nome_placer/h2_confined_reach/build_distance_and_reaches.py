"""H2 redesign step 4: down-channel distance from each reach to the upstream contact.

The predictor. Rasterizes the RI 2024-7 schist-marble contact onto the 10 m
working grid, finds where the DEM stream network crosses it (contact crossings),
and propagates an along-channel distance downstream from every crossing so each
stream cell carries the distance to its NEAREST UPSTREAM contact crossing. This
is the round-5 hydrology.py:distance_downstream_from_lode idea with the contact
(not a mapped lode) as the source, computed in one topological pass over the
stream cells (upstream -> downstream by filled elevation) instead of a per-seed
walk, since the dense contact crosses the streams at many points.

Reaches (gotcha 1): each alluvial placer is snapped to the nearest CONFINED
stream cell, then a reach polyline is traced up- and down-stream along the main
stem inside the confined zone. The predictor reported per reach is the
down-channel distance at the reach's UPSTREAM END (the spec's measurement
point), which is the closest the reach gets to an upstream contact; the snap-
cell distance is kept as a sensitivity. A straight-line distance to the contact
is also computed as the round-5-style baseline.

Outputs: down_channel_dist_to_contact.tif, straight_line_dist_to_contact.tif,
reaches.geojson (the traced reach polylines), reach_features.csv (per-placer
coarseness + distances), distance_meta.json.

Run: uv run python -m scripts.nome_placer.h2_confined_reach.build_distance_and_reaches
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from shapely.geometry import LineString

OUT = Path("data/derived/nome_placer/h2_confined_reach")
CONTACT = OUT / "contact_primary_3338.geojson"
TYPED = OUT / "placers_typed.geojson"

SNAP_TOL_M = 250.0        # max placer->confined-stream snap distance
REACH_TRACE_M = 1500.0    # up/down trace half-length for the reach polyline
MAX_DOWNCHANNEL_M = 12_000.0
MAX_STRAIGHT_M = 12_000.0

# 8-neighbour offsets and their step length on a square grid.
NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def down_channel_distance(streams, rr, rc, filled, accum, crossings, cellsize):
    """Distance along the stream network to the nearest upstream contact crossing."""
    diag = cellsize * np.sqrt(2.0)
    H, W = streams.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    sy, sx = np.where(streams)
    # Upstream -> downstream: descending filled elevation, ties by ascending
    # accumulation (less-accumulated = more upstream) so flats propagate right.
    order = np.lexsort((accum[sy, sx], -filled[sy, sx]))
    sy, sx = sy[order], sx[order]
    for r, c in zip(sy, sx):
        if crossings[r, c]:
            dist[r, c] = 0.0
        d = dist[r, c]
        if not np.isfinite(d):
            continue
        nr, nc = rr[r, c], rc[r, c]
        if nr < 0 or nc < 0:
            continue
        step = diag if (nr != r and nc != c) else cellsize
        nd = d + step
        if nd < dist[nr, nc] and nd <= MAX_DOWNCHANNEL_M:
            dist[nr, nc] = nd
    dist[~np.isfinite(dist)] = np.nan
    return dist


def trace_reach(r0, c0, streams, confined, rr, rc, accum, cellsize, half_m):
    """Trace a reach polyline up- and down-stream along the main stem (confined)."""
    # downstream: follow receivers
    down = [(r0, c0)]
    r, c, d = r0, c0, 0.0
    while d < half_m:
        nr, nc = rr[r, c], rc[r, c]
        if nr < 0 or nc < 0 or not confined[nr, nc]:
            break
        d += cellsize * (np.sqrt(2.0) if (nr != r and nc != c) else 1.0)
        down.append((nr, nc)); r, c = nr, nc
    # upstream: step to the stream parent (receiver==here) with max accumulation
    up = []
    r, c, d = r0, c0, 0.0
    while d < half_m:
        best, ba = None, -1.0
        for dr, dc in NBRS:
            pr, pc = r + dr, c + dc
            if not (0 <= pr < streams.shape[0] and 0 <= pc < streams.shape[1]):
                continue
            if (streams[pr, pc] and confined[pr, pc]
                    and rr[pr, pc] == r and rc[pr, pc] == c and accum[pr, pc] > ba):
                best, ba = (pr, pc), accum[pr, pc]
        if best is None:
            break
        d += cellsize * (np.sqrt(2.0) if (best[0] != r and best[1] != c) else 1.0)
        up.append(best); r, c = best
    cells = list(reversed(up)) + down          # upstream-end first
    return cells


def main() -> None:
    with rasterio.open(OUT / "filled_dem.tif") as src:
        transform = src.transform; crs = src.crs
        H, W = src.height, src.width
        filled = src.read(1).astype(np.float32)
    cellsize = abs(transform.a)
    rr = rasterio.open(OUT / "recv_row.tif").read(1)
    rc = rasterio.open(OUT / "recv_col.tif").read(1)
    streams = rasterio.open(OUT / "streams.tif").read(1) == 1
    confined = rasterio.open(OUT / "confined_valley.tif").read(1) == 1
    accum = rasterio.open(OUT / "flow_accum.tif").read(1)

    # Rasterize the contact, find stream crossings (stream cell touching a contact cell).
    contact = gpd.read_file(CONTACT).to_crs(crs)
    contact_cells = rasterize(
        [(g, 1) for g in contact.geometry], out_shape=(H, W),
        transform=transform, fill=0, all_touched=True).astype(bool)
    crossings = streams & binary_dilation(contact_cells, iterations=1)
    n_cross = int(crossings.sum())
    print(f"contact cells: {int(contact_cells.sum())}, stream crossings: {n_cross}")

    dch = down_channel_distance(streams, rr, rc, filled, accum, crossings, cellsize)
    conf_vals = dch[confined & np.isfinite(dch)]
    print(f"down-channel dist on confined streams: n={conf_vals.size} "
          f"median={np.median(conf_vals):.0f} m p90={np.percentile(conf_vals,90):.0f} m "
          f"max={conf_vals.max():.0f} m")

    # Straight-line distance to contact (baseline), full grid via KDTree on contact cells.
    cy, cx = np.where(contact_cells)
    cxs, cys = rasterio.transform.xy(transform, cy, cx)
    ctree = cKDTree(np.column_stack([np.asarray(cxs), np.asarray(cys)]))

    # Placers -> snap to nearest confined stream cell, trace reach, read distances.
    typed = gpd.read_file(TYPED).to_crs(crs)
    al = typed[typed.geol_type == "alluvial-stream"].copy().reset_index(drop=True)
    fy, fx = np.where(confined)
    conf_xy = np.column_stack([fx, fy]).astype(float)
    snap_tree = cKDTree(conf_xy)
    snap_cells = SNAP_TOL_M / cellsize

    rows = []
    reach_geoms, reach_props = [], []
    for _, p in al.iterrows():
        px, py = p.geometry.x, p.geometry.y
        prow, pcol = rasterio.transform.rowcol(transform, px, py)
        dq, idx = snap_tree.query([pcol, prow], k=1)
        snapped = dq <= snap_cells
        rec = {"ardf_num": p.ardf_num, "site": p.site,
               "coarseness_rank": p.coarseness_rank,
               "snapped": bool(snapped), "snap_off_m": round(float(dq) * cellsize, 1)}
        sl, _ = ctree.query([px, py], k=1)
        rec["straight_contact_m"] = round(float(sl), 1) if sl <= MAX_STRAIGHT_M else None
        if snapped:
            sr, sc = int(fy[idx]), int(fx[idx])
            rec["snap_dch_m"] = (round(float(dch[sr, sc]), 1)
                                 if np.isfinite(dch[sr, sc]) else None)
            cells = trace_reach(sr, sc, streams, confined, rr, rc, accum, cellsize, REACH_TRACE_M)
            up_r, up_c = cells[0]
            rec["reach_head_dch_m"] = (round(float(dch[up_r, up_c]), 1)
                                       if np.isfinite(dch[up_r, up_c]) else None)
            rec["reach_len_m"] = round((len(cells) - 1) * cellsize, 0)
            xs, ys = rasterio.transform.xy(transform, [rc_[0] for rc_ in cells],
                                           [rc_[1] for rc_ in cells])
            if len(cells) >= 2:
                reach_geoms.append(LineString(list(zip(xs, ys))))
                reach_props.append({"ardf_num": p.ardf_num, "site": p.site,
                                    "coarseness_rank": p.coarseness_rank,
                                    "reach_head_dch_m": rec["reach_head_dch_m"]})
        else:
            rec["snap_dch_m"] = None
            rec["reach_head_dch_m"] = None
            rec["reach_len_m"] = None
        rows.append(rec)

    import pandas as pd
    feat = pd.DataFrame(rows)
    feat.to_csv(OUT / "reach_features.csv", index=False)
    if reach_geoms:
        gpd.GeoDataFrame(reach_props, geometry=reach_geoms, crs=crs).to_file(
            OUT / "reaches.geojson", driver="GeoJSON")

    # Write the two distance rasters (confined-clipped down-channel; full straight-line).
    prof = dict(driver="GTiff", dtype="float32", count=1, width=W, height=H,
                crs=crs, transform=transform, nodata=-1.0, compress="lzw",
                tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    with rasterio.open(OUT / "down_channel_dist_to_contact.tif", "w", **prof) as d:
        out = np.where(confined & np.isfinite(dch), dch, -1.0).astype(np.float32)
        d.write(out, 1)

    n_tag = int(feat.coarseness_rank.notna().sum())
    n_tag_snap = int((feat.coarseness_rank.notna() & feat.reach_head_dch_m.notna()).sum())
    meta = {
        "n_contact_crossings": n_cross,
        "snap_tol_m": SNAP_TOL_M, "reach_trace_half_m": REACH_TRACE_M,
        "n_alluvial": int(len(al)),
        "n_snapped": int(feat.snapped.sum()),
        "n_coarseness_tagged": n_tag,
        "n_tagged_with_reach_distance": n_tag_snap,
        "confined_dch_median_m": round(float(np.median(conf_vals)), 1),
        "confined_dch_p90_m": round(float(np.percentile(conf_vals, 90)), 1),
    }
    (OUT / "distance_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
