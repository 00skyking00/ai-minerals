"""v3.7.0 Quaternary relabel: USMIN-only positives with channel-aligned kernel.

Replaces the v3.6 MRDS-derived Quaternary labels (437 records, median 555 m
positional offset) with USMIN-derived labels (436 AOI records, sub-cell
positional accuracy at named anchors). Each USMIN placer/gravel point is
turned into a multi-cell weighted positive by stamping an anisotropic
2D Gaussian aligned with the nearest NHD HR flowline:

  sigma_along  = 1500 m  (Singer & Menzie 2010 placer-run length)
  sigma_cross  =  250 m  (within-channel half-width of a typical Sierra
                          alluvial placer; aligned with the 250 m grid)

Output: `data/derived/northern_sierra_placer/v37_quaternary_kernel_weights.parquet`
with columns (row, col, x, y, weight). Mirrors the v3.6 Tertiary
polygon-rasterization weight column structure, so the assemble step can
load it into `placer_quaternary_weight` and derive
`is_placer_quaternary = (weight > 0)`.

Aggregation across USMIN points is `max` (not sum) so overlapping kernels
cap at 1.0, and the highest-confidence USMIN point dominates per cell.

Per-county audit (see also `northern_sierra_placer_audit_usmin_motherlode.py`):
4 of 10 Mother Lode counties sit below the 20-positive gate (Butte 8,
Yuba 8, Amador 18, Mariposa 4). Sky's plan H2.5 fallback: MRDS placer-Au
records in those counties stay out of training but are scored as a
held-out set after v3.7.0 ships; a v3.7.0.1 augmentation patch fires
only if median MRDS-cell probability in those counties lands sub-decile.

Usage:
    .venv/bin/python scripts/northern_sierra_placer/relabel_motherlode.py
    .venv/bin/python scripts/northern_sierra_placer/relabel_motherlode.py \\
        --sigma-along 1500 --sigma-cross 250

References:
    Plan H2.1c in ~/.claude/plans/hazy-humming-lynx.md
    v3.5.L hypothesis in research/v35_northern_sierra_placer_plan.md
    Singer & Menzie 2010 USGS Bull. 1693 (placer-run length scale)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.aoi import NORTHERN_SIERRA
from ai_minerals.grid import build_grid

REPO = Path(__file__).resolve().parent.parent
USMIN_GPKG = REPO / "data/raw/usmin/usmin_northernsierraplacer.gpkg"
NHD_GPKG = REPO / "data/raw/nhd_hr/nhd_flowlines_northern_sierra.gpkg"
OUT_PARQUET = REPO / "data/derived/northern_sierra_placer/v37_quaternary_kernel_weights.parquet"
OUT_META = OUT_PARQUET.with_suffix(".meta.json")

WORKING_CRS = "EPSG:3310"
RESOLUTION_M = 250

# Quaternary feature classes from USMIN (Hydraulic Mine is Tertiary, excluded).
QUATERNARY_FEATURE_TYPES = {
    "Placer Mine",
    "Gravel Pit",
    "Sand Pit",
    "Sand and Gravel Pit",
    "Gravel/Borrow Pit - Undifferentiated",
    "Diggings",
    "Tailings - Undifferentiated",
    "Mine Dump",
}

# NHD HR stream-order cutoff: placer doesn't form on tiny headwater rivulets.
# stream_order >= 3 is the conventional "first significant stream" cutoff in
# the Strahler scheme for Sierra-scale drainage networks.
MIN_STREAM_ORDER = 3

# Tangent estimation half-step in meters along the line.
TANGENT_EPS_M = 25.0


def load_usmin_quaternary() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(USMIN_GPKG)
    mask = gdf["FTR_TYPE"].isin(QUATERNARY_FEATURE_TYPES)
    out = gdf[mask].copy()
    print(f"[relabel] USMIN: {len(gdf)} total -> {len(out)} Quaternary candidates")
    return out.to_crs(WORKING_CRS)


def load_flowlines() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(NHD_GPKG, columns=["stream_order", "geometry"])
    print(f"[relabel] NHD HR: {len(gdf):,} total flowlines")
    if "stream_order" in gdf.columns:
        kept = gdf[gdf["stream_order"] >= MIN_STREAM_ORDER].copy()
        print(f"[relabel]   after stream_order >= {MIN_STREAM_ORDER}: {len(kept):,}")
    else:
        kept = gdf.copy()
    return kept.to_crs(WORKING_CRS)


def compute_tangents(usmin: gpd.GeoDataFrame, flow: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each USMIN point, find nearest flowline + local tangent direction.

    Returns (tx, ty) unit vectors in working-CRS coordinates.
    """
    print(f"[relabel] computing nearest flowline + tangent for {len(usmin)} USMIN points...")
    joined = gpd.sjoin_nearest(
        usmin[["geometry"]].reset_index(drop=True),
        flow[["geometry"]].reset_index(drop=True),
        how="left",
        distance_col="_d",
    )
    # Drop duplicates produced by sjoin_nearest when a point is equidistant
    # from multiple lines (keep first match).
    joined = joined.drop_duplicates(subset=joined.columns[0], keep="first")
    # Reindex to original USMIN order
    joined = joined.iloc[: len(usmin)].reset_index(drop=True)

    tx = np.zeros(len(usmin), dtype=np.float64)
    ty = np.zeros(len(usmin), dtype=np.float64)
    dist_to_line = np.zeros(len(usmin), dtype=np.float64)

    flow_geoms = flow.geometry.values
    for i, (pt, line_idx, d) in enumerate(zip(usmin.geometry, joined["index_right"], joined["_d"])):
        if pd.isna(line_idx):
            tx[i], ty[i] = 1.0, 0.0
            dist_to_line[i] = np.nan
            continue
        line = flow_geoms[int(line_idx)]
        # Project point onto line (parameterized arc length)
        s = line.project(pt)
        p_at = line.interpolate(s)
        p_eps = line.interpolate(min(s + TANGENT_EPS_M, line.length))
        if p_eps.distance(p_at) < 1e-6:
            p_eps = line.interpolate(max(s - TANGENT_EPS_M, 0.0))
            dx, dy = p_at.x - p_eps.x, p_at.y - p_eps.y
        else:
            dx, dy = p_eps.x - p_at.x, p_eps.y - p_at.y
        norm = np.hypot(dx, dy)
        if norm > 0:
            tx[i], ty[i] = dx / norm, dy / norm
        else:
            tx[i], ty[i] = 1.0, 0.0
        dist_to_line[i] = d

    finite = dist_to_line[~np.isnan(dist_to_line)]
    if len(finite) > 0:
        print(f"[relabel]   distance USMIN -> nearest flowline: "
              f"median={float(np.median(finite)):.1f} m, "
              f"p90={float(np.percentile(finite, 90)):.1f} m, "
              f"max={float(finite.max()):.1f} m")
    return tx, ty


def stamp_kernel_onto_grid(
    usmin: gpd.GeoDataFrame,
    tx: np.ndarray,
    ty: np.ndarray,
    grid_xs: np.ndarray,
    grid_ys: np.ndarray,
    grid_shape: tuple[int, int],
    sigma_along: float,
    sigma_cross: float,
) -> np.ndarray:
    """Stamp anisotropic Gaussian kernels per USMIN point; aggregate via max.

    Returns a 2D grid (n_rows, n_cols) of float32 weights in [0, 1].
    """
    n_rows, n_cols = grid_shape
    out = np.zeros((n_rows, n_cols), dtype=np.float32)

    bbox_along = 5.0 * sigma_along
    bbox_cross = 5.0 * sigma_cross
    # 5-sigma covers ~99.9999% mass; weights below e^(-12.5) ~= 4e-6 are
    # effectively zero. The bounding-box approximation cuts >99% of the
    # per-point work at no measurable cost.

    # Build (x, y) per-cell arrays once.
    cx = grid_xs  # shape (n_cols,)
    cy = grid_ys  # shape (n_rows,)

    res = abs(cx[1] - cx[0]) if len(cx) >= 2 else RESOLUTION_M
    cells_along = int(np.ceil(bbox_along / res))
    cells_cross = int(np.ceil(bbox_cross / res))
    # Worst-case bounding window in cells (when channel is diagonal, the
    # axis-aligned bbox covers a longer diagonal of the anisotropic ellipse).
    bbox_half_cells = max(cells_along, cells_cross) + 1

    # cx and cy are both increasing (centroid_gdf row-major order with
    # ys[0] = south edge; verified for ai_minerals.grid.Grid).
    t0 = time.monotonic()
    for i, pt in enumerate(usmin.geometry):
        px, py = pt.x, pt.y
        # Cell index of point.
        col0 = int(round((px - cx[0]) / res))
        row0 = int(round((py - cy[0]) / res))
        # 2D window in raster index space
        r_lo = max(0, row0 - bbox_half_cells)
        r_hi = min(n_rows, row0 + bbox_half_cells + 1)
        c_lo = max(0, col0 - bbox_half_cells)
        c_hi = min(n_cols, col0 + bbox_half_cells + 1)
        if r_hi <= r_lo or c_hi <= c_lo:
            continue
        # Local (x, y) arrays for this window
        win_x = cx[c_lo:c_hi]  # shape (W,)
        win_y = cy[r_lo:r_hi]  # shape (H,)
        # Broadcast to 2D
        dx = win_x[None, :] - px  # (1, W)
        dy = win_y[:, None] - py  # (H, 1)
        # Project onto channel-aligned coordinates
        d_along = dx * tx[i] + dy * ty[i]
        d_cross = -dx * ty[i] + dy * tx[i]
        # Anisotropic Gaussian
        w = np.exp(
            -0.5 * (d_along * d_along) / (sigma_along * sigma_along)
            - 0.5 * (d_cross * d_cross) / (sigma_cross * sigma_cross)
        ).astype(np.float32)
        # Aggregate via max
        np.maximum(out[r_lo:r_hi, c_lo:c_hi], w, out=out[r_lo:r_hi, c_lo:c_hi])

    elapsed = time.monotonic() - t0
    print(f"[relabel] stamped {len(usmin)} kernels in {elapsed:.1f}s")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-along", type=float, default=1500.0,
                        help="Along-channel Gaussian sigma in meters (default: 1500).")
    parser.add_argument("--sigma-cross", type=float, default=250.0,
                        help="Cross-channel Gaussian sigma in meters (default: 250).")
    parser.add_argument("--min-weight", type=float, default=0.01,
                        help="Cells below this weight are written but flagged as below threshold "
                             "(default: 0.01, ~3-sigma corner).")
    args = parser.parse_args()

    print(f"==> v3.7.0 Quaternary relabel: USMIN + channel-aligned Gaussian kernel")
    print(f"    sigma_along = {args.sigma_along} m  (Singer & Menzie placer-run length)")
    print(f"    sigma_cross = {args.sigma_cross} m  (within-channel half-width)")

    usmin = load_usmin_quaternary()
    flow = load_flowlines()
    grid = build_grid(NORTHERN_SIERRA, resolution_m=RESOLUTION_M, working_crs=WORKING_CRS)
    print(f"[relabel] grid: {grid.shape} cells, n={grid.n_cells:,}, CRS={grid.crs}")

    tx, ty = compute_tangents(usmin, flow)

    weights_2d = stamp_kernel_onto_grid(
        usmin, tx, ty,
        grid_xs=grid.xs,
        grid_ys=grid.ys,
        grid_shape=grid.shape,
        sigma_along=args.sigma_along,
        sigma_cross=args.sigma_cross,
    )

    # Flatten to (row, col, x, y, weight) parquet matching the assemble script's
    # centroid_gdf row-major order.
    n_rows, n_cols = grid.shape
    rows_arr = np.repeat(np.arange(n_rows), n_cols)
    cols_arr = np.tile(np.arange(n_cols), n_rows)
    xs_flat = np.tile(grid.xs, n_rows)
    ys_flat = np.repeat(grid.ys, n_cols)
    weights_flat = weights_2d.flatten()

    df = pd.DataFrame({
        "row": rows_arr.astype(np.int32),
        "col": cols_arr.astype(np.int32),
        "x": xs_flat.astype(np.float64),
        "y": ys_flat.astype(np.float64),
        "weight": weights_flat,
    })
    # Keep only cells with weight > 0 to reduce file size (about 1-2% of cells
    # carry any positive weight at sigma_along=1500m).
    df_nonzero = df[df["weight"] > 0].copy()

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_nonzero.to_parquet(OUT_PARQUET, index=False)
    print(f"[relabel] wrote {OUT_PARQUET}")
    print(f"          {len(df_nonzero):,} cells with weight > 0 "
          f"({100 * len(df_nonzero) / len(df):.2f}% of grid)")

    above_threshold = int((df_nonzero["weight"] >= args.min_weight).sum())
    print(f"          {above_threshold:,} cells above min_weight={args.min_weight}")
    print(f"          weight stats: max={float(df_nonzero['weight'].max()):.4f} "
          f"median={float(df_nonzero['weight'].median()):.4f} "
          f"sum={float(df_nonzero['weight'].sum()):.2f}")

    OUT_META.write_text(json.dumps({
        "input_usmin_gpkg": str(USMIN_GPKG.relative_to(REPO)),
        "input_nhd_gpkg": str(NHD_GPKG.relative_to(REPO)),
        "min_stream_order": MIN_STREAM_ORDER,
        "n_usmin_points": int(len(usmin)),
        "sigma_along_m": args.sigma_along,
        "sigma_cross_m": args.sigma_cross,
        "min_weight_threshold": args.min_weight,
        "working_crs": WORKING_CRS,
        "resolution_m": RESOLUTION_M,
        "grid_shape": list(grid.shape),
        "n_cells_total": grid.n_cells,
        "n_cells_nonzero": int(len(df_nonzero)),
        "n_cells_above_threshold": above_threshold,
        "aggregation": "max",
        "reference": "Singer & Menzie 2010 USGS Bull. 1693",
    }, indent=2))
    print(f"[relabel] wrote {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
