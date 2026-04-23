"""Geochemistry features — per-cell aggregates of AGDB4 pathfinder elements.

For each grid cell, look up AGDB4 samples within a radius (5 km default),
compute mean / max / count per element, returned as columns like
`cu_mean_5km`, `cu_max_5km`, `cu_count_5km`.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ai_minerals.grid import Grid


# Mapping pathfinder element → (BV file, column name in that file)
# AGDB4 "text" files are comma-delimited despite the .txt extension.
# Column name convention: <El>_ppm for trace elements.
PATHFINDER_ELEMENTS = {
    "Ag": ("BV_Ag_Br.txt", "Ag_ppm"),
    "As": ("BV_Ag_Br.txt", "As_ppm"),
    "Au": ("BV_Ag_Br.txt", "Au_ppm"),
    "Bi": ("BV_Ag_Br.txt", "Bi_ppm"),
    "Cu": ("BV_C_Gd.txt",  "Cu_ppm"),
    "Mo": ("BV_Ge_Os.txt", "Mo_ppm"),
    "Pb": ("BV_P_Te.txt",  "Pb_ppm"),
    "Sb": ("BV_P_Te.txt",  "Sb_ppm"),
    "Te": ("BV_P_Te.txt",  "Te_ppm"),
    "Zn": ("BV_Th_Zr.txt", "Zn_ppm"),
}


def _load_bv_element(zip_path: Path, bv_file: str, column: str) -> pd.DataFrame:
    """Read just [DDPD_ID, column] from one BV_*.txt, streaming via usecols."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"AGDB4_text/{bv_file}") as f:
            raw = f.read()
    df = pd.read_csv(
        io.BytesIO(raw),
        sep=",",
        usecols=["DDPD_ID", column],
        low_memory=False,
        encoding="latin-1",
    )
    # AGDB4 encodes below-detection-limit values as negatives (e.g. -50 means "<50").
    # Mask them as NaN rather than treat the sentinel as a real value.
    df.loc[df[column] < 0, column] = np.nan
    return df


def load_pathfinder_assays(
    samples: pd.DataFrame,
    zip_path: Path,
    elements: tuple[str, ...] = tuple(PATHFINDER_ELEMENTS),
) -> pd.DataFrame:
    """Left-join pathfinder-element concentrations onto the AOI sample table."""
    out = samples.copy()
    for el in elements:
        bv_file, col = PATHFINDER_ELEMENTS[el]
        assay = _load_bv_element(zip_path, bv_file, col)
        out = out.merge(assay, on="DDPD_ID", how="left")
        # rename to standard lowercase
        out = out.rename(columns={col: el.lower()})
    return out


def aggregate_in_radius(
    grid: Grid,
    samples_gdf,
    *,
    radius_m: float = 5000.0,
    elements: tuple[str, ...] = tuple(PATHFINDER_ELEMENTS),
) -> dict[str, np.ndarray]:
    """For each grid cell, compute (mean, max, count) of each element within `radius_m`.

    Parameters
    ----------
    grid : Grid
    samples_gdf : GeoDataFrame with geometry (Points in grid.crs) and element columns.

    Returns
    -------
    dict of feature-name → 2-D arrays of shape grid.shape. Feature names are
    `<el>_mean_Nkm`, `<el>_max_Nkm`, `<el>_count_Nkm`.
    """
    samples_gdf = samples_gdf.to_crs(grid.crs)
    sample_xy = np.column_stack([
        samples_gdf.geometry.x.to_numpy(),
        samples_gdf.geometry.y.to_numpy(),
    ])

    # Grid centroids as a 2-D array, paired with row/col.
    xv, yv = np.meshgrid(grid.xs, grid.ys)
    grid_xy = np.column_stack([xv.ravel(), yv.ravel()])
    tree = cKDTree(grid_xy)

    # For each sample, find grid cells within radius.
    # `query_ball_point` returns a list of grid-cell indices per sample.
    neighbor_lists = tree.query_ball_point(sample_xy, r=radius_m)

    km = int(round(radius_m / 1000))
    out: dict[str, np.ndarray] = {}

    for el in elements:
        values = samples_gdf[el.lower()].to_numpy(dtype=np.float32)
        # Accumulate per-cell sum, max, count (skipping NaN samples).
        sum_arr = np.zeros(grid.n_cells, dtype=np.float64)
        max_arr = np.full(grid.n_cells, np.nan, dtype=np.float32)
        cnt_arr = np.zeros(grid.n_cells, dtype=np.int32)

        for sample_idx, cell_indices in enumerate(neighbor_lists):
            v = values[sample_idx]
            if not np.isfinite(v) or not cell_indices:
                continue
            cell_arr = np.asarray(cell_indices, dtype=np.int64)
            sum_arr[cell_arr] += v
            cnt_arr[cell_arr] += 1
            # max: need to compare element-wise
            existing = max_arr[cell_arr]
            updated = np.where(np.isnan(existing) | (v > existing), v, existing)
            max_arr[cell_arr] = updated

        mean_arr = np.full(grid.n_cells, np.nan, dtype=np.float32)
        nonzero = cnt_arr > 0
        mean_arr[nonzero] = (sum_arr[nonzero] / cnt_arr[nonzero]).astype(np.float32)

        out[f"{el.lower()}_mean_{km}km"] = mean_arr.reshape(grid.shape)
        out[f"{el.lower()}_max_{km}km"]  = max_arr.reshape(grid.shape)
        out[f"{el.lower()}_count_{km}km"] = cnt_arr.reshape(grid.shape).astype(np.float32)

    return out
