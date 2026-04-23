"""Label assignment — which grid cells are known deposits."""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.grid import Grid

PORPHYRY_FAMILY_CODES = ("17", "20c", "21a", "21b")
PORPHYRY_STRICT_CODES = ("21a",)


def _code_mask(series: pd.Series, codes: tuple[str, ...]) -> pd.Series:
    pat = r"\b(?:" + "|".join(re.escape(c) for c in codes) + r")\b"
    return series.fillna("").str.contains(pat, case=False, regex=True)


def porphyry_positives(ardf: gpd.GeoDataFrame, strict: bool = False) -> gpd.GeoDataFrame:
    """Return the ARDF rows matching our porphyry family or strict filter."""
    codes = PORPHYRY_STRICT_CODES if strict else PORPHYRY_FAMILY_CODES
    return ardf[_code_mask(ardf["model_code"], codes)].copy()


def assign_cells(points: gpd.GeoDataFrame, grid: Grid) -> pd.DataFrame:
    """Return a DataFrame with ('row', 'col') of the grid cell each point falls in.

    Points outside the grid extent are dropped with a warning.
    """
    pts = points.to_crs(grid.crs)
    r = grid.resolution_m
    x0, y0 = grid.xs[0] - r / 2, grid.ys[0] - r / 2
    x1, y1 = grid.xs[-1] + r / 2, grid.ys[-1] + r / 2

    xs = pts.geometry.x.to_numpy()
    ys = pts.geometry.y.to_numpy()
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    n_out = (~inside).sum()
    if n_out:
        print(f"  [assign_cells] {n_out} point(s) outside grid extent; dropping")

    cols = np.floor((xs - x0) / r).astype(int)
    rows = np.floor((ys - y0) / r).astype(int)
    out = pts.copy()
    out["row"] = rows
    out["col"] = cols
    return out[inside].reset_index(drop=True)
