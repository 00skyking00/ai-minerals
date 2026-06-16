"""Label assignment — which grid cells are known deposits.

Canonical occurrence GeoDataFrames (from `data/adapters/occurrences/*`)
carry a `deposit_codes: tuple[str, ...]` column with jurisdiction-prefixed
codes (e.g. `"usgs:21a"`, `"bc:L03"`). `deposit_positives` filters to
rows matching any of a target code tuple.

v1 `porphyry_positives(ardf, strict=...)` is preserved as a thin wrapper
for back-compat with notebook cells that were written against the raw
ARDF schema (with the free-text `model_code` field).
"""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.grid import Grid


# v1 constants preserved for the back-compat wrapper below.
PORPHYRY_FAMILY_CODES = ("17", "20c", "21a", "21b")
PORPHYRY_STRICT_CODES = ("21a",)


def deposit_positives(
    occurrences: gpd.GeoDataFrame, codes: tuple[str, ...]
) -> gpd.GeoDataFrame:
    """Filter canonical occurrences to those whose deposit_codes intersects `codes`.

    `codes` should be jurisdiction-prefixed (e.g. `("usgs:21a",)`). Matches
    are exact (no substring/regex), so the adapter is the only place that
    parses free-text model-code fields.
    """
    want = frozenset(codes)
    mask = occurrences["deposit_codes"].apply(lambda cs: bool(want.intersection(cs)))
    return occurrences[mask].copy()


# --- Back-compat wrapper (raw ARDF schema, for any caller that hasn't migrated) ---

def _code_mask(series: pd.Series, codes: tuple[str, ...]) -> pd.Series:
    pat = r"\b(?:" + "|".join(re.escape(c) for c in codes) + r")\b"
    return series.fillna("").str.contains(pat, case=False, regex=True)


def porphyry_positives(ardf: gpd.GeoDataFrame, strict: bool = False) -> gpd.GeoDataFrame:
    """Filter raw ARDF GeoDataFrame to porphyry family (or strict 21a) rows.

    Accepts both the raw ARDF schema (has `model_code` column) and the
    canonical schema (has `deposit_codes` column).
    """
    codes = PORPHYRY_STRICT_CODES if strict else PORPHYRY_FAMILY_CODES
    if "deposit_codes" in ardf.columns:
        prefixed = tuple(f"usgs:{c}" for c in codes)
        return deposit_positives(ardf, prefixed)
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


def aux_positives_from_mask(
    mask: pd.Series,
    grid: Grid,
    *,
    weight: float = 0.25,
    label_value: int = 1,
    source: str = "aux_mask",
) -> pd.DataFrame:
    """Convert a per-cell mask into a weighted auxiliary positive-label table.

    Phase 2 supervised model context: drill holes carry weight 1.0
    (measured grade); polygon-derived "was-mined" or "is-prospective"
    masks carry a lower aux weight. PDF 94-39 ``Qh`` placer-tailings
    polygons are the canonical example, sampled via
    ``features.surficial.tailings_mask``.

    Returns a DataFrame with columns ``row``, ``col``, ``lon``,
    ``lat``, ``label``, ``weight``, ``source``. One row per
    positive (mask > 0) cell; cells with mask = 0 are dropped. The
    ``weight`` is constant per call so callers can stack aux sources
    with different weights and concat the resulting tables before
    feeding into a training loop.
    """
    if len(mask) != grid.n_cells:
        raise ValueError(
            f"mask length {len(mask)} != grid.n_cells {grid.n_cells}"
        )

    centroids_gdf = grid.centroid_gdf()
    H, W = grid.shape
    rows = np.repeat(np.arange(H), W)
    cols = np.tile(np.arange(W), H)

    centroids_4326 = centroids_gdf.to_crs("EPSG:4326")
    lons = centroids_4326.geometry.x.to_numpy()
    lats = centroids_4326.geometry.y.to_numpy()

    mask_arr = mask.to_numpy()
    keep = mask_arr > 0
    return pd.DataFrame({
        "row": rows[keep],
        "col": cols[keep],
        "lon": lons[keep],
        "lat": lats[keep],
        "label": np.full(keep.sum(), label_value, dtype=np.int32),
        "weight": np.full(keep.sum(), weight, dtype=np.float32),
        "source": np.full(keep.sum(), source, dtype=object),
    })
