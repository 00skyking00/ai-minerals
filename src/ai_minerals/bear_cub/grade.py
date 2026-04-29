"""Per-interval Au-grade computation + resource estimation for Bear Cub.

Implements the methodology Tweet and Jesse use, derived from the *Handbook for
the Alaskan Prospector* ch. 14 + the operator-side conventions captured
empirically from our 24-log corpus:

  grade (oz/cu yd) = (mg × fineness × 27) / (31103.5 × bit_area_sq_ft × interval_ft)

where 31103.5 mg/troy oz, 27 cu ft/cu yd, and `fineness` is the gold purity
(0.890 for Bear Cub per the empirical "890 Fineness" annotation found on
multiple back-of-sheet calculations).

Two block-volumetric methods are implemented for resource estimation:

- ``polygon_method``: Voronoi tessellation around drill collars; each
  polygon's value is the centroid-hole's grade × area × depth.
- ``triangle_method``: Delaunay triangulation; each triangle's value is the
  three-vertex average grade × area × depth.

These match Tweet's TOP→BR+2 framework and the handbook's three-point + diamond
methods (handbook ch. 14, p. 260).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Conversion + property constants
MG_PER_TROY_OZ = 31103.5
CU_FT_PER_CU_YD = 27.0
DEFAULT_FINENESS = 0.890        # Bear Cub: empirical from "890 Fineness" annotation
DEFAULT_BIT_DIAMETER_IN = 5.5   # 5" / 6" casing average if unparseable


# Jesse's color scheme for hole-level pay-zone classification (from cover letter)
JESSE_GRADE_BINS = [
    (0.000, 0.005, "white",   "trace"),
    (0.005, 0.010, "lightblue", "marginal"),
    (0.010, 0.020, "yellow",  "low pay"),
    (0.020, 0.050, "orange",  "medium pay"),
    (0.050, 0.100, "red",     "good pay"),
    (0.100, 999.0, "magenta", "high pay"),
]

# Tweet's color scheme for surface-to-BOP grade map (from polygon analysis)
TWEET_GRADE_BINS = [
    (0.0000, 0.002,  "#08306b", "0.0000–0.002"),    # dark blue
    (0.002,  0.004,  "#9ecae1", "0.002–0.004"),     # light blue
    (0.004,  0.006,  "#fff7bc", "0.004–0.006"),     # cream
    (0.006,  0.008,  "#fdae6b", "0.006–0.008"),     # peach
    (0.008,  0.015,  "#e6550d", "0.008–0.015"),     # red
    (0.015,  999.0,  "#67000d", "0.015+"),          # dark red
]


# =============================================================================
# Casing diameter parsing
# =============================================================================

_FRACTION_RE = re.compile(r"(\d+)\s+(\d+)/(\d+)")
_BARE_FRACTION_RE = re.compile(r"(\d+)/(\d+)")
_DECIMAL_RE = re.compile(r"\d+\.\d+|\d+")


def _parse_inches_value(s: str) -> float | None:
    """Parse a single inch-value like '5 1/2', '5.625', '5 11/16'."""
    s = s.replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4")
    s = s.replace('"', "").strip()
    m = _FRACTION_RE.search(s)
    if m:
        return int(m[1]) + int(m[2]) / int(m[3])
    m = _BARE_FRACTION_RE.search(s)
    if m:
        return int(m[1]) / int(m[2])
    m = _DECIMAL_RE.search(s)
    if m:
        try:
            return float(m[0])
        except ValueError:
            return None
    return None


def hole_avg_bit_diameter_in(casing_text: str | None) -> float:
    """Best-effort: extract a single representative bit/casing diameter (in)
    from the operator's free-text notes.

    Many logs have multi-zone casings ("5 11/16-5 1/8; 5 9/16-5 1/4"); this
    simplification averages all numeric inch-values found. For Tweet-grade
    accuracy we'd need depth-aware lookup — flagged as future work.
    """
    if not casing_text or not isinstance(casing_text, str):
        return DEFAULT_BIT_DIAMETER_IN

    # Find every inch-like token and average plausible ones (3"-9" range)
    candidates: list[float] = []
    # Mixed fractions first
    for m in _FRACTION_RE.finditer(casing_text):
        v = int(m[1]) + int(m[2]) / int(m[3])
        if 3.0 <= v <= 9.0:
            candidates.append(v)
    # Then bare decimals (avoiding double-counting whole parts of mixed)
    masked = _FRACTION_RE.sub(" ", casing_text)
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", masked):
        try:
            v = float(m[1])
            if 3.0 <= v <= 9.0:
                candidates.append(v)
        except ValueError:
            pass

    if not candidates:
        return DEFAULT_BIT_DIAMETER_IN
    return float(np.mean(candidates))


def bit_area_sq_ft(bit_diameter_in: float) -> float:
    """Cross-sectional area (sq ft) for a circular bit/shoe."""
    return math.pi * (bit_diameter_in / 2.0 / 12.0) ** 2


# =============================================================================
# Per-interval grade
# =============================================================================


def interval_grade_oz_per_cu_yd(
    weight_mg: float,
    interval_ft: float,
    bit_area_ft2: float,
    fineness: float = DEFAULT_FINENESS,
) -> float:
    """Compute interval grade in fine troy ounces per cubic yard.

    grade = (mg × fineness × 27 cu_ft/cu_yd) /
            (31103.5 mg/oz × bit_area_sq_ft × interval_ft)
    """
    if (
        weight_mg is None or pd.isna(weight_mg)
        or interval_ft is None or pd.isna(interval_ft) or interval_ft <= 0
        or bit_area_ft2 <= 0
    ):
        return 0.0
    fine_oz = float(weight_mg) * fineness / MG_PER_TROY_OZ
    volume_cu_ft = bit_area_ft2 * float(interval_ft)
    volume_cu_yd = volume_cu_ft / CU_FT_PER_CU_YD
    return fine_oz / volume_cu_yd if volume_cu_yd > 0 else 0.0


def add_grades_to_intervals(
    intervals: pd.DataFrame, collars: pd.DataFrame, fineness: float = DEFAULT_FINENESS
) -> pd.DataFrame:
    """Augment intervals with grade_oz_per_cu_yd, interval_ft, bit_area_ft2."""
    out = intervals.copy()

    # Per-hole bit area lookup
    hole_bit_dia = {
        r.file_stem: hole_avg_bit_diameter_in(r.casing_or_bit_diameter_text)
        for _, r in collars.iterrows()
    }
    out["bit_diameter_in"] = out["file_stem"].map(hole_bit_dia)
    out["bit_area_ft2"] = out["bit_diameter_in"].apply(bit_area_sq_ft)
    out["interval_ft"] = (out["depth_to_ft"] - out["depth_from_ft"]).astype(float)
    out["grade_oz_per_cu_yd"] = [
        interval_grade_oz_per_cu_yd(w, l, a, fineness)
        for w, l, a in zip(out["estimated_weight_mg"], out["interval_ft"], out["bit_area_ft2"])
    ]
    return out


# =============================================================================
# Hole-level rollups
# =============================================================================


@dataclass
class HoleGradeRollup:
    file_stem: str
    bedrock_depth_ft: float
    surface_to_br_grade: float        # vertically-integrated grade over [0, BR]
    pay_zone_grade: float             # max grade interval
    pay_zone_top_ft: float            # depth where pay-zone (>0.005 oz/cuyd) starts
    pay_zone_bottom_ft: float         # depth where pay-zone ends
    pay_zone_thickness_ft: float
    total_oz_in_hole: float           # sum mg × fineness across all intervals → oz


def hole_rollups(intervals: pd.DataFrame, collars: pd.DataFrame, fineness: float = DEFAULT_FINENESS) -> pd.DataFrame:
    """Per-hole grade summary statistics."""
    rows: list[dict] = []
    for fs, g in intervals.groupby("file_stem"):
        coll = collars[collars.file_stem == fs]
        bedrock = float(coll["depth_to_bedrock_ft"].iloc[0]) if len(coll) and pd.notna(coll["depth_to_bedrock_ft"].iloc[0]) else 0.0
        bit_area = float(g["bit_area_ft2"].iloc[0]) if len(g) else bit_area_sq_ft(DEFAULT_BIT_DIAMETER_IN)

        g = g.sort_values("depth_from_ft")
        # Surface-to-BR grade: total fine oz in [0, BR] / total volume in [0, BR]
        in_br = g[g["depth_to_ft"] <= bedrock] if bedrock > 0 else g
        if len(in_br) > 0:
            total_mg = (in_br["estimated_weight_mg"].fillna(0) * fineness).sum()
            total_vol_cuyd = (in_br["interval_ft"].fillna(0) * bit_area).sum() / CU_FT_PER_CU_YD
            sbr_grade = float(total_mg / MG_PER_TROY_OZ / total_vol_cuyd) if total_vol_cuyd > 0 else 0.0
            total_oz = float(total_mg / MG_PER_TROY_OZ)
        else:
            sbr_grade = 0.0
            total_oz = 0.0

        # Pay zone: HIGHEST-DENSITY contiguous slice (Tweet-style sliding window).
        # Bear Cub typical pay zones are 4-20 ft thick; we constrain max thickness
        # to 20 ft to avoid the degenerate "whole-hole pay zone" that arises when
        # color-distributed mg spreads small amounts across many intervals.
        sorted_g = g.sort_values("depth_from_ft").reset_index(drop=True)
        max_payzone_ft = 20.0
        min_payzone_ft = 2.0
        best_density = 0.0
        best = None
        for i in range(len(sorted_g)):
            cum_mg = 0.0
            cum_thick = 0.0
            for j in range(i, len(sorted_g)):
                row_mg = float(sorted_g.iloc[j]["estimated_weight_mg"] or 0)
                row_iv = float(sorted_g.iloc[j]["interval_ft"] or 0)
                cum_mg += row_mg
                cum_thick += row_iv
                if cum_thick > max_payzone_ft:
                    break
                if cum_thick < min_payzone_ft:
                    continue
                density = cum_mg / cum_thick if cum_thick > 0 else 0.0
                if density > best_density:
                    best_density = density
                    best = (i, j, cum_mg, cum_thick)
        if best is not None:
            i, j, pay_total_mg, pay_span_ft = best
            pay_top = float(sorted_g.iloc[i]["depth_from_ft"])
            pay_bot = float(sorted_g.iloc[j]["depth_to_ft"])
            pay_grade = float(sorted_g.iloc[i:j+1]["grade_oz_per_cu_yd"].max())
            pay_grade_avg = (
                pay_total_mg * fineness / MG_PER_TROY_OZ
                / (pay_span_ft * bit_area / CU_FT_PER_CU_YD)
                if pay_span_ft > 0 and bit_area > 0 else 0.0
            )
        else:
            pay_top, pay_bot, pay_grade, pay_grade_avg = 0.0, 0.0, 0.0, 0.0

        rows.append({
            "file_stem": fs,
            "bedrock_depth_ft": bedrock,
            "surface_to_br_grade": sbr_grade,
            "pay_zone_grade": pay_grade,
            "pay_zone_avg_grade": pay_grade_avg,
            "pay_zone_top_ft": pay_top,
            "pay_zone_bottom_ft": pay_bot,
            "pay_zone_thickness_ft": pay_bot - pay_top,
            "total_fine_oz_in_hole": total_oz,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Block-volumetric resource estimation (Polygon + Triangle)
# =============================================================================


def polygon_resource(
    points_xy: np.ndarray,           # (N, 2) hole positions in ft
    grades: np.ndarray,              # (N,) hole-level grade in oz/cu yd
    depths: np.ndarray,              # (N,) bedrock depths in ft
    bbox: tuple[float, float, float, float],  # (xmin, ymin, xmax, ymax) bbox for ghost-point sizing
    clip_polygon=None,               # optional shapely Polygon — clips cells to claim shape, not the bounding rectangle
) -> dict:
    """Voronoi-polygon block-volumetric estimate.

    Each hole owns a Voronoi cell (clipped to `clip_polygon` if provided, else
    to the rectangular `bbox`). The cell's contained gold is
    grade × cell_area_sq_ft × hole_depth_ft / 27.

    Boundary holes (those on the convex hull of the point set) have unbounded
    Voronoi regions in the standard tessellation. We add 4 far-away "ghost"
    points outside the bbox to ensure every real hole gets a bounded cell;
    the ghost cells themselves are discarded.
    """
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, box

    clip = clip_polygon if clip_polygon is not None else box(*bbox)
    xmin, ymin, xmax, ymax = bbox
    pad = max(xmax - xmin, ymax - ymin) * 10  # far enough to dominate the tessellation
    ghosts = np.array([
        [xmin - pad, ymin - pad],
        [xmax + pad, ymin - pad],
        [xmax + pad, ymax + pad],
        [xmin - pad, ymax + pad],
    ])
    n_real = len(points_xy)
    augmented_points = np.vstack([points_xy, ghosts])
    vor = Voronoi(augmented_points)
    rows: list[dict] = []
    for i in range(n_real):  # only iterate real holes, not ghosts
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            # Should be rare with ghost padding, but skip if it still happens
            continue
        verts = vor.vertices[region]
        try:
            poly = Polygon(verts).intersection(clip)
        except Exception:
            continue
        if poly.is_empty or poly.area <= 0:
            continue
        area_ft2 = float(poly.area)
        vol_cuyd = area_ft2 * float(depths[i]) / CU_FT_PER_CU_YD
        oz = float(grades[i]) * vol_cuyd
        rows.append({
            "hole_idx": i, "area_ft2": area_ft2,
            "depth_ft": float(depths[i]), "volume_cuyd": vol_cuyd,
            "grade_oz_per_cuyd": float(grades[i]), "fine_troy_oz": oz,
        })
    df = pd.DataFrame(rows)
    return {
        "per_polygon": df,
        "total_area_ft2": float(df["area_ft2"].sum()),
        "total_volume_cuyd": float(df["volume_cuyd"].sum()),
        "total_fine_oz": float(df["fine_troy_oz"].sum()),
        "weighted_avg_grade": (
            float(df["fine_troy_oz"].sum() / df["volume_cuyd"].sum())
            if df["volume_cuyd"].sum() > 0 else 0.0
        ),
    }


def triangle_resource(
    points_xy: np.ndarray, grades: np.ndarray, depths: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> dict:
    """Delaunay-triangle block-volumetric estimate.

    Each triangle's value = avg-of-3-vertex-grades × triangle_area × avg-depth / 27.
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import Polygon, box

    clip = box(*bbox)
    tri = Delaunay(points_xy)
    rows: list[dict] = []
    for simplex in tri.simplices:
        verts = points_xy[simplex]
        try:
            poly = Polygon(verts).intersection(clip)
        except Exception:
            continue
        if poly.is_empty or poly.area <= 0:
            continue
        area_ft2 = float(poly.area)
        avg_grade = float(np.mean(grades[simplex]))
        avg_depth = float(np.mean(depths[simplex]))
        vol_cuyd = area_ft2 * avg_depth / CU_FT_PER_CU_YD
        oz = avg_grade * vol_cuyd
        rows.append({
            "vertices": tuple(int(s) for s in simplex), "area_ft2": area_ft2,
            "avg_depth_ft": avg_depth, "volume_cuyd": vol_cuyd,
            "avg_grade_oz_per_cuyd": avg_grade, "fine_troy_oz": oz,
        })
    df = pd.DataFrame(rows)
    return {
        "per_triangle": df,
        "total_area_ft2": float(df["area_ft2"].sum()),
        "total_volume_cuyd": float(df["volume_cuyd"].sum()),
        "total_fine_oz": float(df["fine_troy_oz"].sum()),
        "weighted_avg_grade": (
            float(df["fine_troy_oz"].sum() / df["volume_cuyd"].sum())
            if df["volume_cuyd"].sum() > 0 else 0.0
        ),
    }


# =============================================================================
# Color helpers
# =============================================================================


def grade_to_jesse_color(g: float) -> str:
    for lo, hi, color, _ in JESSE_GRADE_BINS:
        if lo <= g < hi:
            return color
    return "magenta"


def grade_to_tweet_color(g: float) -> str:
    for lo, hi, color, _ in TWEET_GRADE_BINS:
        if lo <= g < hi:
            return color
    return TWEET_GRADE_BINS[-1][2]
