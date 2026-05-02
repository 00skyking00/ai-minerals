"""Bear Cub 3D model: bedrock surface + grade-colored drill sticks.

Builds a 3D scene from the 24 drill collars by:

1. Interpolating the bedrock surface across the claim quadrilateral
   using a thin-plate spline (RBF) on the 24 holes' bedrock depths.
   Three NBR holes carry their KNN-IDW imputed bedrock from the
   resource pipeline (see ``tools/bear_cub_resource_analysis.py``);
   the imputation is documented in ``internal.qmd``'s bedrock review.
2. Building a vertical drill stick per hole, segmented at each
   captured per-2-ft interval, colored by ``grade_oz_per_cu_yd``
   on the same Tweet color scheme used by the 2D figures.
3. Drawing the patent quadrilateral outline at the surface (z=0).
4. Optionally drawing the pay-zone interval as a highlighted band on
   each stick.

The default orientation is oblique-from-south with 8× vertical
exaggeration — the claim is ~600 × 1100 ft in plan but ~80 ft deep,
so depth features only register at 5-10× vertical scale.

Convention: z is *depth* below surface (positive downward in the data,
flipped to negative-z in the scene so 'up' is geologically up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

VERTICAL_EXAGGERATION = 8.0


# ---------------------------------------------------------------------------
# Data loaders + utilities
# ---------------------------------------------------------------------------


@dataclass
class BearCubInputs:
    """All the per-hole + per-interval data the 3D scene needs."""

    holes: pd.DataFrame             # one row per hole (rollups + collar)
    intervals: pd.DataFrame         # one row per captured interval
    corners_local: np.ndarray       # (4, 2) patent corners in local feet, CCW

    @property
    def n_holes(self) -> int:
        return len(self.holes)


def load_inputs(repo_root: Path | str = ".") -> BearCubInputs:
    """Load hole_rollups + intervals_with_grade + claim corners."""
    repo = Path(repo_root)
    holes = pd.read_csv(
        repo / "data/derived/bear_cub_resource/hole_rollups.csv"
    )
    intervals = pd.read_parquet(
        repo / "data/derived/bear_cub_resource/intervals_with_grade.parquet"
    )

    # Derive the claim quadrilateral in local-feet coords from the
    # WGS84 corners stored in src/ai_minerals/bear_cub/georef.py. We
    # transform via the same lat/lon → local approximation that
    # tools/bear_cub_resource_analysis.py uses.
    from ai_minerals.bear_cub.georef import MS_1178_CORNERS

    lat0 = float(holes["lat_wgs84"].mean())
    lon0 = float(holes["lon_wgs84"].mean())
    import math
    ft_per_lat = 364400.0
    ft_per_lon = ft_per_lat * math.cos(math.radians(lat0))

    # Project the patent corners into the same local-feet frame as the
    # holes, then re-anchor to the claim's actual mean easting/northing.
    corners_xy_local_origin = []
    for k in ("TL", "TR", "BR", "BL"):
        lat, lon = MS_1178_CORNERS[k]
        x = (lon - lon0) * ft_per_lon
        y = (lat - lat0) * ft_per_lat
        corners_xy_local_origin.append((x, y))
    corners_xy_local_origin = np.array(corners_xy_local_origin)

    # Re-anchor: translate so the centroid of corners aligns with the
    # centroid of hole easting/northing positions.
    corner_centroid = corners_xy_local_origin.mean(axis=0)
    holes_centroid_e = float(holes["easting_local_ft"].mean())
    holes_centroid_n = float(holes["northing_local_ft"].mean())
    shift = np.array([holes_centroid_e - corner_centroid[0],
                      holes_centroid_n - corner_centroid[1]])
    corners_local = corners_xy_local_origin + shift

    return BearCubInputs(holes=holes, intervals=intervals,
                         corners_local=corners_local)


# ---------------------------------------------------------------------------
# Bedrock surface
# ---------------------------------------------------------------------------


def fit_bedrock_surface(
    holes: pd.DataFrame,
    n_grid: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a thin-plate-spline bedrock surface and sample it on a regular grid.

    Returns (xx, yy, zz) in local feet. ``zz`` is *depth* (positive down).
    Caller is responsible for negating z when rendering with PyVista.
    """
    from scipy.interpolate import RBFInterpolator

    pts = holes[["easting_local_ft", "northing_local_ft"]].values
    z = holes["bedrock_depth_ft"].values

    rbf = RBFInterpolator(pts, z, kernel="thin_plate_spline", smoothing=2.0)

    xmin, ymin = pts.min(axis=0) - 50
    xmax, ymax = pts.max(axis=0) + 50
    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    zz = rbf(grid_pts).reshape(xx.shape)
    return xx, yy, zz


# ---------------------------------------------------------------------------
# Drill stick geometry
# ---------------------------------------------------------------------------


def grade_color(grade: float) -> tuple[float, float, float]:
    """Map an interval grade (oz/cu yd) to an RGB triplet on the Jesse
    color scheme used in the 2D figures.

    Bands: white < 0.005, light blue 0.005–0.010, yellow 0.010–0.020,
    orange 0.020–0.050, red 0.050–0.100, magenta > 0.100.
    """
    if grade < 0.005:
        return (1.0, 1.0, 1.0)
    if grade < 0.010:
        return (0.68, 0.85, 0.90)        # light blue
    if grade < 0.020:
        return (1.0, 0.92, 0.23)         # yellow
    if grade < 0.050:
        return (1.0, 0.65, 0.0)          # orange
    if grade < 0.100:
        return (0.86, 0.08, 0.24)        # red
    return (1.0, 0.0, 1.0)               # magenta


def hole_segments(
    hole: pd.Series,
    intervals: pd.DataFrame,
) -> list[dict]:
    """Return one dict per per-2-ft interval for a hole. Each dict has
    ``z_top`` (depth at top, positive), ``z_bot``, ``grade``, and
    ``color`` ready for PyVista.
    """
    sub = intervals[intervals.file_stem == hole.file_stem].sort_values(
        "depth_from_ft"
    )
    out: list[dict] = []
    for _, r in sub.iterrows():
        out.append({
            "z_top": float(r.depth_from_ft),
            "z_bot": float(r.depth_to_ft),
            "grade": float(r.grade_oz_per_cu_yd),
            "color": grade_color(float(r.grade_oz_per_cu_yd)),
        })
    return out


# ---------------------------------------------------------------------------
# PyVista scene assembly
# ---------------------------------------------------------------------------


def build_scene(
    inputs: BearCubInputs,
    *,
    z_exag: float = VERTICAL_EXAGGERATION,
    stick_radius: float = 6.0,
    bedrock_alpha: float = 0.45,
):
    """Build the PyVista scene without rendering.

    Returns (plotter, scene_dict) where ``scene_dict`` contains references
    to the meshes so the caller can do post-build customization (e.g.,
    swap camera positions for cross-sections without rebuilding).

    `z_exag` multiplies the vertical axis to compensate for Bear Cub's
    plan-to-depth aspect ratio (~10:1).
    """
    import pyvista as pv

    holes = inputs.holes
    intervals = inputs.intervals
    corners = inputs.corners_local

    pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pl.set_background("#f4f4f4")

    # ---------- Bedrock surface ----------
    xx, yy, zz = fit_bedrock_surface(holes)
    bedrock_mesh = pv.StructuredGrid(xx, yy, -zz * z_exag)
    bedrock_mesh["depth_ft"] = zz.ravel(order="F")  # positive depth, for cmap
    pl.add_mesh(
        bedrock_mesh, scalars="depth_ft",
        cmap="terrain", opacity=bedrock_alpha,
        show_scalar_bar=True, scalar_bar_args={
            "title": "Bedrock depth (ft)",
            "n_labels": 5,
            "fmt": "%.0f",
            "position_x": 0.85, "position_y": 0.05,
            "width": 0.04, "height": 0.4,
        },
    )

    # ---------- Drill sticks (per-interval colored cylinders) ----------
    for _, hole in holes.iterrows():
        x0 = float(hole.easting_local_ft)
        y0 = float(hole.northing_local_ft)
        for seg in hole_segments(hole, intervals):
            # Cylinder along z-axis from z=top to z=bot
            center_z = -((seg["z_top"] + seg["z_bot"]) / 2.0) * z_exag
            height = (seg["z_bot"] - seg["z_top"]) * z_exag
            cyl = pv.Cylinder(
                center=(x0, y0, center_z),
                direction=(0, 0, 1),
                radius=stick_radius,
                height=height,
                resolution=12,
            )
            pl.add_mesh(cyl, color=seg["color"], smooth_shading=True)
        # Surface marker (small sphere at top) so the hole is visible
        # even when intervals are all white (barren).
        pl.add_mesh(
            pv.Sphere(radius=stick_radius * 1.6,
                      center=(x0, y0, 0.0)),
            color="black",
        )
    # Hole-id labels for a *subset* of holes (the bedrock-contact pay
    # signature holes from internal.qmd, plus the two Hammon Prospect
    # NBR holes for context). Labeling all 24 makes the figure unreadable.
    LABEL_HOLES = {"L2 H4", "L6900 H6952", "L7300 H7350",
                   "L7700 H7754", "L7100 H7156", "L7100 H7160"}
    label_pts = []
    label_text = []
    for _, h in holes.iterrows():
        if h.file_stem in LABEL_HOLES:
            label_pts.append(
                (float(h.easting_local_ft),
                 float(h.northing_local_ft),
                 80.0)
            )
            label_text.append(h.file_stem.split()[-1].replace("H", ""))
    if label_pts:
        pl.add_point_labels(
            label_pts, label_text,
            font_size=12,
            text_color="black",
            shape_color="white",
            shape_opacity=0.85,
            point_size=0,
        )

    # ---------- Claim outline ----------
    closed = np.vstack([corners, corners[:1]])
    line = pv.lines_from_points(
        np.column_stack([closed[:, 0], closed[:, 1], np.zeros(len(closed))])
    )
    pl.add_mesh(line, color="red", line_width=4)

    # ---------- Camera default: oblique from south ----------
    cx = float(holes["easting_local_ft"].mean())
    cy = float(holes["northing_local_ft"].mean())
    cz = -float(holes["bedrock_depth_ft"].mean()) * z_exag * 0.5
    span = float(max(
        holes["easting_local_ft"].max() - holes["easting_local_ft"].min(),
        holes["northing_local_ft"].max() - holes["northing_local_ft"].min(),
    ))
    pl.camera_position = [
        (cx, cy - span * 1.5, span * 0.8),    # camera location
        (cx, cy, cz),                          # focal point
        (0, 0, 1),                             # view up
    ]

    return pl, {
        "bedrock_mesh": bedrock_mesh,
        "z_exag": z_exag,
        "center": (cx, cy, cz),
        "span": span,
    }


def render_views(
    inputs: BearCubInputs,
    out_dir: Path | str,
    *,
    interactive_html: bool = True,
) -> dict[str, Path]:
    """Render the standard set of views to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # 1. Oblique-from-south
    pl, ctx = build_scene(inputs)
    pl.add_title("Bear Cub 3D model — oblique-from-south\n"
                 f"vertical exaggeration {ctx['z_exag']}×",
                 font_size=10)
    p1 = out / "fig_3d_oblique.png"
    pl.screenshot(str(p1))
    paths["oblique"] = p1

    # 2. Profile view from west (looking east) — drops the plan view
    # since the existing 2D maps already cover plan, and the 3D plan
    # view doesn't add geological information.
    pl, ctx = build_scene(inputs)
    cx, cy, cz = ctx["center"]
    span = ctx["span"]
    pl.camera_position = [
        (cx - span * 1.6, cy, cz),
        (cx, cy, cz),
        (0, 0, 1),
    ]
    pl.add_title("Bear Cub 3D model — profile from west (E-W cross-section view)",
                 font_size=10)
    p3 = out / "fig_3d_profile_east.png"
    pl.screenshot(str(p3))
    paths["profile_east"] = p3

    # 4. Interactive HTML (static scene; widget for rotate/pan/zoom)
    if interactive_html:
        pl, _ = build_scene(inputs)
        pl.add_title("Bear Cub 3D model — interactive", font_size=10)
        html_path = out / "bear_cub_3d.html"
        pl.export_html(str(html_path))
        paths["html"] = html_path

    return paths
