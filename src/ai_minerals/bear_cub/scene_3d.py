"""3D scene assembly for the Bear Cub Murray drill subset.

Renders the 24 drill collars + an IDW-interpolated bedrock surface, exports
to interactive HTML for embedding in the integrated notebook.

Coord system: local UTM-like meters, anchored on the BL/SW BLM monument
(WGS84 64.530167, -165.335444). Surface elevation is held constant at 0 m
visually; depths plot downward in negative-z. (We're modeling the placer
deposit's bedrock topology, not regional terrain — so a flat datum reads
cleanly.)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import RBFInterpolator


REPO = Path(__file__).resolve().parents[3]
COLLARS_CSV = REPO / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"

# BL/Corner 4 GPS-anchored origin
ORIGIN_LAT = 64.530167
ORIGIN_LON = -165.335444
FT_PER_DEG_LAT = 364400.0
FT_PER_DEG_LON = FT_PER_DEG_LAT * math.cos(math.radians(ORIGIN_LAT))
FT_PER_M = 3.28084


def latlon_to_local_m(lat: float, lon: float) -> tuple[float, float]:
    """Project lat/lon to local meters anchored at the BL monument."""
    dx_ft = (lon - ORIGIN_LON) * FT_PER_DEG_LON
    dy_ft = (lat - ORIGIN_LAT) * FT_PER_DEG_LAT
    return dx_ft / FT_PER_M, dy_ft / FT_PER_M


def build_scene(out_html: Path | None = None) -> pv.Plotter:
    """Build the 3D scene (drill collars + bedrock surface). Returns a Plotter."""
    df = pd.read_csv(COLLARS_CSV)
    # local x/y meters
    xy = np.array([latlon_to_local_m(r.lat_wgs84, r.lon_wgs84) for r in df.itertuples()])
    df["x_m"] = xy[:, 0]
    df["y_m"] = xy[:, 1]
    df["bedrock_depth_m"] = df["bedrock_depth_ft"].astype(float) / FT_PER_M
    df["total_depth_m"] = df["total_depth_ft"].astype(float) / FT_PER_M

    # IDW / RBF bedrock surface from holes that reached bedrock
    has_br = df["bedrock_depth_ft"].notna() & (df["bedrock_depth_ft"] > 0)
    pts = df.loc[has_br, ["x_m", "y_m"]].to_numpy()
    z_br = -df.loc[has_br, "bedrock_depth_m"].to_numpy()  # negative = downward

    # Make a regular grid spanning the cluster + a margin
    margin_m = 30.0
    xg = np.linspace(pts[:, 0].min() - margin_m, pts[:, 0].max() + margin_m, 80)
    yg = np.linspace(pts[:, 1].min() - margin_m, pts[:, 1].max() + margin_m, 80)
    XG, YG = np.meshgrid(xg, yg)
    grid_pts = np.column_stack([XG.ravel(), YG.ravel()])
    rbf = RBFInterpolator(pts, z_br, kernel="thin_plate_spline", smoothing=2.0)
    ZG = rbf(grid_pts).reshape(XG.shape)

    # Build PyVista grid surface
    surface = pv.StructuredGrid(XG, YG, ZG)
    surface["bedrock_depth_m"] = -ZG.ravel(order="F")  # positive depth values

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.add_mesh(
        surface,
        cmap="viridis_r",
        opacity=0.85,
        scalar_bar_args={"title": "Bedrock depth (m)", "n_labels": 5},
    )

    # Drill collars as colored vertical lines from surface to total depth
    for r in df.itertuples():
        if not (r.total_depth_ft and r.total_depth_ft > 0):
            continue
        x, y = r.x_m, r.y_m
        depth_m = r.total_depth_ft / FT_PER_M
        line = pv.Line((x, y, 0.0), (x, y, -depth_m))
        plotter.add_mesh(line, color="firebrick", line_width=4)

        # Mark bedrock contact with a small sphere
        if r.bedrock_depth_ft and r.bedrock_depth_ft > 0:
            sphere = pv.Sphere(radius=2.0, center=(x, y, -r.bedrock_depth_ft / FT_PER_M))
            plotter.add_mesh(sphere, color="orange")

        # Hole-id label at surface
        plotter.add_point_labels(
            np.array([[x, y, 1.0]]),
            [str(r.file_stem)],
            font_size=10,
            point_size=3,
            text_color="black",
            shape_opacity=0.6,
        )

    plotter.add_axes()
    plotter.show_grid(color="gray")
    plotter.set_background("white")
    plotter.camera_position = [(80, -120, 60), (40, 40, -25), (0, 0, 1)]

    if out_html is not None:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        plotter.export_html(str(out_html))
        print(f"Saved → {out_html.relative_to(REPO)}")

    return plotter


if __name__ == "__main__":
    out = REPO / "data" / "derived" / "bear_cub_3d.html"
    build_scene(out_html=out)
