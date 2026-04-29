"""Aerial-imagery overlay for Bear Cub drill holes.

Plots hole positions + claim outline + Voronoi pay-zone cells on top of a
high-resolution satellite basemap (ESRI World Imagery via contextily). The
Bear Cub patent corners are used as the claim polygon.

Outputs:
  data/derived/bear_cub_resource/fig_aerial_overlay.png       (clean aerial + holes)
  data/derived/bear_cub_resource/fig_aerial_pay_zone_grade.png (aerial + Voronoi-grade cells)

Run:
    uv run python tools/bear_cub_aerial_overlay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

from ai_minerals.bear_cub.grade import grade_to_tweet_color

REPO = Path(__file__).resolve().parents[1]
ROLLUPS = REPO / "data" / "derived" / "bear_cub_resource" / "hole_rollups.csv"
OUT_DIR = REPO / "data" / "derived" / "bear_cub_resource"

# Bear Cub MS 1178 corners in WGS84 (per SOURCE.md)
CORNERS_WGS = {
    "TL": (64.531854, -165.341649),  # Corner 1, NW
    "TR": (64.532556, -165.338029),  # Corner 2, NE
    "BR": (64.531226, -165.332208),  # Corner 3, SE
    "BL": (64.530167, -165.335444),  # Corner 4, SW
}

# Margin around claim extent (degrees) for the basemap window
LAT_MARGIN = 0.0008
LON_MARGIN = 0.0015


def main() -> None:
    rollups = pd.read_csv(ROLLUPS)
    valid = rollups.dropna(subset=["lat_wgs84", "lon_wgs84"]).copy()
    print(f"Loaded {len(valid)} holes with WGS84 coords")

    # WGS84 → Web Mercator (EPSG:3857) for contextily
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    valid["x_merc"], valid["y_merc"] = to_merc.transform(
        valid["lon_wgs84"].values, valid["lat_wgs84"].values
    )

    corner_xs = []
    corner_ys = []
    for k in ("TL", "TR", "BR", "BL", "TL"):  # close the loop
        lat, lon = CORNERS_WGS[k]
        x, y = to_merc.transform(lon, lat)
        corner_xs.append(x)
        corner_ys.append(y)

    # Map extent: claim bbox + margin (in degrees, then convert)
    lats = [v[0] for v in CORNERS_WGS.values()]
    lons = [v[1] for v in CORNERS_WGS.values()]
    lat_min = min(lats) - LAT_MARGIN
    lat_max = max(lats) + LAT_MARGIN
    lon_min = min(lons) - LON_MARGIN
    lon_max = max(lons) + LON_MARGIN
    x_min, y_min = to_merc.transform(lon_min, lat_min)
    x_max, y_max = to_merc.transform(lon_max, lat_max)

    # ---------------- Figure 1: clean aerial + holes ---------------- #
    print("Rendering aerial overlay (clean) ...")
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add ESRI World Imagery basemap
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.Esri.WorldImagery,
                    zoom=18, attribution=False)

    # Claim outline
    ax.plot(corner_xs, corner_ys, color="red", linewidth=2.5,
            label="Bear Cub MS 1178")

    # Holes color-coded by pay-zone avg grade
    for _, h in valid.iterrows():
        color = (grade_to_tweet_color(h.pay_zone_avg_grade)
                 if h.get("pay_zone_thickness_ft", 0) > 0 else "white")
        ax.scatter(h.x_merc, h.y_merc, s=110, c=color,
                   edgecolor="black", linewidth=0.8, zorder=3)
        # Hole-id label
        ax.annotate(h.file_stem.split()[-1].replace("H", ""),
                    (h.x_merc, h.y_merc), fontsize=6,
                    ha="left", va="bottom", xytext=(5, 5),
                    textcoords="offset points", zorder=4,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55,
                              edgecolor="none", pad=1))

    ax.set_axis_off()
    ax.set_title(
        "Bear Cub MS 1178 — drill holes on satellite imagery\n"
        "Marker color = pay-zone avg grade (Tweet scheme)\n"
        "Imagery © Esri / USGS / GeoEye",
        fontsize=11,
    )
    ax.legend(loc="lower left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_aerial_overlay.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT_DIR / 'fig_aerial_overlay.png').relative_to(REPO)}")

    # ---------------- Figure 2: aerial + Voronoi-grade cells ---------------- #
    print("Rendering aerial overlay with Voronoi pay-zone cells ...")
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.Esri.WorldImagery,
                    zoom=18, attribution=False)

    # Voronoi cells in Web Mercator, clipped to claim outline
    interior = valid[valid["pay_zone_thickness_ft"] > 0].copy()
    pts = interior[["x_merc", "y_merc"]].values
    if len(pts) >= 4:
        vor = Voronoi(pts)
        claim_poly = Polygon(list(zip(corner_xs[:-1], corner_ys[:-1])))
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            verts = vor.vertices[region]
            try:
                cell = Polygon(verts).intersection(claim_poly)
            except Exception:
                continue
            if cell.is_empty or cell.area <= 0:
                continue
            grades = float(interior.iloc[i]["pay_zone_avg_grade"])
            color = grade_to_tweet_color(grades)
            geoms = [cell] if cell.geom_type == "Polygon" else list(cell.geoms)
            for g in geoms:
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, color=color, alpha=0.45,
                        edgecolor="black", linewidth=0.4)

    # Holes on top
    for _, h in valid.iterrows():
        color = (grade_to_tweet_color(h.pay_zone_avg_grade)
                 if h.get("pay_zone_thickness_ft", 0) > 0 else "white")
        ax.scatter(h.x_merc, h.y_merc, s=85, c=color,
                   edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(h.file_stem.split()[-1].replace("H", ""),
                    (h.x_merc, h.y_merc), fontsize=5,
                    ha="left", va="bottom", xytext=(4, 4),
                    textcoords="offset points", zorder=4,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5,
                              edgecolor="none", pad=1))

    # Claim outline last so it's on top
    ax.plot(corner_xs, corner_ys, color="red", linewidth=2.5,
            label="Bear Cub MS 1178")

    ax.set_axis_off()
    ax.set_title(
        "Bear Cub — Voronoi pay-zone-grade cells over satellite imagery\n"
        "Cell color = pay-zone avg grade (Tweet scheme)",
        fontsize=11,
    )
    ax.legend(loc="lower left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_aerial_pay_zone_grade.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT_DIR / 'fig_aerial_pay_zone_grade.png').relative_to(REPO)}")


if __name__ == "__main__":
    main()
