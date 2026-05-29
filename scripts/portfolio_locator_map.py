"""Build the top-of-page locator map for index.qmd.

A small North America banner showing the regions touched by the
portfolio: Bear Cub at Nome (Ch 1, separate methodology), plus the
four regional MPM pipelines (Ch 2). One image, no chrome, banner
aspect — sits under the title on the landing page.

Natural Earth 50m countries shapefile is fetched once from
naciscdn.org and cached under data/raw/natural_earth/ (gitignored).
"""
from __future__ import annotations
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

NE_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
NE_CACHE = Path("data/raw/natural_earth/ne_50m_admin_0_countries.zip")
OUT = Path("data/derived/portfolio_charts/region_locator_map.png")

NA_NAMES = {"United States of America", "Canada", "Mexico"}

REGIONS = [
    {"name": "Bear Cub",          "lat": 64.50, "lon": -165.40, "color": "#d7301f"},
    {"name": "Tanacross",         "lat": 63.40, "lon": -143.30, "color": "#2c7bb6"},
    {"name": "BC Golden Triangle", "lat": 56.92, "lon": -130.05, "color": "#2c7bb6"},
    {"name": "Mother Lode",        "lat": 38.05, "lon": -120.50, "color": "#2c7bb6"},
    {"name": "Arizona",            "lat": 32.20, "lon": -110.70, "color": "#2c7bb6"},
]


def get_countries() -> gpd.GeoDataFrame:
    if not NE_CACHE.exists():
        NE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {NE_URL} -> {NE_CACHE}")
        urllib.request.urlretrieve(NE_URL, NE_CACHE)
    return gpd.read_file(f"zip://{NE_CACHE}")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    world = get_countries()
    na = world[world["NAME"].isin(NA_NAMES)]

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=160)
    na.plot(ax=ax, color="#f4ede0", edgecolor="#9a8e7a", linewidth=0.7)

    for r in REGIONS:
        ax.plot(r["lon"], r["lat"], "o",
                color=r["color"], markersize=11,
                markeredgecolor="#222", markeredgewidth=0.9, zorder=5)
        ax.annotate(
            r["name"], (r["lon"], r["lat"]),
            xytext=(9, 8), textcoords="offset points",
            fontsize=9.5, ha="left", va="bottom",
            color="#222",
            bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec="#bbb", lw=0.5, alpha=0.85),
        )

    ax.set_xlim(-172, -68)
    ax.set_ylim(22, 73)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_aspect(1.4)  # mild equirectangular squish so AK looks reasonable

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
