"""Render the front-page locator map as a clickable SVG.

Produces data/derived/portfolio_charts/region_locator_map.svg with each
region dot wrapped in an <a xlink:href="..."> so readers can click through
to the relevant chapter page.

Projection: Albers Equal Area centred on North America, wide enough to fit
Alaska and the lower 48 in the same frame. We do the projection manually
via pyproj (cartopy isn't in this env's pin set) and draw with matplotlib.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer


REPO_ROOT = Path(__file__).resolve().parents[1]
NE_DIR = REPO_ROOT / "data" / "raw" / "natural_earth"
OUT_PATH = REPO_ROOT / "data" / "derived" / "portfolio_charts" / "region_locator_map.svg"

# Albers Equal Area conic, two standard parallels chosen to span
# southern Arizona up to interior Alaska without too much distortion.
ALBERS = CRS.from_proj4(
    "+proj=aea +lat_1=30 +lat_2=62 +lat_0=45 +lon_0=-130 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
TO_ALBERS = Transformer.from_crs(WGS84, ALBERS, always_xy=True)


@dataclass(frozen=True)
class Region:
    name: str
    lon: float
    lat: float
    href: str
    # color family: regional MPM vs placer vs Bear Cub OCR
    family: str
    label_dx: int = 8   # text offset in pixel-ish projected units
    label_dy: int = 0


REGIONS: list[Region] = [
    Region(
        name="Tanacross + Eastern Alaska",
        lon=-143.0, lat=63.5,
        href="regional.html",
        family="regional",
        label_dx=10, label_dy=4,
    ),
    Region(
        name="BC Golden Triangle",
        lon=-130.0, lat=57.0,
        href="regional.html",
        family="regional",
        label_dx=10, label_dy=0,
    ),
    Region(
        name="California Mother Lode",
        lon=-120.7, lat=38.0,
        href="regional.html",
        family="regional",
        label_dx=-10, label_dy=-6, # label to the left
    ),
    Region(
        name="Northern Sierra placer",
        lon=-120.7, lat=39.5,
        href="placer.html",
        family="placer",
        label_dx=-10, label_dy=6, # label to the left, above
    ),
    Region(
        name="Arizona SE porphyry belt",
        lon=-110.5, lat=32.0,
        href="regional.html",
        family="regional",
        label_dx=10, label_dy=0,
    ),
    Region(
        name="Bear Cub (Cape Nome)",
        lon=-165.0, lat=64.5,
        href="https://johnsondevco.com/bearcub/",
        family="bearcub",
        label_dx=10, label_dy=0,
    ),
]

FAMILY_COLORS = {
    "regional": "#c0392b", # red
    "placer":   "#d4a017", # gold
    "bearcub":  "#2c6fbb", # blue
}


def _load_countries() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(NE_DIR / "ne_50m_admin_0_countries.zip")
    return gdf[gdf["ADMIN"].isin(["United States of America", "Canada", "Mexico"])].to_crs(ALBERS)


def _load_states() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(NE_DIR / "ne_50m_admin_1_states_provinces.zip")
    return gdf[gdf["admin"].isin(["United States of America", "Canada"])].to_crs(ALBERS)


def _draw_basemap(ax: plt.Axes) -> None:
    countries = _load_countries()
    states = _load_states()

    countries.plot(
        ax=ax,
        facecolor="#e8e8e8",
        edgecolor="#888888",
        linewidth=0.6,
    )
    states.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#aaaaaa",
        linewidth=0.3,
    )


def _project(lon: float, lat: float) -> tuple[float, float]:
    return TO_ALBERS.transform(lon, lat)


def _draw_dots(ax: plt.Axes) -> list[tuple[Region, plt.Line2D]]:
    """Plot one matplotlib Line2D per region so each becomes one SVG path we can wrap."""
    handles: list[tuple[Region, plt.Line2D]] = []
    for region in REGIONS:
        x, y = _project(region.lon, region.lat)
        color = FAMILY_COLORS[region.family]
        # gid lets us find the SVG element after savefig
        gid = f"region-{re.sub(r'[^a-z0-9]+', '-', region.name.lower()).strip('-')}"
        (line,) = ax.plot(
            [x], [y],
            marker="o",
            markersize=12,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.6,
            linestyle="None",
            gid=gid,
            zorder=5,
        )
        handles.append((region, line))
    return handles


def _draw_labels(ax: plt.Axes) -> None:
    # rough projected-meter offset for label placement; chosen by eye
    OFFSET_M = 60_000  # ~60 km of projection-space offset per unit of label_dx
    for region in REGIONS:
        x, y = _project(region.lon, region.lat)
        dx = region.label_dx * OFFSET_M / 10.0
        dy = region.label_dy * OFFSET_M / 10.0
        ha = "left" if region.label_dx >= 0 else "right"
        va = "center" if region.label_dy == 0 else ("bottom" if region.label_dy > 0 else "top")
        ax.text(
            x + dx, y + dy,
            region.name,
            ha=ha, va=va,
            fontsize=9,
            color="#222222",
            zorder=6,
            path_effects=None,
        )


def _set_extent(ax: plt.Axes) -> None:
    # extent: from western Aleutians (-175, 50) up to (-65, 72) for Alaska,
    # down to southern Arizona (~-105, 30). Build a box that contains all
    # corners in projected space.
    corners_lonlat = [
        (-175.0, 50.0),
        (-175.0, 72.0),
        (-65.0, 50.0),
        (-65.0, 30.0),
        (-105.0, 30.0),
        (-160.0, 72.0),
    ]
    xs, ys = zip(*(_project(lo, la) for lo, la in corners_lonlat))
    pad = 200_000  # 200 km pad
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()


def render_svg(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=120)
    _draw_basemap(ax)
    _draw_dots(ax)
    _draw_labels(ax)
    _set_extent(ax)
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# --- SVG post-processing: wrap dot elements in <a xlink:href="..."> ---

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def wrap_dots_with_links(svg_path: Path) -> int:
    """Find every element whose id matches one of our region gids and wrap it in <a>.

    Returns the number of anchors inserted.
    """
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)

    tree = ET.parse(svg_path)
    root = tree.getroot()

    region_by_gid = {
        f"region-{re.sub(r'[^a-z0-9]+', '-', r.name.lower()).strip('-')}": r
        for r in REGIONS
    }

    # Build a parent map so we can splice children
    parent_map = {child: parent for parent in root.iter() for child in parent}

    wrapped = 0
    for elem in list(root.iter()):
        elem_id = elem.get("id", "")
        if elem_id in region_by_gid:
            region = region_by_gid[elem_id]
            parent = parent_map.get(elem)
            if parent is None:
                continue
            idx = list(parent).index(elem)
            anchor = ET.Element(f"{{{SVG_NS}}}a")
            anchor.set(f"{{{XLINK_NS}}}href", region.href)
            anchor.set("href", region.href)  # SVG2 fallback
            anchor.set("target", "_self" if region.href.endswith(".html") else "_blank")
            anchor.set("aria-label", region.name)
            # cursor pointer for older renderers
            anchor.set("style", "cursor: pointer;")
            # title tooltip
            title = ET.SubElement(anchor, f"{{{SVG_NS}}}title")
            title.text = region.name
            # move elem under the anchor
            parent.remove(elem)
            anchor.append(elem)
            parent.insert(idx, anchor)
            wrapped += 1

    tree.write(svg_path, xml_declaration=True, encoding="utf-8")
    return wrapped


def main() -> None:
    render_svg(OUT_PATH)
    wrapped = wrap_dots_with_links(OUT_PATH)
    size = OUT_PATH.stat().st_size
    print(f"wrote {OUT_PATH} ({size:,} bytes); wrapped {wrapped} dots in <a>")
    if wrapped < len(REGIONS):
        raise SystemExit(
            f"only wrapped {wrapped}/{len(REGIONS)} regions; check gid matching"
        )


if __name__ == "__main__":
    main()
