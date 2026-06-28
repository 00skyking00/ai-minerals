"""H2 redesign: one map figure tying the result together.

Shows, over the RI 2024-7 map area: the major DOm marble belts (>=1 km^2), the
schist-hosted 36a lodes, and the coarseness-tagged placers coloured by ordinal
class. The visible pattern is the result: the coarse/nuggety (class-3) placers
sit near the lodes and AWAY from the marble, while distance to the marble does
not order the coarseness in the local-source direction.

Run: uv run python -m scripts.nome_placer.h2_confined_reach.make_figure
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from rasterio.enums import Resampling

OUT = Path("data/derived/nome_placer/h2_confined_reach")
GEMS = Path("data/raw/dggs_ri2024_7/extracted/pkg/"
            "casadepaga_bedrock_gems_db_wo_stations-open/GM_MapUnitPolys.shp")
STAGED = Path("data/derived/nome_placer/peninsula_phase2/peninsula_ardf_placer_lode_3338.geojson")
DEM = Path("data/raw/ifsar_dggs/ifsar_dtm_5m_bighurrah_council_bluff_3338.tif")
FIG = OUT / "h2_confined_reach_map.png"
CLS_COLOR = {1: "#f2e000", 2: "#f08000", 3: "#d01010"}
CLS_NAME = {1: "1 fine/flaky", 2: "2 coarse", 3: "3 rough/nuggety"}


def main() -> None:
    polys = gpd.read_file(GEMS).to_crs(3338)
    marble = polys[polys.MapUnit.isin(["DOm", "Dm"])].copy()
    marble["a"] = marble.geometry.area / 1e6
    major = marble[marble.a >= 1.0]
    staged = gpd.read_file(STAGED).to_crs(3338)
    lode = staged[(staged.deposit_class == "lode")
                  & ~staged.model_code.astype(str).str.contains("39")]
    typed = gpd.read_file(OUT / "placers_typed.geojson").to_crs(3338)
    tagged = typed[typed.coarseness_rank.notna()].copy()
    mb = polys.total_bounds

    with rasterio.open(DEM) as src:
        sc = max(1, int(max(src.width, src.height) / 1600))
        dem = src.read(1, out_shape=(src.height // sc, src.width // sc),
                       resampling=Resampling.average).astype("float32")
        b = src.bounds; nod = src.nodata
    z = np.where(dem == nod, np.nan, dem)
    hs = LightSource(315, 45).hillshade(np.nan_to_num(z, nan=np.nanmin(z)), dx=10 * sc, dy=10 * sc)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(hs, cmap="gray", extent=(b.left, b.right, b.bottom, b.top), alpha=0.6)
    major.plot(ax=ax, facecolor="#3a6ea5", edgecolor="none", alpha=0.45)
    lode.plot(ax=ax, marker="^", color="black", markersize=42, label="36a lode (schist-hosted)")
    for k in (1, 2, 3):
        sub = tagged[tagged.coarseness_rank == k]
        if len(sub):
            sub.plot(ax=ax, color=CLS_COLOR[k], edgecolor="k", markersize=70,
                     label=f"placer coarseness {CLS_NAME[k]}")
    ax.set_xlim(mb[0] - 1500, mb[2] + 1500)
    ax.set_ylim(mb[1] - 1500, mb[3] + 1500)
    ax.set_title("Big Hurrah-Council-Bluff: coarse gold tracks the schist-hosted lodes,\n"
                 "not the marble belts (blue = DOm marble >= 1 km², RI 2024-7)", fontsize=12)
    ax.set_xlabel("Easting (m, EPSG:3338)"); ax.set_ylabel("Northing (m, EPSG:3338)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
