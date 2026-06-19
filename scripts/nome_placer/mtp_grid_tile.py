"""Export a georeferenced BLM MTP supplemental tile with a lon/lat grid for tracing.

Usage: python mtp_grid_tile.py <xmin> <ymin> <xmax> <ymax> <tag>
Writes data/derived/nome_placer/mtp/tiles/<tag>.png — the MapServer MTP raster for
the bbox at high zoom, with a 0.002-degree lon/lat grid + labelled ticks so claim
corners can be read directly in WGS84. The 28 residual claims are labelled by MS
number on these plats; trace each bold boundary by reading its corners off the grid.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/nome_placer/mtp/tiles")
EXPORT = ("https://gis.blm.gov/akarcgis/rest/services/Land_Status/"
          "BLM_AK_Master_Title_Plats/MapServer/export")


def main():
    xmin, ymin, xmax, ymax = map(float, sys.argv[1:5])
    tag = sys.argv[5]
    OUT.mkdir(parents=True, exist_ok=True)
    ar = (ymax - ymin) / (xmax - xmin)
    w = 4000
    h = int(w * ar)
    raw = OUT / f"{tag}_raw.png"
    url = (f"{EXPORT}?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=4326&imageSR=4326"
           f"&size={w},{h}&format=png&transparent=false&dpi=300&layers=show:0&f=image")
    subprocess.run(["curl", "-s", "--max-time", "150", "-o", str(raw), url], check=True)
    img = np.asarray(Image.open(raw).convert("RGB"))

    fig, ax = plt.subplots(figsize=(16, 16 * ar))
    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], zorder=0, interpolation="bilinear")
    # 0.002-deg grid
    import math
    gx = np.arange(math.ceil(xmin / 0.002) * 0.002, xmax, 0.002)
    gy = np.arange(math.ceil(ymin / 0.002) * 0.002, ymax, 0.002)
    for x in gx:
        ax.axvline(x, color="red", lw=0.4, alpha=0.35)
    for y in gy:
        ax.axhline(y, color="red", lw=0.4, alpha=0.35)
    ax.set_xticks(gx); ax.set_yticks(gy)
    ax.set_xticklabels([f"{x:.3f}" for x in gx], fontsize=6, rotation=90)
    ax.set_yticklabels([f"{y:.3f}" for y in gy], fontsize=6)
    ax.set_title(f"MTP supplemental {tag}  bbox {xmin},{ymin},{xmax},{ymax}", fontsize=9)
    fig.savefig(OUT / f"{tag}.png", dpi=160, bbox_inches="tight")
    print(f"wrote {OUT / f'{tag}.png'}  ({w}x{h} raster)")


if __name__ == "__main__":
    main()
