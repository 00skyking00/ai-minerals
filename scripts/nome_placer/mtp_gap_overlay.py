"""Overlay the BLM MTP raster against the merged vector claims; locate the residual gaps.

Inputs:
  - data/derived/nome_placer/mineral_surveys_merged.geojson  (goldbug's 373 merged surveys)
  - the 148-claim register MS list (from the datadredge KARDEX registers)
Outputs:
  - data/derived/nome_placer/mtp/mtp_overlay.png   (MTP raster + vector polygons + gap markers)
  - research/claim_polygon_gap_worklist_2026-06-19.md  (the residual claims to trace, located)

Each residual claim (in the register but absent from every vector source) is placed at
the centroid midpoint of its nearest-by-number present neighbours — a search anchor for
finding its drawn polygon on the georeferenced MTP plat.
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path("/home/sky/src/learning/ai-minerals")
MERGED = REPO / "data/derived/nome_placer/mineral_surveys_merged.geojson"
MTP_DIR = REPO / "data/derived/nome_placer/mtp"
OVERLAY = MTP_DIR / "mtp_overlay.png"
WORKLIST = REPO / "research/claim_polygon_gap_worklist_2026-06-19.md"
MTP_EXPORT = ("https://gis.blm.gov/akarcgis/rest/services/Land_Status/"
              "BLM_AK_Master_Title_Plats/MapServer/export")


def poly_centroid(geom) -> tuple[float, float]:
    pts = []
    polys = geom["coordinates"] if geom["type"] == "Polygon" else [r for p in geom["coordinates"] for r in p]
    for ring in polys:
        for pt in ring:
            pts.append((pt[0], pt[1]))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def register_ms() -> set[str]:
    reg = set()
    for p in glob.glob("/home/sky/src/learning/datadredge/samples/kardex/jsonld/*register*.jsonld"):
        txt = Path(p).read_text()
        for m in re.findall(r"msNumber\"\s*:\s*(\[[^\]]*\]|\"[^\"]*\"|\{[^}]*\})", txt):
            for v in re.findall(r"\"([0-9]{2,5})\"", m):
                reg.add(v)
    return reg


def export_mtp(bbox, px=5000) -> Path:
    xmin, ymin, xmax, ymax = bbox
    ar = (ymax - ymin) / (xmax - xmin)
    out = MTP_DIR / "mtp_overlay_basemap.png"
    url = (f"{MTP_EXPORT}?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=4326&imageSR=4326"
           f"&size={px},{int(px*ar)}&format=png&transparent=false&dpi=300&layers=show:0&f=image")
    subprocess.run(["curl", "-s", "--max-time", "180", "-o", str(out), url], check=True)
    return out


def main() -> None:
    MTP_DIR.mkdir(parents=True, exist_ok=True)
    feats = json.loads(MERGED.read_text())["features"]
    cent = {}
    for f in feats:
        ms = str(f["properties"]["MS"]).strip()
        if f.get("geometry"):
            cent[ms] = poly_centroid(f["geometry"])
    present = set(cent)
    reg = register_ms()
    missing = sorted(reg - present, key=lambda x: int(x) if x.isdigit() else 0)
    name_by_ms = {str(f["properties"]["MS"]).strip(): f["properties"].get("name") for f in feats}

    # locate each missing MS at the midpoint of its nearest present neighbours by number
    pres_sorted = sorted((int(m) for m in present if m.isdigit()))
    anchors = {}
    for ms in missing:
        if not ms.isdigit():
            continue
        n = int(ms)
        below = [p for p in pres_sorted if p < n]
        above = [p for p in pres_sorted if p > n]
        picks = ([below[-1]] if below else []) + ([above[0]] if above else [])
        if not picks:
            continue
        cs = [cent[str(p)] for p in picks]
        anchors[ms] = (sum(c[0] for c in cs) / len(cs), sum(c[1] for c in cs) / len(cs), picks)

    # overlay bbox: bound the present-neighbour anchors + a pad (the claim core)
    axs = [a[0] for a in anchors.values()]; ays = [a[1] for a in anchors.values()]
    pad = 0.02
    bbox = (min(axs) - pad, min(ays) - pad, max(axs) + pad, max(ays) + pad)
    base = export_mtp(bbox)
    img = np.asarray(Image.open(base).convert("RGB"))

    fig, ax = plt.subplots(figsize=(16, 16 * (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])))
    ax.imshow(img, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], zorder=0)
    # present polygons (thin blue)
    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        rings = g["coordinates"] if g["type"] == "Polygon" else [r for p in g["coordinates"] for r in p]
        for ring in rings:
            xs = [pt[0] for pt in ring]; ys = [pt[1] for pt in ring]
            ax.plot(xs, ys, color="tab:blue", lw=0.4, alpha=0.5, zorder=2)
    # gap anchors (red)
    for ms, (x, y, _) in anchors.items():
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            ax.plot(x, y, "o", color="red", ms=5, zorder=3)
            ax.annotate(ms, (x, y), color="red", fontsize=7, zorder=4,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"BLM MTP raster + {len(present)} merged claims (blue) + {len(anchors)} residual gaps (red)")
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    fig.savefig(OVERLAY, dpi=130, bbox_inches="tight")
    print(f"wrote {OVERLAY}")

    # worklist
    lines = ["# Claim-polygon gap worklist (residual after 4-source vector merge)", "",
             f"Generated 2026-06-19. {len(missing)} register claims have no polygon in any vector",
             "source (BLM CadNSDI + DNR + Nome taxroll + curated). Trace each from the georeferenced",
             "BLM MTP raster (overlay: `data/derived/nome_placer/mtp/mtp_overlay.png`). Anchor = midpoint",
             "of nearest present claims by MS number; look near it on the plat.", "",
             "| MS | name (if known) | anchor lon | anchor lat | nearest present MS |",
             "|----|-----------------|-----------|-----------|--------------------|"]
    for ms in missing:
        a = anchors.get(ms)
        nm = name_by_ms.get(ms) or ""
        if a:
            lines.append(f"| {ms} | {nm} | {a[0]:.4f} | {a[1]:.4f} | {', '.join(map(str, a[2]))} |")
        else:
            lines.append(f"| {ms} | {nm} | ? | ? | (no numeric neighbour) |")
    WORKLIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {WORKLIST}: {len(missing)} residual claims")


if __name__ == "__main__":
    main()
