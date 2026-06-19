"""Stage the 28 residual claims for tracing: export a tight georeferenced MTP crop per
claim at its neighbour-anchor, then build a labelled contact sheet to triage which are
drawn on the base MTP vs deferred to a supplemental plat (→ Tuck).

Outputs:
  data/derived/nome_placer/mtp/claims/MS<ms>_mtp.png   (per-claim georeferenced crop + bbox sidecar)
  data/derived/nome_placer/mtp/claims_contact_sheet.png
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/home/sky/src/learning/ai-minerals")
MERGED = REPO / "data/derived/nome_placer/mineral_surveys_merged.geojson"
OUT = REPO / "data/derived/nome_placer/mtp/claims"
EXPORT = ("https://gis.blm.gov/akarcgis/rest/services/Land_Status/"
          "BLM_AK_Master_Title_Plats/MapServer/export")
# crop half-size in degrees (~1km lon, ~1.1km lat at 64.5N) and a pad for offset anchors
HALF_LON, HALF_LAT = 0.020, 0.010
PX = 1400


def centroid(geom):
    polys = geom["coordinates"] if geom["type"] == "Polygon" else [r for p in geom["coordinates"] for r in p]
    xs = [pt[0] for r in polys for pt in r]; ys = [pt[1] for r in polys for pt in r]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def register_ms():
    reg = set()
    for p in glob.glob("/home/sky/src/learning/datadredge/samples/kardex/jsonld/*register*.jsonld"):
        for m in re.findall(r"msNumber\"\s*:\s*(\[[^\]]*\]|\"[^\"]*\"|\{[^}]*\})", Path(p).read_text()):
            reg.update(re.findall(r"\"([0-9]{2,5})\"", m))
    return reg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = json.loads(MERGED.read_text())["features"]
    cent = {str(f["properties"]["MS"]).strip(): centroid(f["geometry"]) for f in feats if f.get("geometry")}
    present = set(cent)
    missing = sorted((m for m in register_ms() - present if m.isdigit()), key=int)
    pres = sorted(int(m) for m in present if m.isdigit())

    tiles = []
    for ms in missing:
        n = int(ms)
        below = [p for p in pres if p < n]; above = [p for p in pres if p > n]
        picks = ([below[-1]] if below else []) + ([above[0]] if above else [])
        cs = [cent[str(p)] for p in picks]
        cx = sum(c[0] for c in cs) / len(cs); cy = sum(c[1] for c in cs) / len(cs)
        bbox = (cx - HALF_LON, cy - HALF_LAT, cx + HALF_LON, cy + HALF_LAT)
        png = OUT / f"MS{ms}_mtp.png"
        url = (f"{EXPORT}?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&bboxSR=4326&imageSR=4326"
               f"&size={PX},{int(PX*HALF_LAT/HALF_LON)}&format=png&transparent=false&dpi=300&layers=show:0&f=image")
        subprocess.run(["curl", "-s", "--max-time", "90", "-o", str(png), url], check=True)
        (OUT / f"MS{ms}_mtp.json").write_text(json.dumps({"ms": ms, "bbox": bbox, "px": [PX, int(PX*HALF_LAT/HALF_LON)],
                                                          "neighbours": picks}))
        ink = float((np.asarray(Image.open(png).convert("L")) < 120).mean())
        tiles.append((ms, png, ink))
        print(f"MS{ms}: ink={ink:.3f} neighbours={picks}")

    # contact sheet: 4 cols
    cols = 4; rows = (len(tiles) + cols - 1) // cols
    tw, th = 460, 250
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, (ms, png, ink) in enumerate(tiles):
        r, c = divmod(i, cols)
        im = Image.open(png).convert("RGB").resize((tw, th - 26))
        sheet.paste(im, (c * tw, r * th + 26))
        d.rectangle([c * tw, r * th, c * tw + tw - 1, r * th + 24], fill="black")
        d.text((c * tw + 4, r * th + 3), f"MS {ms}   ink {ink:.2f}", fill="white", font=font)
    sheet.save(OUT.parent / "claims_contact_sheet.png")
    print(f"\nwrote contact sheet ({len(tiles)} claims)")


if __name__ == "__main__":
    main()
