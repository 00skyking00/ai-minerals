"""Build a durable, BLM-independent tile pyramid of the BLM Master Title Plat (MTP)
raster for the Nome claim area (ADR-019).

Re-fetches a high-res mosaic from the BLM MTP MapServer ONCE at build time, then
georeferences it (EPSG:3857), adds an alpha channel so the white plat background
is transparent, and cuts an XYZ tile pyramid with gdal2tiles. Serving is then
BLM-independent: goldbug points a L.tileLayer at the local {z}/{x}/{y} tree with a
fixed maxNativeZoom and zero per-view BLM round-trips.

Outputs to data/derived/nome_placer/mtp_pyramid/ (gitignored). XYZ tiles preferred;
falls back to a Cloud-Optimized GeoTIFF if gdal2tiles is unavailable.
"""
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# claim-area bbox (W,S,E,N) — the 373 cluster, not the empty config bbox
W, S, E, N = -165.62, 64.44, -165.05, 64.66
TARGET_W = 16384          # total mosaic width in px -> ~1.7 m/px over the ~27 km bbox
MAX_CELL = 4096           # per-export cap (MapServer)
DPI = 200
MAX_FETCHES = 25
EXPORT = ("https://gis.blm.gov/akarcgis/rest/services/Land_Status/"
          "BLM_AK_Master_Title_Plats/MapServer/export")

OUTDIR = Path("data/derived/nome_placer/mtp_pyramid")
GEOTIFF = OUTDIR / "mtp_nome_3857.tif"
TILES = OUTDIR / "tiles"
COG = OUTDIR / "mtp_nome_pyramid_cog.tif"
META = OUTDIR / "mtp_pyramid_meta.json"
MAXNATIVE_Z = 16          # native ~1.66 m/px sits between z15 (2.05) and z16 (1.03)
MINZ = 8


def merc(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 lon/lat -> EPSG:3857 (Web Mercator) metres."""
    x = lon * 20037508.342789244 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * 6378137.0
    return x, y


def fetch_cell(bx0: float, by0: float, bx1: float, by1: float, w: int, h: int, dest: str) -> int:
    """Fetch one MapServer export tile (EPSG:3857 bbox) to dest. Returns curl exit code."""
    url = (f"{EXPORT}?bbox={bx0},{by0},{bx1},{by1}&bboxSR=3857&imageSR=3857"
           f"&size={w},{h}&format=png&dpi={DPI}&layers=show:0&f=image")
    r = subprocess.run(["curl", "-sS", "--max-time", "120", "-o", dest, url])
    return r.returncode


def build_mosaic() -> tuple[Image.Image, tuple[float, float, float, float], int, int]:
    """Fetch the 4x4 grid and paste into one RGB mosaic. Returns (img, bbox3857, fetches, fails)."""
    xW, yS = merc(W, S)
    xE, yN = merc(E, N)
    wm, hm = xE - xW, yN - yS
    total_w = TARGET_W
    total_h = int(round(total_w * hm / wm))
    cols = math.ceil(total_w / MAX_CELL)
    rows = math.ceil(total_h / MAX_CELL)
    cw = total_w // cols
    ch = total_h // rows
    n_fetch = cols * rows
    gpx = wm * math.cos(math.radians((S + N) / 2)) / total_w
    print(f"mosaic {cw*cols}x{ch*rows}  grid {cols}x{rows}={n_fetch} fetches  "
          f"cell {cw}x{ch}  ground ~{gpx:.2f} m/px")
    if n_fetch > MAX_FETCHES:
        raise SystemExit(f"grid {n_fetch} exceeds MAX_FETCHES={MAX_FETCHES}")

    mosaic = Image.new("RGB", (cw * cols, ch * rows), "white")
    fetches = 0
    fails = 0
    for i in range(cols):
        for j in range(rows):
            bx0 = xW + i * wm / cols
            bx1 = xW + (i + 1) * wm / cols
            by1 = yN - j * hm / rows          # row 0 = top (north)
            by0 = yN - (j + 1) * hm / rows
            tmp = f"/tmp/_mtppyr_{i}_{j}.png"
            ok = False
            for attempt in (1, 2):
                fetches += 1
                code = fetch_cell(bx0, by0, bx1, by1, cw, ch, tmp)
                if code == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    try:
                        cell = Image.open(tmp).convert("RGB")
                        ok = True
                        break
                    except Exception as exc:
                        print(f"  cell {i},{j} attempt {attempt}: decode failed {exc}")
                else:
                    print(f"  cell {i},{j} attempt {attempt}: curl exit {code}")
            if not ok:
                fails += 1
                print(f"  cell {i},{j}: FAILED after 2 attempts; leaving white")
                continue
            mosaic.paste(cell, (i * cw, j * ch))
    # actual mosaic corners (the grid is integer-cell, so recompute exact 3857 extent)
    exW, exN = xW, yN
    exE = xW + cw * cols * (wm / total_w)
    exS = yN - ch * rows * (hm / total_h)
    return mosaic, (exW, exS, exE, exN), fetches, fails


def add_alpha(mosaic: Image.Image) -> Image.Image:
    """White plat background -> transparent; linework/labels stay opaque (binary alpha)."""
    a = np.asarray(mosaic)
    # white-ish background = all channels high; treat >=245 on min channel as background
    bg = a.min(axis=2) >= 245
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    rgba = np.dstack([a, alpha])
    return Image.fromarray(rgba, "RGBA")


def write_geotiff(rgba: Image.Image, bbox3857: tuple[float, float, float, float]) -> None:
    """Write the RGBA mosaic as an EPSG:3857 GeoTIFF with a north-up geotransform."""
    x0, y0, x1, y1 = bbox3857
    w, h = rgba.size
    px = (x1 - x0) / w
    py = (y1 - y0) / h
    tmp_png = OUTDIR / "_mtp_rgba.png"
    rgba.save(tmp_png)
    # gdal_translate: -a_srs EPSG:3857, -a_ullr ulx uly lrx lry (north-up)
    cmd = [
        "gdal_translate", "-of", "GTiff",
        "-a_srs", "EPSG:3857",
        "-a_ullr", str(x0), str(y1), str(x1), str(y0),
        "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
        str(tmp_png), str(GEOTIFF),
    ]
    subprocess.run(cmd, check=True)
    tmp_png.unlink(missing_ok=True)
    print(f"wrote {GEOTIFF}  px={px:.3f} py={py:.3f} m")


def build_xyz() -> bool:
    """Cut XYZ pyramid with gdal2tiles. Returns True on success."""
    if shutil.which("gdal2tiles.py") is None:
        return False
    if TILES.exists():
        shutil.rmtree(TILES)
    cmd = [
        "gdal2tiles.py", "--xyz",
        "-z", f"{MINZ}-{MAXNATIVE_Z}",
        "-r", "average",
        "-w", "none",
        "--processes", "4",
        str(GEOTIFF), str(TILES),
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd)
    return r.returncode == 0 and TILES.exists()


def build_cog() -> int:
    """Fallback: Cloud-Optimized GeoTIFF + overviews. Returns overview count."""
    cmd = [
        "gdal_translate", "-of", "COG",
        "-co", "COMPRESS=DEFLATE",
        "-co", "OVERVIEW_RESAMPLING=AVERAGE",
        str(GEOTIFF), str(COG),
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(["gdaladdo", "-r", "average", str(COG), "2", "4", "8", "16", "32"], check=False)
    # count overviews
    out = subprocess.run(["gdalinfo", str(COG)], capture_output=True, text=True).stdout
    return out.count("Overview ") if "Overview" in out else out.count("Overviews:")


def dir_size_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def count_tiles(p: Path) -> int:
    return sum(1 for f in p.rglob("*.png"))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mosaic, bbox3857, fetches, fails = build_mosaic()
    print(f"fetches={fetches} fails={fails}  mosaic={mosaic.size}")
    rgba = add_alpha(mosaic)
    write_geotiff(rgba, bbox3857)

    xyz_ok = build_xyz()
    gpx = (merc(E, S)[0] - merc(W, S)[0]) * math.cos(math.radians((S + N) / 2)) / mosaic.size[0]

    meta = {
        "description": "BLM Master Title Plat (MTP) raster pyramid, Nome claim area. "
                       "Built once from BLM MapServer; serving is BLM-independent (ADR-019).",
        "source": EXPORT,
        "bounds_wgs84_WSEN": [W, S, E, N],
        "crs": "EPSG:3857 (tiles); EPSG:4326 bounds",
        "mosaic_px": list(mosaic.size),
        "native_ground_res_m_per_px": round(gpx, 3),
        "fetches": fetches,
        "fetch_failures": fails,
        "built_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                    capture_output=True, text=True).stdout.strip(),
    }

    if xyz_ok:
        n_tiles = count_tiles(TILES)
        size_b = dir_size_bytes(TILES)
        rel = "data/derived/nome_placer/mtp_pyramid/tiles/{z}/{x}/{y}.png"
        meta.update({
            "pyramid_type": "xyz",
            "tile_scheme": "xyz",
            "tile_size_px": 256,
            "tile_path_template_on_disk": rel,
            "tiles_dir": str(TILES),
            "minZoom": MINZ,
            "maxNativeZoom": MAXNATIVE_Z,
            "tiles_count": n_tiles,
            "size_on_disk_bytes": size_b,
            "size_on_disk_mb": round(size_b / 1e6, 1),
            "serving_note": (
                f"Mount {TILES}/ at some URL prefix and use that prefix in the "
                "leaflet template below (here '/tiles/mtp/' is a placeholder). "
                "Tiles are XYZ (OSM/Google) scheme, transparent PNG (white plat "
                "background -> alpha 0). Set maxNativeZoom so Leaflet upscales the "
                "z16 tiles past native zoom instead of requesting absent deeper "
                "levels."
            ),
            "leaflet_snippet": (
                "L.tileLayer('/tiles/mtp/{z}/{x}/{y}.png', {\n"
                f"  minZoom: {MINZ}, maxNativeZoom: {MAXNATIVE_Z}, maxZoom: 22,\n"
                "  tileSize: 256, opacity: 0.85,\n"
                f"  bounds: L.latLngBounds([[{S}, {W}], [{N}, {E}]]),\n"
                "  attribution: 'BLM AK Master Title Plats'\n"
                "}).addTo(map);"
            ),
        })
        print(f"\nXYZ pyramid: {n_tiles} tiles, {size_b/1e6:.1f} MB, z{MINZ}-{MAXNATIVE_Z}")
    else:
        print("\ngdal2tiles unavailable -> COG fallback")
        n_ov = build_cog()
        size_b = COG.stat().st_size
        meta.update({
            "pyramid_type": "cog",
            "cog_path": str(COG),
            "overview_count": n_ov,
            "maxNativeZoom": MAXNATIVE_Z,
            "size_on_disk_bytes": size_b,
            "size_on_disk_mb": round(size_b / 1e6, 1),
            "leaflet_note": (
                "Serve mtp_nome_pyramid_cog.tif via a COG-aware client "
                "(georaster-layer-for-leaflet or a tile endpoint). "
                f"Internal overviews give pyramid down to native ~{gpx:.2f} m/px."
            ),
        })
        print(f"\nCOG: {size_b/1e6:.1f} MB, {n_ov} overviews -> {COG}")

    META.write_text(json.dumps(meta, indent=2))
    print(f"wrote {META}")


if __name__ == "__main__":
    main()
