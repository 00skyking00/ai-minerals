"""Generate fixed-width thumbnails of each chapter hero for index.qmd.

Each chapter on the portfolio landing page gets an inline thumbnail
right of its title, 320 px wide, height computed to preserve the
source aspect ratio. The full hero stays at level-2 (each chapter
page renders the original PNG).
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image

THUMB_WIDTH = 320

SOURCES = {
    "ch1_bear_cub.png": "data/derived/bear_cub_resource/fig_aerial_pay_zone_grade.png",
    "ch2_regional.png": "data/derived/motherlode/fig_prospectivity_motherlode_cleaned.png",
    "ch3_goldbug.png":  "data/derived/portfolio_charts/goldbug_screenshot.png",
    "ch4_reproductions.png": "data/derived/portfolio_charts/lawley_waterfall.png",
    "ch5_cross_region.png":  "data/derived/portfolio_charts/cross_region_top1.png",
    "ch6_drill_planning.png": "data/derived/bcgt/fig_pomcp_discovery_curves.png",
}

OUT_DIR = Path("data/derived/portfolio_charts/thumbs")


def make_thumb(src: Path, dst: Path, width: int = THUMB_WIDTH) -> None:
    img = Image.open(src)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    new_h = int(round(h * width / w))
    img = img.resize((width, new_h), Image.LANCZOS)
    img.save(dst, "PNG", optimize=True)
    print(f"{src} ({w}x{h}) -> {dst} ({width}x{new_h})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for out_name, src_path in SOURCES.items():
        src = Path(src_path)
        if not src.exists():
            print(f"  SKIP: source missing: {src}")
            continue
        make_thumb(src, OUT_DIR / out_name)


if __name__ == "__main__":
    main()
