"""Streamlit corner-picker for the Bear Cub MS 1178 outline on BearCubDHMap.pdf.

Click the four corners of the cartographer-drawn red parallelogram in order
(TL → TR → BR → BL). As soon as all four are set, the tool solves the
lat/lon ↔ pixel affine transform from the BLM ground-truth corners and
overlays the 24 drill holes (computed from `bear_cub_collars.csv` lat/lon),
so you can verify they land where the dh map's own dots are.

Run:
    uv run streamlit run tools/bear_cub_corner_picker.py

Output:
    data/raw/bear_cub/dhmap_corners_user.json — pixel coords in original-
    resolution dh map space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

REPO = Path(__file__).resolve().parents[1]
DHMAP_PNG = REPO / "data" / "raw" / "bear_cub" / "page_pngs" / "BearCubDHMap__p1.png"
COLLARS_CSV = REPO / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"
OUT_JSON = REPO / "data" / "raw" / "bear_cub" / "dhmap_corners_user.json"

CORNERS = ("TL", "TR", "BR", "BL")
CORNER_COLORS = {"TL": "#3b82f6", "TR": "#10b981", "BR": "#f59e0b", "BL": "#ef4444"}

# BLM Master Title Plat — MS 1178 (Bear Cub Placer, Cape Nome district), WGS84.
BLM_LATLON = {
    "TL": (64.531784, -165.341551),
    "TR": (64.532488, -165.337952),
    "BR": (64.531171, -165.332170),
    "BL": (64.530095, -165.335329),
}

DISPLAY_WIDTH = 1500  # browser-friendly; clicks scaled back to original res


def init_state() -> None:
    if "corners" not in st.session_state:
        st.session_state.corners = {k: None for k in CORNERS}
    if "active" not in st.session_state:
        st.session_state.active = "TL"
    if "notes" not in st.session_state:
        st.session_state.notes = {k: "" for k in CORNERS}
    if "last_click" not in st.session_state:
        st.session_state.last_click = None


def advance_active() -> None:
    """Set active to the first un-set corner; None if all are set."""
    for k in CORNERS:
        if st.session_state.corners[k] is None:
            st.session_state.active = k
            return
    st.session_state.active = None


@st.cache_data
def load_dhmap() -> Image.Image:
    return Image.open(DHMAP_PNG).convert("RGB")


@st.cache_data
def load_collars() -> pd.DataFrame:
    return pd.read_csv(COLLARS_CSV)


def solve_affine(corners_px: dict[str, tuple[int, int]]) -> np.ndarray:
    """Solve lat,lon,1 → px,py affine using all 4 BLM corners as controls."""
    LL = np.array([(*BLM_LATLON[k], 1.0) for k in CORNERS])
    PX = np.array([corners_px[k] for k in CORNERS], dtype=float)
    A, *_ = np.linalg.lstsq(LL, PX, rcond=None)
    return A  # shape (3, 2)


def latlon_to_px(A: np.ndarray, lat: float, lon: float) -> tuple[float, float]:
    px, py = np.array([lat, lon, 1.0]) @ A
    return float(px), float(py)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
    )


def render_grid(img: Image.Image, n: int = 10) -> Image.Image:
    """Faint labeled grid overlay (A1..J10 default), rendered at full res."""
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = img.size
    label_font = font(28)
    for i in range(1, n):
        x = int(w * i / n)
        draw.line([(x, 0), (x, h)], fill=(80, 80, 80, 100), width=2)
        y = int(h * i / n)
        draw.line([(0, y), (w, y)], fill=(80, 80, 80, 100), width=2)
    for col in range(n):
        for row in range(n):
            x = int(w * col / n) + 8
            y = int(h * row / n) + 6
            label = f"{chr(ord('A') + col)}{row + 1}"
            draw.text((x, y), label, fill=(80, 80, 80, 160), font=label_font)
    return out


def render_overlay(
    img: Image.Image,
    corners_px: dict[str, tuple[int, int] | None],
    collars: pd.DataFrame | None,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    name_font = font(40)
    hole_font = font(22)

    # Drill holes — render *first* so corners draw on top.
    all_set = all(corners_px[k] is not None for k in CORNERS)
    if all_set and collars is not None:
        A = solve_affine(corners_px)
        for _, row in collars.iterrows():
            try:
                lat = float(row["lat_wgs84"])
                lon = float(row["lon_wgs84"])
            except (TypeError, ValueError):
                continue
            x, y = latlon_to_px(A, lat, lon)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="#dc2626", outline="#7f1d1d", width=3)
            draw.text((x + 16, y - 12), str(row["hole_id"]), fill="#7f1d1d", font=hole_font)

    # Parallelogram outline (only if all 4 set)
    if all_set:
        pts = [corners_px[k] for k in CORNERS]
        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            draw.line([a, b], fill="#1d4ed8", width=5)

    # Corner markers (always drawn, even partial)
    for name in CORNERS:
        c = corners_px[name]
        if c is None:
            continue
        x, y = c
        color = CORNER_COLORS[name]
        draw.ellipse((x - 22, y - 22, x + 22, y + 22), outline=color, width=5)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x + 26, y - 18), name, fill=color, font=name_font)

    return out


def main() -> None:
    st.set_page_config(layout="wide", page_title="Bear Cub corner picker")
    init_state()

    if not DHMAP_PNG.exists():
        st.error(f"dh map PNG not found at {DHMAP_PNG}")
        return
    if not COLLARS_CSV.exists():
        st.error(f"collars CSV not found at {COLLARS_CSV}")
        return

    dhmap = load_dhmap()
    collars = load_collars()
    orig_w, orig_h = dhmap.size
    scale = DISPLAY_WIDTH / orig_w

    col_image, col_panel = st.columns([4, 1])

    # --- Sidebar / right panel ---------------------------------------------------
    with col_panel:
        st.markdown("### Pick corners")

        active = st.session_state.active
        if active is None:
            st.success("All 4 corners set — overlay live.")
        else:
            color = CORNER_COLORS[active]
            st.markdown(
                f"**Next click:** "
                f"<span style='background:{color};color:white;padding:4px 10px;"
                f"border-radius:6px;font-weight:600'>{active}</span>",
                unsafe_allow_html=True,
            )

        show_grid = st.checkbox("Faint labeled grid (A1..J10)", value=True)
        show_holes = st.checkbox("Overlay drill holes (after 4 corners set)", value=True)

        st.divider()
        for name in CORNERS:
            c = st.session_state.corners[name]
            color = CORNER_COLORS[name]
            badge = (
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:4px'>{name}</span>"
            )
            if c is not None:
                st.markdown(f"{badge} ({c[0]}, {c[1]})", unsafe_allow_html=True)
            else:
                st.markdown(f"{badge} _not set_", unsafe_allow_html=True)
            st.session_state.notes[name] = st.text_input(
                f"Note for {name}",
                value=st.session_state.notes[name],
                key=f"note_{name}",
                placeholder="e.g., near hole 7754",
                label_visibility="collapsed",
            )
            if c is not None and st.button(f"Redo {name}", key=f"redo_{name}"):
                st.session_state.corners[name] = None
                st.session_state.active = name
                st.session_state.last_click = None
                st.rerun()

        st.divider()
        if st.button("Reset all"):
            st.session_state.corners = {k: None for k in CORNERS}
            st.session_state.active = "TL"
            st.session_state.last_click = None
            st.rerun()

        if all(st.session_state.corners[k] is not None for k in CORNERS):
            if st.button("💾 Save to JSON", type="primary"):
                payload = {
                    "corners_px": {k: list(st.session_state.corners[k]) for k in CORNERS},
                    "notes": dict(st.session_state.notes),
                    "blm_latlon": BLM_LATLON,
                    "image": str(DHMAP_PNG.relative_to(REPO)),
                    "image_size": list(dhmap.size),
                }
                OUT_JSON.write_text(json.dumps(payload, indent=2))
                st.success(f"Saved → {OUT_JSON.relative_to(REPO)}")

    # --- Main image pane ---------------------------------------------------------
    with col_image:
        canvas = render_grid(dhmap) if show_grid else dhmap
        canvas = render_overlay(
            canvas,
            st.session_state.corners,
            collars if show_holes else None,
        )
        # Resize for display while preserving original-resolution coordinate math
        disp = canvas.resize((DISPLAY_WIDTH, int(orig_h * scale)), Image.LANCZOS)

        click = streamlit_image_coordinates(disp, key="dhmap_canvas")

        if click is not None and click != st.session_state.last_click:
            st.session_state.last_click = click
            active = st.session_state.active
            if active is not None:
                # Scale display click → original-resolution pixel coords
                x_orig = int(click["x"] / scale)
                y_orig = int(click["y"] / scale)
                st.session_state.corners[active] = (x_orig, y_orig)
                advance_active()
                st.rerun()

        st.caption(
            f"Display {DISPLAY_WIDTH}×{int(orig_h * scale)} px · "
            f"original {orig_w}×{orig_h} px · click maps to original-resolution coords"
        )


if __name__ == "__main__":
    main()
