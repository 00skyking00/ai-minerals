"""Streamlit drill-hole picker for the Bear Cub dh map.

Calibrate the local-grid-feet → dh-map-pixels affine by clicking on labeled
drill-hole dots whose hole IDs we can read off the map. This bypasses the
cartographer's red parallelogram entirely — the OCR'd local-grid coordinates
are the authoritative position record, so matching them to the cartographer's
own drill-hole dots is the cleanest calibration we can do.

Workflow:
  1. Pick a hole from the dropdown (e.g., 7754).
  2. Click that hole's dot on the dh map.
  3. Repeat for 3+ holes (more = better fit, more redundancy).
  4. Once 3+ picks land, the tool solves the affine and shows predicted
     positions for ALL 24 holes as faint guide markers, helping you
     locate subsequent holes.
  5. Save → data/raw/bear_cub/dhmap_drillhole_picks.json.

Run:
    uv run streamlit run tools/bear_cub_drillhole_picker.py
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
CORNERS_JSON = REPO / "data" / "raw" / "bear_cub" / "dhmap_corners_user.json"
OUT_JSON = REPO / "data" / "raw" / "bear_cub" / "dhmap_drillhole_picks.json"

DISPLAY_WIDTH = 1600
CROP_PAD_PX = 700  # pad around Bear Cub bbox so adjacent drill holes are visible


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
    )


@st.cache_data
def load_dhmap_and_crop() -> tuple[Image.Image, tuple[int, int]]:
    """Crop the dh map to a Bear-Cub-area window for closer view.

    Returns (cropped_image, (x_offset, y_offset)) — offset is crop origin
    in original-image pixel coords; saved picks are translated back so
    they're stored in original-image coords.
    """
    img = Image.open(DHMAP_PNG).convert("RGB")

    if CORNERS_JSON.exists():
        with open(CORNERS_JSON) as f:
            data = json.load(f)
        xs = [c[0] for c in data["corners_px"].values()]
        ys = [c[1] for c in data["corners_px"].values()]
        x0 = max(0, min(xs) - CROP_PAD_PX)
        y0 = max(0, min(ys) - CROP_PAD_PX)
        x1 = min(img.width, max(xs) + CROP_PAD_PX)
        y1 = min(img.height, max(ys) + CROP_PAD_PX)
    else:
        x0, y0, x1, y1 = 0, 0, img.width, img.height

    return img.crop((x0, y0, x1, y1)), (x0, y0)


@st.cache_data
def load_collars() -> pd.DataFrame:
    df = pd.read_csv(COLLARS_CSV)
    df["hole_id_str"] = df["hole_id"].astype(str)
    # `file_stem` is the source PDF basename (e.g. "L2 H4", "L6500 H6554") — the
    # canonical disambiguator since the 1949 Drill-Report-for-Frozen-Ground forms
    # use bare 1-digit hole IDs (2,3,4,5) that collide with nothing on the
    # Hammon dh map, while the Hammon forms use composite 4-digit IDs.
    df["display"] = df["file_stem"] + "  (id=" + df["hole_id_str"] + ")"
    return df.sort_values("file_stem").reset_index(drop=True)


def label_for(collars: pd.DataFrame, hole_id_str: str) -> str:
    match = collars[collars["hole_id_str"] == hole_id_str]
    if len(match):
        return str(match.iloc[0]["display"])
    return hole_id_str


def solve_local_to_pixel(
    picks: dict[str, tuple[int, int]], collars: pd.DataFrame
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Solve full affine transform (6 DOF) from (e_local_ft, n_local_ft) → (px, py).

    Why affine, not similarity: the dh map has a Y-axis reflection relative to
    local-grid (image-y increases downward; local-N increases northward), which
    direct 4-DOF similarity (rotation + uniform scale + translation) cannot
    represent. Empirical fit on real picks shows the dh map IS very close to
    conformal-with-reflection (E/N scale ratio ≈ 1.00, shear ≈ 0°), but the
    reflection makes 6-DOF affine the cleanest minimum sufficient model.

    With 3 non-collinear picks this is exactly determined (zero residual).
    With 4+ picks it's overdetermined and residuals reveal real non-affine
    distortion in the cartographer's projection.

    Model: [e, n, 1] @ A → (px, py)  where A is (3, 2).

    Returns (A_3x2, residuals_per_pick) or (None, None) if < 3 picks.
    """
    rows: list[list[float]] = []
    targets: list[tuple[float, float]] = []
    for hole_id, pixel in picks.items():
        m = collars[collars["hole_id_str"] == str(hole_id)]
        if not len(m):
            continue
        rows.append(
            [
                float(m.iloc[0]["easting_local_ft"]),
                float(m.iloc[0]["northing_local_ft"]),
                1.0,
            ]
        )
        targets.append(pixel)

    if len(rows) < 3:
        return None, None

    M = np.array(rows)
    T = np.array(targets, dtype=float)
    A, *_ = np.linalg.lstsq(M, T, rcond=None)
    predicted = M @ A
    residuals = np.linalg.norm(predicted - T, axis=1)
    return A, residuals


def apply_local_to_pixel(A: np.ndarray, e: float, n: float) -> tuple[float, float]:
    px, py = np.array([e, n, 1.0]) @ A
    return float(px), float(py)


def init_state() -> None:
    if "picks" not in st.session_state:
        st.session_state.picks = {}  # dict[hole_id_str, (x_orig, y_orig)]
    if "active_hole" not in st.session_state:
        st.session_state.active_hole = None
    if "last_click" not in st.session_state:
        st.session_state.last_click = None
    if "show_predictions" not in st.session_state:
        st.session_state.show_predictions = True


def render_overlay(
    cropped: Image.Image,
    offset: tuple[int, int],
    picks: dict[str, tuple[int, int]],
    A: np.ndarray | None,
    collars: pd.DataFrame,
    show_predictions: bool,
    active_hole: str | None,
) -> Image.Image:
    """Render recorded picks (red) + predicted positions for unpicked holes (faint blue).

    All math is in original-image coords; we translate to crop coords for drawing.
    """
    out = cropped.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    pick_font = font(28)
    pred_font = font(20)

    ox, oy = offset

    # Predicted positions for all holes (faint blue) — guides for finding remaining holes
    if A is not None and show_predictions:
        for _, row in collars.iterrows():
            hole_id = row["hole_id_str"]
            file_stem = str(row["file_stem"])
            e = float(row["easting_local_ft"])
            n = float(row["northing_local_ft"])
            px_orig, py_orig = apply_local_to_pixel(A, e, n)
            px = px_orig - ox
            py = py_orig - oy
            if not (0 <= px <= cropped.width and 0 <= py <= cropped.height):
                continue
            color = (37, 99, 235, 200) if hole_id != active_hole else (245, 158, 11, 230)
            draw.ellipse((px - 14, py - 14, px + 14, py + 14), outline=color, width=3)
            draw.text((px + 18, py - 10), file_stem, fill=color, font=pred_font)

    # Recorded picks (solid red)
    by_id = collars.set_index("hole_id_str")["file_stem"].to_dict()
    for hole_id, (x_orig, y_orig) in picks.items():
        x = x_orig - ox
        y = y_orig - oy
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#dc2626", outline="#7f1d1d", width=3)
        draw.text((x + 14, y - 14), str(by_id.get(hole_id, hole_id)), fill="#7f1d1d", font=pick_font)

    return out


def main() -> None:
    st.set_page_config(layout="wide", page_title="Bear Cub drill-hole picker")
    init_state()

    if not DHMAP_PNG.exists() or not COLLARS_CSV.exists():
        st.error("Required input files missing.")
        return

    cropped, offset = load_dhmap_and_crop()
    collars = load_collars()

    crop_w, crop_h = cropped.size
    scale = DISPLAY_WIDTH / crop_w
    disp_h = int(crop_h * scale)

    A, residuals = solve_local_to_pixel(st.session_state.picks, collars)

    col_image, col_panel = st.columns([4, 1])

    with col_panel:
        st.markdown("### Pick drill holes")

        all_ids = collars["hole_id_str"].tolist()
        unpicked = [h for h in all_ids if h not in st.session_state.picks]

        if st.session_state.active_hole is None and unpicked:
            st.session_state.active_hole = unpicked[0]

        if unpicked:
            st.session_state.active_hole = st.selectbox(
                "Currently picking:",
                options=unpicked,
                index=unpicked.index(st.session_state.active_hole)
                if st.session_state.active_hole in unpicked
                else 0,
                format_func=lambda h: label_for(collars, h),
            )
            st.caption(
                "Click that hole's dot on the dh map. "
                "Skip the L2/L3 (1949) holes if not visible — they sit on adjacent claims."
            )
        else:
            st.success("All 24 holes picked!")
            st.session_state.active_hole = None

        st.session_state.show_predictions = st.checkbox(
            "Show predicted positions for all holes (after 3+ non-collinear picks)",
            value=st.session_state.show_predictions,
        )

        st.divider()
        st.markdown(f"**Recorded** ({len(st.session_state.picks)})")
        for hid, px in list(st.session_state.picks.items()):
            cols = st.columns([3, 1])
            cols[0].markdown(f"`{label_for(collars, hid)}` → ({px[0]}, {px[1]})")
            if cols[1].button("✗", key=f"del_{hid}"):
                del st.session_state.picks[hid]
                st.session_state.last_click = None
                st.rerun()

        if A is not None and residuals is not None:
            st.divider()
            # Decompose affine: A is (3,2) so [e,n,1]@A = (px,py)
            #   px = A[0,0]*e + A[1,0]*n + A[2,0]
            #   py = A[0,1]*e + A[1,1]*n + A[2,1]
            ax, ay = A[0, 0], A[0, 1]   # E-axis vector in pixel space
            bx, by = A[1, 0], A[1, 1]   # N-axis vector in pixel space
            scale_e = float(np.hypot(ax, ay))
            scale_n = float(np.hypot(bx, by))
            cos_angle = (ax * bx + ay * by) / (scale_e * scale_n + 1e-12)
            shear_deg = 90.0 - float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
            st.markdown(
                f"**E-axis:** {scale_e:.3f} px/ft · **N-axis:** {scale_n:.3f} px/ft "
                f"(ratio **{scale_e / scale_n:.3f}**)"
            )
            st.markdown(f"**Shear** (deviation from orthogonal): {shear_deg:+.2f}°")
            st.caption(
                "Healthy dh map: ratio near 1.000, shear near 0°. Big departures "
                "mean the cartographer's projection isn't conformal."
            )
            st.markdown("**Residuals** (orig-image px)")
            picks_list = list(st.session_state.picks.keys())
            for k, r in zip(picks_list, residuals):
                color = "#10b981" if r < 5 else "#f59e0b" if r < 20 else "#ef4444"
                st.markdown(
                    f"<span style='color:{color}'>`{label_for(collars, k)}`: {r:.1f} px</span>",
                    unsafe_allow_html=True,
                )
            st.caption(f"Mean residual: {residuals.mean():.1f} px")

        st.divider()
        if st.button("Reset all picks"):
            st.session_state.picks = {}
            st.session_state.active_hole = None
            st.session_state.last_click = None
            st.rerun()

        if len(st.session_state.picks) >= 3:
            if st.button("💾 Save picks to JSON", type="primary"):
                # Solve and save the affine + per-hole predicted px positions
                payload = {
                    "picks_px": {
                        hid: list(p) for hid, p in st.session_state.picks.items()
                    },
                    "image": str(DHMAP_PNG.relative_to(REPO)),
                    "image_size": list(Image.open(DHMAP_PNG).size),
                    "crop_offset": list(offset),
                    "affine_local_to_pixel": A.tolist() if A is not None else None,
                    "residuals_px": (
                        residuals.tolist() if residuals is not None else None
                    ),
                    "all_predicted_px": {
                        row["hole_id_str"]: list(
                            apply_local_to_pixel(
                                A,
                                float(row["easting_local_ft"]),
                                float(row["northing_local_ft"]),
                            )
                        )
                        for _, row in collars.iterrows()
                    }
                    if A is not None
                    else None,
                }
                OUT_JSON.write_text(json.dumps(payload, indent=2))
                st.success(f"Saved → {OUT_JSON.relative_to(REPO)}")

    with col_image:
        canvas = render_overlay(
            cropped,
            offset,
            st.session_state.picks,
            A,
            collars,
            st.session_state.show_predictions,
            st.session_state.active_hole,
        )
        disp = canvas.resize((DISPLAY_WIDTH, disp_h), Image.LANCZOS)

        click = streamlit_image_coordinates(disp, key="dhmap_canvas")

        if click is not None and click != st.session_state.last_click:
            st.session_state.last_click = click
            active = st.session_state.active_hole
            if active is not None:
                # Display click → cropped-image pixel coords → original-image pixel coords
                x_crop = int(click["x"] / scale)
                y_crop = int(click["y"] / scale)
                x_orig = x_crop + offset[0]
                y_orig = y_crop + offset[1]
                st.session_state.picks[active] = (x_orig, y_orig)
                # advance to next unpicked (re-derived after rerun)
                st.session_state.active_hole = None
                st.rerun()

        st.caption(
            f"Display {DISPLAY_WIDTH}×{disp_h} px · "
            f"crop {crop_w}×{crop_h} px @ offset ({offset[0]}, {offset[1]}) · "
            f"clicks stored in original {Image.open(DHMAP_PNG).size[0]}×"
            f"{Image.open(DHMAP_PNG).size[1]} px coords"
        )


if __name__ == "__main__":
    main()
