"""Streamlit spot-checker for Bear Cub OCR'd drill-log header data.

Run with:

    uv run streamlit run tools/bear_cub_ocr_review.py

Left pane shows the source page image, right pane shows editable fields for the
parsed CSV row. Saving recomputes lat/lon from the (possibly edited) local-grid
easting/northing using a fixed anchor.
"""

from __future__ import annotations

from datetime import date
from math import cos, radians
from pathlib import Path

import pandas as pd
import streamlit as st

# --- paths ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"
COMBO_DIR = REPO_ROOT / "data" / "raw" / "bear_cub" / "combo"
HEADER_DIR = REPO_ROOT / "data" / "raw" / "bear_cub" / "header_crops"

# --- coordinate anchor (matches notebooks pipeline) ----------------------

ANCHOR_E_LOCAL_FT = 77696
ANCHOR_N_LOCAL_FT = 22702
ANCHOR_LAT = 64.531171
ANCHOR_LON = -165.332170
FT_PER_DEG_LAT = 364400.0
FT_PER_DEG_LON = FT_PER_DEG_LAT * cos(radians(ANCHOR_LAT))  # ~156699

# --- known controlled-vocab values --------------------------------------

FORM_TYPES = [
    "Hammon Field Log",
    "Hammon Prospect Drilling Log",
    "Drill Report Frozen Ground",
    "Alaska Gold Company",
]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

STRING_COLS = [
    "file_stem",
    "line_id",
    "district",
    "claim",
    "panner",
    "driller",
    "ocr_notes",
]
INT_COLS = ["hole_id"]
FLOAT_COLS = [
    "easting_local_ft",
    "northing_local_ft",
    "elevation_ft",
    "bedrock_depth_ft",
    "total_depth_ft",
]
READONLY_FLOAT_COLS = ["lat_wgs84", "lon_wgs84"]


# --- helpers -------------------------------------------------------------


def local_to_latlon(e_local: float, n_local: float) -> tuple[float, float]:
    lat = ANCHOR_LAT + (n_local - ANCHOR_N_LOCAL_FT) / FT_PER_DEG_LAT
    lon = ANCHOR_LON + (e_local - ANCHOR_E_LOCAL_FT) / FT_PER_DEG_LON
    return lat, lon


@st.cache_data
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date_drilled"] = pd.to_datetime(df["date_drilled"], errors="coerce")
    df = df.sort_values("file_stem").reset_index(drop=True)
    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    # Format date back to YYYY-MM-DD strings; keep NaT empty
    if "date_drilled" in out.columns:
        out["date_drilled"] = out["date_drilled"].apply(
            lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else ""
        )
    out.to_csv(path, index=False)
    load_df.clear()


def confidence_badge(level: str) -> str:
    color = {"high": "#1a7f37", "medium": "#d97706", "low": "#b91c1c"}.get(
        str(level).lower(), "#6b7280"
    )
    label = str(level).upper() if isinstance(level, str) else "—"
    return (
        f"<span style='background:{color};color:white;padding:4px 10px;"
        f"border-radius:6px;font-weight:600;font-size:0.9rem'>{label}</span>"
    )


def safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def safe_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


# --- app -----------------------------------------------------------------

st.set_page_config(page_title="Bear Cub OCR review", layout="wide")

df = load_df(CSV_PATH)

# session state
if "edits" not in st.session_state:
    st.session_state.edits = {}  # file_stem -> dict of edited fields
if "current_stem" not in st.session_state:
    st.session_state.current_stem = df["file_stem"].iloc[0]

# --- sidebar: filters --------------------------------------------------

st.sidebar.header("Filter")
show_high = st.sidebar.checkbox("high confidence", value=True)
show_med = st.sidebar.checkbox("medium confidence", value=True)
show_low = st.sidebar.checkbox("low confidence", value=True)

allowed_levels = set()
if show_high:
    allowed_levels.add("high")
if show_med:
    allowed_levels.add("medium")
if show_low:
    allowed_levels.add("low")

filtered = df[df["ocr_confidence"].isin(allowed_levels)].reset_index(drop=True)
if filtered.empty:
    st.warning("No rows match the current filter. Toggle a confidence level on.")
    st.stop()

# Make sure current_stem is in filtered list, else snap to first
if st.session_state.current_stem not in filtered["file_stem"].values:
    st.session_state.current_stem = filtered["file_stem"].iloc[0]

# --- title bar w/ unsaved indicator -----------------------------------

current_stem = st.session_state.current_stem
unsaved = current_stem in st.session_state.edits and bool(
    st.session_state.edits[current_stem]
)
title_suffix = "  •  <span style='color:#d97706'>● Unsaved changes</span>" if unsaved else ""
st.markdown(
    f"## Bear Cub OCR review{title_suffix}",
    unsafe_allow_html=True,
)

# --- navigation row ----------------------------------------------------

stems = filtered["file_stem"].tolist()
cur_idx = stems.index(current_stem)

nav_l, nav_m, nav_r, nav_pos = st.columns([1, 3, 1, 2])
with nav_l:
    if st.button("← Prev", disabled=cur_idx == 0, width="stretch"):
        st.session_state.current_stem = stems[cur_idx - 1]
        st.rerun()
with nav_m:
    picked = st.selectbox(
        "Jump to hole",
        stems,
        index=cur_idx,
        label_visibility="collapsed",
    )
    if picked != current_stem:
        st.session_state.current_stem = picked
        st.rerun()
with nav_r:
    if st.button("Next →", disabled=cur_idx == len(stems) - 1, width="stretch"):
        st.session_state.current_stem = stems[cur_idx + 1]
        st.rerun()
with nav_pos:
    st.markdown(
        f"<div style='padding-top:0.4rem'>Hole <b>{cur_idx + 1}</b> of "
        f"<b>{len(stems)}</b> (filtered) · {len(df)} total</div>",
        unsafe_allow_html=True,
    )

# --- pull row from canonical df (with any pending in-memory edits applied for display only)

row = df[df["file_stem"] == current_stem].iloc[0].copy()
pending = st.session_state.edits.get(current_stem, {})
for k, v in pending.items():
    row[k] = v

# --- two-column layout -------------------------------------------------

img_col, form_col = st.columns([6, 4])

with img_col:
    view = st.radio(
        "Image view",
        ["combo (full page)", "header crop", "header+footer"],
        index=0,
        horizontal=True,
    )
    combo_path = COMBO_DIR / f"{current_stem}.png"
    hdr_path = HEADER_DIR / f"{current_stem}__p1_HDR.png"
    ftr_path = HEADER_DIR / f"{current_stem}__p1_FTR.png"

    if view == "combo (full page)":
        if combo_path.exists():
            st.image(str(combo_path), width="stretch")
        else:
            st.warning(f"Missing image: {combo_path}")
    elif view == "header crop":
        if hdr_path.exists():
            st.image(str(hdr_path), width="stretch")
        else:
            st.warning(f"Missing image: {hdr_path}")
    else:  # header+footer
        if hdr_path.exists():
            st.image(str(hdr_path), width="stretch", caption="header")
        else:
            st.warning(f"Missing image: {hdr_path}")
        if ftr_path.exists():
            st.image(str(ftr_path), width="stretch", caption="footer")
        else:
            st.info(f"No footer crop: {ftr_path.name}")

with form_col:
    # confidence badge + ocr_notes
    badge_col, stem_col = st.columns([1, 2])
    with badge_col:
        st.markdown(
            confidence_badge(safe_str(row["ocr_confidence"])),
            unsafe_allow_html=True,
        )
    with stem_col:
        st.markdown(f"**`{row['file_stem']}`**")

    notes_display = safe_str(row.get("ocr_notes"))
    st.text_area(
        "OCR notes (read-only context)",
        value=notes_display,
        height=70,
        disabled=True,
        key=f"notes_ro_{current_stem}",
    )

    st.divider()

    # collect edits into a dict; we'll commit to session_state.edits on diff
    new_vals: dict = {}

    # file_stem (don't actually allow editing since it's the join key, but show)
    new_vals["file_stem"] = st.text_input(
        "file_stem (key, do not edit)",
        value=safe_str(row["file_stem"]),
        disabled=True,
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        new_vals["hole_id"] = st.number_input(
            "hole_id",
            value=int(row["hole_id"]) if pd.notna(row["hole_id"]) else 0,
            step=1,
            format="%d",
        )
        new_vals["line_id"] = st.text_input(
            "line_id", value=safe_str(row["line_id"])
        )
        new_vals["district"] = st.text_input(
            "district", value=safe_str(row["district"])
        )
        new_vals["claim"] = st.text_input("claim", value=safe_str(row["claim"]))
        new_vals["form_type"] = st.selectbox(
            "form_type",
            FORM_TYPES,
            index=(
                FORM_TYPES.index(row["form_type"])
                if row["form_type"] in FORM_TYPES
                else 0
            ),
        )
        new_vals["ocr_confidence"] = st.selectbox(
            "ocr_confidence",
            CONFIDENCE_LEVELS,
            index=(
                CONFIDENCE_LEVELS.index(row["ocr_confidence"])
                if row["ocr_confidence"] in CONFIDENCE_LEVELS
                else 0
            ),
        )
    with cc2:
        new_vals["easting_local_ft"] = st.number_input(
            "easting_local_ft",
            value=float(row["easting_local_ft"])
            if pd.notna(row["easting_local_ft"])
            else 0.0,
            step=1.0,
            format="%.2f",
        )
        new_vals["northing_local_ft"] = st.number_input(
            "northing_local_ft",
            value=float(row["northing_local_ft"])
            if pd.notna(row["northing_local_ft"])
            else 0.0,
            step=1.0,
            format="%.2f",
        )
        # elevation/depths can be NaN — use a separate "blank?" toggle approach
        new_vals["elevation_ft"] = st.number_input(
            "elevation_ft (0 = blank)",
            value=float(row["elevation_ft"]) if pd.notna(row["elevation_ft"]) else 0.0,
            step=0.1,
            format="%.2f",
        )
        new_vals["bedrock_depth_ft"] = st.number_input(
            "bedrock_depth_ft (0 = blank)",
            value=float(row["bedrock_depth_ft"])
            if pd.notna(row["bedrock_depth_ft"])
            else 0.0,
            step=0.1,
            format="%.2f",
        )
        new_vals["total_depth_ft"] = st.number_input(
            "total_depth_ft (0 = blank)",
            value=float(row["total_depth_ft"])
            if pd.notna(row["total_depth_ft"])
            else 0.0,
            step=0.1,
            format="%.2f",
        )

    cc3, cc4 = st.columns(2)
    with cc3:
        # date_drilled — accept NaT
        cur_date = row["date_drilled"]
        if pd.isna(cur_date):
            cur_date = date(1925, 1, 1)
        else:
            cur_date = cur_date.date() if hasattr(cur_date, "date") else cur_date
        new_vals["date_drilled"] = st.date_input(
            "date_drilled",
            value=cur_date,
            min_value=date(1900, 1, 1),
            max_value=date(2000, 1, 1),
        )
        new_vals["panner"] = st.text_input("panner", value=safe_str(row["panner"]))
        new_vals["driller"] = st.text_input("driller", value=safe_str(row["driller"]))
    with cc4:
        new_vals["ocr_notes"] = st.text_area(
            "ocr_notes (editable)", value=safe_str(row["ocr_notes"]), height=120
        )

    st.divider()

    # read-only computed lat/lon based on the (possibly just-edited) easting/northing
    e_preview = new_vals["easting_local_ft"]
    n_preview = new_vals["northing_local_ft"]
    lat_preview, lon_preview = local_to_latlon(e_preview, n_preview)
    st.markdown("**Computed (read-only) WGS84 — recomputed on save:**")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.text_input(
            "lat_wgs84 (preview)",
            value=f"{lat_preview:.7f}",
            disabled=True,
            key=f"lat_preview_{current_stem}",
        )
    with rcol2:
        st.text_input(
            "lon_wgs84 (preview)",
            value=f"{lon_preview:.7f}",
            disabled=True,
            key=f"lon_preview_{current_stem}",
        )

    # detect dirty state by comparing widget values to canonical row
    def _val_for_compare(col: str, raw):
        if col == "date_drilled":
            return raw.isoformat() if raw is not None else ""
        return raw

    dirty: dict = {}
    for col, val in new_vals.items():
        if col == "file_stem":
            continue
        original = row[col]
        if col in FLOAT_COLS or col == "elevation_ft":
            original_f = (
                float(original) if pd.notna(original) else 0.0
            )  # we display NaN as 0.0
            if abs(float(val) - original_f) > 1e-9:
                dirty[col] = float(val)
        elif col in INT_COLS:
            original_i = int(original) if pd.notna(original) else 0
            if int(val) != original_i:
                dirty[col] = int(val)
        elif col == "date_drilled":
            orig_iso = (
                original.date().isoformat() if pd.notna(original) else None
            )
            new_iso = val.isoformat() if val else None
            if orig_iso != new_iso:
                dirty[col] = val
        else:
            if safe_str(val) != safe_str(original):
                dirty[col] = val

    if dirty:
        st.session_state.edits[current_stem] = dirty
    elif current_stem in st.session_state.edits:
        del st.session_state.edits[current_stem]

    # --- save / discard ---
    save_col, discard_col = st.columns([1, 1])
    with save_col:
        if st.button("Save changes", type="primary", width="stretch"):
            # Apply pending edits to df, recompute lat/lon, write CSV
            mask = df["file_stem"] == current_stem
            for col, val in new_vals.items():
                if col == "file_stem":
                    continue
                if col == "date_drilled":
                    df.loc[mask, col] = pd.to_datetime(val) if val else pd.NaT
                elif col in FLOAT_COLS or col == "elevation_ft":
                    # treat 0 as "user means 0", not blank — only blank if user typed nothing
                    df.loc[mask, col] = float(val)
                elif col in INT_COLS:
                    df.loc[mask, col] = int(val)
                else:
                    df.loc[mask, col] = val if val != "" else None
            # recompute lat/lon
            e_new = float(df.loc[mask, "easting_local_ft"].iloc[0])
            n_new = float(df.loc[mask, "northing_local_ft"].iloc[0])
            lat_new, lon_new = local_to_latlon(e_new, n_new)
            df.loc[mask, "lat_wgs84"] = lat_new
            df.loc[mask, "lon_wgs84"] = lon_new

            save_df(df, CSV_PATH)
            st.session_state.edits.pop(current_stem, None)
            st.toast(f"Saved {current_stem}", icon="✅")
            st.rerun()
    with discard_col:
        if st.button(
            "Discard edits",
            width="stretch",
            disabled=current_stem not in st.session_state.edits,
        ):
            st.session_state.edits.pop(current_stem, None)
            st.rerun()
