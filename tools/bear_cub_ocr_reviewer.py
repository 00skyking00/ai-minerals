"""Row-by-row Streamlit reviewer for Bear Cub problematic holes.

For each interval in the OCR'd table, the left column shows a cropped image
strip of just that row from the original PDF page (columns 1-5: TIME, DEPTH,
COLORS, Weight of gold, Volume), and the right column has aligned editable
fields (depth_from, depth_to, mg, colors, notes).

Pre-populates from the most-detailed available OCR source per hole:
  1. Existing user-saved structured corrections (`ocr_corrections.json`)
  2. Latest v2 re-OCR (`<stem>_v2.json`)
  3. Original OCR backup (`<stem>_v1backup.json`)
  4. Current v1 state (`<stem>.json`)

Crop boundaries adjustable per hole via top sliders; saved to
`ocr_review_crops.json` so each hole keeps its own settings.

Run:
    uv run streamlit run tools/bear_cub_ocr_reviewer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "bear_cub"
PNG_DIR = RAW / "page_pngs"
OCR_DIR = RAW / "full_ocr"
ROLLUPS = REPO / "data" / "derived" / "bear_cub_resource" / "hole_rollups.csv"
CORRECTIONS = RAW / "ocr_corrections.json"
CROPS_JSON = RAW / "ocr_review_crops.json"

# Default crop region (x_min, x_max, y_top, y_bottom, interval_start, interval_end) —
# covers cols 1-5 of a typical Hammon Field / Hammon Prospect log on the 1226×2016
# page render. interval_start/end specify which slice of the hole's intervals are
# rendered on this page. Override per page-of-hole via the sliders.
# Page 1 has a header to skip; subsequent pages start near the top.
DEFAULT_CROP_P1 = {"x_min": 20, "x_max": 720, "y_top": 420, "y_bottom": 1780}
DEFAULT_CROP_PN = {"x_min": 20, "x_max": 720, "y_top": 100, "y_bottom": 1900}

# Holes available to review — the 9 problem holes + H6760
PROBLEM_HOLES = [
    ("L6700 H6760", "convention_C", "Hammon Prospect — S1=44 / S2+S3=101 / S4=33 mg"),
    ("L6900 H6954", "convention_A", "Hammon Field — 17 mg total"),
    ("L6900 H6960", "convention_B", "Hammon Prospect Conv B (red-pen mg in vol col), 117 mg"),
    ("L6900 H6964", "no_bedrock",   "Alaska Gold Co. — NBR, 48.5 mg"),
    ("L7100 H7156", "convention_C", "Hammon Prospect Conv C (back-page assays), 157 mg"),
    ("L7100 H7160", "convention_C", "Hammon Prospect Conv C — 78 mg"),
    ("L7300 H7354", "no_bedrock",   "Hammon Field — bedrock 70.5; review needed"),
    ("L7300 H7360", "convention_C", "Hammon Prospect Conv C — 48 mg"),
    ("L7500 H7560", "convention_A", "Hammon Field — 6 mg total"),
    ("L7700 H7752", "convention_A", "Hammon Field — 41 mg"),
    ("L7700 H7754", "convention_A",
     "Hammon Field — Σ user mg = 98 mg, peak 0.132 at 13-15 ft (well above bedrock 62 ft). "
     "Sample-anchor redistribution rank #7 in the diff list (Σ|Δ|=7.7 mg, 8% change). "
     "Per-interval mg are authoritative on Conv A — likely candidate for the 🔓 Unlink button."),
]

DEFAULT_SAMPLE_ROW = {
    "sample_num": 1,
    "depth_from_ft": 0.0,
    "depth_to_ft": 0.0,
    "mg_total": 0.0,
    "source": "back-page",
    "notes": "",
}


@st.cache_data
def load_rollups() -> pd.DataFrame:
    if ROLLUPS.exists():
        return pd.read_csv(ROLLUPS)
    return pd.DataFrame()


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def list_page_pngs(file_stem: str) -> list[Path]:
    return sorted(PNG_DIR.glob(f"{file_stem}__p*.png"))


def _infer_missing_depths(iv: list[dict]) -> list[dict]:
    """Fill in missing depth_from / depth_to from neighboring intervals so
    rows with partial OCR depths aren't silently dropped (otherwise we lose
    bedrock-contact intervals when the operator wrote depths only on every
    other row)."""
    out = []
    prev_to = 0.0
    n = len(iv)
    for i, r in enumerate(iv):
        d_from = r.get("depth_from_ft")
        d_to = r.get("depth_to_ft")
        mg = r.get("estimated_weight_mg")
        colors = r.get("no_of_colors_total")
        notes = r.get("estimated_yield_raw") or ""
        if d_from is None and d_to is None and not mg and not colors and not notes:
            continue
        if d_from is None and d_to is not None:
            d_from = prev_to
        if d_to is None and d_from is not None:
            d_to = None
            for j in range(i + 1, n):
                nd_from = iv[j].get("depth_from_ft")
                nd_to = iv[j].get("depth_to_ft")
                if nd_from is not None:
                    d_to = nd_from
                    break
                if nd_to is not None:
                    d_to = nd_to
                    break
            if d_to is None or d_to <= d_from:
                d_to = d_from + 3.0
        if d_from is None or d_to is None:
            d_from = d_from or 0.0
            d_to = d_to or 0.0
        if d_to <= d_from:
            continue
        prev_to = d_to
        out.append({**r, "depth_from_ft": d_from, "depth_to_ft": d_to})
    return out


def get_best_intervals(stem: str, existing: dict) -> tuple[str, list[dict]]:
    """Return (source_label, intervals)."""
    if existing.get("intervals_structured"):
        rows = []
        for r in existing["intervals_structured"]:
            rows.append({
                "depth_from_ft": float(r.get("depth_from_ft") or 0),
                "depth_to_ft": float(r.get("depth_to_ft") or 0),
                "mg": float(r.get("mg") or 0),
                "colors": int(r.get("colors") or 0),
                "sample_num": int(r.get("sample_num") or 0),
                "notes": r.get("notes", "") or "",
            })
        # Sort by depth_from so a gap-fill row that historically landed at
        # the end (pre-fix) renders in its correct position now.
        rows.sort(key=lambda r: r["depth_from_ft"])
        if rows:
            return "user-saved (your edits)", rows

    for suffix, label in [("_v2", "v2 re-OCR"), ("_v1backup", "v1 backup"), ("", "v1 current")]:
        p = OCR_DIR / f"{stem}{suffix}.json"
        if not p.exists():
            continue
        j = load_json(p)
        raw_iv = (j.get("front") or {}).get("intervals") or [] if "front" in j else j.get("intervals") or []
        iv = _infer_missing_depths(raw_iv)
        rows = []
        for i in iv:
            d_from = i.get("depth_from_ft")
            d_to = i.get("depth_to_ft")
            if d_from is None or d_to is None:
                continue
            rows.append({
                "depth_from_ft": float(d_from),
                "depth_to_ft": float(d_to),
                "mg": float(i.get("estimated_weight_mg") or 0),
                "colors": int(i.get("no_of_colors_total") or 0),
                "sample_num": int(i.get("sample_num") or 0),
                "notes": i.get("estimated_yield_raw") or "",
            })
        if rows:
            return label, rows

    return "(no OCR found)", []


def compute_effective_mg(intervals: list[dict], samples: list[dict]) -> list[float]:
    """For each interval, return the mg value the grade calc should use.

    If interval has sample_num linked to a sample in the samples table:
      effective_mg = sample.mg_total × (interval.colors / Σ sample colors)
      (uniform across sample if all colors=0)
    Otherwise:
      effective_mg = interval.mg (user's per-interval value)
    """
    samples_by_num = {
        int(s.get("sample_num") or 0): s
        for s in samples
        if s.get("sample_num") and s.get("mg_total")
    }
    groups: dict[int, list[int]] = {}
    for i, r in enumerate(intervals):
        sn = int(r.get("sample_num") or 0)
        if sn > 0 and sn in samples_by_num:
            groups.setdefault(sn, []).append(i)

    eff = [float(r.get("mg") or 0) for r in intervals]
    for sn, idx_list in groups.items():
        sample_mg = float(samples_by_num[sn].get("mg_total") or 0)
        if sample_mg == 0:
            continue
        total_colors = sum(int(intervals[i].get("colors") or 0) for i in idx_list)
        if total_colors > 0:
            for i in idx_list:
                c = int(intervals[i].get("colors") or 0)
                eff[i] = sample_mg * (c / total_colors)
        else:
            per_iv = sample_mg / len(idx_list)
            for i in idx_list:
                eff[i] = per_iv
    return eff


def merge_to_canonical(
    stem: str,
    intervals: list[dict],
    bedrock_text: str,
    samples: list[dict] | None = None,
) -> Path:
    """Write user edits back into the canonical `<stem>.json`.

    When intervals are linked to samples (via `sample_num`), computes
    effective_mg via color-weighted distribution and stores that in
    `estimated_weight_mg` (so grade calc just works downstream). The raw
    sample_num + per-interval mg are also preserved in the canonical for
    traceability.
    """
    canon = OCR_DIR / f"{stem}.json"
    backup = OCR_DIR / f"{stem}_v1backup.json"
    if not canon.exists():
        return canon
    if not backup.exists():
        import shutil
        shutil.copy(canon, backup)
    doc = load_json(canon)
    front = doc.setdefault("front", {})

    samples = samples or []
    eff_mgs = compute_effective_mg(intervals, samples)

    new_intervals = []
    for i, r in enumerate(intervals, 1):
        sn = int(r.get("sample_num") or 0) or None
        eff = eff_mgs[i - 1]
        new_intervals.append({
            "interval_index": i,
            "time_text": "",
            "depth_from_ft": r["depth_from_ft"],
            "depth_to_ft": r["depth_to_ft"],
            "core_measured_volume_cu_ft": None,
            "core_before_pump_in": None,
            "core_after_pump_in": None,
            "formation_text": "",
            "no_of_colors_total": r.get("colors") or None,
            "no_of_colors_1": None, "no_of_colors_2": None, "no_of_colors_3": None,
            "estimated_weight_mg": float(eff) if eff else None,
            "user_mg_raw": float(r["mg"]) if r["mg"] else None,
            "sample_num": sn,
            "estimated_yield_raw": r.get("notes", "") or "",
            "remarks": "[user-reviewed via Streamlit]",
        })
    front["intervals"] = new_intervals
    if bedrock_text:
        bt = bedrock_text.strip()
        if bt and not bt.upper().startswith("N"):
            try:
                front["depth_to_bedrock_ft"] = float(bt)
            except ValueError:
                pass
        elif bt.upper().startswith("N"):
            front["depth_to_bedrock_ft"] = None

    if samples:
        back = doc.setdefault("back", {})
        back["samples"] = samples

    existing_notes = front.get("ocr_notes") or ""
    if "[user-reviewed" not in existing_notes:
        front["ocr_notes"] = existing_notes + " | [user-reviewed via Streamlit reviewer]"
    save_json(canon, doc)
    return canon


def get_crop_config(stem: str, page_idx: int, all_crops: dict, n_pages: int, intervals: list[dict]) -> dict:
    """Return the crop config for this hole's page, with sensible depth-range defaults.

    Defaults split the total drilled depth evenly across pages.
    """
    key = f"{stem}__p{page_idx + 1}"
    base = DEFAULT_CROP_P1 if page_idx == 0 else DEFAULT_CROP_PN

    # Compute default depth bounds for even split
    if intervals:
        depths_from = [r.get("depth_from_ft", 0) for r in intervals if r.get("depth_to_ft", 0) > 0]
        depths_to = [r.get("depth_to_ft", 0) for r in intervals]
        d_min = min(depths_from) if depths_from else 0
        d_max = max(depths_to) if depths_to else 0
    else:
        d_min, d_max = 0, 100

    span = d_max - d_min
    default_depth_from = d_min + (page_idx * span) / max(n_pages, 1)
    default_depth_to = d_min + ((page_idx + 1) * span) / max(n_pages, 1)

    if key in all_crops:
        cfg = dict(all_crops[key])
        # Backfill depth_from_ft / depth_to_ft if missing (legacy configs may have interval_start/end)
        if "depth_from_ft" not in cfg or "depth_to_ft" not in cfg:
            cfg["depth_from_ft"] = float(default_depth_from)
            cfg["depth_to_ft"] = float(default_depth_to)
        return cfg

    cfg = dict(base)
    cfg["depth_from_ft"] = float(default_depth_from)
    cfg["depth_to_ft"] = float(default_depth_to)
    return cfg


def save_crop_config(stem: str, page_idx: int, crop: dict, all_crops: dict) -> None:
    key = f"{stem}__p{page_idx + 1}"
    all_crops[key] = crop
    save_json(CROPS_JSON, all_crops)


@st.cache_data
def open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_strip(img: Image.Image, crop: dict, n_rows: int, row_idx: int) -> Image.Image:
    if n_rows <= 0:
        return img
    x0 = max(0, int(crop["x_min"]))
    x1 = min(img.width, int(crop["x_max"]))
    y_top = max(0, int(crop["y_top"]))
    y_bot = min(img.height, int(crop["y_bottom"]))
    h = (y_bot - y_top) / n_rows
    y0 = int(y_top + row_idx * h)
    y1 = int(y_top + (row_idx + 1) * h)
    return img.crop((x0, y0, x1, y1))


def main() -> None:
    st.set_page_config(page_title="Bear Cub OCR reviewer", layout="wide")
    st.title("Bear Cub OCR reviewer — row-by-row")
    st.caption(
        "Left: cropped strip from the PDF page (cols 1-5: TIME, DEPTH, COLORS, "
        "Weight, Volume). Right: editable depth_from / depth_to / mg / colors / "
        "notes. Pre-populated from the latest OCR source available."
    )

    rollups = load_rollups()
    corrections = load_json(CORRECTIONS)
    all_crops = load_json(CROPS_JSON)

    # ---------- Sidebar: hole picker ----------
    with st.sidebar:
        st.header("Holes")
        labels = []
        for stem, conv, note in PROBLEM_HOLES:
            existing = corrections.get(stem, {})
            badge = "✓" if existing.get("intervals_structured") else " "
            labels.append(f"{badge} {stem}")
        idx = st.radio(
            "Pick a hole",
            range(len(PROBLEM_HOLES)),
            format_func=lambda i: labels[i],
            key="hole_picker",  # persists across reruns (saves, data_editor edits)
        )
        st.markdown("---")
        st.caption("✓ = per-interval saved")

    file_stem, convention, note = PROBLEM_HOLES[idx]
    existing = corrections.get(file_stem, {})

    # ---------- Header ----------
    h1, h2, h3 = st.columns([3, 1, 1])
    with h1:
        st.subheader(file_stem)
        st.caption(f"**{convention}** — {note}")
    if not rollups.empty:
        rr = rollups[rollups["file_stem"] == file_stem]
        if not rr.empty:
            r = rr.iloc[0]
            with h2:
                st.metric("Surface-to-BR (oz/yd³)", f"{r['surface_to_br_grade']:.4f}")
                st.metric("Bedrock (ft)", f"{r['bedrock_depth_ft']:.0f}")
            with h3:
                st.metric("Total fine oz", f"{r['total_fine_oz_in_hole']:.4f}")
                st.metric("Pay-zone peak", f"{r['pay_zone_grade']:.4f}")

    # ---------- Page selector + crop sliders ----------
    pages = list_page_pngs(file_stem)
    if not pages:
        st.warning(f"No page PNGs found for {file_stem}")
        return

    src_label, intervals = get_best_intervals(file_stem, existing)

    # Initialize session-state row store
    rows_key = f"rows_{file_stem}"
    if rows_key not in st.session_state or st.session_state.get(f"_init_{file_stem}") != src_label:
        st.session_state[rows_key] = list(intervals)
        st.session_state[f"_init_{file_stem}"] = src_label

    st.caption(f"📄 Source: **{src_label}** · {len(st.session_state[rows_key])} intervals pre-populated")

    # ---------- Per-sample mg editor (rendered up here so its state is available
    #            for the auto-link button + effective-mg preview + save block) ----------
    s_key = f"samples_{file_stem}"

    def _samples_from_back(back_dict: dict) -> list[dict]:
        """Best-effort: parse 'N-MG' pairs from back_raw_text."""
        import re
        text = back_dict.get("back_raw_text", "") or ""
        pairs = re.findall(r"\b([1-9])\s*[-–]\s*(\d{1,4})\b(?!\s*ft)", text)
        if not pairs:
            return []
        return [
            {"sample_num": int(n), "depth_from_ft": 0.0, "depth_to_ft": 0.0,
             "mg_total": float(mg), "source": "back-page",
             "notes": "auto-parsed from back_raw_text — verify"}
            for n, mg in pairs[:8]
        ]

    canonical_doc_for_back = load_json(OCR_DIR / f"{file_stem}.json")
    back_for_samples = (canonical_doc_for_back.get("back") or {})

    if s_key not in st.session_state:
        existing_samples = existing.get("samples") or []
        if not existing_samples:
            existing_samples = _samples_from_back(back_for_samples)
        if not existing_samples:
            existing_samples = [DEFAULT_SAMPLE_ROW.copy()]
        st.session_state[s_key] = pd.DataFrame(existing_samples)

    # Show any pending auto-link diagnostic from the previous click
    if "_autolink_msg" in st.session_state:
        if st.session_state["_autolink_msg"].startswith("Auto-link result"):
            st.info(st.session_state.pop("_autolink_msg"))
        else:
            st.warning(st.session_state.pop("_autolink_msg"))

    st.markdown("### 🧮 Per-sample mg totals (back-page authoritative)")
    st.caption(
        "Add a row per sample from the back of the page. "
        "Auto-parsed from back_raw_text on first open — verify and edit as needed."
    )
    edited_s = st.data_editor(
        st.session_state[s_key],
        column_config={
            "sample_num": st.column_config.NumberColumn("Sample #", format="%d"),
            "depth_from_ft": st.column_config.NumberColumn("Depth from (ft)", format="%.1f"),
            "depth_to_ft": st.column_config.NumberColumn("Depth to (ft)", format="%.1f"),
            "mg_total": st.column_config.NumberColumn("Total mg", format="%.1f"),
            "source": st.column_config.SelectboxColumn(
                "Src", options=["back-page", "front-red", "front-text-code", "inferred"],
            ),
            "notes": st.column_config.TextColumn("Notes"),
        },
        num_rows="dynamic",
        hide_index=True,
        key=f"editor_s_{file_stem}",
        use_container_width=True,
    )
    # Don't write `st.session_state[s_key] = edited_s` here — that would
    # cause the data_editor to reset on the next rerun and drop every-other
    # cell edit. The widget manages its own state via the `key=` arg.
    from ai_minerals.bear_cub.row_editor_ui import render_sample_delete_button
    render_sample_delete_button(file_stem, s_key, edited_s)
    sample_total = float(edited_s["mg_total"].sum()) if not edited_s.empty else 0.0
    st.caption(f"**Σ samples mg total: {sample_total:.1f}**")

    # Pre-emptive validation of the samples table — flag issues before
    # the user clicks auto-link
    if not edited_s.empty:
        issues = []
        for i, s in edited_s.iterrows():
            sn_raw = s.get("sample_num")
            try:
                sn = int(sn_raw) if sn_raw not in (None, "") and pd.notna(sn_raw) else 0
            except (TypeError, ValueError):
                sn = 0
            df_v = float(s.get("depth_from_ft") or 0)
            dt_v = float(s.get("depth_to_ft") or 0)
            mg = float(s.get("mg_total") or 0)
            label = f"Sample {sn}" if sn > 0 else f"Row {i+1}"
            if sn == 0:
                issues.append(f"**{label}**: missing or zero `Sample #` — this row won't link any intervals.")
            if dt_v <= df_v and not (df_v == 0 and dt_v == 0):
                issues.append(
                    f"**{label}**: depth_from ({df_v:.1f}) ≥ depth_to ({dt_v:.1f}) — "
                    f"inverted or empty range, can't link to intervals. "
                    f"Likely a typo (e.g. `76.8` should be `86`?)."
                )
            elif df_v == 0 and dt_v == 0:
                issues.append(f"**{label}**: depth range is 0-0 — fill in `Depth from` and `Depth to`.")
        if issues:
            st.warning("**Sample table issues** (fix before clicking auto-link):\n\n"
                       + "\n".join(f"- {x}" for x in issues))
    st.markdown("---")

    # ---------- Review checklist (auto-generated anomaly items) ----------
    from ai_minerals.bear_cub.checklist_ui import render_checklist
    from ai_minerals.bear_cub.row_editor_ui import (
        render_reload_from_ocr,
        render_row_generator,
        wipe_iv_widget_state,
    )
    render_checklist(file_stem)
    st.markdown("---")

    # ---------- Single-page-at-a-time row-by-row editor ----------
    rows = st.session_state[rows_key]
    n_rows = len(rows)
    n_pages = len(pages)

    # Page selector — radio in horizontal mode, persists per hole
    page_state_key = f"current_page_{file_stem}"
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 0
    page_idx = st.radio(
        "Page",
        range(n_pages),
        index=st.session_state[page_state_key],
        format_func=lambda p: f"📄 Page {p+1}",
        horizontal=True,
        key=f"page_radio_{file_stem}",
    )
    st.session_state[page_state_key] = page_idx

    img = open_image(str(pages[page_idx]))
    crop = get_crop_config(file_stem, page_idx, all_crops, n_pages, rows)

    # Render the crop + depth-range inputs FIRST so user values flow into the filter
    with st.expander(
        f"Crop + depth-range for page {page_idx + 1} of {n_pages}",
        expanded=True,
    ):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            crop["x_min"] = st.slider("x_min", 0, img.width, int(crop["x_min"]), step=10,
                                      key=f"xmin_{file_stem}_p{page_idx}")
        with c2:
            crop["x_max"] = st.slider("x_max", 0, img.width, int(crop["x_max"]), step=10,
                                      key=f"xmax_{file_stem}_p{page_idx}")
        with c3:
            crop["y_top"] = st.slider("y_top", 0, img.height, int(crop["y_top"]), step=10,
                                      key=f"ytop_{file_stem}_p{page_idx}")
        with c4:
            crop["y_bottom"] = st.slider("y_bottom", 0, img.height, int(crop["y_bottom"]), step=10,
                                         key=f"ybot_{file_stem}_p{page_idx}")
        st.markdown(
            "**Depth range on this page** — what depths does the table on this PDF page cover? "
            "(e.g., page 1 = 0-40 ft, page 2 = 40-82 ft)"
        )
        r1, r2 = st.columns(2)
        with r1:
            crop["depth_from_ft"] = st.number_input(
                "Page depth from (ft)",
                value=float(crop["depth_from_ft"]),
                step=2.0, key=f"dfrom_{file_stem}_p{page_idx}",
            )
        with r2:
            crop["depth_to_ft"] = st.number_input(
                "Page depth to (ft)",
                value=float(crop["depth_to_ft"]),
                step=2.0, key=f"dto_{file_stem}_p{page_idx}",
            )
        prev_box = img.crop((crop["x_min"], crop["y_top"], crop["x_max"], crop["y_bottom"]))
        st.image(prev_box, caption="Crop window preview", width=400)
        if st.button(f"💾 Save crop config for page {page_idx + 1}",
                     key=f"savecrop_{file_stem}_p{page_idx}"):
            save_crop_config(file_stem, page_idx, crop, all_crops)
            st.success("Saved")

    # NOW filter intervals using the user-entered depth bounds
    depth_from = float(crop["depth_from_ft"])
    depth_to = float(crop["depth_to_ft"])
    page_indices = [
        i for i, r in enumerate(rows)
        if (r.get("depth_to_ft", 0) > depth_from) and (r.get("depth_from_ft", 0) < depth_to)
    ]
    page_rows = [rows[i] for i in page_indices]
    n_rows_this_page = len(page_rows)

    if page_rows:
        first_d = f"{page_rows[0]['depth_from_ft']:.0f}-{page_rows[0]['depth_to_ft']:.0f}"
        last_d = f"{page_rows[-1]['depth_from_ft']:.0f}-{page_rows[-1]['depth_to_ft']:.0f}"
        st.markdown(
            f"### 📄 Page {page_idx + 1} of {n_pages} · "
            f"depth {depth_from:.0f}-{depth_to:.0f} ft → "
            f"**{n_rows_this_page} intervals selected** (first: {first_d} ft · last: {last_d} ft)"
        )

        # Auto-link ALL intervals (across all pages) to samples by depth midpoint
        if st.button(
            "🔗 Auto-link ALL intervals (this hole, all pages) to samples by depth",
            key=f"autolink_{file_stem}_p{page_idx}",
            help="Iterates every interval in this hole — not just the current page — "
                 "and sets sample# from whichever sample's depth range contains the midpoint",
        ):
            if edited_s is not None and not edited_s.empty:
                samples_list = edited_s.to_dict("records")
                all_rows = st.session_state[rows_key]
                per_sample_count: dict[int, int] = {}
                n_linked = 0
                for actual_idx, row in enumerate(all_rows):
                    df_key = f"{file_stem}_iv{actual_idx}_from"
                    dt_key = f"{file_stem}_iv{actual_idx}_to"
                    d_from = float(st.session_state.get(df_key, row.get("depth_from_ft") or 0))
                    d_to = float(st.session_state.get(dt_key, row.get("depth_to_ft") or 0))
                    if d_to <= d_from:
                        continue
                    midpoint = (d_from + d_to) / 2
                    matched = 0
                    for s in samples_list:
                        s_from = float(s.get("depth_from_ft") or 0)
                        s_to = float(s.get("depth_to_ft") or 0)
                        if s_from <= midpoint <= s_to and s.get("sample_num"):
                            matched = int(s["sample_num"])
                            break
                    st.session_state[f"{file_stem}_iv{actual_idx}_sample"] = matched
                    if matched > 0:
                        n_linked += 1
                        per_sample_count[matched] = per_sample_count.get(matched, 0) + 1
                lines = []
                for s in samples_list:
                    sn = int(s.get("sample_num") or 0)
                    if sn == 0:
                        continue
                    s_from = float(s.get("depth_from_ft") or 0)
                    s_to = float(s.get("depth_to_ft") or 0)
                    lines.append(
                        f"Sample {sn}: depth {s_from:.0f}-{s_to:.0f} ft → "
                        f"{per_sample_count.get(sn, 0)} intervals linked"
                    )
                st.session_state["_autolink_msg"] = (
                    f"Auto-link result: **{n_linked} of {len(all_rows)}** intervals linked "
                    f"to {len([s for s in samples_list if s.get('sample_num')])} sample(s):\n\n"
                    + "\n".join(f"- {ln}" for ln in lines)
                )
                if n_linked == 0:
                    st.session_state["_autolink_msg"] += (
                        "\n\n⚠️ **0 intervals linked.** Most common cause: sample depth "
                        "ranges still at 0-0 (defaults). Edit the samples table at the top "
                        "of the page to fill in `depth_from_ft` and `depth_to_ft` per sample, "
                        "then click again."
                    )
                st.rerun()
            else:
                st.warning(
                    "No samples defined yet — fill in the 'Per-sample mg totals' table "
                    "above first."
                )

        # Inverse: clear all sample_num linkages on this hole. Use this for
        # Convention A/B holes where the per-interval mg are authoritative
        # and shouldn't be overwritten by sample-anchored color-weighted
        # redistribution.
        if st.button(
            "🔓 Unlink ALL intervals from samples (per-interval mg authoritative)",
            key=f"unlink_{file_stem}_p{page_idx}",
            help="Sets sample# = 0 on every interval row. Use for holes where "
                 "the operator recorded mg per interval (Convention A/B) so "
                 "color-weighted redistribution does NOT overwrite them on save.",
        ):
            all_rows = st.session_state[rows_key]
            n_cleared = 0
            for actual_idx in range(len(all_rows)):
                key = f"{file_stem}_iv{actual_idx}_sample"
                prev = int(st.session_state.get(key, 0) or 0)
                if prev != 0:
                    n_cleared += 1
                st.session_state[key] = 0
            st.session_state["_autolink_msg"] = (
                f"Cleared sample# on {n_cleared} of {len(all_rows)} intervals. "
                f"On Save, per-interval mg will be used directly (no sample-anchored "
                f"redistribution). The samples table itself is unchanged — delete rows "
                f"there if any are bogus."
            )
            st.rerun()
    else:
        st.markdown(
            f"### 📄 Page {page_idx + 1} of {n_pages} · "
            f"depth {depth_from:.0f}-{depth_to:.0f} ft → **0 intervals match**"
        )
        st.warning(
            "No intervals in this depth range — use the row generator below to "
            "fill the page (set `from`/`to` to this page's depth range)."
        )

    # Always-visible row generator (for filling missed bands or extending past TD)
    render_row_generator(file_stem, rows_key)
    st.markdown("")

    if n_rows_this_page == 0:
        st.caption("(no intervals on this page — set the interval range above and click Generate)")
    else:
        # Column header
        col_widths = [5, 0.8, 0.8, 1, 0.8, 0.7, 1.2, 0.7]
        hcols = st.columns(col_widths)
        hcols[0].markdown("**Cropped strip · TIME · DEPTH · COLORS · WEIGHT · VOLUME**")
        hcols[1].markdown("**from**")
        hcols[2].markdown("**to**")
        hcols[3].markdown("**mg**")
        hcols[4].markdown("**colors**")
        hcols[5].markdown("**sample#**")
        hcols[6].markdown("**notes**")
        hcols[7].markdown("**🗑️/➕**")

        for j, row in enumerate(page_rows):
            actual_idx = page_indices[j]
            strip = crop_strip(img, crop, n_rows_this_page, j)
            cols = st.columns(col_widths)
            with cols[0]:
                st.image(strip, use_container_width=True)
            with cols[1]:
                st.number_input(
                    "from", value=float(row.get("depth_from_ft", 0)),
                    step=1.0, label_visibility="collapsed",
                    key=f"{file_stem}_iv{actual_idx}_from",
                )
            with cols[2]:
                st.number_input(
                    "to", value=float(row.get("depth_to_ft", 0)),
                    step=1.0, label_visibility="collapsed",
                    key=f"{file_stem}_iv{actual_idx}_to",
                )
            with cols[3]:
                st.number_input(
                    "mg", value=float(row.get("mg", 0)),
                    step=0.5, label_visibility="collapsed",
                    key=f"{file_stem}_iv{actual_idx}_mg",
                )
            with cols[4]:
                st.number_input(
                    "colors", value=int(row.get("colors", 0)),
                    step=1, label_visibility="collapsed",
                    key=f"{file_stem}_iv{actual_idx}_colors",
                )
            with cols[5]:
                st.number_input(
                    "sample#", value=int(row.get("sample_num", 0) or 0),
                    min_value=0, step=1, label_visibility="collapsed",
                    key=f"{file_stem}_iv{actual_idx}_sample",
                    help="Link this interval to a sample (1, 2, ...). Use 0 = not linked.",
                )
            with cols[6]:
                st.text_input(
                    "notes", value=row.get("notes", ""),
                    label_visibility="collapsed",
                    key=f"{file_stem}_iv{actual_idx}_notes",
                )
            with cols[7]:
                if st.button("🗑️", key=f"del_{file_stem}_iv{actual_idx}",
                             help="Delete this interval row"):
                    cur = st.session_state[rows_key]
                    if 0 <= actual_idx < len(cur):
                        cur.pop(actual_idx)
                        st.session_state[rows_key] = cur
                        wipe_iv_widget_state(file_stem)
                        st.rerun()
                if st.button("➕", key=f"ins_{file_stem}_iv{actual_idx}",
                             help="Insert a new row after this one"):
                    cur = st.session_state[rows_key]
                    if 0 <= actual_idx < len(cur):
                        r = cur[actual_idx]
                        d_from = float(r.get("depth_to_ft") or 0)
                        d_step = float((r.get("depth_to_ft") or 0) - (r.get("depth_from_ft") or 0)) or 2.0
                        cur.insert(actual_idx + 1, {
                            "depth_from_ft": d_from,
                            "depth_to_ft": d_from + d_step,
                            "mg": 0.0,
                            "colors": 0,
                            "sample_num": 0,
                            "notes": "",
                        })
                        st.session_state[rows_key] = cur
                        wipe_iv_widget_state(file_stem)
                        st.rerun()

    # Collect ALL pages' edits from session state (widget keys persist across page nav)
    updated_rows: list[dict] = []
    for i, row in enumerate(rows):
        k_from = f"{file_stem}_iv{i}_from"
        if k_from in st.session_state:
            updated_rows.append({
                "depth_from_ft": float(st.session_state[k_from]),
                "depth_to_ft": float(st.session_state.get(f"{file_stem}_iv{i}_to", row.get("depth_to_ft", 0))),
                "mg": float(st.session_state.get(f"{file_stem}_iv{i}_mg", row.get("mg", 0))),
                "colors": int(st.session_state.get(f"{file_stem}_iv{i}_colors", row.get("colors", 0))),
                "sample_num": int(st.session_state.get(f"{file_stem}_iv{i}_sample", row.get("sample_num", 0)) or 0),
                "notes": st.session_state.get(f"{file_stem}_iv{i}_notes", row.get("notes", "")),
            })
        else:
            updated_rows.append({**row, "sample_num": int(row.get("sample_num", 0) or 0)})

    st.markdown("---")
    mg_sum = sum(r["mg"] for r in updated_rows)
    st.caption(f"**Σ mg across {n_rows} intervals: {mg_sum:.1f}**")

    # ---------- Reload from OCR (with confirmation) ----------
    render_reload_from_ocr(file_stem, rows_key, src_label, corrections, CORRECTIONS)

    # ---------- Back-of-page data (read from OCR's `back` field) ----------
    canonical_doc = load_json(OCR_DIR / f"{file_stem}.json")
    back = (canonical_doc.get("back") or {})
    if back:
        with st.expander(
            f"📄 Back-of-page data (assayed total = {back.get('actual_assayed_weight_mg', '?')} mg)",
            expanded=False,
        ):
            # Show the back page image alongside the raw text
            back_png = pages[-1] if len(pages) > 1 else None
            if back_png:
                bcol1, bcol2 = st.columns([1, 1])
                with bcol1:
                    st.image(open_image(str(back_png)), caption=f"Back page ({back_png.name})")
                with bcol2:
                    st.markdown("**Raw OCR'd back text:**")
                    st.code(back.get("back_raw_text", ""), language="text")
                    st.markdown(f"**Operator initials:** {back.get('operator_initials_raw', '')}")
                    st.markdown(f"**Green-pencil notes:** {back.get('green_pencil_notes', '')}")
            yield_calcs = back.get("yield_calcs") or []
            if yield_calcs:
                st.markdown(f"**{len(yield_calcs)} yield-calc formulas captured:**")
                yc_df = pd.DataFrame(yield_calcs)
                show_cols = [c for c in ["depth_from_ft", "depth_to_ft", "result_value",
                                         "result_unit_as_written", "result_qualifier"] if c in yc_df.columns]
                st.dataframe(yc_df[show_cols] if show_cols else yc_df, use_container_width=True, height=180)

    # ---------- Per-interval Σ vs per-sample Σ sanity warning ----------
    if mg_sum > 0 and sample_total > 0:
        diff_pct = (mg_sum - sample_total) / sample_total * 100
        if abs(diff_pct) > 5:
            st.warning(
                f"Per-interval Σ ({mg_sum:.1f}) vs per-sample Σ ({sample_total:.1f}) "
                f"differ by {diff_pct:+.1f}%."
            )

    with st.expander("📝 Hole notes + bedrock"):
        new_bedrock = st.text_input(
            "Corrected bedrock depth (ft)",
            value=existing.get("bedrock_depth_ft", ""),
            help="Numeric ft, or 'NBR' / 'No Bedrock Reached'",
            key=f"bedrock_{file_stem}",
        )
        general_notes = st.text_area(
            "General notes",
            value=existing.get("general_notes", ""),
            height=80,
            key=f"gennotes_{file_stem}",
        )
        mg_notes = st.text_area(
            "mg-capture notes (form variant, where mg appears, etc.)",
            value=existing.get("mg_notes", ""),
            height=120,
            key=f"mgnotes_{file_stem}",
        )

    # ---------- Effective-mg summary (after sample-anchor distribution) ----------
    samples_now = edited_s.to_dict("records") if (edited_s is not None and not edited_s.empty) else []
    eff = compute_effective_mg(updated_rows, samples_now)
    n_linked = sum(1 for r in updated_rows if int(r.get("sample_num") or 0) > 0)
    user_mg_sum = sum(float(r.get("mg") or 0) for r in updated_rows)
    eff_mg_sum = sum(eff)
    sample_mg_sum = sum(float(s.get("mg_total") or 0) for s in samples_now)

    st.markdown("---")
    st.markdown("### 🔍 Effective-mg preview (what the grade calc will use)")
    sm_cols = st.columns(4)
    sm_cols[0].metric("Per-interval Σ mg (your row inputs)", f"{user_mg_sum:.1f}")
    sm_cols[1].metric("Per-sample Σ mg (samples table)", f"{sample_mg_sum:.1f}")
    sm_cols[2].metric("Effective Σ mg (post-distribution)", f"{eff_mg_sum:.1f}")
    sm_cols[3].metric("Intervals linked to samples", f"{n_linked} / {len(updated_rows)}")
    if eff_mg_sum > 0 and sample_mg_sum > 0 and abs(eff_mg_sum - sample_mg_sum) / sample_mg_sum > 0.05:
        st.caption(
            f"ℹ️ Effective Σ ({eff_mg_sum:.1f}) ≠ per-sample Σ ({sample_mg_sum:.1f}) — "
            f"means some sample mg isn't being attributed to any interval. "
            f"Likely cause: sample's depth range doesn't overlap any interval, "
            f"or sample has sample_num=0."
        )

    if n_linked > 0 and samples_now:
        # Show per-sample distribution
        with st.expander("Per-sample distribution detail", expanded=False):
            samples_by_num = {int(s.get("sample_num") or 0): s for s in samples_now if s.get("sample_num")}
            for sn in sorted(samples_by_num.keys()):
                s = samples_by_num[sn]
                idx_in_sample = [i for i, r in enumerate(updated_rows) if int(r.get("sample_num") or 0) == sn]
                if not idx_in_sample:
                    continue
                colors_sum = sum(int(updated_rows[i].get("colors") or 0) for i in idx_in_sample)
                method = "color-weighted" if colors_sum > 0 else "uniform (all colors=0)"
                st.markdown(
                    f"**Sample {sn}** (depth {s.get('depth_from_ft')}-{s.get('depth_to_ft')} ft, "
                    f"total = {s.get('mg_total')} mg, {len(idx_in_sample)} intervals, distribution: {method}):"
                )
                preview_rows = [
                    {
                        "depth": f"{updated_rows[i]['depth_from_ft']:.0f}-{updated_rows[i]['depth_to_ft']:.0f}",
                        "colors": updated_rows[i].get("colors", 0),
                        "user_mg": updated_rows[i].get("mg", 0),
                        "effective_mg": round(eff[i], 2),
                    }
                    for i in idx_in_sample
                ]
                st.dataframe(pd.DataFrame(preview_rows), height=200, use_container_width=True)

    # ---------- Save ----------
    st.markdown("---")
    if st.button("💾 Save corrections for this hole", type="primary", use_container_width=True):
        # Filter out blank rows
        iv_filtered = [
            r for r in updated_rows
            if not (
                r["depth_from_ft"] == 0 and r["depth_to_ft"] == 0
                and r["mg"] == 0 and int(r.get("sample_num") or 0) == 0
            )
        ]
        s_filtered = (
            edited_s[~((edited_s["depth_from_ft"] == 0) & (edited_s["depth_to_ft"] == 0) & (edited_s["mg_total"] == 0))]
            .to_dict("records")
            if edited_s is not None and not edited_s.empty else []
        )

        # 1. Journal
        corrections[file_stem] = {
            "issue": convention,
            "bedrock_depth_ft": new_bedrock.strip(),
            "intervals_structured": iv_filtered,
            "samples": s_filtered,
            "mg_notes": mg_notes.strip(),
            "general_notes": general_notes.strip(),
        }
        save_json(CORRECTIONS, corrections)

        # 2. Canonical with sample-distributed effective_mg as estimated_weight_mg
        canon_path = merge_to_canonical(file_stem, iv_filtered, new_bedrock, samples=s_filtered)

        # 3. Refresh rows_key + wipe widget state so subsequent saves don't
        # read from stale pre-edit rows_key values (see suspect reviewer for
        # full reasoning).
        sorted_rows = sorted(iv_filtered,
                             key=lambda r: float(r.get("depth_from_ft") or 0))
        st.session_state[rows_key] = [
            {
                "depth_from_ft": float(r.get("depth_from_ft") or 0),
                "depth_to_ft":   float(r.get("depth_to_ft") or 0),
                "mg":            float(r.get("mg") or 0),
                "colors":        int(r.get("colors") or 0),
                "sample_num":    int(r.get("sample_num") or 0),
                "notes":         r.get("notes", "") or "",
            }
            for r in sorted_rows
        ]
        st.session_state[f"_init_{file_stem}"] = "user-saved (ocr_corrections)"
        from ai_minerals.bear_cub.row_editor_ui import wipe_iv_widget_state
        wipe_iv_widget_state(file_stem)

        st.success(
            f"Saved {file_stem}: {len(iv_filtered)} intervals (Σ effective mg = {eff_mg_sum:.1f}), "
            f"{len(s_filtered)} samples, bedrock={new_bedrock or '(blank)'}"
        )
        st.caption(
            f"📝 Journal: `{CORRECTIONS.relative_to(REPO)}`  ·  "
            f"📦 Canonical: `{canon_path.relative_to(REPO)}` (v1backup preserved)  ·  "
            "Run `uv run python tools/bear_cub_aggregate_ocr.py && "
            "uv run python tools/bear_cub_resource_analysis.py` to refresh resource numbers."
        )


if __name__ == "__main__":
    main()
