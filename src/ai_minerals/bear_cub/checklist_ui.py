"""Streamlit UI helpers for the per-hole review checklist.

Used by `tools/bear_cub_ocr_reviewer.py` and `tools/bear_cub_suspect_reviewer.py`
to render the checklist section + persist user resolutions to
`data/raw/bear_cub/review_checklist.json`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[3]
CHECKLIST_FILE = REPO / "data" / "raw" / "bear_cub" / "review_checklist.json"

STATUS_OPTIONS = ["open", "not_an_issue", "fixed"]
STATUS_BADGES = {
    "open":         "🟡 Open",
    "not_an_issue": "⚪ Not an issue",
    "fixed":        "🟢 Fixed",
}


def _load() -> dict:
    if CHECKLIST_FILE.exists():
        return json.loads(CHECKLIST_FILE.read_text())
    return {}


def _save(data: dict) -> None:
    CHECKLIST_FILE.write_text(json.dumps(data, indent=2))


def render_checklist(file_stem: str) -> None:
    """Render the checklist section for one hole. Saves on every change."""
    data = _load()
    hole_entry = data.get(file_stem)
    if not hole_entry or not hole_entry.get("items"):
        st.markdown("### 📋 Review checklist")
        st.caption(
            "No checklist items for this hole. Run "
            "`uv run python tools/bear_cub_generate_checklist.py` "
            "to (re)generate items from anomaly checks."
        )
        return

    items = hole_entry["items"]
    n_open = sum(1 for it in items if it.get("status", "open") == "open")
    n_fixed = sum(1 for it in items if it.get("status") == "fixed")
    n_nai = sum(1 for it in items if it.get("status") == "not_an_issue")

    st.markdown(
        f"### 📋 Review checklist · "
        f"🟡 {n_open} open · 🟢 {n_fixed} fixed · ⚪ {n_nai} not-an-issue"
    )
    if n_open == 0:
        st.success("All checklist items resolved for this hole. ✓")
    st.caption(
        "Auto-generated from `bear_cub_generate_checklist.py`. Item IDs are stable, "
        "so your status + comment persist across regenerations. "
        "Edits save immediately."
    )

    for idx, item in enumerate(items):
        item_id = item.get("id", f"item_{idx}")
        status = item.get("status", "open")
        category = item.get("category", "")
        description = item.get("description", "")
        comment = item.get("comment", "") or ""

        # Visual: status badge + ID + category, then description, then status radio + comment
        with st.container(border=True):
            cols = st.columns([3, 2])
            with cols[0]:
                st.markdown(f"**{STATUS_BADGES.get(status, status)}** · `{item_id}` · *{category}*")
            with cols[1]:
                st.caption(item.get("resolved_at") or "")
            st.markdown(description)

            status_key = f"checklist_{file_stem}_{item_id}_status"
            comment_key = f"checklist_{file_stem}_{item_id}_comment"

            new_status = st.radio(
                "Status",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(status) if status in STATUS_OPTIONS else 0,
                horizontal=True,
                label_visibility="collapsed",
                key=status_key,
            )
            new_comment = st.text_area(
                "Comment",
                value=comment,
                placeholder="(optional) explain how this was resolved or why it's not an issue",
                height=68,
                label_visibility="collapsed",
                key=comment_key,
            )

            # Persist if anything changed
            if new_status != status or new_comment != comment:
                items[idx]["status"] = new_status
                items[idx]["comment"] = new_comment
                if new_status in ("not_an_issue", "fixed"):
                    items[idx]["resolved_at"] = datetime.now().isoformat(timespec="seconds")
                else:
                    items[idx]["resolved_at"] = None
                hole_entry["items"] = items
                data[file_stem] = hole_entry
                _save(data)
