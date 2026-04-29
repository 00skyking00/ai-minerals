"""Streamlit helpers for the per-2-ft interval row editor.

Used by `tools/bear_cub_suspect_reviewer.py` and
`tools/bear_cub_ocr_reviewer.py`. Three helpers:

- `wipe_iv_widget_state(file_stem)`: drop every `{file_stem}_iv*` key from
  `st.session_state`. Call this before `st.rerun()` after any structural
  change (delete, insert-after, regenerate, reload) so widgets re-init
  from the new row data instead of carrying stale index-keyed state.
- `render_row_generator(file_stem, rows_key)`: an always-visible block
  for adding N new rows from `from` to `to` at `step` ft. Defaults
  extend past the last existing row.
- `render_reload_from_ocr(...)`: a checkbox-confirm flow for replacing
  user-saved interval edits with the OCR source. Sample-level mg +
  bedrock are preserved (different `corrections` keys).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st


def wipe_iv_widget_state(file_stem: str) -> None:
    """Drop every `{file_stem}_iv*` widget key from session_state.

    After a delete/insert/regenerate, widget state is keyed by old
    indices that no longer match positions in `rows_key`. Wiping forces
    every per-row widget to re-initialize from row data via `value=`.
    """
    pattern = re.compile(rf"^{re.escape(file_stem)}_iv\d+_")
    for k in [k for k in list(st.session_state.keys()) if pattern.match(k)]:
        del st.session_state[k]


def render_row_generator(file_stem: str, rows_key: str) -> None:
    """Always-visible block for generating N new rows of a fixed step.

    Defaults: from=max existing depth_to, to=from+30, step=2.0. Appends
    to `st.session_state[rows_key]` (does not replace) and wipes widget
    state so existing rows re-render cleanly under their new indices.
    """
    rows = st.session_state.get(rows_key, [])
    last_to = max((float(r.get("depth_to_ft") or 0) for r in rows), default=0.0)

    st.markdown("**➕ Generate rows** (appends; useful when OCR missed an interval band)")
    g = st.columns([1.2, 1.2, 1.2, 1.4, 4])
    with g[0]:
        gen_from = st.number_input(
            "from (ft)", min_value=0.0, max_value=500.0,
            value=float(last_to), step=2.0,
            key=f"gen_from_{file_stem}",
        )
    with g[1]:
        gen_to = st.number_input(
            "to (ft)", min_value=0.0, max_value=500.0,
            value=float(last_to + 30.0), step=2.0,
            key=f"gen_to_{file_stem}",
        )
    with g[2]:
        gen_step = st.number_input(
            "step (ft)", min_value=0.5, max_value=20.0,
            value=2.0, step=0.5,
            key=f"gen_step_{file_stem}",
        )
    n_to_make = max(0, int((gen_to - gen_from) / gen_step)) if gen_step > 0 else 0
    with g[3]:
        if st.button(
            f"Generate {n_to_make} rows",
            key=f"gen_btn_{file_stem}",
            disabled=n_to_make == 0,
        ):
            new_rows = []
            d = gen_from
            while d < gen_to - 1e-6:
                new_rows.append({
                    "depth_from_ft": float(d),
                    "depth_to_ft": float(min(d + gen_step, gen_to)),
                    "mg": 0.0,
                    "colors": 0,
                    "sample_num": 0,
                    "notes": "",
                })
                d += gen_step
            st.session_state[rows_key] = rows + new_rows
            wipe_iv_widget_state(file_stem)
            st.rerun()


def render_sample_delete_button(
    file_stem: str,
    s_key: str,
    edited_s,
) -> None:
    """Drop a sample row from the per-sample-mg table by sample_num.

    Streamlit's `data_editor` with `num_rows="dynamic"` supports row deletion
    via leftmost-cell select + DEL key, but that's hidden UX. This helper
    adds an explicit 🗑️ button. On click: pops the matching row from
    `session_state[s_key]`, wipes the data_editor widget key so it re-inits
    from the updated source, then reruns.
    """
    if edited_s is None or edited_s.empty:
        return
    nums = sorted({int(n) for n in edited_s["sample_num"].dropna().tolist() if int(n) > 0})
    if not nums:
        return
    cols = st.columns([1.4, 1, 5])
    with cols[0]:
        target = st.selectbox(
            "Drop sample #",
            options=nums,
            key=f"del_sample_pick_{file_stem}",
            label_visibility="collapsed",
        )
    with cols[1]:
        if st.button("🗑️ Delete sample row", key=f"del_sample_btn_{file_stem}"):
            current = list(st.session_state.get(s_key, []))
            new_rows = [r for r in current if int(r.get("sample_num") or 0) != int(target)]
            st.session_state[s_key] = new_rows
            # Wipe the data_editor widget key so it re-initializes from s_key
            st.session_state.pop(f"editor_s_{file_stem}", None)
            st.rerun()


def render_reload_from_ocr(
    file_stem: str,
    rows_key: str,
    src_label: str,
    corrections: dict,
    corrections_path: Path,
) -> None:
    """Two-step confirm flow for replacing saved interval edits with OCR source.

    Pops `intervals_structured` from `corrections[file_stem]` so the
    next page-load falls back to the OCR-priority chain. Sample-level
    mg, bedrock, and notes are preserved.
    """
    confirm_key = f"reload_confirm_{file_stem}"

    if not st.session_state.get(confirm_key):
        if st.button("🔄 Reload from OCR", key=f"reload_btn_{file_stem}"):
            st.session_state[confirm_key] = True
            st.rerun()
        return

    st.warning(
        "⚠️ This will discard your saved per-2-ft interval edits for this hole "
        "and reload from the OCR source. Sample-level mg, bedrock, and notes "
        "are kept (they're separate fields)."
    )
    ack = st.checkbox(
        "I understand — discard my interval edits",
        key=f"reload_ack_{file_stem}",
    )
    cc = st.columns([1, 1, 4])
    with cc[0]:
        if st.button("Confirm reload", type="primary", disabled=not ack,
                     key=f"reload_confirm_btn_{file_stem}"):
            entry = corrections.get(file_stem, {})
            entry.pop("intervals_structured", None)
            if entry:
                corrections[file_stem] = entry
            elif file_stem in corrections:
                del corrections[file_stem]
            corrections_path.write_text(json.dumps(corrections, indent=2))

            st.session_state.pop(rows_key, None)
            st.session_state.pop(f"_init_{file_stem}", None)
            st.session_state.pop(confirm_key, None)
            st.session_state.pop(f"reload_ack_{file_stem}", None)
            wipe_iv_widget_state(file_stem)
            st.rerun()
    with cc[1]:
        if st.button("Cancel", key=f"reload_cancel_{file_stem}"):
            st.session_state.pop(confirm_key, None)
            st.session_state.pop(f"reload_ack_{file_stem}", None)
            st.rerun()
