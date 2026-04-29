"""Re-OCR the 9 problematic Bear Cub holes with form-aware prompts.

Targets the holes whose resource-analysis output is zero or partial:
  - 7 zero-grade holes (likely missing estimated_weight_mg)
  - 2 zero-bedrock holes (collar OCR missed bedrock depth)

Reads optional reviewer notes from `data/raw/bear_cub/ocr_corrections.json`
and embeds them into the per-hole prompt. Writes results to
`data/raw/bear_cub/full_ocr/<stem>_v2.json` so the v1 results are preserved
for diff. Skip-on-exists: re-running won't repeat completed holes.

After re-OCR, optionally re-aggregates and re-runs the resource analysis
(use --refresh-downstream).

Run:
    uv run python tools/bear_cub_reocr_problem_holes.py
    uv run python tools/bear_cub_reocr_problem_holes.py --refresh-downstream
    uv run python tools/bear_cub_reocr_problem_holes.py --only "L7100 H7156"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

PAGE_DIR = REPO / "data" / "raw" / "bear_cub" / "page_pngs"
OCR_DIR = REPO / "data" / "raw" / "bear_cub" / "full_ocr"
CORRECTIONS = REPO / "data" / "raw" / "bear_cub" / "ocr_corrections.json"

# Same 9 holes as the Streamlit reviewer
PROBLEM_HOLES = [
    ("L6900 H6954", "zero_grade"),
    ("L6900 H6960", "zero_grade"),
    ("L6900 H6964", "no_bedrock"),
    ("L7100 H7156", "zero_grade"),
    ("L7100 H7160", "zero_grade"),
    ("L7300 H7354", "no_bedrock"),
    ("L7300 H7360", "both"),
    ("L7500 H7560", "zero_grade"),
    ("L7700 H7752", "zero_grade"),
]


def _import_full_log_ocr():
    """Import helper functions from bear_cub_full_log_ocr.py without making it a package."""
    src = REPO / "tools" / "bear_cub_full_log_ocr.py"
    spec = importlib.util.spec_from_file_location("bear_cub_full_log_ocr", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vp(msg: str) -> None:
    print(msg, flush=True)


# Form-aware re-OCR prompt — focuses the model on the fields v1 missed.
FOCUSED_PROMPT_TEMPLATE = """You are RE-READING a 1900s-era placer-Au drill log from the Bear Cub claim, Cape Nome mining district, Alaska. A previous OCR pass on this log produced incomplete data — your job is to fill in the gaps.

# Known issue with this hole

{issue_description}

# Reviewer notes (from human review of the v1 OCR output)

{reviewer_notes}

# Form-type guidance

{form_guidance}

# What to capture

Read the entire log and return a SINGLE JSON object (and nothing else) inside a ```json code block. The schema is the same as the v1 pass:

{{
  "hole_id": "string",
  "form_type": "Hammon Field Log | Hammon Prospect Drilling Log | Drill Report Frozen Ground | Alaska Gold Company",
  "easting_local_ft": number or null,
  "northing_local_ft": number or null,
  "elevation_ft": number or null,
  "total_depth_ft": number or null,
  "depth_to_bedrock_ft": number or null,
  "depth_into_bedrock_ft": number or null,
  "casing_or_bit_diameter_text": "string verbatim",
  "intervals": [
    {{
      "interval_index": int (1-based),
      "depth_from_ft": number,
      "depth_to_ft": number,
      "core_measured_volume_cu_ft": number or null,
      "no_of_colors_total": number or null,
      "estimated_weight_mg": number or null,
      "estimated_yield_raw": "string verbatim (e.g. 'VF', '5', '0.3 mg', '$0.15')",
      "formation_text": "string verbatim",
      "remarks": "string verbatim"
    }}
  ],
  "ocr_confidence": "high | medium | low",
  "ocr_notes": "string with any reading uncertainties — especially flag if the mg column is genuinely blank vs unreadable"
}}

# CRITICAL focus areas (this is why we're re-OCRing)

1. **`estimated_weight_mg`** — the weight of gold recovered from each interval, in milligrams. There are TWO conventions in this archive that the v1 OCR missed:

   **Convention A (most Hammon forms): mg in the "Weight of gold" column.** Column 4 of the per-interval table on Hammon Field Log + Hammon Prospect forms is labeled "Weight of gold" (or "Wt"). It contains numeric mg values — typically 0.1-50 mg, often integer (1, 2, 5, 17) but sometimes decimal. **DO NOT confuse this with the `No. of Colors` column (column 3)** — colors is a count of visible gold flakes (small integers like 1, 2, 5, 12); weight is a weighed mass. Both columns can be populated independently.

   **Convention B (some Hammon Prospect logs): text codes in "Weight of gold", numbers in RED inside the Volume column.** On a subset of Hammon Prospect Drilling Logs, the "Weight of gold" column does NOT contain numbers — it contains text codes ('VF' = very fine, 'F' = fine, '-' = none). On these logs, the actual mg values are handwritten in **RED pencil** inside the CORE/Measured-Volume column, alongside the larger black-pencil volume number. The red number is mg; the larger number is volume in cu ft. **You must read the red annotations in the volume column when the Weight column shows only text codes.** Capture the red-pencil numbers as `estimated_weight_mg`, and capture the text codes ('VF', 'F', etc.) as `estimated_yield_raw`.

   **PER-INTERVAL RED-PEN mg HAVE PRIORITY when present.** On Convention B and on some Hammon Field Logs, even when sample-level totals are written on the back page, individual 2-ft intervals may also have red-pencil mg numbers handwritten in the Volume column or in a small space above/below the volume number. CAPTURE THESE PER-INTERVAL VALUES if visible. They allow a finer-grained grade calculation than sample-level lumping. Per-interval reds typically sum to (or close to) the sample's back-page total — if they don't, note the discrepancy in `ocr_notes` but capture both. **Reading priority: front-page red-pen per-interval mg > back-page sample totals > front-page text codes.**

   **Convention C (Hammon Prospect with assay-on-back): per-sample mg only on the back page.** On some Hammon Prospect Drilling Logs the front-page "Weight of gold" column has ONLY text codes (VF / F / - / single-digit color counts) and NO red-pencil mg in the volume column. On these logs, the authoritative mg values live on the BACK of the page (page 3 typically) as a numbered list of sample sums:
   ```
   1 - 115
   2 - 11
   3 - 13
   4 - 13
   5 - 5
   ─────
       157
   ```
   These are per-sample (not per-interval) assayed weights — the operator submitted samples to a lab when only fines were visible. **The back-page assayed mg are MORE AUTHORITATIVE than any front-page numbers.**

   **CRITICAL output instruction for Convention C:** When the front-page mg column is empty (only VF/F/-/colors) AND the back page has sample-level mg sums, you MUST return the sample sums as `intervals` rows in the output, one row per sample. Set `depth_from_ft` and `depth_to_ft` to the sample's depth range, set `estimated_weight_mg` to the sample's mg, and put `"sample N (back-page assayed total)"` in `remarks`. Sample boundaries: read them off the front page (operators often wrote "Sample 1", "Sample 2" in a TIME or REMARKS cell at the row where each sample started); if not explicit, infer from context (e.g., samples often align with 14-ft / 24-ft / total-depth boundaries) and note the inference in `ocr_notes`. **DO NOT return 30 per-interval rows with null mg PLUS an explanation in ocr_notes — return the sample-level rows directly.** Example output for a hole with 2 samples (note: this is the actual JSON output you should produce, with the same schema as the per-interval case but only N rows where N = number of samples):

       intervals[0]: depth_from_ft=0,  depth_to_ft=14, estimated_weight_mg=68, remarks="sample 1 (back-page assayed total)"
       intervals[1]: depth_from_ft=14, depth_to_ft=68, estimated_weight_mg=10, remarks="sample 2 (back-page assayed total)"

   **Operator-totalled cross-check.** At the bottom of the table or after the last interval, the operator often wrote a column total (e.g., a line followed by "17 mg" or "76 mg" or "157 mg"). Use this as a sum-check on your captured mg values; mention any mismatch in `ocr_notes`.

2. **`depth_to_bedrock_ft`**: Look in the header AND the body for an explicit bedrock-depth notation:
   - Header field: "Depth to Bedrock" or "Bedrock at"
   - Inline remarks: "BR @ 67' 6\\"", "Bedrock at 70'", "Bedrock"
   - **NBR notation**: if the log says "NBR" anywhere, that means **No Bedrock Reached** — bedrock was not struck within the drilled depth. In this case, set `depth_to_bedrock_ft` to `null` and put `"NBR — no bedrock reached"` in `ocr_notes`. Do NOT set it to 0 or to the total depth.

3. **Casing/bit diameter**: Critical for grade calculation. Capture the exact text — typically '5⅞"', '6 1/4"', '6 in', '5.53', etc.

# JSON validity rules — non-negotiable

- Use double quotes only (no single quotes, no smart quotes).
- INSIDE string values, escape any literal double quote as \\\\".
- INSIDE string values, replace literal newlines with a single space.
- NO trailing commas, NO comments.
- For numeric fields where you cannot read a value, use null (NOT 0).

Return ONLY the JSON code block, no prose."""


ISSUE_DESCRIPTIONS = {
    "zero_grade": (
        "The v1 OCR captured the intervals but every `estimated_weight_mg` value "
        "is null. The downstream grade calculation needs mg values, so the resource "
        "estimate for this hole came out as 0 fine oz — almost certainly wrong, since "
        "neighboring holes from the same form type returned non-zero grades."
    ),
    "no_bedrock": (
        "The v1 OCR captured `depth_to_bedrock_ft` as null or 0. This excluded the "
        "hole from the volumetric resource estimate. The interval data for this hole "
        "was captured fine — we just need bedrock depth."
    ),
    "both": (
        "BOTH `depth_to_bedrock_ft` is null/0 AND every interval's `estimated_weight_mg` "
        "is null. Treat this as a from-scratch re-read."
    ),
}


def get_form_guidance(form_type: str) -> str:
    if "Hammon Prospect" in (form_type or ""):
        return (
            "**Hammon Prospect Drilling Log** — variant Hammon layout. The mg column "
            "is on the right portion of the per-interval table, often labeled 'Wt' or "
            "'Weight'. Values are handwritten in pencil, sometimes faint. The "
            "`estimated_yield_raw` column on this form often contains text codes "
            "('VF' = very fine, plus single-digit counts) that are a SEPARATE column "
            "from mg — capture BOTH if both exist. If the v1 saw only text codes, "
            "the mg column may be entirely blank in this log (operator used the text "
            "codes as the primary yield indicator), in which case ocr_notes should "
            "say so explicitly."
        )
    if "Hammon Field" in (form_type or ""):
        return (
            "**Hammon Field Log** — printed form, handwritten fields. mg column is "
            "typically labeled 'Weight of gold'. Values 0.1-50 mg per interval."
        )
    if "Frozen Ground" in (form_type or ""):
        return (
            "**Drill Report for Frozen Ground Only** — different layout from Hammon. "
            "Yield is under EST. YIELD with possibly a separate weight subcolumn. "
            "COLORS has 1, 2, 3 sub-columns for size classifications."
        )
    if "Alaska Gold" in (form_type or ""):
        return (
            "**Alaska Gold Company** form (1955) — same layout as Hammon Field Log. "
            "Bedrock-depth notation may be only in remarks ('BR @ X')."
        )
    return "Form type unknown — read carefully and pick the matching form_type from the schema."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Only re-OCR this single file_stem")
    parser.add_argument("--refresh-downstream", action="store_true",
                        help="After re-OCR, run aggregator + resource analysis")
    parser.add_argument("--force", action="store_true",
                        help="Re-OCR even if _v2.json already exists")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="API retry attempts per hole (default 5; set 1 to fail fast)")
    parser.add_argument("--max-img-bytes", type=int, default=5_000_000,
                        help="Force downsample to this raw byte ceiling (default 5 MB; "
                             "set lower e.g. 1_500_000 to avoid Opus 90s-timeout on large pages)")
    args = parser.parse_args()

    # Optional reviewer notes
    notes_by_stem: dict[str, dict] = {}
    if CORRECTIONS.exists():
        with open(CORRECTIONS) as f:
            notes_by_stem = json.load(f)
        vp(f"Loaded reviewer notes for {len(notes_by_stem)} hole(s) from {CORRECTIONS.relative_to(REPO)}")
    else:
        vp(f"No reviewer notes file at {CORRECTIONS.relative_to(REPO)} — proceeding with form-only guidance")

    # Import helper functions from the existing OCR script
    helpers = _import_full_log_ocr()
    client = anthropic.Anthropic()

    targets = [(s, i) for s, i in PROBLEM_HOLES if (args.only is None or s == args.only)]
    if not targets:
        vp(f"No matching holes for --only={args.only!r}")
        return

    vp(f"\nRe-OCR {len(targets)} hole(s); writing to *_v2.json under full_ocr/")
    total_in = total_out = 0

    for i, (stem, issue) in enumerate(targets, 1):
        out_json = OCR_DIR / f"{stem}_v2.json"
        if out_json.exists() and not args.force:
            vp(f"[{i}/{len(targets)}] {stem}: SKIP (v2 exists)")
            continue

        # Load v1 to learn form_type
        v1_path = OCR_DIR / f"{stem}.json"
        form_type = ""
        if v1_path.exists():
            with open(v1_path) as f:
                v1 = json.load(f)
            form_type = v1.get("front", {}).get("form_type", "")

        notes = notes_by_stem.get(stem, {})
        notes_blob = ""
        if notes:
            notes_blob = json.dumps(
                {k: v for k, v in notes.items() if v},
                indent=2,
            )
        else:
            notes_blob = "(no reviewer notes — review via tools/bear_cub_ocr_reviewer.py for best results)"

        prompt = FOCUSED_PROMPT_TEMPLATE.format(
            issue_description=ISSUE_DESCRIPTIONS[issue],
            reviewer_notes=notes_blob,
            form_guidance=get_form_guidance(form_type),
        )

        # Send ALL pages — the back-page assay-mg cross-check (Convention C) needs them.
        all_pages = sorted(PAGE_DIR.glob(f"{stem}__p*.png"))
        if not all_pages:
            vp(f"[{i}/{len(targets)}] {stem}: no page PNGs, skipping")
            continue

        vp(f"[{i}/{len(targets)}] {stem} ({form_type or '?'}, {issue}): {len(all_pages)} page(s)")
        try:
            # Build content with a per-call image-size override
            content = []
            for p in all_pages:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": helpers.encode(p, max_b64_bytes=args.max_img_bytes),
                    },
                })
            content.append({"type": "text", "text": prompt})

            import anthropic as _ant
            import time as _t
            last_err = None
            payload = None
            usage = None
            for attempt in range(args.max_retries):
                try:
                    response = client.with_options(timeout=90.0).messages.create(
                        model="claude-opus-4-7",
                        max_tokens=16384,
                        thinking={"type": "adaptive"},
                        messages=[{"role": "user", "content": content}],
                    )
                    text = "".join(b.text for b in response.content if b.type == "text")
                    payload = helpers.parse_json_response(text)
                    usage = response.usage
                    break
                except (_ant.APIStatusError, _ant.APITimeoutError, _ant.APIConnectionError) as e:
                    last_err = e
                    wait = min(60.0, 2.0 ** attempt)
                    status = getattr(e, "status_code", "n/a")
                    vp(f"      [retry {attempt + 1}/{args.max_retries} after {wait:.0f}s — {type(e).__name__} status={status}]")
                    if attempt < args.max_retries - 1:
                        _t.sleep(wait)
            if payload is None:
                raise last_err or RuntimeError("no payload after retries")
        except Exception as e:
            vp(f"    FAIL: {type(e).__name__}: {e}")
            continue

        total_in += usage.input_tokens
        total_out += usage.output_tokens
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        n_intervals = len(payload.get("intervals", []))
        n_with_mg = sum(1 for iv in payload.get("intervals", [])
                        if iv.get("estimated_weight_mg") is not None)
        bedrock = payload.get("depth_to_bedrock_ft")
        vp(f"    → {n_intervals} intervals, {n_with_mg} with mg, bedrock={bedrock}")

    vp(f"\nTotal token usage: in={total_in:,}, out={total_out:,}")

    if args.refresh_downstream:
        vp("\nRefreshing downstream artifacts...")
        # Aggregator + resource analysis read the v1 jsons by default. Inform the user
        # they need to merge v2 → v1 manually, OR adjust those scripts. For now, just
        # report what to do next.
        vp(
            "  NOTE: aggregator (tools/bear_cub_aggregate_ocr.py) reads <stem>.json, not\n"
            "  <stem>_v2.json. Diff each v2 against v1, decide which fields to keep, then\n"
            "  either replace v1 with v2 (preserving v1 as <stem>_v1backup.json) or merge\n"
            "  selectively. Once merged, run:\n"
            "      uv run python tools/bear_cub_aggregate_ocr.py\n"
            "      uv run python tools/bear_cub_resource_analysis.py"
        )


if __name__ == "__main__":
    main()
