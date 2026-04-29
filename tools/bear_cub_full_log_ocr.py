"""Per-page OCR of all 24 Bear Cub drill logs into a structured corpus.

Two API calls per log:
  1. Front of sheet — header, drilling table, water measurements, footer.
  2. Back of sheet — yield-calculation formulas verbatim, decomposed into
     numeric terms and operators (no semantic labeling).

Uses `messages.create` with prompt-embedded JSON instructions + manual
parsing — bypasses the structured-output grammar compiler which was hanging
on nested-list schemas. Same data fidelity, more reliable.

Output: per-log JSON files at `data/raw/bear_cub/full_ocr/<file_stem>.json`.

Run:
    uv run python tools/bear_cub_full_log_ocr.py
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import anthropic
import io
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

PAGE_DIR = REPO / "data" / "raw" / "bear_cub" / "page_pngs"
COLLARS_CSV = REPO / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"
OUT_DIR = REPO / "data" / "raw" / "bear_cub" / "full_ocr"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def vp(msg: str) -> None:
    """Verbose-print: flushed immediately so progress shows in tail -f."""
    print(msg, flush=True)


# =============================================================================
# Prompts — the model returns JSON in a code block; we parse it manually.
# =============================================================================

FRONT_PROMPT = """You are reading the FRONT of a 1900s-era placer-Au drill-log sheet from the Cape Nome mining district, Alaska. There are four form types in the archive:

1. **Hammon Field Log** (~1925) — printed form, handwritten fields. Per-interval table columns: TIME, DEPTH, No. of Colors, Weight of gold, CORE (Measured Volume / Before Pump / After Pump), FORMATION, WATER MEASUREMENTS (Depth From, Depth To, Volume in Gallons, Time), REMARKS. The TIME field carries dates (e.g., "8/6"), shift labels ("D.S.", "N.S."), or sample IDs ("Sample 1") rather than clock time.
2. **Hammon Prospect Drilling Log** (1925-1936) — variant Hammon layout, same fundamentals.
3. **Drill Report for Frozen Ground Only** (likely 1919) — different layout. Column headers: TIME, DEPTH, FORMATION, CORE, COLORS (1, 2, 3 sub-columns for size/weight classifications), EST. YIELD, REMARKS. TIME is sometimes clock time on this form.
4. **Alaska Gold Company** (1955) — same physical form layout as Hammon, just different operator name.

Local-grid coordinates are in feet. Easting range typically 76,000-78,000; northing 22,000-24,000. Read each digit carefully. Capture as the raw number (no "E76,502" prefix, just 76502).

# Output format

Return a SINGLE JSON object (and nothing else) inside a ```json code block. Schema:

{
  "hole_id": "string",
  "line_id": "string or empty",
  "district": "string or empty",
  "claim": "string or empty",
  "form_type": "Hammon Field Log | Hammon Prospect Drilling Log | Drill Report Frozen Ground | Alaska Gold Company",
  "easting_local_ft": number or null,
  "northing_local_ft": number or null,
  "elevation_ft": number or null,
  "date_started": "string (ISO YYYY-MM-DD if recoverable, else raw)",
  "date_finished": "string",
  "panner": "string or empty",
  "driller": "string or empty",
  "day_shift_crew": "string or empty",
  "night_shift_crew": "string or empty",
  "total_depth_ft": number or null,
  "depth_to_bedrock_ft": number or null,
  "depth_into_bedrock_ft": number or null,
  "depth_of_muck_ft": number or null,
  "casing_or_bit_diameter_text": "string (verbatim, e.g. '5⅞\"', '6 in', '5.53')",
  "width_from_text": "string verbatim",
  "width_to_text": "string verbatim",
  "intervals": [
    {
      "interval_index": int (1-based),
      "time_text": "string verbatim",
      "depth_from_ft": number or null,
      "depth_to_ft": number or null,
      "core_measured_volume_cu_ft": number or null,
      "core_before_pump_in": number or null,
      "core_after_pump_in": number or null,
      "formation_text": "string verbatim (preserve abbreviation codes)",
      "no_of_colors_total": number or null,
      "no_of_colors_1": number or null,
      "no_of_colors_2": number or null,
      "no_of_colors_3": number or null,
      "estimated_weight_mg": number or null,
      "estimated_yield_raw": "string verbatim",
      "remarks": "string verbatim"
    }
  ],
  "water_measurements": [
    {
      "meas_index": int,
      "depth_from_ft": number or null,
      "depth_to_ft": number or null,
      "volume_value": number or null,
      "volume_unit_as_written": "string (e.g. 'Gal', 'cu ft', or empty)",
      "time_text": "string verbatim",
      "raw_row_text": "string verbatim full row"
    }
  ],
  "ocr_confidence": "high | medium | low",
  "ocr_notes": "string with any reading uncertainties"
}

For numeric fields where you cannot read a value, use null (NOT 0). For string fields use empty string "". Do NOT add fields not in the schema. Return ONLY the JSON code block, no prose before or after.

# JSON validity rules — non-negotiable

- Use double quotes only (no single quotes, no smart quotes).
- INSIDE string values, escape any literal double quote as \\".
- INSIDE string values, replace literal newlines with a single space (do NOT emit a real newline within a string value).
- NO trailing commas after the last item in any array or object.
- NO comments.
- All keys must match the schema exactly (case-sensitive).
- If a string field contains arithmetic or fractions, write them in plain ASCII (e.g. "63½" → "63 1/2", "5.53²" → "5.53^2"). Numeric fields stay numeric (e.g. 63.5)."""

BACK_PROMPT = """You are reading the BACK of a 1900s-era placer-Au drill-log sheet. The back is mostly blank printed form (the form is double-sided) with handwritten yield calculations in pencil. Some marks may be 180° rotated or mirrored if the sheet was scanned upside-down.

# What's typically on the back

1. **Actual amalgamated gold weight** at the top — e.g. "actual wt = 126 mg".
2. **One or more yield-calculation formulas** verbatim, like:
   - `5.024 × 126 / 63½ = 9.94`
   - `(282.6 × 87) / (5.53² × 60.7) = 13.3¢`
   - `36 / [(.200)(6)] × 1.588 = 47.6 ¢/yd`
   Each may have a description ("Top 15 ft", "0-15") and a unit suffix.
3. **Operator initials** (e.g. "W.W.B.", "JDH", "F.K.").
4. **Green-pencil annotations** (cross-checks or revisions in different ink).
5. **Geological interpretation** — free-text pay-zone calls.

# Critical: do NOT invent semantic labels

Capture each formula's terms verbatim with no interpretation. For "5.024 × 126 / 63½":
- term 0: value=5.024, raw_token="5.024", operator_to_prev=""
- term 1: value=126, raw_token="126", operator_to_prev="*"
- term 2: value=63.5, raw_token="63½", operator_to_prev="/"

For "5.53²" prefer a single term {value: 5.53, raw_token: "5.53²", operator_to_prev: "*"} — the square is captured in the raw_token.

# Output format

Return a SINGLE JSON object inside a ```json code block. Schema:

{
  "actual_assayed_weight_mg": number or null,
  "yield_calcs": [
    {
      "calc_index": int (1-based),
      "description_text": "string verbatim (e.g. 'Top 15 ft', 'full hole')",
      "depth_from_ft": number or null,
      "depth_to_ft": number or null,
      "formula_raw_text": "string verbatim",
      "terms": [
        {"value": number, "operator_to_prev": "string ('' for first, '*', '/', '+', '-', '(', ')')", "raw_token": "string verbatim"}
      ],
      "result_value": number or null,
      "result_unit_as_written": "string ('¢', 'cents/cu yd', 'BRF', or empty)",
      "result_qualifier": "string (e.g. 'disregarding shrinkage', or empty)"
    }
  ],
  "operator_initials_raw": "string (semicolon-joined if multiple)",
  "green_pencil_notes": "string verbatim",
  "geological_interpretation": "string verbatim",
  "back_raw_text": "string with all readable handwriting verbatim"
}

If the back is essentially blank, return yield_calcs=[] and put any text in back_raw_text. Return ONLY the JSON code block, no prose.

# JSON validity rules — non-negotiable

- Use double quotes only (no single quotes, no smart quotes).
- INSIDE string values, escape any literal double quote as \\".
- INSIDE string values, replace literal newlines with a single space (do NOT emit a real newline within a string value).
- NO trailing commas after the last item in any array or object.
- NO comments.
- All keys must match the schema exactly (case-sensitive)."""


def encode(p: Path, max_b64_bytes: int = 5_000_000) -> str:
    """Base64-encode an image, downsampling if needed to fit under the API's 5 MB limit."""
    raw = p.read_bytes()
    # Base64 inflates by ~4/3 (33%), so source under (max / 1.34) is safe.
    safe_raw_max = int(max_b64_bytes * 0.74)
    if len(raw) <= safe_raw_max:
        return base64.standard_b64encode(raw).decode("ascii")

    # Need to downsample. Open + resize until the encoded blob fits.
    img = Image.open(p)
    scale = (safe_raw_max / len(raw)) ** 0.5  # area-scale heuristic
    for attempt in range(5):
        scale = max(0.05, scale * (0.92 ** attempt))
        new_w = max(800, int(img.width * scale))
        new_h = max(800, int(img.height * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
        encoded_size = len(out) * 4 // 3 + 4
        if encoded_size <= max_b64_bytes:
            vp(f"      [downsampled {p.name}: {img.size} → {(new_w, new_h)}, "
               f"{len(raw)} → {len(out)} bytes raw]")
            return base64.standard_b64encode(out).decode("ascii")
    # Last-ditch: aggressive shrink
    resized = img.resize((1000, int(1000 * img.height / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def parse_json_response(text: str) -> dict:
    """Pull a JSON object out of a model response (with or without ```json fences).

    Falls back to json-repair when strict parsing fails (handles trailing
    commas, unescaped quotes, single quotes, comments — all the typical
    LLM-emitted JSON imperfections).
    """
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        if start < 0:
            raise ValueError("no JSON object in response")
        text = text[start:]
        end = text.rfind("}")
        text = text[: end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to json-repair (lenient parser)
        try:
            from json_repair import repair_json
        except ImportError as e:
            raise json.JSONDecodeError(
                "strict parse failed and json-repair not installed",
                text, 0,
            ) from e
        repaired = repair_json(text, return_objects=True)
        if not isinstance(repaired, dict):
            raise ValueError(f"json-repair returned non-dict: {type(repaired).__name__}")
        return repaired


def call_with_retry(client, prompt: str, images: list[Path], retries: int = 5) -> tuple[dict, anthropic.types.Usage]:
    content = []
    for p in images:
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": encode(p)},
            }
        )
    content.append({"type": "text", "text": prompt})

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.with_options(timeout=90.0).messages.create(
                model="claude-opus-4-7",
                max_tokens=16384,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            try:
                payload = parse_json_response(text)
            except (ValueError, json.JSONDecodeError) as e:
                vp(f"      JSON-parse error: {e}; first 200 chars: {text[:200]!r}")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                raise
            return payload, response.usage
        except (anthropic.APIStatusError, anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            last_err = e
            wait = min(60.0, 2.0 ** attempt)
            status = getattr(e, "status_code", "n/a")
            vp(f"      [retry {attempt + 1}/{retries} after {wait:.0f}s — {type(e).__name__} status={status}]")
            time.sleep(wait)
    if last_err:
        raise last_err
    raise RuntimeError("Exhausted retries without exception")


def main() -> None:
    client = anthropic.Anthropic()
    df = pd.read_csv(COLLARS_CSV)
    file_stems = df["file_stem"].tolist()

    total_in = total_out = 0
    vp(f"Full per-page OCR for {len(file_stems)} drill logs (~$3-5 expected)\n")

    for i, fs in enumerate(file_stems, 1):
        out_json = OUT_DIR / f"{fs}.json"
        if out_json.exists():
            vp(f"[{i:2d}/{len(file_stems)}] {fs}: SKIP (already OCR'd)")
            continue

        vp(f"[{i:2d}/{len(file_stems)}] {fs}")
        all_pages = sorted(PAGE_DIR.glob(f"{fs}__p*.png"))
        if not all_pages:
            vp(f"    no pages found, skipping")
            continue

        # Convention: the BACK of the form is the LAST page; everything before
        # it is front (header + per-interval table + continuation pages). For
        # 2-page logs that's p1=front, p2=back. For 3-page logs that's
        # p1+p2=front (sheet 1 + continuation), p3=back. The back is the only
        # page with light-pencil yield calcs over a near-blank printed grid;
        # everything else is filled-in form.
        front_pages = all_pages[:-1] if len(all_pages) >= 2 else all_pages
        back_pages = [all_pages[-1]] if len(all_pages) >= 2 else []

        try:
            vp(f"    front ({len(front_pages)} page(s))...")
            front_payload, front_usage = call_with_retry(client, FRONT_PROMPT, front_pages)
            total_in += front_usage.input_tokens
            total_out += front_usage.output_tokens
            n_intervals = len(front_payload.get("intervals", []) or [])
            n_water = len(front_payload.get("water_measurements", []) or [])
            vp(f"      → {n_intervals} intervals, {n_water} water rows  "
               f"(in={front_usage.input_tokens} out={front_usage.output_tokens})")

            back_payload = None
            if back_pages:
                vp(f"    back...")
                back_payload, back_usage = call_with_retry(client, BACK_PROMPT, back_pages)
                total_in += back_usage.input_tokens
                total_out += back_usage.output_tokens
                n_calcs = len(back_payload.get("yield_calcs", []) or [])
                vp(f"      → {n_calcs} yield calcs  "
                   f"(in={back_usage.input_tokens} out={back_usage.output_tokens})")

            payload = {
                "file_stem": fs,
                "front": front_payload,
                "back": back_payload,
            }
            out_json.write_text(json.dumps(payload, indent=2, default=str))
            vp(f"    saved → {out_json.relative_to(REPO)}")
        except Exception as e:
            vp(f"    ERROR: {type(e).__name__}: {e}")
            continue

    vp(f"\nDone.")
    vp(f"Total cost: ${total_in*5/1e6:.4f} input + ${total_out*25/1e6:.4f} output "
       f"= ${(total_in*5 + total_out*25)/1e6:.4f}")
    vp(f"Per-log JSON files in: {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
