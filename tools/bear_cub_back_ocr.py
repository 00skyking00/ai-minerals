"""OCR the back of drill-log sheets to extract yield calculation text.

The fronts of Bear Cub drill logs hold the per-interval drilling table; the
backs contain the operator's free-form yield calculations. These are
typically light pencil writing on a printed grid that's bled-through from
the front, so direct visual reading is hard. Extract via Claude vision with
a prompt focused on free-form text + formulas.

Run:
    uv run python tools/bear_cub_back_ocr.py
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

PAGE_DIR = REPO / "data" / "raw" / "bear_cub" / "page_pngs"
OUT_JSON = REPO / "data" / "raw" / "bear_cub" / "back_of_sheet_ocr.json"

# Sample logs across form types and vintages
SAMPLES = [
    "L7700 H7754",   # Hammon Field Log 1925
    "L7100 H7160",   # Hammon Prospect Drilling Log 1936
    "L3 H2",         # Frozen Ground (likely 1919)
    "L6900 H6964",   # Alaska Gold Company 1955
]


class BackOfSheet(BaseModel):
    raw_text: str           # all readable handwriting verbatim
    yield_calculations: str # the formula(s) and arithmetic
    geological_notes: str   # operator's pay-zone calls, bedrock notes, etc.
    other_annotations: str  # anything else (initials, dates, marks)


SYSTEM = """You are reading the BACK of a 1900s-era placer-Au drill-log sheet from the Cape Nome mining district, Alaska. The form is printed on both sides; the back is mostly blank printed grid, with the operator's handwritten yield calculations and geological notes in pencil overlaid on top.

The front of these sheets carries the per-interval drilling-table data. The BACK is where the operator computes the hole's yield value (cents per cubic yard) from the actual gold recovered + the volume drilled. Common formula form per the Alaskan-Prospector handbook:

    value (cents/cu yd) = (total mg gold × cents-per-mg × 27) / volume in cu ft

For a CASED (thawed-ground) hole the volume comes from depth × casing factor (e.g. 0.307 sq ft for 6" casing, or 0.27 Radford-factor). For an OPEN (frozen-ground) hole the volume comes from water-fill measurements.

Your job:
- Read EVERY handwritten mark, even if faint. The pencil is light; the printed form bleeds through and clutters; some marks may be smudged. Do your best.
- Distinguish the operator's HANDWRITING (pencil, mostly numbers + short notes) from the printed FORM TEXT (printed labels of column headers, abbreviation keys, etc.). Ignore the printed form text — only transcribe the handwriting.
- Some marks may be 180° rotated or mirrored if the form was scanned upside down. Read whatever is recognizable.
- Extract verbatim wherever possible (preserve exact numbers and arithmetic operators); paraphrase only when the writing is too damaged to transcribe.

Return four sections:
- raw_text: all handwriting verbatim, line by line
- yield_calculations: the formula(s) and any arithmetic results
- geological_notes: any pay-zone calls, lithology comments, bedrock observations
- other_annotations: dates, initials, sample IDs, anything else"""


def encode(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    client = anthropic.Anthropic()
    out: dict = {}
    total_in = total_out = 0

    for stem in SAMPLES:
        print(f"\n=== {stem} ===")
        # Some logs have more than 2 pages; back-of-sheet for our purposes is p2 (and p3 if exists)
        candidates = sorted(PAGE_DIR.glob(f"{stem}__p[2-9].png"))
        result_per_page: dict[str, dict] = {}

        for page_path in candidates:
            page_label = page_path.stem.split("__")[1]
            print(f"  {page_label}: ...")
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encode(page_path),
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"This is `{stem}` page `{page_label}` — the BACK of the drill-log "
                        f"sheet. Extract every handwritten mark."
                    ),
                },
            ]

            for attempt in range(5):
                try:
                    response = client.with_options(timeout=90.0).messages.parse(
                        model="claude-opus-4-7",
                        max_tokens=4096,
                        thinking={"type": "adaptive"},
                        system=SYSTEM,
                        messages=[{"role": "user", "content": content}],
                        output_format=BackOfSheet,
                    )
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code in (429, 503, 529) or e.status_code >= 500:
                        wait = 2.0 ** attempt
                        print(f"    [retry {attempt + 1}/5 after {wait:.0f}s — {e.status_code}]")
                        time.sleep(wait)
                        continue
                    raise

            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens
            result_per_page[page_label] = response.parsed_output.model_dump()
            print(f"    yields: {response.parsed_output.yield_calculations[:120]!r}")

        out[stem] = result_per_page

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {OUT_JSON.relative_to(REPO)}")
    print(
        f"Cost: ${total_in*5/1e6:.4f} input + ${total_out*25/1e6:.4f} output "
        f"= ${(total_in*5 + total_out*25)/1e6:.4f}"
    )


if __name__ == "__main__":
    main()
