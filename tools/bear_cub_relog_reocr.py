"""Re-OCR specific drill-log headers flagged by the dh-map geometric cross-check.

The dh-map affine fit (`tools/bear_cub_dhmap_ocr.py`) flagged holes 4, 5,
6964 as having local-grid coordinates inconsistent with their cartographer-
plotted positions on `BearCubDHMap.pdf`. Holes 2 and 3 are also flagged at
lower residuals. Re-OCR'ing the source PDFs with a tighter prompt that
emphasizes the easting/northing numeric fields catches transcription
errors that the original wide-prompt OCR pass missed.

Output:
    data/raw/bear_cub/relog_reocr.json — per-hole new vs old comparison

Run:
    uv run python tools/bear_cub_relog_reocr.py
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

HEADER_DIR = REPO / "data" / "raw" / "bear_cub" / "header_crops"
COLLARS_CSV = REPO / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"
OUT_JSON = REPO / "data" / "raw" / "bear_cub" / "relog_reocr.json"

# Holes flagged by the dh-map cross-check
RELOG_FILE_STEMS = ["L2 H4", "L2 H5", "L3 H2", "L3 H3", "L6900 H6964"]


class DrillLogHeader(BaseModel):
    """Minimum-complexity schema — focus on the load-bearing E/N fields.

    Sticking to non-optional types with defaults keeps the grammar compiler
    happy. Use empty-string sentinels for missing values rather than `None`
    types in the Union.
    """

    hole_id: str
    easting_local_ft: float
    northing_local_ft: float
    line_id: str = ""
    elevation_ft: float = 0.0
    total_depth_ft: float = 0.0
    bedrock_depth_ft: float = 0.0
    date_drilled: str = ""
    form_type: str = ""
    ocr_confidence: str = "medium"
    ocr_notes: str = ""


SYSTEM_PROMPT = """You are an expert at transcribing handwritten data from historical mining drill-log forms in the Cape Nome mining district, Alaska, drilled 1925-1955.

# Form types

Four form layouts:
1. **Hammon Field Log** (1925-1936) — printed form, handwritten fields. Header lists hole id, line id (e.g. "HV", "E1/2"), district, claim, local-grid easting and northing in feet, elevation, then drilling-detail fields below.
2. **Hammon Prospect Drilling Log** (1925-1936) — variant Hammon layout, same fundamentals.
3. **Drill Report for Frozen Ground Only** (1949) — typeset form by individual operators. Less standardized field placement; smaller fields. The "Line" field on this form is often "HV-3" or similar. Hole numbers on these forms are often single-digit (1, 2, 3, 4, 5).
4. **Alaska Gold Company** (1955) — different layout, often weathered.

# Coordinate conventions

Local-grid coordinates are in **feet**. The region's drilling spans:
- Easting: 74,000 - 80,000 ft (typical) — but possibly other ranges for adjacent claims
- Northing: 22,000 - 24,000 ft

These are large 5-digit numbers. **Read each digit carefully — handwritten 4s, 7s, and 1s are easy to confuse**, as are 6s and 0s. If a digit is unclear, mark `ocr_confidence` as "medium" or "low" and call out the specific ambiguity in `ocr_notes`.

# Your task

Read the form header (and footer if shown) carefully. Extract every field you can confidently read. For uncertain fields, return null rather than guess.

Critical rule: **the easting and northing are the load-bearing position fields**. If you read them, double-check each digit. If unsure of even one digit, mark confidence as "medium" or "low" and explain in `ocr_notes`.

Use `ocr_notes` to flag specific uncertainties — e.g., "easting third digit ambiguous, could be 4 or 7", or "northing partially obscured by water stain". Better to flag a concern than have a wrong number propagate downstream.

Output strictly to the schema. For dates, return ISO format YYYY-MM-DD if the form gives day/month/year; if only year is recoverable, return YYYY (and note in `ocr_notes`)."""


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def reocr_log(
    client: anthropic.Anthropic, file_stem: str
) -> tuple[DrillLogHeader, anthropic.types.Usage]:
    hdr = HEADER_DIR / f"{file_stem}__p1_HDR.png"
    ftr = HEADER_DIR / f"{file_stem}__p1_FTR.png"

    content: list[dict] = []
    for path in (hdr, ftr):
        if path.exists():
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encode_image(path),
                    },
                }
            )
    content.append(
        {
            "type": "text",
            "text": (
                f"Drill log `{file_stem}`. The first image is the form HEADER (top "
                f"of page 1, contains hole id / line / coordinates / elevation / "
                f"district / claim). The second image is the FOOTER (bottom of "
                f"page 1, often contains total depth / date / signatures). Extract "
                f"every field you can read confidently. Read the easting and "
                f"northing digit-by-digit and call out any ambiguity."
            ),
        }
    )

    # Grammar-compilation-overload retries with exponential backoff
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            response = client.with_options(timeout=60.0).messages.parse(
                model="claude-opus-4-7",
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_format=DrillLogHeader,
            )
            return response.parsed_output, response.usage
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 503, 529) or e.status_code >= 500:
                last_err = e
                wait = 2.0 ** attempt
                print(f"    [retry {attempt + 1}/5 after {wait:.0f}s — {e.status_code}]")
                time.sleep(wait)
                continue
            raise
    raise last_err if last_err else RuntimeError("Exhausted retries")


def main() -> None:
    client = anthropic.Anthropic()
    df = pd.read_csv(COLLARS_CSV)

    out: dict[str, dict] = {}
    total_in = total_out = 0

    print(f"Re-OCR'ing {len(RELOG_FILE_STEMS)} flagged drill logs\n")
    for fs in RELOG_FILE_STEMS:
        print(f"=== {fs} ===")
        new, usage = reocr_log(client, fs)
        total_in += usage.input_tokens
        total_out += usage.output_tokens

        existing_rows = df[df.file_stem == fs]
        if len(existing_rows):
            row = existing_rows.iloc[0]
            print(f"  OLD CSV:  E={row['easting_local_ft']}  N={row['northing_local_ft']}  "
                  f"conf={row['ocr_confidence']}")
            print(f"  NEW OCR:  E={new.easting_local_ft}  N={new.northing_local_ft}  "
                  f"conf={new.ocr_confidence}")
            de = (
                (new.easting_local_ft or 0) - row["easting_local_ft"]
                if pd.notna(row["easting_local_ft"])
                else None
            )
            dn = (
                (new.northing_local_ft or 0) - row["northing_local_ft"]
                if pd.notna(row["northing_local_ft"])
                else None
            )
            if de is not None and dn is not None:
                print(f"  Δ:        E={de:+.0f} ft   N={dn:+.0f} ft")
            if new.ocr_notes:
                print(f"  notes:    {new.ocr_notes}")

        out[fs] = {
            "old": existing_rows.iloc[0].to_dict() if len(existing_rows) else None,
            "new": new.model_dump(),
        }
        print(f"  tokens in={usage.input_tokens} out={usage.output_tokens}\n")

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved → {OUT_JSON.relative_to(REPO)}")
    print(
        f"Total cost: ${total_in*5/1e6:.4f} input + ${total_out*25/1e6:.4f} output "
        f"= ${(total_in*5 + total_out*25)/1e6:.4f}"
    )


if __name__ == "__main__":
    main()
