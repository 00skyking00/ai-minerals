"""Re-OCR only the back-of-sheet for logs with empty back data.

Some 3-page logs were OCR'd with the old (incorrect) front/back assignment
that sent p2 as "back" when the actual back is p3. This tool detects logs
with empty `back.yield_calcs` AND a 3rd page available, re-OCRs only the
correct back page, and merges into the existing per-log JSON. Saves cost
vs. a full re-OCR.

Run AFTER `bear_cub_full_log_ocr.py` (and only after Anthropic credits are
topped up).
"""

from __future__ import annotations

import json
from pathlib import Path

# Reuse the OCR machinery from the main tool
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from bear_cub_full_log_ocr import (  # noqa: E402
    BACK_PROMPT,
    PAGE_DIR,
    OUT_DIR,
    call_with_retry,
    vp,
)
import anthropic
from dotenv import load_dotenv

load_dotenv(REPO / ".env")


def main() -> None:
    client = anthropic.Anthropic()
    suspect = []
    for f in sorted(OUT_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        back = d.get("back") or {}
        if not (back.get("yield_calcs") or []):
            stem = d["file_stem"]
            pages = sorted(PAGE_DIR.glob(f"{stem}__p*.png"))
            if len(pages) >= 3:
                suspect.append((f, stem, pages[-1]))

    if not suspect:
        vp("No empty-back logs with 3+ pages — nothing to fix.")
        return

    vp(f"Re-OCR'ing back of {len(suspect)} log(s):\n")
    total_in = total_out = 0

    for json_path, stem, last_page in suspect:
        vp(f"  {stem} → re-OCR back page {last_page.name}")
        try:
            payload, usage = call_with_retry(client, BACK_PROMPT, [last_page])
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            vp(f"    → {len(payload.get('yield_calcs', []) or [])} yield calcs  "
               f"(in={usage.input_tokens} out={usage.output_tokens})")
            existing = json.loads(json_path.read_text())
            existing["back"] = payload
            json_path.write_text(json.dumps(existing, indent=2, default=str))
            vp(f"    merged into {json_path.relative_to(REPO)}")
        except Exception as e:
            vp(f"    ERROR: {type(e).__name__}: {e}")

    vp(f"\nDone.")
    vp(f"Cost: ${total_in*5/1e6:.4f} input + ${total_out*25/1e6:.4f} output "
       f"= ${(total_in*5 + total_out*25)/1e6:.4f}")


if __name__ == "__main__":
    main()
