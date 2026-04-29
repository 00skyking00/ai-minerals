"""Re-OCR all 24 Bear Cub drill-log headers via the Anthropic API.

Original-pass OCR on these logs was done via multi-modal Read in
conversation (no API key was set at the time). The 5-log re-OCR pass
(`tools/bear_cub_relog_reocr.py`) demonstrated the API + structured-output
path catches errors the conversation-based OCR missed (5 corrections out
of 5 flagged). This script extends that pass to all 24 logs.

Output:
    data/raw/bear_cub/full_reocr.json — per-hole new vs old comparison
    Updates bear_cub_collars.csv in place (lat/lon recomputed)

Run:
    uv run python tools/bear_cub_full_reocr.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")
sys.path.insert(0, str(REPO / "tools"))
from bear_cub_relog_reocr import DrillLogHeader, SYSTEM_PROMPT, encode_image, HEADER_DIR  # noqa: E402

CSV_PATH = REPO / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"
OUT_JSON = REPO / "data" / "raw" / "bear_cub" / "full_reocr.json"

# Cardinal-anchor lat/lon model (matches src/ai_minerals/bear_cub/georef.py)
ANCHOR_E = 77696.0
ANCHOR_N = 22702.0
ANCHOR_LAT = 64.531171
ANCHOR_LON = -165.332170
FT_PER_DEG_LAT = 364400.0
FT_PER_DEG_LON = FT_PER_DEG_LAT * math.cos(math.radians(ANCHOR_LAT))


def latlon_from_local(e: float, n: float) -> tuple[float, float]:
    return (
        ANCHOR_LAT + (n - ANCHOR_N) / FT_PER_DEG_LAT,
        ANCHOR_LON + (e - ANCHOR_E) / FT_PER_DEG_LON,
    )


def reocr_log(client, file_stem: str) -> tuple[DrillLogHeader, anthropic.types.Usage]:
    """Same flow as bear_cub_relog_reocr.reocr_log but inlined for direct usage."""
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
                f"of page 1). The second image is the FOOTER (bottom of page 1, "
                f"often contains total depth / date / signatures). Extract every "
                f"field you can read confidently. Read the easting and northing "
                f"digit-by-digit and call out any ambiguity."
            ),
        }
    )

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
    df = pd.read_csv(CSV_PATH)

    out: dict[str, dict] = {}
    total_in = total_out = 0
    n_changed = n_unchanged = 0
    changes_summary: list[str] = []

    file_stems = df["file_stem"].tolist()
    print(f"Re-OCR'ing all {len(file_stems)} drill logs via API\n")

    for i, fs in enumerate(file_stems, 1):
        print(f"[{i:2d}/{len(file_stems)}] {fs}")
        new, usage = reocr_log(client, fs)
        total_in += usage.input_tokens
        total_out += usage.output_tokens

        idx = df[df.file_stem == fs].index[0]
        old_e = float(df.at[idx, "easting_local_ft"]) if pd.notna(df.at[idx, "easting_local_ft"]) else None
        old_n = float(df.at[idx, "northing_local_ft"]) if pd.notna(df.at[idx, "northing_local_ft"]) else None
        new_e = new.easting_local_ft
        new_n = new.northing_local_ft

        de = (new_e - old_e) if (old_e is not None and new_e is not None) else None
        dn = (new_n - old_n) if (old_n is not None and new_n is not None) else None

        # Apply if E/N change > 5 ft (keeps tiny noise from rewriting the CSV)
        e_changed = de is not None and abs(de) >= 5
        n_changed_flag = dn is not None and abs(dn) >= 5
        meaningful = e_changed or n_changed_flag

        if meaningful:
            df.at[idx, "easting_local_ft"] = new_e
            df.at[idx, "northing_local_ft"] = new_n
            lat, lon = latlon_from_local(new_e, new_n)
            df.at[idx, "lat_wgs84"] = lat
            df.at[idx, "lon_wgs84"] = lon
            df.at[idx, "ocr_confidence"] = new.ocr_confidence
            old_notes = str(df.at[idx, "ocr_notes"]) if pd.notna(df.at[idx, "ocr_notes"]) else ""
            correction = f"reocr v2 2026-04-25: was E={old_e}, N={old_n}"
            new_notes = (new.ocr_notes or "").strip()
            df.at[idx, "ocr_notes"] = (
                ((new_notes + " | " if new_notes else "") + correction).strip()
            )
            n_changed += 1
            changes_summary.append(f"  {fs}: ΔE={de:+.0f} ΔN={dn:+.0f} ft  (conf {new.ocr_confidence})")
            print(f"    CHANGED: E {old_e}→{new_e} (Δ{de:+.0f}), N {old_n}→{new_n} (Δ{dn:+.0f})")
        else:
            n_unchanged += 1
            print(f"    unchanged (Δ < 5 ft)")

        out[fs] = {
            "old": {"E": old_e, "N": old_n},
            "new": new.model_dump(),
            "applied": meaningful,
        }

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    df.to_csv(CSV_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Changed: {n_changed} of {len(file_stems)}")
    print(f"  Unchanged: {n_unchanged}")
    if changes_summary:
        print("\nChange summary:")
        for s in changes_summary:
            print(s)
    print(
        f"\nTotal cost: ${total_in*5/1e6:.4f} input + ${total_out*25/1e6:.4f} output "
        f"= ${(total_in*5 + total_out*25)/1e6:.4f}"
    )
    print(f"\nSaved diff → {OUT_JSON.relative_to(REPO)}")
    print(f"Updated CSV in place: {CSV_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
