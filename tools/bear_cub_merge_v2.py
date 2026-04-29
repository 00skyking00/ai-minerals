"""Merge v2 re-OCR results into v1 for selected holes.

For holes that did NOT have a manual reviewer patch, the v2 OCR is the new
authoritative front. We merge v2 → v1 by:
  1. Backing v1 up to <stem>_v1backup.json (if not already backed up).
  2. Replacing v1.front.* with v2.* (preserving v1.back, which v2 doesn't carry).
  3. Saving in place so the aggregator picks up the new data.

For the 4 holes with manual patches (H6954/H6960/H6964/H7156), v2 was OCR'd
purely as a cross-check; we do NOT overwrite the user's authoritative patch.

Run:
    uv run python tools/bear_cub_merge_v2.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OCR_DIR = REPO / "data" / "raw" / "bear_cub" / "full_ocr"

# Holes whose v2 should overwrite v1's front (no manual patch existed)
MERGE_HOLES = [
    "L7100 H7160",
    "L7300 H7354",
    "L7300 H7360",
    "L7500 H7560",
    "L7700 H7752",
]

# Holes whose v1 was manually patched — DO NOT overwrite. v2 is cross-check only.
PROTECTED_HOLES = [
    "L6900 H6954",
    "L6900 H6960",
    "L6900 H6964",
    "L7100 H7156",
]


def merge_one(stem: str) -> None:
    v1_path = OCR_DIR / f"{stem}.json"
    v2_path = OCR_DIR / f"{stem}_v2.json"
    backup = OCR_DIR / f"{stem}_v1backup.json"

    if not v2_path.exists():
        print(f"  {stem}: SKIP — no v2 OCR available (likely failed)")
        return
    if not v1_path.exists():
        print(f"  {stem}: SKIP — no v1 to merge into")
        return

    if not backup.exists():
        shutil.copy(v1_path, backup)
        print(f"  {stem}: backed up v1 → {backup.name}")

    v1 = json.loads(v1_path.read_text())
    v2 = json.loads(v2_path.read_text())

    front = v1.setdefault("front", {})
    old_intervals = len(front.get("intervals") or [])
    old_total_mg = sum(
        (iv.get("estimated_weight_mg") or 0) for iv in (front.get("intervals") or [])
    )

    # Replace front.* with v2.* (v2 is flat, contains the new front data)
    for k, v in v2.items():
        front[k] = v

    new_intervals = len(front.get("intervals") or [])
    new_total_mg = sum(
        (iv.get("estimated_weight_mg") or 0) for iv in (front.get("intervals") or [])
    )

    v1_path.write_text(json.dumps(v1, indent=2))
    print(
        f"  {stem}: intervals {old_intervals}→{new_intervals}, "
        f"mg sum {old_total_mg:.1f}→{new_total_mg:.1f}, "
        f"bedrock={front.get('depth_to_bedrock_ft')}"
    )


def main() -> None:
    print(f"Merging {len(MERGE_HOLES)} v2 result(s) into v1 in-place")
    for stem in MERGE_HOLES:
        merge_one(stem)
    print(f"\nProtected holes (manual patch retained, v2 NOT merged):")
    for stem in PROTECTED_HOLES:
        v2 = OCR_DIR / f"{stem}_v2.json"
        if v2.exists():
            d = json.loads(v2.read_text())
            mg = sum((iv.get("estimated_weight_mg") or 0) for iv in d.get("intervals", []))
            print(f"  {stem}: v2 cross-check mg={mg:.1f} (manual patch retained)")
    print("\nNext: uv run python tools/bear_cub_aggregate_ocr.py")


if __name__ == "__main__":
    main()
