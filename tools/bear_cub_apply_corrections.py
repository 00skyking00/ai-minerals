"""Apply user-curated mg + bedrock corrections to per-hole OCR JSONs.

The reviewer transcribed ground-truth mg values for the first 4 problem holes
via `tools/bear_cub_ocr_reviewer.py`. This script translates those notes into
patched `front.intervals` arrays + bedrock-depth fixes, in-place on the
per-hole JSONs that the aggregator reads.

For each patched hole, the v1 OCR is preserved as `<stem>_v1backup.json`.

Run:
    uv run python tools/bear_cub_apply_corrections.py
    uv run python tools/bear_cub_apply_corrections.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OCR_DIR = REPO / "data" / "raw" / "bear_cub" / "full_ocr"
CORRECTIONS = REPO / "data" / "raw" / "bear_cub" / "ocr_corrections.json"


# Each entry: list of (depth_from, depth_to, mg) tuples from the user's transcription.
# vf / f / tr / "-" / "--" coded as 0 mg per the reviewer's stated convention.
PATCHES: dict[str, dict] = {
    # H6954: Hammon Field Log. mg in column 4 'Weight of gold'. Total = 17 mg.
    # Bedrock at 76. vf and tr → 0.
    "L6900 H6954": {
        "depth_to_bedrock_ft": 76,
        "ocr_notes": (
            "Manual mg correction from reviewer (2026-04-27). 17 mg total per "
            "operator's column-sum cross-check. vf and tr coded as 0 mg. "
            "Footer remark 'Rerocking 3 tr.' captured below."
        ),
        "intervals": [
            (6, 9, 1), (9, 12, 1), (12, 15, 1),
            (15, 18, 0), (18, 21, 0), (21, 24, 0),
            (24, 27, 1), (27, 30, 6), (30, 33, 2), (33, 36, 2),
            (39, 42, 0), (42, 45, 0), (45, 48, 0),
            (54, 57, 0),
            (71, 73, 2), (73, 75, 1),
        ],
    },
    # H6960: Hammon Prospect, red-pencil-mg-in-volume-column variant. Total ≈ 117 mg
    # across 3 sample sections (37 + 76 + 4). Bedrock 83.
    "L6900 H6960": {
        "depth_to_bedrock_ft": 83,
        "ocr_notes": (
            "Manual mg correction from reviewer (2026-04-27). 'Weight of gold' col "
            "had VF/F/- text codes; actual mg in red pencil inside volume column. "
            "Sample sums: 0-18=37, 18-56=76, 56-85=4 (total 117 mg). Back-page "
            "calc 130×0.90=117 supports this (raw weight 130 × fineness 0.90)."
        ),
        "intervals": [
            # 0-18 ft: sums to 37
            (0, 8, 2), (8, 10, 1), (10, 12, 14), (12, 14, 8),
            (14, 16, 6), (16, 18, 6),
            # 18-56 ft: sums to ~75 (user said 76 — within rounding)
            (18, 20, 5), (20, 22, 13), (24, 26, 8), (26, 28, 3),
            (28, 30, 5), (30, 32, 3), (32, 34, 1), (34, 36, 1),
            (36, 38, 2), (38, 40, 1), (42, 44, 5), (44, 46, 9),
            (46, 48, 12), (48, 50, 5), (54, 56, 2),
            # 56-85 ft: lumped sum 4 mg over 29 ft (sample-level only)
            (56, 85, 4),
        ],
    },
    # H6964: Alaska Gold Co., 1955. NBR (No Bedrock Reached). Total depth 66.
    # 0-30 ft: barren (no measurement; operator assumed 3 cents/yd background — set 0).
    # 30-66 ft: 48.5 mg total per back-of-page sums.
    "L6900 H6964": {
        "depth_to_bedrock_ft": None,
        "ocr_notes": (
            "Manual mg correction from reviewer (2026-04-27). NBR — bedrock not "
            "reached at total-depth 66 ft (1 ft moss + 9 ft muck + 56 ft gravel). "
            "Bedrock estimated 70+ ft. 0-30 ft mg unmeasured (operator assumed "
            "3 cents/yd background); set to 0 here. 30-66 ft sums to 48.5 mg."
        ),
        "intervals": [
            (0, 30, 0),  # unmeasured / assumed background
            (30, 36, 36), (36, 42, 2), (42, 48, 4),
            (48, 54, 3), (54, 60, 2.5), (60, 66, 1),
        ],
    },
    # H7156: Hammon Prospect, sample-assayed-mg variant. Front 'Weight of gold' col
    # has VF/F/- text codes (and stray 5, 8). Back of page has authoritative
    # sample-level mg sums: S1(0-14)=115, S2=11, S3=13, S4=13, S5=5, total=157.
    # Sample boundaries 2-5 not fully specified; user's calc fixes (0-38)=152 mg
    # → S2+S3+S4 collectively = 152-115 = 37 mg over 14-38 ft.
    "L7100 H7156": {
        "depth_to_bedrock_ft": 68,
        "ocr_notes": (
            "Manual mg correction from reviewer (2026-04-27). Hammon Prospect "
            "with VF/F/- text codes on front; assayed sample mg on back of page. "
            "5 samples totalling 157 mg: S1(0-14)=115, S2+S3+S4(14-38)=37, "
            "S5(38-72)=5. Boundaries 2-4 not individually specified; lumped."
        ),
        "intervals": [
            (0, 14, 115),   # Sample 1
            (14, 38, 37),   # Samples 2-4 lumped
            (38, 72, 5),    # Sample 5
        ],
    },
    # H6760: Hammon Prospect with red-pen mg in volume column AND back-page
    # sample sums. Reviewer-extracted authoritative totals (2026-04-28):
    #   S1(0-40)=44, S2(40-48)=94, S3(40-48)=7 (lumped with S2 → 101), S4(48-82)=33.
    # Bit diameter 5⅝".
    "L6700 H6760": {
        "depth_to_bedrock_ft": 77,
        "ocr_notes": (
            "Manual mg correction from reviewer (2026-04-28). Hammon Prospect "
            "Convention B/C — red-pen mg in volume column + back-page sample sums "
            "as authoritative. 4 samples totalling 178 mg: S1(0-40)=44, "
            "S2+S3(40-48)=101 (S3 within S2 depth range, lumped), S4(48-82)=33. "
            "Bit diameter 5⅝ in."
        ),
        "intervals": [
            (0, 40, 44),     # Sample 1
            (40, 48, 101),   # Samples 2+3 lumped (S3 is part of 40-48)
            (48, 82, 33),    # Sample 4
        ],
    },
}


def make_interval(idx: int, d_from: float, d_to: float, mg: float) -> dict:
    return {
        "interval_index": idx,
        "time_text": "",
        "depth_from_ft": d_from,
        "depth_to_ft": d_to,
        "core_measured_volume_cu_ft": None,
        "core_before_pump_in": None,
        "core_after_pump_in": None,
        "formation_text": "",
        "no_of_colors_total": None,
        "no_of_colors_1": None,
        "no_of_colors_2": None,
        "no_of_colors_3": None,
        "estimated_weight_mg": float(mg),
        "estimated_yield_raw": "",
        "remarks": "[reviewer-corrected mg]",
    }


def patch_one(stem: str, patch: dict, dry_run: bool) -> None:
    src = OCR_DIR / f"{stem}.json"
    backup = OCR_DIR / f"{stem}_v1backup.json"
    if not src.exists():
        print(f"  {stem}: SKIP — source JSON missing")
        return

    if not backup.exists():
        if dry_run:
            print(f"  {stem}: would back up v1 → {backup.name}")
        else:
            shutil.copy(src, backup)
    else:
        print(f"  {stem}: v1 backup already exists at {backup.name}")

    with open(src) as f:
        doc = json.load(f)
    front = doc.setdefault("front", {})

    new_intervals = [
        make_interval(i + 1, *t) for i, t in enumerate(patch["intervals"])
    ]
    old_n = len(front.get("intervals") or [])
    old_total_mg = sum(
        (iv.get("estimated_weight_mg") or 0)
        for iv in (front.get("intervals") or [])
    )
    new_total_mg = sum(t[2] for t in patch["intervals"])

    front["intervals"] = new_intervals
    if "depth_to_bedrock_ft" in patch:
        front["depth_to_bedrock_ft"] = patch["depth_to_bedrock_ft"]
    if "ocr_notes" in patch:
        existing_notes = front.get("ocr_notes") or ""
        front["ocr_notes"] = (existing_notes + " | " if existing_notes else "") + patch["ocr_notes"]

    print(
        f"  {stem}: intervals {old_n}→{len(new_intervals)}, "
        f"mg sum {old_total_mg:.1f}→{new_total_mg:.1f}, "
        f"bedrock={front.get('depth_to_bedrock_ft')}"
    )

    if not dry_run:
        with open(src, "w") as f:
            json.dump(doc, f, indent=2)


def load_structured_corrections() -> dict[str, dict]:
    """Read ocr_corrections.json and convert structured fields to PATCHES format.

    Streamlit reviewer writes:
      {
        "<stem>": {
          "issue": str,
          "bedrock_depth_ft": str,
          "intervals_structured": [{depth_from_ft, depth_to_ft, mg, colors, notes}, ...],
          "samples":              [{sample_num, depth_from_ft, depth_to_ft, mg_total, source, notes}, ...],
          "mg_notes": str, "general_notes": str,
        }
      }

    For each hole, prefer per-interval data when present; fall back to sample-level.
    Returns a PATCHES-shaped dict (intervals as list of (d_from, d_to, mg) tuples).
    """
    if not CORRECTIONS.exists():
        return {}
    with open(CORRECTIONS) as f:
        raw = json.load(f)

    patches: dict[str, dict] = {}
    for stem, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        intervals_struct = entry.get("intervals_structured") or []
        samples = entry.get("samples") or []

        # Pick the source: per-interval if any rows exist, else samples
        intervals: list[tuple[float, float, float]] = []
        if intervals_struct:
            for r in intervals_struct:
                d_from = r.get("depth_from_ft")
                d_to = r.get("depth_to_ft")
                mg = r.get("mg")
                if d_from is None or d_to is None or mg is None:
                    continue
                intervals.append((float(d_from), float(d_to), float(mg)))
        elif samples:
            for r in samples:
                d_from = r.get("depth_from_ft")
                d_to = r.get("depth_to_ft")
                mg = r.get("mg_total")
                if d_from is None or d_to is None or mg is None:
                    continue
                intervals.append((float(d_from), float(d_to), float(mg)))

        if not intervals:
            continue

        # Parse bedrock — accept numeric strings, "NBR" / blank → None
        br_text = (entry.get("bedrock_depth_ft") or "").strip()
        br: float | None
        if not br_text or br_text.upper().startswith("N"):
            br = None
        else:
            try:
                br = float(br_text)
            except ValueError:
                br = None

        notes_parts = []
        if entry.get("mg_notes"):
            notes_parts.append(entry["mg_notes"])
        if entry.get("general_notes"):
            notes_parts.append(entry["general_notes"])
        ocr_notes = (
            "Manual mg correction from reviewer. " + " | ".join(notes_parts)
            if notes_parts else "Manual mg correction from reviewer."
        )

        patches[stem] = {
            "depth_to_bedrock_ft": br,
            "ocr_notes": ocr_notes,
            "intervals": intervals,
        }
    return patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--source",
        choices=["builtin", "streamlit", "both"],
        default="both",
        help=(
            "Where to read patches from. 'builtin' = the hard-coded PATCHES dict in this "
            "file (4 holes user transcribed early). 'streamlit' = ocr_corrections.json "
            "structured fields (Streamlit reviewer output). 'both' (default) = streamlit "
            "first, then builtin for any hole NOT covered by streamlit."
        ),
    )
    args = parser.parse_args()

    streamlit_patches = (
        load_structured_corrections() if args.source in ("streamlit", "both") else {}
    )
    builtin_patches = PATCHES if args.source in ("builtin", "both") else {}

    # Resolve: streamlit takes precedence over builtin per-hole when both exist
    merged: dict[str, dict] = dict(builtin_patches)
    merged.update(streamlit_patches)

    print(
        f"Applying {len(merged)} hole correction(s) "
        f"({len(streamlit_patches)} from Streamlit, "
        f"{len(builtin_patches) - len(set(builtin_patches) & set(streamlit_patches))} "
        f"from builtin only){' [DRY RUN]' if args.dry_run else ''}"
    )
    for stem, patch in merged.items():
        source_tag = "[streamlit]" if stem in streamlit_patches else "[builtin]"
        print(f"  {source_tag}", end=" ")
        patch_one(stem, patch, args.dry_run)

    if not args.dry_run:
        print(
            "\nDone. Next:\n"
            "  uv run python tools/bear_cub_aggregate_ocr.py\n"
            "  uv run python tools/bear_cub_resource_analysis.py\n"
            "  uv run python tools/bear_cub_unified_comparison.py"
        )


if __name__ == "__main__":
    main()
