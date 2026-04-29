"""Aggregate per-log OCR JSON into 5 relational tables.

Tables emitted:
  drillhole_collars       1 row/hole
  drillhole_intervals     many rows/hole (per-interval drilling table)
  drillhole_water         many rows/hole (water measurements)
  drillhole_yield_calcs   many rows/hole (back-of-sheet formulas)
  drillhole_back_summary  1 row/hole

Outputs:
  data/raw/bear_cub/structured/*.parquet  (Parquet primary)
  data/raw/bear_cub/structured/*.csv      (CSV exports for review)
  data/raw/bear_cub/structured/bear_cub.sqlite  (SQLite for ad-hoc joins)

Run:
    uv run python tools/bear_cub_aggregate_ocr.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

# Numeric columns that may have come back as strings (fractions, percents, units)
NUMERIC_COLUMNS = {
    "drillhole_collars": [
        "easting_local_ft", "northing_local_ft", "elevation_ft",
        "total_depth_ft", "depth_to_bedrock_ft",
        "depth_into_bedrock_ft", "depth_of_muck_ft",
    ],
    "drillhole_intervals": [
        "depth_from_ft", "depth_to_ft",
        "core_measured_volume_cu_ft", "core_before_pump_in", "core_after_pump_in",
        "no_of_colors_total", "no_of_colors_1", "no_of_colors_2", "no_of_colors_3",
        "estimated_weight_mg",
    ],
    "drillhole_water": [
        "depth_from_ft", "depth_to_ft", "volume_value",
    ],
    "drillhole_yield_calcs": [
        "depth_from_ft", "depth_to_ft", "result_value",
    ],
    "drillhole_back_summary": ["actual_assayed_weight_mg"],
}


def coerce_numeric(v):
    """Coerce value to float; handles fraction strings ('3/4', '1 1/2'), etc."""
    if v is None or v == "" or v == "None":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    # Pure number
    try:
        return float(s)
    except ValueError:
        pass
    # Mixed fraction "1 3/4"
    m = re.fullmatch(r"(\d+)\s+(\d+)/(\d+)", s)
    if m:
        whole, num, den = int(m[1]), int(m[2]), int(m[3])
        return whole + num / den
    # Bare fraction "3/4"
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    if m:
        return int(m[1]) / int(m[2])
    # Strip trailing units ("ft", "in", etc.) and retry
    m = re.match(r"^([\d.]+)", s)
    if m:
        try:
            return float(m[1])
        except ValueError:
            pass
    return None  # unparseable

REPO = Path(__file__).resolve().parents[1]
OCR_DIR = REPO / "data" / "raw" / "bear_cub" / "full_ocr"
OUT_DIR = REPO / "data" / "raw" / "bear_cub" / "structured"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Canonical per-hole files only — exclude _v2 (re-OCR side-output) and _v1backup
    files = sorted(
        f for f in OCR_DIR.glob("*.json")
        if not f.stem.endswith("_v2") and not f.stem.endswith("_v1backup")
    )
    if not files:
        print(f"No OCR JSON files found in {OCR_DIR}")
        return
    print(f"Aggregating {len(files)} per-log OCR files\n")

    collars: list[dict] = []
    intervals: list[dict] = []
    water: list[dict] = []
    yield_calcs: list[dict] = []
    back_summary: list[dict] = []

    for f in files:
        payload = json.loads(f.read_text())
        fs = payload["file_stem"]
        front = payload.get("front", {}) or {}
        back = payload.get("back", {}) or {}

        # === collars ===
        collar = {k: front.get(k) for k in (
            "hole_id", "line_id", "district", "claim", "form_type",
            "easting_local_ft", "northing_local_ft", "elevation_ft",
            "date_started", "date_finished",
            "panner", "driller", "day_shift_crew", "night_shift_crew",
            "total_depth_ft", "depth_to_bedrock_ft",
            "depth_into_bedrock_ft", "depth_of_muck_ft",
            "casing_or_bit_diameter_text",
            "width_from_text", "width_to_text",
            "ocr_confidence", "ocr_notes",
        )}
        collar["file_stem"] = fs
        collars.append(collar)

        # === intervals ===
        for r in front.get("intervals", []) or []:
            row = dict(r)
            row["file_stem"] = fs
            intervals.append(row)

        # === water measurements ===
        for r in front.get("water_measurements", []) or []:
            row = dict(r)
            row["file_stem"] = fs
            water.append(row)

        # === yield calcs ===
        for c in back.get("yield_calcs", []) or []:
            row = dict(c)
            row["file_stem"] = fs
            row["terms_json"] = json.dumps(row.pop("terms", []))
            yield_calcs.append(row)

        # === back summary ===
        back_summary.append({
            "file_stem": fs,
            "actual_assayed_weight_mg": back.get("actual_assayed_weight_mg", 0.0),
            "operator_initials_raw": back.get("operator_initials_raw", ""),
            "green_pencil_notes": back.get("green_pencil_notes", ""),
            "geological_interpretation": back.get("geological_interpretation", ""),
            "back_raw_text": back.get("back_raw_text", ""),
        })

    tables = {
        "drillhole_collars": pd.DataFrame(collars),
        "drillhole_intervals": pd.DataFrame(intervals),
        "drillhole_water": pd.DataFrame(water),
        "drillhole_yield_calcs": pd.DataFrame(yield_calcs),
        "drillhole_back_summary": pd.DataFrame(back_summary),
    }

    # Coerce numeric columns
    for name, tbl in tables.items():
        for col in NUMERIC_COLUMNS.get(name, []):
            if col in tbl.columns:
                tbl[col] = tbl[col].apply(coerce_numeric)
        tables[name] = tbl

    # Write Parquet + CSV
    for name, tbl in tables.items():
        if len(tbl) == 0:
            print(f"  {name}: empty (skipping)")
            continue
        cols = ["file_stem"] + [c for c in tbl.columns if c != "file_stem"]
        tbl = tbl[cols]
        tbl.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        tbl.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  {name}: {len(tbl)} rows  →  {name}.parquet, {name}.csv")

    # SQLite DB
    db_path = OUT_DIR / "bear_cub.sqlite"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        for name, tbl in tables.items():
            if len(tbl) > 0:
                tbl.to_sql(name, conn, index=False)
    print(f"\n  SQLite DB: {db_path.relative_to(REPO)}")

    print(f"\nAll tables written to: {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
