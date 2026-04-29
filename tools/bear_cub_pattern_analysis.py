"""Empirical pattern analysis on the Bear Cub structured corpus.

For each yield-calc term across all 24 logs:
  1. Try to derive the value from front-of-sheet aggregates (sum of weights,
     total depths, single-row values, etc.) within ±3% tolerance.
  2. If matched, annotate with derivation source.
  3. If unmatched, treat as a candidate constant and tabulate occurrences.

Output: data/raw/bear_cub/structured/candidate_constants_report.md

Run:
    uv run python tools/bear_cub_pattern_analysis.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = REPO / "data" / "raw" / "bear_cub" / "structured"
REPORT_PATH = STRUCTURED_DIR / "candidate_constants_report.md"

TOL_PCT = 0.03  # ±3% match tolerance


def near(a: float, b: float, tol: float = TOL_PCT) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) < 0.5
    return abs(a - b) / max(abs(a), abs(b)) <= tol


def front_aggregates(intervals_for_hole: pd.DataFrame, collar: pd.Series, water_for_hole: pd.DataFrame | None = None) -> dict[str, float]:
    """All numeric values that COULD have been a source for a back-side term."""
    out: dict[str, float] = {}
    if len(intervals_for_hole):
        wt = intervals_for_hole["estimated_weight_mg"].astype(float).fillna(0)
        vol = intervals_for_hole["core_measured_volume_cu_ft"].astype(float).fillna(0)
        out["sum_estimated_weight_mg"] = float(wt.sum())
        out["max_single_estimated_weight_mg"] = float(wt.max())
        out["sum_measured_volume_cu_ft"] = float(vol.sum())

    # Water-measurement aggregates (Hammon "Gallons" or Frozen Ground "cu ft")
    if water_for_hole is not None and len(water_for_hole):
        wv = water_for_hole["volume_value"].astype(float).fillna(0)
        out["sum_water_volume_value"] = float(wv.sum())
        # Convert assumed gallons → cu ft (1 cu ft = 7.4805 gal)
        out["sum_water_volume_as_cuft_from_gal"] = float(wv.sum() / 7.4805)
        # Or: assume cu ft directly
        out["sum_water_volume_as_cuft_direct"] = float(wv.sum())
        # Water depth range covered
        df_min = float(water_for_hole["depth_from_ft"].astype(float).fillna(0).min())
        df_max = float(water_for_hole["depth_to_ft"].astype(float).fillna(0).max())
        if df_max > df_min:
            water_rise = df_max - df_min
            out["water_total_rise_ft"] = water_rise
            # D² (in²) = (volume_cu_ft × 4 × 144) / (π × rise_ft)
            for v_label, v in (("gal", out["sum_water_volume_as_cuft_from_gal"]), ("cuft", out["sum_water_volume_as_cuft_direct"])):
                if water_rise > 0 and v > 0:
                    d2 = (v * 4 * 144) / (math.pi * water_rise)
                    out[f"derived_D2_in_assuming_{v_label}"] = d2
    for k in (
        "total_depth_ft", "depth_to_bedrock_ft",
        "depth_into_bedrock_ft", "depth_of_muck_ft",
        "elevation_ft", "easting_local_ft", "northing_local_ft",
    ):
        v = collar.get(k)
        try:
            v = float(v)
            if v != 0:
                out[f"collar.{k}"] = v
        except (TypeError, ValueError):
            pass

    # Cumulative-sum subsets — sum_estimated_weight over depth range [a, b]
    if len(intervals_for_hole):
        df = intervals_for_hole.sort_values("depth_to_ft").reset_index(drop=True)
        # Cumulative sum of weights from surface
        cum_wt = 0.0
        for _, r in df.iterrows():
            cum_wt += float(r.get("estimated_weight_mg", 0) or 0)
            depth_to = float(r.get("depth_to_ft", 0) or 0)
            if depth_to > 0:
                out[f"cum_wt_thru_{depth_to}_ft"] = cum_wt
    return out


def find_match(term_value: float, aggregates: dict[str, float]) -> tuple[str, float] | None:
    if term_value == 0:
        return None
    for label, val in aggregates.items():
        if near(term_value, val):
            return label, val
    return None


def main() -> None:
    collars = pd.read_parquet(STRUCTURED_DIR / "drillhole_collars.parquet")
    intervals = pd.read_parquet(STRUCTURED_DIR / "drillhole_intervals.parquet")
    yield_calcs = pd.read_parquet(STRUCTURED_DIR / "drillhole_yield_calcs.parquet")
    back_summary = pd.read_parquet(STRUCTURED_DIR / "drillhole_back_summary.parquet")
    water_path = STRUCTURED_DIR / "drillhole_water.parquet"
    water = pd.read_parquet(water_path) if water_path.exists() else pd.DataFrame()

    print(f"Loaded {len(collars)} collars, {len(intervals)} intervals, "
          f"{len(yield_calcs)} yield calcs.\n")

    candidate_constants: list[dict] = []  # term value with no match
    matched_terms: list[dict] = []        # term value matched to a front aggregate

    for _, calc in yield_calcs.iterrows():
        fs = calc["file_stem"]
        try:
            terms = json.loads(calc["terms_json"])
        except (TypeError, json.JSONDecodeError):
            continue

        collar = collars[collars.file_stem == fs].iloc[0] if (collars.file_stem == fs).any() else pd.Series()
        ivals = intervals[intervals.file_stem == fs] if "file_stem" in intervals.columns else intervals.iloc[0:0]
        wmeas = water[water.file_stem == fs] if (len(water) and "file_stem" in water.columns) else None
        agg = front_aggregates(ivals, collar, wmeas)
        # Also include the actual_assayed_weight_mg from back summary
        bs = back_summary[back_summary.file_stem == fs]
        if len(bs):
            w = bs.iloc[0].get("actual_assayed_weight_mg", 0)
            try:
                w = float(w)
                if w > 0:
                    agg["back.actual_assayed_weight_mg"] = w
            except (TypeError, ValueError):
                pass

        for i, t in enumerate(terms):
            v = t.get("value", 0) if isinstance(t, dict) else 0
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            match = find_match(v, agg)
            row = {
                "file_stem": fs,
                "form_type": collar.get("form_type", ""),
                "calc_index": calc.get("calc_index"),
                "calc_description": calc.get("description_text", ""),
                "formula": calc.get("formula_raw_text", ""),
                "result_value": calc.get("result_value"),
                "term_index": i,
                "term_value": v,
                "term_raw": t.get("raw_token", "") if isinstance(t, dict) else "",
                "operator_to_prev": t.get("operator_to_prev", "") if isinstance(t, dict) else "",
            }
            if match:
                row["matched_to"] = match[0]
                row["matched_value"] = match[1]
                matched_terms.append(row)
            else:
                row["matched_to"] = None
                candidate_constants.append(row)

    # Tabulate candidate constants
    constants_df = pd.DataFrame(candidate_constants)
    matched_df = pd.DataFrame(matched_terms)

    if len(constants_df):
        # Cluster by rounded value
        constants_df["rounded"] = constants_df["term_value"].round(1)
        clustered = constants_df.groupby("rounded").agg(
            count=("file_stem", "size"),
            distinct_logs=("file_stem", "nunique"),
            form_types=("form_type", lambda s: sorted(set(s))),
            example_formulas=("formula", lambda s: sorted(set(s))[:3]),
        ).reset_index().sort_values(["count", "rounded"], ascending=[False, True])
    else:
        clustered = pd.DataFrame()

    # Write report
    lines = [
        "# Bear Cub yield-calc empirical analysis",
        "",
        f"Corpus: **{len(collars)}** drill logs, **{len(yield_calcs)}** back-of-sheet yield calculations.",
        f"Tolerance for cross-front-back derivation: **±{int(TOL_PCT*100)}%**.",
        "",
        f"- **{len(matched_df)}** yield-calc terms matched to a front-of-sheet aggregate.",
        f"- **{len(constants_df)}** terms unmatched → candidate constants.",
        "",
        "## Candidate constants (clustered by rounded value, sorted by frequency)",
        "",
    ]
    if len(clustered):
        lines.append("| value (rounded) | count | distinct logs | form types | example formulas |")
        lines.append("|---|---|---|---|---|")
        for _, r in clustered.iterrows():
            ft = ", ".join(str(x) for x in r["form_types"][:3])
            ex = " / ".join(str(x) for x in r["example_formulas"][:2])
            lines.append(f"| {r['rounded']} | {r['count']} | {r['distinct_logs']} | {ft} | `{ex}` |")
    else:
        lines.append("(no unmatched terms — all values traced to front of sheet)")

    lines += [
        "",
        "## Matched derivations (front → back)",
        "",
        "Sample of yield-calc terms whose values trace to a front-of-sheet aggregate:",
        "",
    ]
    if len(matched_df):
        sample = matched_df.head(40)
        lines.append("| log | calc | term | matched to | match value |")
        lines.append("|---|---|---|---|---|")
        for _, r in sample.iterrows():
            lines.append(
                f"| {r['file_stem']} | `{r['calc_description'][:30]}` | "
                f"{r['term_value']} (`{r['term_raw']}`) | {r['matched_to']} | {r['matched_value']:.2f} |"
            )

    lines += ["", "## All yield calculations as captured", ""]
    for _, calc in yield_calcs.iterrows():
        try:
            terms = json.loads(calc["terms_json"])
        except (TypeError, json.JSONDecodeError):
            terms = []
        lines.append(f"- **{calc['file_stem']}** [{calc.get('description_text', '')}]: "
                     f"`{calc.get('formula_raw_text', '')}` = {calc.get('result_value', '')} "
                     f"{calc.get('result_unit_as_written', '')}")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written → {REPORT_PATH.relative_to(REPO)}")
    if len(clustered):
        print("\nTop candidate-constant clusters:")
        print(clustered.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
