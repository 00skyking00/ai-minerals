"""Auto-generate per-hole review checklists from anomaly checks.

For each of the 24 holes, runs nine anomaly checks against the current
pipeline state (rollups + corrections + Jesse + bedrock imputation status).
Each check has a stable ID; on regeneration, the user's existing
status + comment for each (hole, id) pair is preserved.

Schema of output `data/raw/bear_cub/review_checklist.json`:

    {
      "L2 H4": {
        "regenerated_at": "2026-04-29T12:00:00",
        "items": [
          {
            "id": "high_peak_grade",
            "category": "anomaly",
            "description": "Pay-zone peak grade 0.576 oz/yd³ is high...",
            "status": "open" | "not_an_issue" | "fixed",
            "comment": "",
            "resolved_at": null
          },
          ...
        ]
      }
    }

Run:
    uv run python tools/bear_cub_generate_checklist.py
    uv run python tools/bear_cub_generate_checklist.py --reset      # discard user resolutions
    uv run python tools/bear_cub_generate_checklist.py --dry-run    # show diff, don't write
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived" / "bear_cub_resource"
RAW = REPO / "data" / "raw" / "bear_cub"
CHECKLIST_FILE = RAW / "review_checklist.json"
OCR_CORRECTIONS = RAW / "ocr_corrections.json"
ORIGINAL_COLLARS = RAW / "bear_cub_collars.csv"

# Tweet's published per-hole grades — only the one entry we have transcribed
TWEET_GRADES_OZ_YD3 = {
    "L6700 H6760": 0.05,
}

# Thresholds (kept here so they're easy to tune)
PEAK_HIGH_THRESHOLD = 0.5
DOMINANT_POLYGON_PCT = 0.25
INTERVAL_VS_SAMPLE_DISAGREE_PCT = 0.05
PAY_ZONE_THIN_FT = 2.0
PAY_ZONE_THIN_MG_MIN = 50
AVG_PEAK_RATIO = 4.0
JESSE_DISAGREE_PCT = 0.5
TWEET_DISAGREE_PCT = 0.2
SPARSE_MG_PCT = 0.3


def load_inputs() -> dict:
    """Load all source data the checks need. Returns a dict with keyed sources."""
    rollups = pd.read_csv(DERIVED / "hole_rollups.csv")
    poly = pd.read_csv(DERIVED / "polygon_cells.csv")
    poly["file_stem"] = rollups.iloc[poly.hole_idx.values].file_stem.values
    intervals = pd.read_parquet(RAW / "structured" / "drillhole_intervals.parquet")
    corrections = json.loads(OCR_CORRECTIONS.read_text()) if OCR_CORRECTIONS.exists() else {}
    orig_collars = pd.read_csv(ORIGINAL_COLLARS)
    if "depth_to_bedrock_ft" in orig_collars.columns:
        orig_br_col = "depth_to_bedrock_ft"
    else:
        orig_br_col = "bedrock_depth_ft"
    nbr_holes = set(
        orig_collars[orig_collars[orig_br_col].isna() | (orig_collars[orig_br_col] == 0)]
        .file_stem.tolist()
    )
    unified_csv = DERIVED / "jesse_ours_unified.csv"
    unified = pd.read_csv(unified_csv) if unified_csv.exists() else pd.DataFrame()
    return {
        "rollups": rollups,
        "poly": poly,
        "total_polygon_oz": poly["fine_troy_oz"].sum(),
        "intervals": intervals,
        "corrections": corrections,
        "nbr_holes": nbr_holes,
        "unified": unified,
    }


def _hole_iv_summary(intervals: pd.DataFrame, fs: str) -> tuple[int, int, float]:
    sub = intervals[intervals.file_stem == fs]
    n = len(sub)
    n_with_mg = int((sub.estimated_weight_mg.fillna(0) > 0).sum())
    mg_sum = float(sub.estimated_weight_mg.fillna(0).sum())
    return n, n_with_mg, mg_sum


def _hole_sample_total_mg(corrections: dict, fs: str) -> float:
    e = corrections.get(fs, {})
    samples = e.get("samples", []) or []
    return sum(float(s.get("mg_total") or 0) for s in samples)


def generate_items_for_hole(fs: str, sources: dict) -> list[dict]:
    """Run all anomaly checks for one hole. Returns a list of items
    (each with id, category, description). Status / comment / resolved_at
    are added later when merging with prior resolutions."""
    rollups = sources["rollups"]
    poly = sources["poly"]
    intervals = sources["intervals"]
    corrections = sources["corrections"]
    nbr_holes = sources["nbr_holes"]
    unified = sources["unified"]
    total_poly_oz = sources["total_polygon_oz"]

    h_row = rollups[rollups.file_stem == fs]
    if h_row.empty:
        return []
    h = h_row.iloc[0]
    pay_peak = float(h.get("pay_zone_grade", 0) or 0)
    pay_avg = float(h.get("pay_zone_avg_grade", 0) or 0)
    pay_thick = float(h.get("pay_zone_thickness_ft", 0) or 0)
    bedrock = float(h.get("bedrock_depth_ft", 0) or 0)

    poly_h = poly[poly.file_stem == fs]
    poly_oz = float(poly_h["fine_troy_oz"].sum()) if not poly_h.empty else 0.0
    poly_pct = poly_oz / total_poly_oz if total_poly_oz > 0 else 0.0

    n_iv, n_with_mg, iv_mg_sum = _hole_iv_summary(intervals, fs)
    sample_mg_sum = _hole_sample_total_mg(corrections, fs)

    items: list[dict] = []

    # 1. High peak grade
    if pay_peak > PEAK_HIGH_THRESHOLD:
        items.append({
            "id": "high_peak_grade",
            "category": "anomaly",
            "description": (
                f"Pay-zone peak grade **{pay_peak:.4f} oz/yd³** is high "
                f"(threshold {PEAK_HIGH_THRESHOLD}). Verify the mg readings of the "
                f"densest interval against the original log scan."
            ),
        })

    # 2. Dominant polygon contribution
    if poly_pct > DOMINANT_POLYGON_PCT:
        items.append({
            "id": "dominant_polygon_contribution",
            "category": "concentration",
            "description": (
                f"Hole contributes **{poly_oz:,.0f} oz** "
                f"({poly_pct*100:.0f}% of {total_poly_oz:,.0f} polygon total). "
                f"Single-hole-dominant resource estimates are sensitive to this hole's data."
            ),
        })

    # 3. Per-interval Σ vs samples Σ disagreement
    if iv_mg_sum > 0 and sample_mg_sum > 0:
        diff_pct = abs(iv_mg_sum - sample_mg_sum) / sample_mg_sum
        if diff_pct > INTERVAL_VS_SAMPLE_DISAGREE_PCT:
            items.append({
                "id": "interval_vs_sample_mismatch",
                "category": "consistency",
                "description": (
                    f"Per-interval Σ ({iv_mg_sum:.0f} mg) vs samples Σ "
                    f"({sample_mg_sum:.0f} mg) differ by {diff_pct*100:.0f}%. "
                    f"Reconcile: either some per-interval values are missing "
                    f"or the samples table double-counts."
                ),
            })

    # 4. Pay zone thin for mg captured
    if iv_mg_sum > PAY_ZONE_THIN_MG_MIN and 0 < pay_thick <= PAY_ZONE_THIN_FT:
        items.append({
            "id": "pay_zone_thin_for_mg",
            "category": "structure",
            "description": (
                f"Pay zone is only **{pay_thick:.0f} ft** thick despite "
                f"{iv_mg_sum:.0f} mg captured in the hole. Either the gold is "
                f"genuinely concentrated in a single 2-ft interval (3rd-beach-line "
                f"bedrock-contact pattern), or the sliding-window pay-zone search "
                f"is picking too narrow a slice of a wider gold-bearing zone."
            ),
        })

    # 5. Avg/peak disagreement (one-interval spike)
    if pay_avg > 0 and pay_peak / pay_avg > AVG_PEAK_RATIO:
        items.append({
            "id": "avg_peak_disagreement",
            "category": "structure",
            "description": (
                f"Pay-zone peak grade ({pay_peak:.4f}) is "
                f"{pay_peak/pay_avg:.1f}× the avg ({pay_avg:.4f}) — single-interval "
                f"spike. The color-weighted distribution is concentrating most of "
                f"the sample mg into one 2-ft interval. Verify whether this is "
                f"geologically right or a distribution artifact."
            ),
        })

    # 6. Jesse comparison disagreement
    if not unified.empty and "hv_id" in unified.columns:
        u = unified[unified.bear_cub_log == fs]
        if not u.empty:
            ur = u.iloc[0]
            jesse_lo = ur.get("jesse_grade_lo")
            jesse_hi = ur.get("jesse_grade_hi")
            ours_avg = ur.get("ours_avg_over_jesse_zone")
            if pd.notna(jesse_lo) and pd.notna(ours_avg) and ours_avg != "":
                try:
                    jesse_mid = (float(jesse_lo) + float(jesse_hi or jesse_lo)) / 2
                    ours_avg_f = float(ours_avg)
                    if jesse_mid > 0:
                        diff = abs(ours_avg_f - jesse_mid) / jesse_mid
                        if diff > JESSE_DISAGREE_PCT:
                            items.append({
                                "id": "jesse_disagreement",
                                "category": "external_validation",
                                "description": (
                                    f"Jesse's pay-zone grade {jesse_lo}-"
                                    f"{jesse_hi or jesse_lo} oz/yd³ over "
                                    f"{ur.get('jesse_pay_zone_ft', '?')} ft. "
                                    f"Our depth-weighted avg over the same zone: "
                                    f"{ours_avg_f:.4f} oz/yd³ — differs by "
                                    f"{diff*100:.0f}%. Likely either Convention C "
                                    f"lumping (we average over a wider sample range) "
                                    f"or Jesse's depth label is off."
                                ),
                            })
                except (ValueError, TypeError):
                    pass

    # 7. Tweet comparison
    if fs in TWEET_GRADES_OZ_YD3:
        tweet_g = TWEET_GRADES_OZ_YD3[fs]
        if pay_peak > 0 and abs(pay_peak - tweet_g) / tweet_g > TWEET_DISAGREE_PCT:
            items.append({
                "id": "tweet_disagreement",
                "category": "external_validation",
                "description": (
                    f"Tweet's published grade for this hole is "
                    f"{tweet_g} oz/yd³. Our pay-zone peak: {pay_peak:.4f} oz/yd³ "
                    f"— differs by {abs(pay_peak-tweet_g)/tweet_g*100:.0f}%. "
                    f"Cross-check the bedrock-contact intervals."
                ),
            })

    # 8. Bedrock imputed
    if fs in nbr_holes:
        items.append({
            "id": "bedrock_imputed",
            "category": "imputation",
            "description": (
                f"Hole's bedrock was NBR (No Bedrock Reached) in the original log. "
                f"Pipeline imputed bedrock = {bedrock:.1f} ft via KNN-IDW (K=4) "
                f"with floor = total_depth + 5 ft. The volume calc uses this "
                f"imputed depth. Default status: not_an_issue (informational)."
            ),
        })

    # 9. Sparse mg capture
    if n_iv > 0:
        mg_pct = n_with_mg / n_iv
        if mg_pct < SPARSE_MG_PCT and iv_mg_sum > 0:
            items.append({
                "id": "sparse_mg_capture",
                "category": "data_quality",
                "description": (
                    f"Only {n_with_mg} of {n_iv} intervals "
                    f"({mg_pct*100:.0f}%) have non-zero mg. The rest may have "
                    f"missed mg readings the OCR didn't capture. Worth verifying "
                    f"the upper or lower portion of the column where mg is sparsely "
                    f"present."
                ),
            })

    return items


def merge_with_existing(new_items: list[dict], existing_items: list[dict]) -> list[dict]:
    """Preserve existing status/comment/resolved_at by (id) match."""
    by_id = {it["id"]: it for it in existing_items}
    merged = []
    for it in new_items:
        prior = by_id.get(it["id"])
        merged.append({
            **it,
            "status": prior.get("status", "open") if prior else (
                "not_an_issue" if it["id"] == "bedrock_imputed" else "open"
            ),
            "comment": prior.get("comment", "") if prior else "",
            "resolved_at": prior.get("resolved_at") if prior else None,
        })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Discard user resolutions; start fresh")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the diff without writing")
    args = parser.parse_args()

    sources = load_inputs()
    rollups = sources["rollups"]
    existing = json.loads(CHECKLIST_FILE.read_text()) if CHECKLIST_FILE.exists() else {}

    out: dict = {}
    diff_lines: list[str] = []
    for fs in sorted(rollups.file_stem.unique()):
        new_items = generate_items_for_hole(fs, sources)
        prior_items = []
        if not args.reset:
            prior_items = (existing.get(fs) or {}).get("items", [])
        merged = merge_with_existing(new_items, prior_items)

        # Diff: which items are new this run vs disappeared since last
        new_ids = {it["id"] for it in new_items}
        prior_ids = {it["id"] for it in prior_items}
        appeared = new_ids - prior_ids
        disappeared = prior_ids - new_ids
        if appeared or disappeared:
            for did in appeared:
                diff_lines.append(f"+ {fs}: {did}")
            for did in disappeared:
                # Preserve in output ONLY if user resolved it (lets them keep the audit trail)
                pri = next((p for p in prior_items if p["id"] == did), None)
                if pri and pri.get("status") in ("not_an_issue", "fixed"):
                    merged.append(pri)
                    diff_lines.append(f"= {fs}: {did} (resolved earlier; kept)")
                else:
                    diff_lines.append(f"- {fs}: {did} (no longer triggers)")

        if merged:
            out[fs] = {
                "regenerated_at": datetime.now().isoformat(timespec="seconds"),
                "items": merged,
            }

    if args.dry_run:
        print("DRY RUN — diff vs existing:")
        if diff_lines:
            for ln in diff_lines:
                print(f"  {ln}")
        else:
            print("  (no changes)")
        n_holes = len(out)
        n_items = sum(len(h["items"]) for h in out.values())
        print(f"\nWould write {n_items} items across {n_holes} holes to "
              f"{CHECKLIST_FILE.relative_to(REPO)}")
        return

    CHECKLIST_FILE.write_text(json.dumps(out, indent=2))
    n_holes = len(out)
    n_items = sum(len(h["items"]) for h in out.values())
    print(f"Wrote {n_items} items across {n_holes} holes to "
          f"{CHECKLIST_FILE.relative_to(REPO)}")
    if diff_lines:
        print("\nChanges since last run:")
        for ln in diff_lines:
            print(f"  {ln}")


if __name__ == "__main__":
    main()
