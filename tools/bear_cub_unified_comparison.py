"""Unified hole-by-hole comparison: Jesse + ours side by side.

Outputs `data/derived/bear_cub_resource/jesse_ours_unified.csv` with one row
per hole (union of our 24 Murray subset + every hole on Jesse's PDF p3-4
summary). For each row, fills whichever fields are available — both, one, or
neither.

Columns:
  hv_id                   Jesse's "HV-XXXX" hole ID (canonical)
  bear_cub_log            Our file_stem if hole exists in our Murray subset
  in_murray_subset        True if we have a paper drill log + OCR
  jesse_year              Year per Jesse's summary (1925/1936/1988)
  jesse_mined             "Mined" / notes from Jesse's summary
  jesse_pay_zone_ft       Jesse's pay-zone depth range, verbatim
  jesse_grade_lo / hi     Jesse's stated pay-zone grade range (oz/yd³)
  jesse_pay_thickness_ft  Jesse's pay-zone thickness
  ours_bedrock_ft         Bedrock depth (incl. KNN-IDW imputation for NBR)
  ours_total_depth_ft     Total drilled depth
  ours_surface_to_br_grade  Vertically-integrated grade in [0, BR]
  ours_pay_zone_grade     Highest single-interval grade
  ours_pay_zone_top_ft    Top of our identified pay zone
  ours_pay_zone_bottom_ft Bottom of our identified pay zone
  ours_total_fine_oz      Total fine oz Au in this hole
  notes                   Comparison notes / flags

Run:
    uv run python tools/bear_cub_unified_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
JESSE_SUMMARY = REPO / "data" / "raw" / "bear_cub" / "reference" / "jesse_hole_summary.csv"
OURS_ROLLUPS = REPO / "data" / "derived" / "bear_cub_resource" / "hole_rollups.csv"
OUT = REPO / "data" / "derived" / "bear_cub_resource" / "jesse_ours_unified.csv"


# Map our file_stem → Jesse's HV-XXXX (one entry per Murray hole)
MURRAY_TO_HV = {
    "L2 H4": "HV-2",         # not in Jesse
    "L2 H5": "HV-3",         # not in Jesse
    "L3 H2": "HV-4",         # not in Jesse
    "L3 H3": "HV-5",         # not in Jesse (HV-5 not in summary either)
    "L6500 H6554": "HV-6554",
    "L6500 H6556": "HV-6556",
    "L6700 H6760": "HV-6760",
    "L6900 H6952": "HV-6952",
    "L6900 H6954": "HV-6954",
    "L6900 H6956": "HV-6956",
    "L6900 H6960": "HV-6960",
    "L6900 H6964": "HV-6964",
    "L7100 H7156": "HV-7156",
    "L7100 H7160": "HV-7160",
    "L7300 H7350": "HV-7350",
    "L7300 H7352": "HV-7352",  # may not be in Jesse
    "L7300 H7354": "HV-7354",
    "L7300 H7356": "HV-7356",
    "L7300 H7360": "HV-7360",
    "L7500 H7556": "HV-7556",
    "L7500 H7560": "HV-7560",
    "L7600 H7656": "HV-7656",
    "L7700 H7752": "HV-7752",
    "L7700 H7754": "HV-7754",
}


def main() -> None:
    jesse = pd.read_csv(JESSE_SUMMARY)
    ours = pd.read_csv(OURS_ROLLUPS)

    # Build Jesse-keyed dict (consolidating multi-row entries — keep the row with grade)
    jesse_keyed: dict[str, dict] = {}
    for hv, group in jesse.groupby("hole_id"):
        with_grade = group.dropna(subset=["pay_zone_grade_oz_per_yd_low"])
        row = with_grade.iloc[0] if not with_grade.empty else group.iloc[0]
        jesse_keyed[hv] = row.to_dict()

    # Build ours-keyed dict
    ours_keyed: dict[str, dict] = {
        row["file_stem"]: row.to_dict() for _, row in ours.iterrows()
    }
    hv_to_ours = {v: ours_keyed.get(k) for k, v in MURRAY_TO_HV.items()}

    # Union of all HV IDs
    all_hvs = sorted(set(jesse_keyed.keys()) | set(MURRAY_TO_HV.values()))

    rows = []
    for hv in all_hvs:
        j = jesse_keyed.get(hv, {})
        ours_row = hv_to_ours.get(hv)
        bear_cub = next((k for k, v in MURRAY_TO_HV.items() if v == hv), "")
        in_murray = ours_row is not None

        notes = []
        if not in_murray and j:
            notes.append("Jesse-only (not in Murray paper-log archive)")
        if in_murray and not j:
            notes.append("Ours-only (Jesse omitted)")
        if in_murray and j and pd.isna(j.get("pay_zone_grade_oz_per_yd_low")):
            notes.append("Both have hole; Jesse no grade data")

        # Compute peak-in-zone AND avg-over-zone if Jesse has a pay zone for this hole
        ours_peak_in_zone = None
        ours_avg_over_zone = None
        ours_avg_over_sample_containing_zone = None
        if in_murray and j and j.get("pay_zone_text") and not pd.isna(j.get("pay_zone_grade_oz_per_yd_low")):
            try:
                txt = str(j["pay_zone_text"]).replace("ft", "").replace("to", "-")
                parts = [p.strip() for p in txt.split("-") if p.strip()]
                if len(parts) == 2:
                    z_lo, z_hi = float(parts[0]), float(parts[1])
                    iv_path = REPO / "data" / "derived" / "bear_cub_resource" / "intervals_with_grade.parquet"
                    iv = pd.read_parquet(iv_path)
                    iv = iv[iv["file_stem"] == bear_cub]
                    in_zone = iv[
                        (iv["depth_to_ft"] > z_lo) & (iv["depth_from_ft"] < z_hi)
                    ]
                    if not in_zone.empty:
                        ours_peak_in_zone = float(in_zone["grade_oz_per_cu_yd"].max())
                        # depth-weighted avg over Jesse's zone
                        num = den = 0.0
                        for _, r in in_zone.iterrows():
                            ov_from = max(float(r["depth_from_ft"]), z_lo)
                            ov_to = min(float(r["depth_to_ft"]), z_hi)
                            if ov_to > ov_from:
                                num += float(r["grade_oz_per_cu_yd"]) * (ov_to - ov_from)
                                den += (ov_to - ov_from)
                        if den > 0:
                            ours_avg_over_zone = num / den
                    # Find the SAMPLE row(s) that fully contain Jesse's zone — for
                    # Convention C/B holes, this gives operator's "sample grade"
                    # (the natural unit Jesse used).
                    containing = iv[
                        (iv["depth_from_ft"] <= z_lo) & (iv["depth_to_ft"] >= z_hi)
                    ]
                    if not containing.empty:
                        ours_avg_over_sample_containing_zone = float(
                            containing.iloc[0]["grade_oz_per_cu_yd"]
                        )
            except Exception:
                pass

        # Flag > 5% off Jesse's stated grade. Jesse's stated value is the
        # depth-AVERAGE across her pay zone, so compare to our depth-weighted
        # average (not the peak interval — peak is naturally higher than avg).
        flag = ""
        # Prefer avg if available, fall back to peak
        ours_compare = ours_avg_over_zone if ours_avg_over_zone is not None else ours_peak_in_zone
        if ours_compare is not None and not pd.isna(j.get("pay_zone_grade_oz_per_yd_low")):
            j_lo = j["pay_zone_grade_oz_per_yd_low"]
            j_hi = j.get("pay_zone_grade_oz_per_yd_high") or j_lo
            j_mid = (j_lo + j_hi) / 2
            in_range = j_lo <= ours_compare <= j_hi
            rel = (ours_compare - j_mid) / j_mid if j_mid > 0 else 0
            if not in_range and abs(rel) > 0.05:
                flag = f"OFF {rel*100:+.0f}%"
            else:
                flag = "MATCH"

        rows.append({
            "hv_id": hv,
            "bear_cub_log": bear_cub,
            "in_murray_subset": in_murray,
            "jesse_year": j.get("year", ""),
            "jesse_mined": j.get("mined", ""),
            "jesse_pay_zone_ft": j.get("pay_zone_text", ""),
            "jesse_pay_thickness_ft": j.get("pay_zone_thickness_ft", ""),
            "jesse_grade_lo": j.get("pay_zone_grade_oz_per_yd_low", ""),
            "jesse_grade_hi": j.get("pay_zone_grade_oz_per_yd_high", ""),
            "ours_bedrock_ft": (
                round(float(ours_row["bedrock_depth_ft"]), 1) if ours_row and ours_row.get("bedrock_depth_ft") else ""
            ),
            "ours_pay_zone_top_ft": (
                int(ours_row["pay_zone_top_ft"]) if ours_row and ours_row.get("pay_zone_top_ft") else ""
            ),
            "ours_pay_zone_bottom_ft": (
                int(ours_row["pay_zone_bottom_ft"]) if ours_row and ours_row.get("pay_zone_bottom_ft") else ""
            ),
            "ours_pay_zone_grade_peak": (
                round(float(ours_row["pay_zone_grade"]), 4) if ours_row else ""
            ),
            "ours_surface_to_br_grade": (
                round(float(ours_row["surface_to_br_grade"]), 4) if ours_row else ""
            ),
            "ours_peak_in_jesse_zone": round(ours_peak_in_zone, 4) if ours_peak_in_zone else "",
            "ours_avg_over_jesse_zone": round(ours_avg_over_zone, 4) if ours_avg_over_zone else "",
            "ours_sample_containing_zone": round(ours_avg_over_sample_containing_zone, 4) if ours_avg_over_sample_containing_zone else "",
            "ours_total_fine_oz": (
                round(float(ours_row["total_fine_oz_in_hole"]), 4) if ours_row else ""
            ),
            "comparison_flag": flag,
            "notes": "; ".join(notes),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # Summary
    def has_grade(v):
        return v != "" and not (isinstance(v, float) and pd.isna(v))

    n_total = len(df)
    n_jesse_listed = df["jesse_year"].apply(has_grade).sum()
    n_jesse_grade = df["jesse_grade_lo"].apply(has_grade).sum()
    n_murray = df["in_murray_subset"].sum()
    n_both = df["in_murray_subset"].sum()  # = Murray; redundant but explicit
    n_both_with_grade = (
        df["in_murray_subset"] & df["jesse_grade_lo"].apply(has_grade)
    ).sum()
    n_match = (df["comparison_flag"] == "MATCH").sum()
    n_off = (df["comparison_flag"].str.startswith("OFF")).sum()

    print(f"Unified Jesse vs ours comparison")
    print(f"  Total unique holes (union of both lists): {n_total}")
    print(f"  Jesse-listed holes:                       {n_jesse_listed}")
    print(f"  Jesse holes with NUMERIC grade:           {n_jesse_grade}")
    print(f"  Our Murray subset:                        {n_murray}")
    print(f"  Murray ∩ Jesse-with-numeric-grade:        {n_both_with_grade}")
    print(f"    of which MATCH (within 5% + Jesse range):  {n_match}")
    print(f"    of which OFF >5%:                          {n_off}")
    print(f"\nWritten: {OUT.relative_to(REPO)}")

    # Show a focused view: holes with both datasets
    both = df[(df["jesse_grade_lo"] != "") & (df["ours_pay_zone_grade_peak"] != "")]
    print(f"\nHoles with grade data on BOTH sides ({len(both)}):")
    cols_show = ["hv_id", "bear_cub_log", "jesse_pay_zone_ft", "jesse_grade_lo", "jesse_grade_hi",
                 "ours_peak_in_jesse_zone", "ours_pay_zone_grade_peak", "comparison_flag"]
    print(both[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
