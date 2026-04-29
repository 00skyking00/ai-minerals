"""List Bear Cub holes where sample-anchored mg redistribution is currently
in effect — i.e., per-interval `estimated_weight_mg` (what the grade calc
uses) differs from `user_mg_raw` (what the operator/reviewer wrote).

For Convention A/B holes (Frozen Ground, most Hammon Field Logs) the
per-interval mg is authoritative and should NOT be redistributed. If a
reviewer accidentally clicked Auto-link on such a hole, every interval
picked up a sample_num and `compute_effective_mg` swapped authoritative
values for color-weighted redistributed ones.

This script walks every canonical `data/raw/bear_cub/full_ocr/<stem>.json`,
sums `|user_mg_raw - estimated_weight_mg|` per hole, and prints the holes
sorted by total magnitude. Re-open them in the Streamlit reviewer (suspect
or standard) and click 🔓 *Unlink ALL intervals from samples* to revert.

Usage:
    uv run python tools/bear_cub_list_redistribution_diffs.py
    uv run python tools/bear_cub_list_redistribution_diffs.py --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OCR_DIR = REPO / "data" / "raw" / "bear_cub" / "full_ocr"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Minimum total |user_mg_raw - estimated_weight_mg| (mg) to list. "
             "Default 0.1 — anything smaller is rounding noise."
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only the top N holes by magnitude.",
    )
    parser.add_argument(
        "--show-intervals", action="store_true",
        help="Print each diverging interval (largest absolute diff first)."
    )
    args = parser.parse_args()

    if not OCR_DIR.exists():
        print(f"OCR canonical dir missing: {OCR_DIR.relative_to(REPO)}")
        return

    rows = []
    for path in sorted(OCR_DIR.glob("*.json")):
        if path.stem.endswith("_v1backup"):
            continue
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        intervals = ((doc.get("front") or {}).get("intervals") or [])
        diffs = []
        for r in intervals:
            user = r.get("user_mg_raw")
            eff = r.get("estimated_weight_mg")
            if user is None or eff is None:
                continue
            d = float(eff) - float(user)
            if abs(d) > 1e-6:
                diffs.append({
                    "depth_from_ft": r.get("depth_from_ft"),
                    "depth_to_ft":   r.get("depth_to_ft"),
                    "sample_num":    r.get("sample_num"),
                    "user_mg_raw":   float(user),
                    "estimated_mg":  float(eff),
                    "delta_mg":      d,
                })
        if not diffs:
            continue
        total_user = sum(d["user_mg_raw"] for d in diffs)
        total_diff_abs = sum(abs(d["delta_mg"]) for d in diffs)
        rows.append({
            "file_stem": path.stem,
            "n_diverging": len(diffs),
            "total_diff_abs_mg": total_diff_abs,
            "total_user_mg": total_user,
            "pct_change": (total_diff_abs / total_user * 100) if total_user > 0 else float("inf"),
            "diffs": diffs,
        })

    rows.sort(key=lambda r: r["total_diff_abs_mg"], reverse=True)
    rows = [r for r in rows if r["total_diff_abs_mg"] >= args.threshold]
    if args.top is not None:
        rows = rows[: args.top]

    if not rows:
        print(f"No holes with |Δ mg| ≥ {args.threshold} mg. Nothing to revert.")
        return

    print(f"Holes with sample-anchor redistribution active (Δ ≥ {args.threshold} mg), "
          f"sorted by total |Δ|:\n")
    print(f"  {'rank':>4}  {'file_stem':<14}  "
          f"{'n_div':>5}  {'Σ|Δ| mg':>10}  {'Σ user mg':>10}  {'%change':>8}")
    for i, r in enumerate(rows, 1):
        print(f"  {i:>4}  {r['file_stem']:<14}  "
              f"{r['n_diverging']:>5}  "
              f"{r['total_diff_abs_mg']:>10.1f}  "
              f"{r['total_user_mg']:>10.1f}  "
              f"{r['pct_change']:>7.0f}%")
        if args.show_intervals:
            for d in sorted(r["diffs"], key=lambda x: -abs(x["delta_mg"]))[:6]:
                print(f"        d={d['depth_from_ft']:>5.0f}-{d['depth_to_ft']:>5.0f} ft  "
                      f"sample#={d['sample_num']}  "
                      f"user={d['user_mg_raw']:>6.1f}  → eff={d['estimated_mg']:>6.1f}  "
                      f"(Δ {d['delta_mg']:+.1f})")
            if len(r["diffs"]) > 6:
                print(f"        ... and {len(r['diffs']) - 6} more")
            print()
    print()
    print(f"Total: {len(rows)} hole(s). To revert: open in the reviewer "
          f"and click 🔓 'Unlink ALL intervals from samples'.")


if __name__ == "__main__":
    main()
