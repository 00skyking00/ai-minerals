"""Path 3 Stage B: build the Arizona porphyry-Cu feature frame.

Uses the standard `build_feature_frame` pipeline against the ARIZONA
Region config. Outputs `data/derived/features_arizona_500m.parquet`.

After this, run scripts/arizona_oof_comparison.py for the 4-cell
RF/DevNet x raw/DEEP-SEAM-style experiment.
"""

from __future__ import annotations

from pathlib import Path
from ai_minerals.regions.arizona import ARIZONA
from ai_minerals.features.assemble import build_feature_frame

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
OUT = DATA_DERIVED / "features_arizona_500m.parquet"


def main() -> None:
    print("=== Path 3 Stage B: Arizona feature frame ===")
    df = build_feature_frame(ARIZONA, resolution_m=500)
    print(f"feature frame: {df.shape}")
    print(f"columns: {df.columns.tolist()[:25]}...")
    print(f"\nlabel counts:")
    for c in df.columns:
        if c.startswith("is_") or c == "any_mineral_occurrence":
            n = int((df[c] == 1).sum()) if df[c].dtype != object else 0
            print(f"  {c}: {n}")
    df.to_parquet(OUT)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
