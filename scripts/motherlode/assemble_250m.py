"""Build the Mother Lode feature frame at 250m for the gldbg integration.

Output: data/derived/features_motherlode_250m.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ai_minerals.features.assemble import build_feature_frame
from ai_minerals.regions.motherlode import MOTHERLODE

OUT = Path("/home/sky/src/learning/ai-minerals/data/derived/features_motherlode_250m.parquet")


def main() -> None:
    t0 = time.time()
    df: pd.DataFrame = build_feature_frame(MOTHERLODE, resolution_m=250)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(df):,} rows, {len(df.columns)} cols) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
