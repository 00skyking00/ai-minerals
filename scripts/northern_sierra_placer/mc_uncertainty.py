"""EW5: AutoBEL-style Monte Carlo uncertainty bracket on the Quaternary
PU-bagging predictions.

Per-cell P05 / P50 / P95 across the 30 Mordelet-Vert bags. Each bag sees a
different random "unlabeled-as-negative" subsample; their disagreement at
any cell is an empirical sample from the PU model's epistemic uncertainty
about which unlabeled cells are actually negatives.

Cited as the entry-level Monte Carlo precursor to the full Bayesian
Evidential Learning framework in Yin, Strebelle, Caers (2020),
DOI 10.5194/gmd-13-651-2020. The general AutoBEL framework computes a
posterior over geological structure conditional on borehole + geophysics
data; this script is the "discrete property of interest" instance, where
the property is the per-cell probability of placer mineralization.

Outputs to ``data/derived/northern_sierra_placer/``:

- ``prospectivity_placer_quaternary_250m_mc_p05_3310.tif`` + ``_4326.tif``
- ``prospectivity_placer_quaternary_250m_mc_p50_3310.tif`` + ``_4326.tif``
- ``prospectivity_placer_quaternary_250m_mc_p95_3310.tif`` + ``_4326.tif``
- ``prospectivity_placer_quaternary_250m_mc_spread_3310.tif`` + ``_4326.tif``
  (= p95 - p05; wide spread = high model uncertainty)

Run via run_capped.sh; ~3 min wall clock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai_minerals.config import QUATERNARY_FEATURE_COLUMNS
from ai_minerals.io.geotiff import write_geotiff_dual_crs
from ai_minerals.model_pu import fit_pu_bagging
from ai_minerals.uncertainty.mc_bracket import monte_carlo_bracket


FEATURES_PARQUET = Path(
    "data/derived/features_northern_sierra_placer_250m.parquet"
)
OUT_DIR = Path("data/derived/northern_sierra_placer")
LABEL_COL = "is_placer_quaternary"
RESOLUTION_M = 250.0
SRC_CRS = "EPSG:3310"

N_BAGS = 30  # matches the v3 pipeline's N_PU_BAGS for direct comparison
RANDOM_STATE = 42  # same seed as v3 so the headline raster aligns


def main() -> None:
    print(f"[mc] Loading features from {FEATURES_PARQUET}")
    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"[mc]   shape: {df.shape}")
    if LABEL_COL not in df.columns:
        raise KeyError(f"{LABEL_COL!r} missing from features parquet")
    n_pos = int((df[LABEL_COL] == 1).sum())
    print(f"[mc]   {n_pos:,} positives ({LABEL_COL})")

    # Restrict to AOI training cells (the same set the v3 pipeline trains on
    # for Quaternary). The features parquet should already be AOI-clipped, so
    # no additional filter is needed; we just confirm the label column is
    # populated and the lithology onehot is present.
    if "lithology_class" not in df.columns:
        raise KeyError(
            "lithology_class missing; expected the v3 assembled parquet"
        )
    top_classes = (
        df["lithology_class"].value_counts().head(10).index.tolist()
    )
    print(f"[mc]   top {len(top_classes)} lithology classes: {top_classes}")

    print(f"[mc] Running PU bagging (n_bags={N_BAGS}, return_per_bag=True)...")
    p_oob, _, per_bag = fit_pu_bagging(
        df, top_classes,
        label_col=LABEL_COL,
        n_bags=N_BAGS,
        random_state=RANDOM_STATE,
        return_per_bag=True,
    )
    print(f"[mc]   p_oob shape: {p_oob.shape}, "
          f"per_bag shape: {per_bag.shape}")

    print("[mc] Computing P05 / P50 / P95 bracket across bag axis...")
    bracket = monte_carlo_bracket(per_bag, lower_q=0.05, upper_q=0.95)
    print(f"[mc]   p05 range:    {bracket['p_lower'].min():.4f} … {bracket['p_lower'].max():.4f}")
    print(f"[mc]   p50 range:    {bracket['p50'].min():.4f} … {bracket['p50'].max():.4f}")
    print(f"[mc]   p95 range:    {bracket['p_upper'].min():.4f} … {bracket['p_upper'].max():.4f}")
    print(f"[mc]   spread mean:  {float(bracket['spread'].mean()):.4f}  "
          f"(wider = higher model uncertainty per cell)")

    df_xy = df[["x", "y"]].copy()
    for key, label in (
        ("p_lower", "mc_p05"),
        ("p50",     "mc_p50"),
        ("p_upper", "mc_p95"),
        ("spread",  "mc_spread"),
    ):
        out_src = OUT_DIR / f"prospectivity_placer_quaternary_250m_{label}_3310.tif"
        out_4326 = OUT_DIR / f"prospectivity_placer_quaternary_250m_{label}_4326.tif"
        print(f"[mc] Writing {label} -> {out_src.name} + 4326 sibling")
        write_geotiff_dual_crs(
            bracket[key], df_xy,
            resolution_m=RESOLUTION_M,
            src_crs=SRC_CRS,
            out_src=out_src,
            out_4326=out_4326,
        )

    print("[mc] Done.")


if __name__ == "__main__":
    main()
