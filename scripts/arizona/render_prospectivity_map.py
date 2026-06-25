"""Render the Arizona porphyry-Cu prospectivity map for the regional chapter.

Reproduces the published Cell D surface (Random Forest on the raw 77-column
feature stack) from `oof_comparison.py` deterministically — every estimator is
seeded with random_state=42, so the out-of-fold probabilities and the headline
23.0% top-1% capture come out identical to the metrics JSON. Nothing is
re-trained to change a number; this script only persists the per-cell OOF
surface (which the experiment never saved) so it can be drawn as a map in the
same style as the Tanacross figure.

Outputs:
  data/derived/arizona/model_predictions_arizona_rf_oof.parquet
  data/derived/arizona/fig_prospectivity_arizona.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the exact experiment harness so the surface is the published model.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oof_comparison import (  # noqa: E402
    LABEL_COL,
    RAW_PARQUET,
    AZ_DIR,
    report,
    run_rf,
    setup,
)

PRED_PARQUET = AZ_DIR / "model_predictions_arizona_rf_oof.parquet"
FIG_PATH = AZ_DIR / "fig_prospectivity_arizona.png"

# Published headline (data/derived/arizona/path3_decomposed_metrics.json, RF_raw).
EXPECTED_TOP1_RATE = 0.230


def main() -> None:
    df, df_oh, feat_cols, n_pos = setup(RAW_PARQUET)
    print(f"AZ raw: {df.shape}, {len(feat_cols)} features, {n_pos} positives", flush=True)

    oof, pos_mask = run_rf(df, df_oh, feat_cols)
    capture = report("RF raw", oof, pos_mask)

    top1 = capture["top_1_pct"]["rate"]
    print(f"\nReproduced top-1% capture: {top1*100:.1f}%  (published 23.0%)", flush=True)
    assert abs(top1 - EXPECTED_TOP1_RATE) < 0.005, (
        f"top-1% capture {top1*100:.1f}% drifted from published 23.0% — "
        "config no longer matches oof_comparison.py"
    )

    # Persist the OOF surface (row, col, x, y, label, probability), mirroring the
    # eastak prediction-parquet convention.
    out = df[["row", "col", "x", "y", LABEL_COL]].copy()
    out["p_rf_oof"] = oof
    out.to_parquet(PRED_PARQUET, index=False)
    print(f"saved {PRED_PARQUET}  ({len(out):,} cells)", flush=True)

    render_map(out, top1, capture["top_5_pct"]["rate"])


def render_map(out: pd.DataFrame, top1: float, top5: float) -> None:
    surface = out[~out["p_rf_oof"].isna()]
    pos = out[out[LABEL_COL] == 1]
    n_pos = len(pos)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        surface["x"], surface["y"], c=surface["p_rf_oof"],
        s=6, cmap="viridis", alpha=0.85, linewidths=0, rasterized=True,
    )
    ax.scatter(
        pos["x"], pos["y"], s=22, marker="+", c="#e8000b", linewidths=0.8,
        label=f"MRDS porphyry-Cu positives (N={n_pos})",
    )
    ax.set_aspect("equal")
    ax.set_title("Random Forest prospectivity — Arizona SE porphyry belt")
    ax.set_xlabel("Easting (m, EPSG:3310)")
    ax.set_ylabel("Northing (m, EPSG:3310)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("P(porphyry-Cu), out-of-fold")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(
        f"saved {FIG_PATH}  (top-1% {top1*100:.1f}%, top-5% {top5*100:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
