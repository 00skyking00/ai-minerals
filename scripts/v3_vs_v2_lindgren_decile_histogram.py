"""Lindgren PP73 Tertiary held-out districts: v2 vs v3 decile histogram.

For each of the Tertiary-named Lindgren secondary diggings that survived
the 500 m dedup buffer against trained positives, look up the decile rank
in both:

  - the v2 calibrated Tertiary raster
    (`_v2_outputs/pop_calibrated_placer_tertiary_250m.parquet`)
  - the v3 calibrated Tertiary raster
    (`pop_calibrated_placer_tertiary_250m.parquet`)

Both deciles use the same `_decile(score) = min(9, int(pct_above * 10))`
rule that the v2 chapter-figures script used. This semantics ties all
zero-valued cells into the bottom bin (d9), so the "v2 puts the held-out
points in the zero bin" claim and the "v3 moves them to d0/d1" claim
are computed the same way against each raster.

Writes a 1200x600 grouped-bar PNG showing the v2 vs v3 distribution side
by side for each decile, plus an annotation of v3 median and top-quintile.

Usage:
    .venv/bin/python scripts/v3_vs_v2_lindgren_decile_histogram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

# Make `ai_minerals` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_minerals.regions._northern_sierra_lindgren_secondaries import (  # noqa: E402
    LINDGREN_TERTIARY_NAMES,
    LINDGREN_SECONDARY_DIGGINGS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "derived" / "northern_sierra_placer"

V2_CAL = DATA / "_v2_outputs" / "pop_calibrated_placer_tertiary_250m.parquet"
V3_CAL = DATA / "pop_calibrated_placer_tertiary_250m.parquet"
V3_RESULTS = DATA / "lindgren_secondary_blind_set_results.csv"
OUT_PNG = DATA / "v3_vs_v2_lindgren_decile_histogram.png"

WORKING_CRS = "EPSG:3310"


def _decile_pct_above(scores: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    """Decile via `pct_above * 10`. Ties at zero land in d9 together.

    This matches `northern_sierra_placer_v2_chapter_figures.py::_decile`,
    which is the rule that generated the published 'v2 8-of-13 in d9' number.
    """
    out = np.empty(len(scores), dtype=int)
    for i, s in enumerate(scores):
        pct_above = float((all_scores >= s).mean())
        out[i] = min(9, int(pct_above * 10))
    return out


def _nearest_cell_value(
    cal_x: np.ndarray, cal_y: np.ndarray, value: np.ndarray, x: float, y: float
) -> float:
    dx = cal_x - x
    dy = cal_y - y
    i = int(np.argmin(dx * dx + dy * dy))
    return float(value[i])


def main() -> int:
    # ---- Pick the surviving Tertiary names from the v3 blind-set CSV (post-dedup) ----
    v3_csv = pd.read_csv(V3_RESULTS)
    v3_tertiary = v3_csv[
        (v3_csv["point_source"] == "Lindgren")
        & (v3_csv["point_name"].isin(LINDGREN_TERTIARY_NAMES))
    ].copy()
    names = v3_tertiary["point_name"].to_list()
    print(f"Tertiary-named Lindgren rows (post-dedup): {len(names)}")

    # ---- Project those centroids to working CRS ----
    transformer = Transformer.from_crs("EPSG:4326", WORKING_CRS, always_xy=True)
    pts_xy: list[tuple[str, float, float]] = []
    for name in names:
        lon, lat = LINDGREN_SECONDARY_DIGGINGS[name]
        x, y = transformer.transform(lon, lat)
        pts_xy.append((name, x, y))

    # ---- v2: load raster, look up nearest cell, compute decile via pct_above ----
    v2_cal = pd.read_parquet(V2_CAL)
    v2_x = v2_cal["x"].to_numpy()
    v2_y = v2_cal["y"].to_numpy()
    v2_p = v2_cal["p_cal"].to_numpy()
    print(f"v2 raster cells: {len(v2_cal):,}")

    v2_point_scores = np.array(
        [_nearest_cell_value(v2_x, v2_y, v2_p, x, y) for (_, x, y) in pts_xy]
    )
    v2_deciles = _decile_pct_above(v2_point_scores, v2_p)
    print("v2 deciles by name:")
    for (n, _, _), s, d in zip(pts_xy, v2_point_scores, v2_deciles):
        print(f"  {n:24s}  p_cal={s:.6f}  d{d}")

    # ---- v3: load raster, same procedure ----
    v3_cal = pd.read_parquet(V3_CAL)
    v3_x = v3_cal["x"].to_numpy()
    v3_y = v3_cal["y"].to_numpy()
    v3_p = v3_cal["p_cal"].to_numpy()
    print(f"v3 raster cells: {len(v3_cal):,}")

    v3_point_scores = np.array(
        [_nearest_cell_value(v3_x, v3_y, v3_p, x, y) for (_, x, y) in pts_xy]
    )
    v3_deciles = _decile_pct_above(v3_point_scores, v3_p)
    print("v3 deciles by name:")
    for (n, _, _), s, d in zip(pts_xy, v3_point_scores, v3_deciles):
        print(f"  {n:24s}  p_cal={s:.6f}  d{d}")

    # ---- Counts per decile (0..9) ----
    bins = np.arange(11)
    v2_counts, _ = np.histogram(v2_deciles, bins=bins)
    v3_counts, _ = np.histogram(v3_deciles, bins=bins)
    n_v2 = int(v2_counts.sum())
    n_v3 = int(v3_counts.sum())
    v3_d9 = int(v3_counts[9])
    v2_d9 = int(v2_counts[9])
    v3_top_quintile = int(v3_counts[0] + v3_counts[1])
    v3_top_quintile_pct = 100.0 * v3_top_quintile / n_v3 if n_v3 else 0.0
    v3_median = int(np.median(v3_deciles))

    print()
    print(f"v2 d9 count: {v2_d9} of {n_v2}")
    print(f"v3 d9 count: {v3_d9} of {n_v3}")
    print(f"v3 top quintile (d0+d1): {v3_top_quintile} of {n_v3} ({v3_top_quintile_pct:.0f}%)")
    print(f"v3 median decile: d{v3_median}")

    # ---- Plot ----
    # Target 1200x600 at dpi=150 = 8x4 inches; tight bbox trims ~15px each side,
    # so go a touch wider/taller and let tight bbox land near 1200x600.
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(10)
    width = 0.4
    v2_color = "#9aa0a6"
    v3_color = "#3a6ea5"
    bars_v2 = ax.bar(
        x - width / 2, v2_counts, width=width,
        color=v2_color, edgecolor="white", linewidth=0.6,
        label=f"v2 (n={n_v2})",
    )
    bars_v3 = ax.bar(
        x + width / 2, v3_counts, width=width,
        color=v3_color, edgecolor="white", linewidth=0.6,
        label=f"v3 (n={n_v3})",
    )

    # Integer y axis ticks; cap at max+2.
    ymax = int(max(v2_counts.max(), v3_counts.max())) + 2
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1, 1))
    ax.set_xticks(x)
    ax.set_xticklabels([f"d{i}" for i in range(10)])
    ax.set_xlabel("Calibrated-P decile (d0 = top 10%, d9 = bottom 10%)")
    ax.set_ylabel("Count of Lindgren PP73 Tertiary secondaries")

    ax.set_title(
        "Lindgren PP73 Tertiary-named held-out districts: decile rank in v2 vs v3 calibrated raster",
        fontsize=10, loc="left", pad=18,
    )
    subtitle = (
        f"v2 put {v2_d9} of {n_v2} in the zero bin (d9); "
        f"v3 puts {v3_d9} of {n_v3} there."
    )
    ax.text(
        0.0, 1.02, subtitle,
        transform=ax.transAxes, fontsize=9, color="#444444",
    )

    # Annotation: v3 median + top-quintile fraction.
    annot = (
        f"v3 median decile: d{v3_median}\n"
        f"v3 top quintile (d0+d1): {v3_top_quintile}/{n_v3} ({v3_top_quintile_pct:.0f}%)"
    )
    ax.text(
        0.98, 0.95, annot,
        transform=ax.transAxes, fontsize=9, color=v3_color,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=v3_color, linewidth=0.8),
    )

    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.legend(loc="upper center", frameon=False, fontsize=9, ncols=2)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"==> wrote {OUT_PNG}")

    # Also stash the numbers as a sidecar for downstream consumers.
    numbers = {
        "n_v2": n_v2,
        "n_v3": n_v3,
        "v2_d9": v2_d9,
        "v3_d9": v3_d9,
        "v3_d0": int(v3_counts[0]),
        "v3_d1": int(v3_counts[1]),
        "v3_top_quintile": v3_top_quintile,
        "v3_top_quintile_pct": round(v3_top_quintile_pct, 1),
        "v3_median_decile": v3_median,
        "v2_counts": v2_counts.tolist(),
        "v3_counts": v3_counts.tolist(),
    }
    print("numbers:", numbers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
