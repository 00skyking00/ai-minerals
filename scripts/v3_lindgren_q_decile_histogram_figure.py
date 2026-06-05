"""Lindgren PP73 Quaternary-named held-out districts: decile histogram in v3.

Reads `p_cal_placer_quaternary_decile` straight from
`lindgren_secondary_blind_set_results.csv` for the five Q-named secondaries
the task chapter calls out (Downieville, Goodyears Bar, Sierra City,
Brandy City, Camptonville) and plots their decile distribution in the v3
Q calibrated raster.

Note: per `_northern_sierra_lindgren_secondaries.py` Brandy City and
Camptonville are catalogued under `_TERTIARY` (PP73 deep-gravel mines on
or near Quaternary channels). The task explicitly lists all five as the
Q-named district set for this chapter figure, so they are all included
here.

If a v2 Q-raster decile column is present in the CSV, it would be plotted
side by side; today the CSV only carries v3 deciles, so the figure is
v3-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "derived" / "northern_sierra_placer"
RESULTS_CSV = DATA / "lindgren_secondary_blind_set_results.csv"
OUT_PATH = DATA / "v3_lindgren_q_decile_histogram.png"

# Per the task: the five Q-named secondaries to display.
Q_NAMES = [
    "Downieville (Quaternary)",  # CSV stores it with the "(Quaternary)" suffix
    "Goodyears Bar",
    "Sierra City",
    "Brandy City",
    "Camptonville",
]
DISPLAY_NAME = {
    "Downieville (Quaternary)": "Downieville",
    "Goodyears Bar": "Goodyears Bar",
    "Sierra City": "Sierra City",
    "Brandy City": "Brandy City",
    "Camptonville": "Camptonville",
}


def main() -> int:
    df = pd.read_csv(RESULTS_CSV)
    sub = df[(df["point_source"] == "Lindgren") & (df["point_name"].isin(Q_NAMES))].copy()
    # Take the first row per district (centroid lookup is unique per name).
    sub = sub.drop_duplicates("point_name", keep="first")

    missing = [n for n in Q_NAMES if n not in sub["point_name"].to_numpy()]
    if missing:
        print(f"WARNING: missing from CSV: {missing}", file=sys.stderr)

    deciles = sub["p_cal_placer_quaternary_decile"].to_numpy()
    names = sub["point_name"].to_numpy()
    n = len(deciles)

    print("Lindgren Q-named decile lookup (v3 Q calibrated raster):")
    for name, d in zip(names, deciles):
        print(f"  {DISPLAY_NAME.get(name, name):20s}  d{int(d)}")

    bins = np.arange(11)
    counts, _ = np.histogram(deciles, bins=bins)
    print(f"counts by decile (d0..d9): {counts.tolist()}")
    median = int(np.median(deciles))
    top_q = int(counts[0] + counts[1])
    top_q_pct = 100.0 * top_q / n if n else 0.0

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(10)
    color = "#1f8f5c"  # match the Q viridis-ish family
    bars = ax.bar(
        x, counts, width=0.7,
        color=color, edgecolor="white", linewidth=0.6,
        label=f"v3 Q (n={n})",
    )
    # Annotate the deciles that actually have districts with their names.
    name_by_decile: dict[int, list[str]] = {}
    for name, d in zip(names, deciles):
        name_by_decile.setdefault(int(d), []).append(DISPLAY_NAME.get(name, name))
    for d, labels in name_by_decile.items():
        ax.text(
            d, counts[d] + 0.12,
            "\n".join(labels),
            ha="center", va="bottom", fontsize=8, color="#333333",
        )

    ymax = int(counts.max()) + 2 + max(0, max(len(v) for v in name_by_decile.values()) - 1)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1, 1))
    ax.set_xticks(x)
    ax.set_xticklabels([f"d{i}" for i in range(10)])
    ax.set_xlabel("Calibrated-P decile (d0 = top 10%, d9 = bottom 10%)")
    ax.set_ylabel("Count of Lindgren PP73 Q-named secondaries")

    ax.set_title(
        "Lindgren PP73 Quaternary-named held-out districts in v3 calibrated raster",
        fontsize=10, loc="left", pad=20,
    )
    subtitle = (
        f"n={n}; median decile: d{median}; "
        f"top quintile (d0+d1): {top_q}/{n} ({top_q_pct:.0f}%)"
    )
    ax.text(
        0.0, 1.02, subtitle,
        transform=ax.transAxes, fontsize=9, color="#444444",
    )

    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
