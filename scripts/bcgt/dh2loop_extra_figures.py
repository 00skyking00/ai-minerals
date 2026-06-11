"""Two diagnostic figures from the dh2loop-formatted BCGS tables.

Output:
  data/derived/bcgt/fig_bcgs_drill_density_split.png
  data/derived/bcgt/fig_bcgs_lithology_vocabulary.png

The first visualizes the substrate for the D.6.B retrospective POMDP
validation: pre-2010 collars vs. 2010+ collars side by side, showing
where industry drilled at each phase. The second shows the top-25
operator lithology terms inside the BCGT AOI with the 7+ overburden
codes highlighted, making the dh2loop-thesaurus-need claim visible
rather than just asserted in prose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BCGT_BBOX = {"x_min": 658_250, "x_max": 792_750,
             "y_min": 1_228_750, "y_max": 1_458_250}

COLLAR_CSV = Path("data/derived/bcgs_dh2loop/Collar.csv")
LITHO_CSV = Path("data/derived/bcgs_dh2loop/Lithology.csv")
OUT_DIR = Path("data/derived/bcgt")

# Same overburden vocabulary as bcgt_bedrock_plate.py — order matters
# for the bar-chart highlight pass below.
OVERBURDEN_TERMS = {
    "OVB", "OVBD", "OB", "OVER", "Overburden", "overburden",
    "WCAS", "CAS", "CASN", "CASE", "Casing", "Casing/Overburden",
    "Soil", "Till", "Alluvium", "Glacial",
    "DHCS", "drillhole casing (DHCS)",
}


def figure_drill_density_split() -> None:
    collar = pd.read_csv(COLLAR_CSV)
    aoi = collar[(collar["X"].between(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"])) &
                 (collar["Y"].between(BCGT_BBOX["y_min"], BCGT_BBOX["y_max"]))].copy()
    aoi["DrillStart"] = pd.to_datetime(aoi["DrillStart"], errors="coerce")
    aoi = aoi.dropna(subset=["DrillStart"])
    pre = aoi[aoi["DrillStart"] < "2010-01-01"]
    post = aoi[aoi["DrillStart"] >= "2010-01-01"]
    print(f"[density] AOI holes: pre-2010 n={len(pre):,}  2010+ n={len(post):,}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, sub, label in [(axes[0], pre, f"Pre-2010 (n = {len(pre):,})"),
                            (axes[1], post, f"2010-2020 (n = {len(post):,})")]:
        # Hexbin density (each bin = 5 km × 5 km on the BC Albers grid).
        hb = ax.hexbin(sub["X"], sub["Y"], gridsize=40,
                       cmap="magma", mincnt=1,
                       extent=(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"],
                               BCGT_BBOX["y_min"], BCGT_BBOX["y_max"]))
        ax.set_xlim(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"])
        ax.set_ylim(BCGT_BBOX["y_min"], BCGT_BBOX["y_max"])
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("Easting (EPSG:3005, m)")
        plt.colorbar(hb, ax=ax, label="Drill collars per hex")
    axes[0].set_ylabel("Northing (EPSG:3005, m)")
    fig.suptitle("BCGS drill density in the BCGT AOI, pre/post-2010\n"
                 "(substrate for the D.6.B retrospective POMDP validation: "
                 "train prior on left, predict right)", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "fig_bcgs_drill_density_split.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[density] wrote {out}")
    plt.close(fig)


def figure_lithology_vocabulary() -> None:
    collar = pd.read_csv(COLLAR_CSV)
    litho = pd.read_csv(LITHO_CSV)
    aoi_collars = collar[(collar["X"].between(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"])) &
                         (collar["Y"].between(BCGT_BBOX["y_min"], BCGT_BBOX["y_max"]))]
    aoi_litho = litho[litho["CollarID"].isin(aoi_collars["CollarID"])]
    print(f"[vocab] AOI intervals: {len(aoi_litho):,}")

    top = aoi_litho["Detailed_Lithology"].value_counts().head(25)
    colors = ["#d62728" if term in OVERBURDEN_TERMS else "#1f77b4" for term in top.index]
    overburden_count = sum(1 for t in top.index if t in OVERBURDEN_TERMS)

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(top))
    ax.barh(y, top.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlabel("Interval count")
    ax.set_title("Top-25 lithology terms inside the BCGT AOI "
                 f"({len(aoi_litho):,} intervals across {len(aoi_collars):,} holes)\n"
                 f"red = surficial / casing ({overburden_count} of top 25 are the same concept "
                 "under different operator codes)\n"
                 "blue = lithology (with case + abbreviation variants that the "
                 "dh2loop 757-term thesaurus would normalize)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig_bcgs_lithology_vocabulary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[vocab] wrote {out}")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_drill_density_split()
    figure_lithology_vocabulary()


if __name__ == "__main__":
    main()
