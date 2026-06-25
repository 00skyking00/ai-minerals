"""Render the two core Nome-placer chapter-redo figures from committed MPM reports.

Figure 1: headline placer comparison (baseline 0.444 vs terrain MPM 0.733/0.741).
Figure 2: the lode mirage (small-box 0.80 -> district 0.62/0.58; geochem circular).

Both read the committed JSON reports under data/derived/nome_placer/ so every
number on the chart is the live computed value, not typed by hand. No CV per-fold
arrays are persisted, so no error bars are drawn; the bars are pooled out-of-fold
AUC point estimates (see captions / the coordinator reply).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/sky/src/learning/ai-minerals")
ONSHORE = ROOT / "data/derived/nome_placer/mpm_onshore/mpm_onshore_presence_report.json"
LODE_BOX = ROOT / "data/derived/nome_placer/mpm_lode/mpm_lode_presence_report.json"
LODE_DIST = ROOT / "data/derived/nome_placer/mpm_lode/mpm_lode_district_report.json"
OUT = ROOT / "data/derived/portfolio_charts/thumbs"

CHANCE = 0.5


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def headline(onshore: dict) -> None:
    labels = [
        "v3.1 knowledge\noverlay\n(live on goldbug)",
        "terrain MPM\nspatial-block CV",
        "terrain MPM\nbuffered 300 m",
    ]
    vals = [
        onshore["baseline_v31_composite_auc"],
        onshore["mpm_geomorph_terrain_spatialCV_auc"],
        onshore["mpm_geomorph_terrain_buffered300m_auc"],
    ]
    colors = ["#b03a2e", "#1f6f54", "#2e7d5b"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    ax.axhline(CHANCE, color="#555", ls="--", lw=1.2, zorder=2)
    ax.text(2.45, CHANCE + 0.008, "chance (0.50)", color="#555",
            ha="right", va="bottom", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 0.85)
    ax.set_ylabel("AUC against 65 ARDF placer occurrences")
    ax.set_title("Placer prospectivity: the model that held up under spatial CV",
                 fontsize=12.5, fontweight="bold", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "ch1_nome_placer_headline_auc.png", facecolor="white")
    plt.close(fig)


def lode_mirage(box: dict, dist: dict) -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150, sharey=True)
    fig.patch.set_facecolor("white")

    box_vals = [
        box["lode_indep_geol_akmag_struct_terrain_spatialCV_auc"],
        box["lode_indep_buffered300m_auc"],
    ]
    dist_vals = [
        dist["structural_geol_akmag_fault_terrain_spatialCV_auc"],
        dist["structural_buffered300m_auc"],
    ]
    tick = ["spatial-block\nCV", "buffered\n300 m"]

    for ax, vals, color, title, sub in [
        (axl, box_vals, "#8e7cc3",
         f"Small box: {box['n_lode']} lodes",
         "lodes in the hills, background on the coastal flat"),
        (axr, dist_vals, "#b03a2e",
         f"Whole district: {dist['n_lode']} lodes",
         "background is now upland too (incl. Big Hurrah)"),
    ]:
        ax.set_facecolor("white")
        bars = ax.bar(tick, vals, color=color, width=0.55, zorder=3)
        ax.axhline(CHANCE, color="#555", ls="--", lw=1.2, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 0.9)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.text(0.5, -0.16, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=9, color="#444")
        _style(ax)

    axl.set_ylabel("AUC, structural/terrain model")
    axl.text(1.45, CHANCE + 0.008, "chance", color="#555", ha="right",
             va="bottom", fontsize=9)
    geo = dist["full_with_geochem_spatialCV_auc"]
    axr.text(0.5, 0.84,
             f"(geochem model reads {geo:.3f}\nbut the pathfinders trail gold:\ncircular, not a finding)",
             transform=axr.transAxes, ha="center", va="top", fontsize=8.5,
             color="#7a2718", style="italic")
    fig.suptitle("The lode mirage: 0.80 was the terrain telling the model where the coast was",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT / "ch1_nome_lode_mirage.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    onshore = json.loads(ONSHORE.read_text())
    box = json.loads(LODE_BOX.read_text())
    dist = json.loads(LODE_DIST.read_text())
    headline(onshore)
    lode_mirage(box, dist)
    print("wrote ch1_nome_placer_headline_auc.png + ch1_nome_lode_mirage.png")


if __name__ == "__main__":
    main()
