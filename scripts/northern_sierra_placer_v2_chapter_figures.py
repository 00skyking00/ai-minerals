"""Build the v2 chapter figures for the placer prospectivity overview page.

Reads the K.5 v2 outputs (Tertiary only, since Quaternary is still training)
and writes three PNGs to data/derived/northern_sierra_placer/ that the
chapter overview page references via absolute URLs:

    v2_calibrated_tertiary_map.png
    v2_fold_auc_tertiary.png
    v2_decile_histogram_tertiary.png

Quaternary versions get appended when K.5 finishes both populations.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS
from ai_minerals.regions._northern_sierra_lindgren_secondaries import (
    _TERTIARY as LINDGREN_TERTIARY,
    _QUATERNARY as LINDGREN_QUATERNARY,
)


REGION = NORTHERN_SIERRA_PLACER
WCRS = REGION.working_crs

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "derived" / "northern_sierra_placer"
OUT_DIR = DATA

POP = "placer_tertiary"
PHASE1_PARQUET = DATA / "phase1_index_250m.parquet"
V2_CAL_PARQUET = DATA / f"pop_calibrated_{POP}_250m.parquet"
V2_FOLDS_CSV = DATA / f"pop_fold_metrics_{POP}.csv"


# Anchor + Lindgren in working CRS.
def _to_xy(name_lonlat: dict[str, tuple[float, float]]) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(
        {"name": list(name_lonlat.keys())},
        geometry=[Point(lon, lat) for (lon, lat) in name_lonlat.values()],
        crs="EPSG:4326",
    ).to_crs(WCRS)
    return pd.DataFrame({
        "name": pts["name"].to_list(),
        "x": pts.geometry.x.to_numpy(),
        "y": pts.geometry.y.to_numpy(),
    })


def _nearest_cell_score(df_scores: pd.DataFrame, score_col: str, x: float, y: float) -> float:
    dx = df_scores["x"] - x
    dy = df_scores["y"] - y
    i = (dx * dx + dy * dy).idxmin()
    return float(df_scores.loc[i, score_col])


def _decile(score: float, all_scores: np.ndarray) -> int:
    pct_above = float((all_scores >= score).mean())
    return min(9, int(pct_above * 10))


# Style helpers reused from the data_exploration notebook.
def _label_anchors(ax, anchor_xy: pd.DataFrame, *, fontsize: int = 8) -> None:
    offsets = {
        "Malakoff Diggins / North Bloomfield": (6, 8),
        "North San Juan":                      (-95, 4),
        "Dutch Flat":                          (6, -12),
        "You Bet":                             (-50, 10),
        "Iowa Hill":                           (-65, 0),
        "Forest Hill":                         (6, -12),
        "Michigan Bluff":                      (6, 6),
    }
    for _, row in anchor_xy.iterrows():
        ax.plot(row["x"], row["y"], "o", color="crimson", markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=10)
        short = row["name"].split(" / ")[0]
        dx, dy = offsets.get(row["name"], (6, 6))
        ax.annotate(short, (row["x"], row["y"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=fontsize,
                    color="black", fontweight="semibold",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
                    zorder=11)


def build_calibrated_map() -> Path:
    """Tertiary v2 calibrated probability raster, anchor cells overlaid.

    Calibrated probability has a long zero tail (85% of cells = 0, 14% at the
    calibrator's mid-bin, ~0.1% in the real-signal upper tail), so a linear
    colorbar collapses into one or two color stops. Display the
    percentile rank of p_cal instead, which preserves the spatial story
    (where the model places mass) and stays interpretable as a heatmap.
    """
    cal = pd.read_parquet(V2_CAL_PARQUET)
    cal["pct_rank"] = cal["p_cal"].rank(method="min", pct=True).astype(np.float32)

    nrows = int(cal["row"].max() + 1)
    ncols = int(cal["col"].max() + 1)
    grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    grid[cal["row"].to_numpy(), cal["col"].to_numpy()] = cal["pct_rank"].to_numpy()

    rx_min = cal["x"].min() - 125
    rx_max = cal["x"].max() + 125
    ry_min = cal["y"].min() - 125
    ry_max = cal["y"].max() + 125

    anchor_xy = _to_xy(ANCHOR_DISTRICTS)
    pad = 25_000
    ax_xmin, ax_xmax = anchor_xy["x"].min() - pad, anchor_xy["x"].max() + pad
    ax_ymin, ax_ymax = anchor_xy["y"].min() - pad, anchor_xy["y"].max() + pad

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(grid, extent=(rx_min, rx_max, ry_min, ry_max), origin="upper",
                   cmap="inferno", vmin=0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75,
                        label="Percentile rank of calibrated P(placer_tertiary)")
    cbar.set_ticks([0.0, 0.5, 0.9, 0.99, 1.0])
    cbar.set_ticklabels(["0", "median", "p90", "p99", "max"])

    # Hydraulic pit outlines for context (the Tertiary positive labels).
    pits = gpd.read_file(REGION.raw_paths["hydraulic_pits"]).to_crs(WCRS)
    pits.boundary.plot(ax=ax, color="cyan", linewidth=0.4, alpha=0.7, zorder=4)

    _label_anchors(ax, anchor_xy)

    ax.set_xlim(ax_xmin, ax_xmax)
    ax.set_ylim(ax_ymin, ax_ymax)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("v2 Tertiary deep-gravel prospectivity (RF + LGBM + stacking + "
                 "isotonic calibration)", fontsize=11)

    out = OUT_DIR / "v2_calibrated_tertiary_map.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fold_auc_chart() -> Path:
    """Per-fold ROC-AUC by model for Tertiary. RF and LGBM get one dot per fold
    plus a mean bar; Stacking is the global OOF score (single value, plotted as
    a horizontal marker with the label "global OOF" so the n=1 difference is
    visible)."""
    folds = pd.read_csv(V2_FOLDS_CSV)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    series = []
    series.append(("Random Forest", folds[folds["model"] == "rf"]["roc_auc"].to_numpy(), "#7c2d12"))
    series.append(("LightGBM", folds[folds["model"] == "lgbm"]["roc_auc"].to_numpy(), "#1d4ed8"))
    stack_row = folds[folds["model"] == "stack"]
    stack_val = float(stack_row["roc_auc"].iloc[0]) if not stack_row.empty else None

    xs = np.arange(len(series))
    for i, (name, values, color) in enumerate(series):
        if values.size == 0:
            continue
        # Per-fold scatter, jittered for visibility.
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=values.size)
        ax.scatter(xs[i] + jitter, values, s=42, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=3)
        # Mean line.
        m = float(values.mean())
        s = float(values.std())
        ax.plot([xs[i] - 0.25, xs[i] + 0.25], [m, m], color="black", linewidth=2, zorder=4)
        ax.text(xs[i], m + 0.012, f"mean {m:.3f}", ha="center", fontsize=10,
                fontweight="semibold", zorder=5)
        ax.text(xs[i], m - 0.025, f"std {s:.3f}", ha="center", fontsize=8,
                color="gray", zorder=5)

    # Stacking: global OOF (one number), shown as a star marker on its own column.
    if stack_val is not None:
        ax.scatter([2.0], [stack_val], marker="*", s=380, color="#15803d",
                   edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(2.0, stack_val + 0.012, f"OOF {stack_val:.3f}", ha="center",
                fontsize=10, fontweight="semibold", zorder=5)
        ax.text(2.0, stack_val - 0.025, "global OOF\n(not per-fold)", ha="center",
                fontsize=8, color="gray", zorder=5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        f"Random Forest\n(14 folds)",
        f"LightGBM\n(14 folds)",
        f"Stacking (LR meta)\n(1 global OOF)",
    ])
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Tertiary deep-gravel: spatial-block CV by model "
                 "(20 km blocks, anchor districts excluded from training)",
                 fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.text(2.45, 0.5, "random", fontsize=8, color="gray", va="center")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xlim(-0.5, 2.65)
    out = OUT_DIR / "v2_fold_auc_tertiary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def build_decile_histogram() -> Path:
    """Decile distribution of the calibrated raster at three populations:
    anchor districts, Lindgren-secondary Tertiary, and Lindgren-secondary Quaternary."""
    cal = pd.read_parquet(V2_CAL_PARQUET)
    all_scores = cal["p_cal"].to_numpy()

    sets = {
        "anchor (trained out)":    ANCHOR_DISTRICTS,
        "Lindgren Tertiary":       LINDGREN_TERTIARY,
        "Lindgren Quaternary":     LINDGREN_QUATERNARY,
    }
    decile_counts: dict[str, np.ndarray] = {}
    for name, fixture in sets.items():
        xy = _to_xy(fixture)
        deciles = []
        for _, row in xy.iterrows():
            s = _nearest_cell_score(cal, "p_cal", row["x"], row["y"])
            deciles.append(_decile(s, all_scores))
        counts = pd.Series(deciles).value_counts().reindex(range(10), fill_value=0)
        decile_counts[name] = counts.to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["crimson", "steelblue", "darkorange"]
    width = 0.27
    for i, (name, counts) in enumerate(decile_counts.items()):
        n = int(counts.sum())
        ax.bar(np.arange(10) + (i - 1) * width, counts, width,
               label=f"{name} (n={n})", color=colors[i], alpha=0.85)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"d{d}" for d in range(10)])
    ax.set_xlabel("Calibrated-P decile (d0 = top 10%, d9 = bottom 10%)")
    ax.set_ylabel("count")
    ax.set_title("v2 Tertiary calibrated probability: anchor + held-out Lindgren cells",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    out = OUT_DIR / "v2_decile_histogram_tertiary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    if not V2_CAL_PARQUET.exists():
        print(f"ERROR: {V2_CAL_PARQUET} missing. K.5 hasn't produced Tertiary yet.")
        return 2
    if not V2_FOLDS_CSV.exists():
        print(f"ERROR: {V2_FOLDS_CSV} missing.")
        return 2

    p1 = build_calibrated_map()
    print(f"wrote {p1}  ({p1.stat().st_size/1024:.0f} KB)")
    p2 = build_fold_auc_chart()
    print(f"wrote {p2}  ({p2.stat().st_size/1024:.0f} KB)")
    p3 = build_decile_histogram()
    print(f"wrote {p3}  ({p3.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
