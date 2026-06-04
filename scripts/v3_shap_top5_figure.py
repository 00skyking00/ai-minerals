"""Generate v3 SHAP top-5 feature importance bar chart for the Tertiary stack."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

CSV_PATH = Path(
    "/home/sky/src/learning/ai-minerals/data/derived/northern_sierra_placer/"
    "feature_importance_placer_tertiary.csv"
)
PNG_PATH = Path(
    "/home/sky/src/learning/ai-minerals/data/derived/northern_sierra_placer/"
    "v3_shap_top5_tertiary.png"
)

HIGHLIGHT_FEATURE = "hydraulic_pit_proximity_m_buffered"
HIGHLIGHT_COLOR = "#dc143c"  # crimson
NEUTRAL_COLOR = "#4682b4"  # steel blue


def fmt_value(v: float) -> str:
    return f"{v:,.0f}" if v >= 10_000 else f"{v:,.2f}"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("mean_abs_shap_combined", ascending=False).head(5).reset_index(drop=True)

    top1 = df.loc[0, "mean_abs_shap_combined"]
    top2 = df.loc[1, "mean_abs_shap_combined"]
    ratio = top1 / top2

    # Plot ordering: largest at top means we plot from bottom-up reversed
    plot_df = df.iloc[::-1].reset_index(drop=True)

    colors = [
        HIGHLIGHT_COLOR if f == HIGHLIGHT_FEATURE else NEUTRAL_COLOR
        for f in plot_df["feature"]
    ]

    plt.rcParams["font.family"] = "DejaVu Sans"

    # 1200 / 150 = 8.0 in wide, 500 / 150 = 3.333 in tall
    fig, ax = plt.subplots(figsize=(8.0, 3.3333), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = range(len(plot_df))
    bars = ax.barh(
        list(y_positions),
        plot_df["mean_abs_shap_combined"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["feature"], fontfamily="DejaVu Sans Mono", fontsize=10)

    ax.set_xlabel("mean(|SHAP|)", fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")

    # Value labels at end of each bar
    x_max = plot_df["mean_abs_shap_combined"].max()
    pad = x_max * 0.01
    for bar, val in zip(bars, plot_df["mean_abs_shap_combined"]):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt_value(val),
            va="center",
            ha="left",
            fontsize=9.5,
            color="#222222",
        )

    # Headroom for the value labels
    ax.set_xlim(0, x_max * 1.18)

    # Annotation between #1 and #2 bars. In plot ordering, #1 is at the top
    # (y = len-1) and #2 is one below it (y = len-2).
    y_top1 = len(plot_df) - 1
    y_top2 = len(plot_df) - 2
    y_mid = (y_top1 + y_top2) / 2
    ann_x = x_max * 0.45
    ax.text(
        ann_x,
        y_mid,
        f"#1 is {ratio:.1f}x larger than #2",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#fff8e1",
            edgecolor="#cccccc",
            linewidth=0.8,
        ),
    )

    ax.set_title(
        "SHAP top-5 features for v3 Tertiary stacked model",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=22,
    )
    ax.text(
        0.0,
        1.02,
        "Bars are mean(|SHAP|) across the top-500 cells by calibrated probability. "
        "The buffered pit-proximity feature dominates, which prompted the ablation.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#555555",
        ha="left",
        va="bottom",
    )

    # Allocate generous left margin for the feature-name labels and headroom
    # for the title + subtitle, then save without tight cropping to land at
    # the requested 1200x500 pixel canvas.
    fig.subplots_adjust(left=0.34, right=0.97, top=0.80, bottom=0.16)

    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # bbox_inches="tight" can drift the final pixel dims off the requested
    # 1200x500. Pin to spec via PIL.
    from PIL import Image as _PILImage

    with _PILImage.open(PNG_PATH) as im:
        if im.size != (1200, 500):
            im = im.convert("RGBA")
            bg = _PILImage.new("RGBA", (1200, 500), (255, 255, 255, 255))
            im_resized = im.resize((1200, 500), _PILImage.LANCZOS)
            bg.alpha_composite(im_resized)
            bg.convert("RGB").save(PNG_PATH, "PNG", dpi=(150, 150))

    print(f"Wrote {PNG_PATH}")
    print(f"Top-1 / Top-2 ratio: {ratio:.4f}")
    print(df[["feature", "mean_abs_shap_combined"]].to_string(index=False))


if __name__ == "__main__":
    main()
