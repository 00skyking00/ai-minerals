"""Generate the v2 hard-cap vs v3 gaussian-falloff illustration.

Synthetic figure for data_overview.qmd. No source data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(
    "/home/sky/src/learning/ai-minerals/data/derived/"
    "northern_sierra_placer/gaussian_vs_hard_cap_falloff.png"
)

FOREST_HILL_KM = 12.0
SIGMA_KM = 12.0
HARD_CAP_KM = 15.0


def v2_hard_cap(d_km: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - d_km / HARD_CAP_KM)


def v3_gaussian(d_km: np.ndarray) -> np.ndarray:
    return np.exp(-(d_km**2) / (2.0 * SIGMA_KM**2))


def main() -> None:
    d = np.linspace(0.0, 30.0, 601)
    v2 = v2_hard_cap(d)
    v3 = v3_gaussian(d)

    v2_fh = float(v2_hard_cap(np.array([FOREST_HILL_KM]))[0])
    v3_fh = float(v3_gaussian(np.array([FOREST_HILL_KM]))[0])

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        d,
        v2,
        color="#444444",
        linestyle="--",
        linewidth=2.0,
        label=f"v2 hard cap: max(0, 1 - d/{HARD_CAP_KM:g} km)",
    )
    ax.plot(
        d,
        v3,
        color="steelblue",
        linestyle="-",
        linewidth=2.2,
        label=f"v3 gaussian: exp(-d^2 / (2 sigma^2)), sigma = {SIGMA_KM:g} km",
    )

    ax.axvline(
        FOREST_HILL_KM,
        color="#888888",
        linestyle=":",
        linewidth=1.2,
    )

    ax.scatter(
        [FOREST_HILL_KM, FOREST_HILL_KM],
        [v2_fh, v3_fh],
        color=["#444444", "steelblue"],
        zorder=5,
        s=40,
    )
    ax.annotate(
        f"Forest Hill, v3 = {v3_fh:.2f}",
        xy=(FOREST_HILL_KM, v3_fh),
        xytext=(FOREST_HILL_KM + 1.0, v3_fh + 0.08),
        fontsize=10,
        color="steelblue",
        arrowprops=dict(arrowstyle="-", color="steelblue", lw=0.8),
    )
    ax.annotate(
        f"Forest Hill, v2 = {v2_fh:.2f}",
        xy=(FOREST_HILL_KM, v2_fh),
        xytext=(FOREST_HILL_KM + 1.0, v2_fh - 0.10),
        fontsize=10,
        color="#444444",
        arrowprops=dict(arrowstyle="-", color="#444444", lw=0.8),
    )
    ax.text(
        FOREST_HILL_KM + 0.2,
        0.98,
        "Forest Hill (12 km)",
        fontsize=9,
        color="#666666",
        va="top",
    )

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Distance to nearest lode seed (km)")
    ax.set_ylabel("Score")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "v2 hard-cap vs v3 gaussian falloff: distance_to_lode_m",
        fontsize=13,
        loc="left",
        pad=22,
    )
    subtitle = (
        "Forest Hill sits 12 km from the nearest lode seed. "
        "v2 gives it ~0.2 (the hard-cap floor); "
        "v3 with sigma=12 km gives ~0.61 (matching its recognized geology)."
    )
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=10,
        color="#444444",
        va="bottom",
    )

    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {OUT}")
    print(f"v2(12 km) = {v2_fh:.4f}")
    print(f"v3(12 km) = {v3_fh:.4f}")
    print(f"v3(36 km) = {float(v3_gaussian(np.array([36.0]))[0]):.4f}")


if __name__ == "__main__":
    main()
