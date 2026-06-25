"""Render the Tanacross SHAP top-feature figure for the regional chapter.

Loads the cached SHAP values from the Eastern Alaska "no count features" RF
(data/derived/shap_rf_nocount.npz — the same model drawn in the Tanacross
prospectivity map) and draws the top-15 features by mean |SHAP|. No SHAP is
recomputed; this only renders the cached array.

Output:
  data/derived/eastak/fig_shap_tanacross.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")
SHAP_NPZ = DATA_DERIVED / "shap_rf_nocount.npz"
FIG_PATH = DATA_DERIVED / "eastak" / "fig_shap_tanacross.png"

ELEMENTS = {
    "ag": "Ag", "te": "Te", "mo": "Mo", "cu": "Cu", "sb": "Sb", "au": "Au",
    "pb": "Pb", "zn": "Zn", "bi": "Bi", "as": "As", "w": "W",
}
# Pathfinder geochem (Ag/Te/Mo/Cu/...) is the porphyry-halo story; colour those
# bars one way and the geophysical / terrain features another so the figure
# reads at a glance.
PATHFINDER_COLOR = "#cc6600"
OTHER_COLOR = "#3a6ea5"


def prettify(name: str) -> tuple[str, bool]:
    """Return (display label, is_pathfinder_geochem)."""
    parts = name.split("_")
    if parts[0] in ELEMENTS and len(parts) >= 2:
        stat = parts[1]
        return f"{ELEMENTS[parts[0]]} {stat} (5 km)", True
    labels = {
        "magnetic": "Magnetic intensity",
        "elevation": "Elevation",
        "distance_to_fault_m": "Distance to fault",
        "gravity": "Bouguer gravity",
        "slope": "Slope",
    }
    return labels.get(name, name), False


def main() -> None:
    pack = np.load(SHAP_NPZ, allow_pickle=True)
    sv = pack["sv"]
    feature_names = pack["feature_names"].tolist()
    print(f"Loaded SHAP: {sv.shape} over {len(feature_names)} features", flush=True)

    mean_abs = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top = shap_df.head(15).iloc[::-1]
    labels, colors = [], []
    for name in top["feature"]:
        label, is_path = prettify(name)
        labels.append(label)
        colors.append(PATHFINDER_COLOR if is_path else OTHER_COLOR)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, top["mean_abs_shap"], color=colors)
    ax.set_xlabel("mean |SHAP value| (impact on predicted probability)")
    ax.set_title("Tanacross porphyry-Cu: top 15 features by SHAP")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PATHFINDER_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOR),
    ]
    ax.legend(handles, ["stream-sediment pathfinder", "geophysics / terrain"],
              loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_PATH}", flush=True)
    print("\nTop 10 by mean |SHAP|:", flush=True)
    print(shap_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
