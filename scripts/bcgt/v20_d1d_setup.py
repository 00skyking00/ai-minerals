"""bcgt-v2.0 D.1.D: visualize the BCGT real-deposit-type prior factory.

Renders a 5-panel chart over the 4 deposit-type prior surfaces
(porphyry, skarn, epithermal, VMS) plus a 6th panel showing the
expected deposit probability per cell under uniform belief with the
top-K candidate cells highlighted. Also prints the pairwise correlation
matrix so a reviewer can see how distinct the four real priors are.

Output: data/derived/bcgt/fig_v20_d1d_priors_setup.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_minerals.decision.v20.bcgt_scale import (
    DEFAULT_N_SIDE,
    DEFAULT_TOP_K,
    expected_deposit_per_cell,
)
from ai_minerals.decision.v20.hypotheses import (
    make_bcgt_deposit_type_hypothesis_set,
)

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_d1d_priors_setup.png"


def main() -> int:
    n_side = DEFAULT_N_SIDE
    hset, _ = make_bcgt_deposit_type_hypothesis_set(n_side=n_side)

    fields = [h.prior_mean_field for h in hset.hypotheses]
    names = [h.name for h in hset.hypotheses]

    belief = hset.initial_prior()
    ep = expected_deposit_per_cell(hset, belief)
    top_k_cells = np.argsort(-ep)[:DEFAULT_TOP_K]

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4))
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)
    for ax, field, name in zip(axes[:4], fields, names):
        im = ax.imshow(
            field.reshape(n_side, n_side),
            origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
            extent=(0, n_side - 1, 0, n_side - 1),
        )
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[4]
    im = ax.imshow(
        ep.reshape(n_side, n_side),
        origin="lower", cmap="plasma",
        extent=(0, n_side - 1, 0, n_side - 1),
    )
    for c in top_k_cells:
        r, col = divmod(c, n_side)
        ax.plot(col, r, "ro", markersize=4, markeredgecolor="white",
                markeredgewidth=0.5)
    ax.set_title(
        f"Expected deposit prob (uniform belief)\ntop {DEFAULT_TOP_K} candidates (red)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "D.1.D: BCGT real-deposit-type prior setup. "
        f"{n_side}x{n_side} grid downsampled from 460x270 native BCGS labels, "
        "Matern v=2.5, l=1500 m.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")

    print("\nPairwise prior-surface correlation matrix:")
    print("            " + "  ".join(f"{n:>11}" for n in names))
    for n1, f1 in zip(names, fields):
        row = f"{n1:>11} "
        for f2 in fields:
            r = float(np.corrcoef(f1, f2)[0, 1])
            row += f"  {r:>+11.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
