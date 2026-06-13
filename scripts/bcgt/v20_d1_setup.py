"""bcgt-v2.0 D.1.A: visualize the BCGT-scale multi-hypothesis prior
factory.

Renders a 4-panel chart:
  - Panel 1: H_NW prior mean field (Gaussian blob in the NW quadrant)
  - Panel 2: H_SE prior mean field (Gaussian blob in the SE quadrant)
  - Panel 3: H_null prior mean field (flat zero)
  - Panel 4: Expected deposit probability per cell under uniform belief,
            with the top-K candidate cells highlighted.

Also runs a quick SARSOP solve at the top-K subproblem to confirm the
D.1.B per-step solver works at the chosen K and reports timing.

Output: data/derived/bcgt/fig_v20_d1_priors_setup.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai_minerals.decision.v20.bcgt_scale import (
    BcgtScaleSARSOPPolicy,
    DEFAULT_N_SIDE,
    DEFAULT_TOP_K,
    expected_deposit_per_cell,
    make_bcgt_synthetic_hypothesis_set,
    realize_deposit_sets,
)

REPO = Path(__file__).resolve().parents[2]
OUT_PNG = REPO / "data/derived/bcgt/fig_v20_d1_priors_setup.png"
POMDPSOL = REPO / "vendor/sarsop/pomdpsol"


def main() -> int:
    n_side = DEFAULT_N_SIDE
    hset, coords = make_bcgt_synthetic_hypothesis_set(n_side=n_side)
    n_cells = hset.hypotheses[0].n_cells

    fields = [
        hset.hypotheses[0].prior_mean_field,
        hset.hypotheses[1].prior_mean_field,
        np.zeros(n_cells),  # null
    ]
    titles = ["H_NW: deposit blob NW quadrant", "H_SE: deposit blob SE quadrant",
              "H_null: no deposit"]

    belief = hset.initial_prior()
    ep = expected_deposit_per_cell(hset, belief)

    top_k_cells = np.argsort(-ep)[:DEFAULT_TOP_K]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))

    vmin = min(np.min(f) for f in fields)
    vmax = max(np.max(f) for f in fields[:2])

    for ax, field, title in zip(axes[:3], fields, titles):
        im = ax.imshow(
            field.reshape(n_side, n_side),
            origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
            extent=(0, n_side - 1, 0, n_side - 1),
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Panel 4: expected deposit per cell + top-K markers
    ax = axes[3]
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
        f"Expected deposit prob (uniform belief)\n"
        f"top {DEFAULT_TOP_K} candidate cells (red)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "D.1.A: BCGT-scale multi-hypothesis prior setup. "
        f"{n_side}x{n_side} grid, 2 synthetic GP-correlated hypotheses "
        "plus null, Matern v=2.5, l=1500 m.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")

    # D.1.B per-step SARSOP timing check
    print("\nD.1.B per-step SARSOP timing check:")
    rng = np.random.default_rng(20260613)
    deposit_sets = realize_deposit_sets(hset, rng)
    print(f"  realized deposit cells per h: "
          f"{[len(deposit_sets[i]) for i in deposit_sets]}")

    policy = BcgtScaleSARSOPPolicy(
        hypothesis_set=hset,
        deposit_sets=deposit_sets,
        pomdpsol_path=POMDPSOL,
        top_k=DEFAULT_TOP_K,
    )

    t0 = time.perf_counter()
    chosen = policy.choose_action()
    elapsed = time.perf_counter() - t0
    print(f"  SARSOP per-step solve: {elapsed:.2f} s, chose cell {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
