"""Build B.2 retrospective inputs: 50x50 subarea around the densest BCGS Cu+ cluster.

Produces THREE prior variants for the same subarea:
  b2_inputs.npz                  Gaussian-smoothed `any_mineral_occurrence` from
                                 BCGS MINFILE; the maximally-informative prior
                                 (likely temporally contaminated since MINFILE is
                                 a current snapshot).
  b2_inputs_uniform.npz          uniform 0.1 prior across the grid; the planner
                                 starts blind and has to learn from observations.
  b2_inputs_pre2010_only.npz     prior built from pre-2010 BCGS drilling alone:
                                 Gaussian-smoothed Cu+ rate among pre-2010 holes.
                                 Leak-free by construction but very sparse signal.

All three share the same post-2010 ground truth + pre-2010 drilled set + cell
coordinates; only prior_mean differs.

The cluster choice (largest DBSCAN cluster of post-2010 Cu+ cells, centroid at
row=131 / col=101 in the full 500 m BCGT grid; KSM / Kerr-Sulphurets-Mitchell
area) is hard-coded; rebuild this script to retarget.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "data/derived/bcgt/bcgs_pre_post_2010_overlay.parquet"
FEATURES = REPO / "data/derived/features_bcgt_500m.parquet"
OUT_DIR = REPO / "data/derived/bcgt"
OUT_INFORMATIVE = OUT_DIR / "b2_inputs.npz"
OUT_UNIFORM = OUT_DIR / "b2_inputs_uniform.npz"
OUT_PRE2010 = OUT_DIR / "b2_inputs_pre2010_only.npz"

# 50 x 50 subarea centered on the largest BCGS Cu+ cluster (DBSCAN row 131,
# col 101 in the full 500 m grid).
CENTER_ROW = 131
CENTER_COL = 101
GRID_SIZE = 50          # cells per side -> 25 km x 25 km at 500 m spacing
SMOOTH_SIGMA_CELLS = 3  # Gaussian smoothing sigma in cells (1500 m)


def main() -> int:
    print(f"[load] overlay + features...")
    overlay = pd.read_parquet(OVERLAY)
    feat = pd.read_parquet(FEATURES)

    half = GRID_SIZE // 2
    r0, r1 = CENTER_ROW - half, CENTER_ROW + half
    c0, c1 = CENTER_COL - half, CENTER_COL + half

    # Build a regular (row, col) grid for this subarea.
    rr, cc = np.meshgrid(
        np.arange(r0, r1), np.arange(c0, c1), indexing="ij",
    )
    sub_rows = rr.ravel()
    sub_cols = cc.ravel()
    sub_df = pd.DataFrame({"row": sub_rows, "col": sub_cols})

    # Join features (gives x/y EPSG:3005 and the mineral-occurrence indicator)
    sub_df = sub_df.merge(
        feat[["row", "col", "x", "y", "any_mineral_occurrence"]],
        on=["row", "col"], how="left",
    )
    # Join BCGS overlay
    sub_df = sub_df.merge(
        overlay[[
            "row", "col",
            "pre_2010_n_holes", "pre_2010_cu_positive_n_holes",
            "post_2010_n_holes",
            "post_2010_cu_positive_n_holes", "post_2010_max_cu_ppm",
        ]],
        on=["row", "col"], how="left",
    )
    for c in (
        "any_mineral_occurrence",
        "pre_2010_n_holes", "pre_2010_cu_positive_n_holes",
        "post_2010_n_holes", "post_2010_cu_positive_n_holes",
    ):
        sub_df[c] = sub_df[c].fillna(0).astype(int)
    for c in ("post_2010_max_cu_ppm",):
        sub_df[c] = sub_df[c].fillna(0.0).astype(float)

    print(f"  subarea: row {r0}-{r1}, col {c0}-{c1}  ({GRID_SIZE * GRID_SIZE} cells)")
    print(f"  any_mineral_occurrence: {sub_df['any_mineral_occurrence'].sum()} cells")
    print(f"  pre-2010 drilled cells: {(sub_df['pre_2010_n_holes'] > 0).sum()}")
    print(f"  post-2010 drilled cells: {(sub_df['post_2010_n_holes'] > 0).sum()}")
    print(f"  post-2010 Cu+ cells (>=2000 ppm): "
          f"{(sub_df['post_2010_cu_positive_n_holes'] > 0).sum()}")

    n_cells = GRID_SIZE * GRID_SIZE
    if n_cells != len(sub_df):
        raise RuntimeError(
            f"subgrid mismatch: expected {n_cells} cells, got {len(sub_df)}"
        )

    # Three prior fields, same other arrays.
    occ = sub_df["any_mineral_occurrence"].to_numpy(dtype=float).reshape(GRID_SIZE, GRID_SIZE)
    smoothed_occ = gaussian_filter(occ, sigma=SMOOTH_SIGMA_CELLS, mode="reflect")
    prior_informative = (smoothed_occ / max(smoothed_occ.max(), 1e-9) * 0.4).ravel()

    prior_uniform = np.full(GRID_SIZE * GRID_SIZE, 0.1, dtype=float)

    pre_cu_pos = (
        sub_df["pre_2010_cu_positive_n_holes"]
        .to_numpy(dtype=float)
        .reshape(GRID_SIZE, GRID_SIZE)
    )
    smoothed_pre = gaussian_filter(pre_cu_pos, sigma=SMOOTH_SIGMA_CELLS, mode="reflect")
    # Anchor pre-2010 prior at the same 0.4 peak height as the informative
    # variant when there's signal; otherwise fall back to uniform 0.1 to keep
    # the planner well-conditioned.
    pre_max = smoothed_pre.max()
    if pre_max > 1e-9:
        prior_pre2010 = (smoothed_pre / pre_max * 0.4).ravel()
    else:
        prior_pre2010 = np.full(GRID_SIZE * GRID_SIZE, 0.1, dtype=float)

    # Shared arrays
    post_pos_field = (sub_df["post_2010_cu_positive_n_holes"] > 0).to_numpy(dtype=int)
    post_grade_field = (sub_df["post_2010_max_cu_ppm"] / 10000.0).to_numpy(dtype=float)
    pre_drilled_field = (sub_df["pre_2010_n_holes"] > 0).to_numpy(dtype=int)
    coords = sub_df[["x", "y"]].to_numpy(dtype=float)

    print(f"\n[stats]")
    print(f"  prior_informative: min={prior_informative.min():.3f}  max={prior_informative.max():.3f}  "
          f"mean={prior_informative.mean():.3f}")
    print(f"  prior_uniform:     constant 0.1 over {len(prior_uniform)} cells")
    print(f"  prior_pre2010:     min={prior_pre2010.min():.3f}  max={prior_pre2010.max():.3f}  "
          f"mean={prior_pre2010.mean():.3f}")
    print(f"  pre-2010 Cu+ cells used in pre2010 prior: "
          f"{(pre_cu_pos.ravel() > 0).sum()}")
    print(f"  post-2010 positive cells: {post_pos_field.sum()}")
    print(f"  pre-2010 drilled cells:   {pre_drilled_field.sum()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common = dict(
        cell_coords_m=coords,
        post_2010_positive=post_pos_field,
        post_2010_grade=post_grade_field,
        pre_2010_drilled=pre_drilled_field,
        row_min=r0, col_min=c0,
        n_rows=GRID_SIZE, n_cols=GRID_SIZE,
    )
    np.savez(OUT_INFORMATIVE, prior_mean=prior_informative, **common)
    np.savez(OUT_UNIFORM, prior_mean=prior_uniform, **common)
    np.savez(OUT_PRE2010, prior_mean=prior_pre2010, **common)
    print(f"\n[wrote] {OUT_INFORMATIVE}")
    print(f"[wrote] {OUT_UNIFORM}")
    print(f"[wrote] {OUT_PRE2010}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
