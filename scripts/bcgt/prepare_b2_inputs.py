"""Build B.2 retrospective inputs: 50x50 subarea around a chosen BCGT mining district.

For each district (or all of them, via `--all`), produces THREE prior
variants for the same 50x50 subarea:

  b2_inputs_{district}.npz
      Gaussian-smoothed `any_mineral_occurrence` from BCGS MINFILE; the
      maximally informative prior (likely temporally contaminated since
      MINFILE is a current snapshot).
  b2_inputs_uniform_{district}.npz
      uniform 0.1 prior across the grid; the planner starts blind and
      has to learn from observations.
  b2_inputs_pre2010_only_{district}.npz
      prior built from pre-2010 BCGS drilling alone; Gaussian-smoothed
      Cu+ rate among pre-2010 holes. Leak-free by construction but very
      sparse signal.

All three share the same post-2010 ground truth + pre-2010 drilled
set + cell coordinates; only `prior_mean` differs.

District centroids come from `src.ai_minerals.regions.bcgt.BCGT_B2_CLUSTERS`
(identified by DBSCAN on the post-2010 Cu+ point cloud).

Backward compatibility: when invoked with no arguments, the script
writes outputs named `b2_inputs.npz`, `b2_inputs_uniform.npz`, and
`b2_inputs_pre2010_only.npz` (no district suffix) for the KSM district,
matching the existing path conventions the v20_b2_retrospective_benchmark.py
script reads from.

Usage:

    .venv/bin/python scripts/bcgt/prepare_b2_inputs.py
        # KSM, files at b2_inputs.npz (legacy paths)

    .venv/bin/python scripts/bcgt/prepare_b2_inputs.py --district Red_Chris
        # files at b2_inputs_Red_Chris.npz etc.

    .venv/bin/python scripts/bcgt/prepare_b2_inputs.py --all
        # all 7 BCGT clusters; legacy paths still written for KSM
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ai_minerals.regions.bcgt import BCGT_B2_CLUSTERS

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "data/derived/bcgt/bcgs_pre_post_2010_overlay.parquet"
FEATURES = REPO / "data/derived/features_bcgt_500m.parquet"
OUT_DIR = REPO / "data/derived/bcgt"

LEGACY_DEFAULT_DISTRICT = "KSM"
GRID_SIZE = 50  # cells per side -> 25 km x 25 km at 500 m spacing
SMOOTH_SIGMA_CELLS = 3  # Gaussian smoothing sigma in cells (1500 m)
PRIOR_INFORMATIVE_PEAK = 0.4
PRIOR_UNIFORM_LEVEL = 0.1


def output_paths(district: str, *, use_legacy_paths: bool) -> tuple[Path, Path, Path]:
    """Return the three output NPZ paths for a district.

    When `use_legacy_paths` is True (KSM with no `--district` flag), the
    files take the un-suffixed historic names so the existing benchmark
    script keeps working without modification.
    """
    if use_legacy_paths:
        return (
            OUT_DIR / "b2_inputs.npz",
            OUT_DIR / "b2_inputs_uniform.npz",
            OUT_DIR / "b2_inputs_pre2010_only.npz",
        )
    return (
        OUT_DIR / f"b2_inputs_{district}.npz",
        OUT_DIR / f"b2_inputs_uniform_{district}.npz",
        OUT_DIR / f"b2_inputs_pre2010_only_{district}.npz",
    )


def build_subarea_inputs(
    district: str,
    *,
    use_legacy_paths: bool = False,
) -> dict:
    """Build the 50x50 subarea inputs centered on the named district.

    Returns a summary dict (cell counts, prior stats) for the run-log.
    Writes three NPZ files at the paths from `output_paths`.
    """
    cluster = BCGT_B2_CLUSTERS[district]
    center_row = cluster["center_row"]
    center_col = cluster["center_col"]

    print(f"[district] {district}: row={center_row}, col={center_col}, "
          f"lat={cluster['center_lat']:.3f}, lon={cluster['center_lon']:.3f}")
    print(f"[district] description: {cluster['description']}")

    print("[load] overlay + features...")
    overlay = pd.read_parquet(OVERLAY)
    feat = pd.read_parquet(FEATURES)

    half = GRID_SIZE // 2
    r0, r1 = center_row - half, center_row + half
    c0, c1 = center_col - half, center_col + half

    row_grid, col_grid = np.meshgrid(
        np.arange(r0, r1), np.arange(c0, c1), indexing="ij",
    )
    sub_df = pd.DataFrame({
        "row": row_grid.ravel(),
        "col": col_grid.ravel(),
    })

    sub_df = sub_df.merge(
        feat[["row", "col", "x", "y", "any_mineral_occurrence"]],
        on=["row", "col"], how="left",
    )
    sub_df = sub_df.merge(
        overlay[[
            "row", "col",
            "pre_2010_n_holes", "pre_2010_cu_positive_n_holes",
            "post_2010_n_holes",
            "post_2010_cu_positive_n_holes", "post_2010_max_cu_ppm",
        ]],
        on=["row", "col"], how="left",
    )
    for col_name in (
        "any_mineral_occurrence",
        "pre_2010_n_holes", "pre_2010_cu_positive_n_holes",
        "post_2010_n_holes", "post_2010_cu_positive_n_holes",
    ):
        sub_df[col_name] = sub_df[col_name].fillna(0).astype(int)
    for col_name in ("post_2010_max_cu_ppm",):
        sub_df[col_name] = sub_df[col_name].fillna(0.0).astype(float)

    n_cells = GRID_SIZE * GRID_SIZE
    if n_cells != len(sub_df):
        raise RuntimeError(
            f"subgrid mismatch: expected {n_cells} cells, got {len(sub_df)}"
        )

    n_occ = int(sub_df["any_mineral_occurrence"].sum())
    n_pre_drilled = int((sub_df["pre_2010_n_holes"] > 0).sum())
    n_post_drilled = int((sub_df["post_2010_n_holes"] > 0).sum())
    n_post_positive = int((sub_df["post_2010_cu_positive_n_holes"] > 0).sum())
    n_pre_positive = int((sub_df["pre_2010_cu_positive_n_holes"] > 0).sum())
    print(f"  subarea: row {r0}-{r1}, col {c0}-{c1}  ({n_cells} cells)")
    print(f"  any_mineral_occurrence:  {n_occ} cells")
    print(f"  pre-2010 drilled cells:  {n_pre_drilled}")
    print(f"  pre-2010 Cu+ cells:      {n_pre_positive}")
    print(f"  post-2010 drilled cells: {n_post_drilled}")
    print(f"  post-2010 Cu+ cells:     {n_post_positive}")

    # Informative prior: Gaussian-smoothed mineral-occurrence flag
    occ = (
        sub_df["any_mineral_occurrence"]
        .to_numpy(dtype=float).reshape(GRID_SIZE, GRID_SIZE)
    )
    smoothed_occ = gaussian_filter(occ, sigma=SMOOTH_SIGMA_CELLS, mode="reflect")
    if smoothed_occ.max() > 1e-9:
        prior_informative = (
            smoothed_occ / smoothed_occ.max() * PRIOR_INFORMATIVE_PEAK
        ).ravel()
    else:
        prior_informative = np.full(n_cells, PRIOR_UNIFORM_LEVEL, dtype=float)

    # Uniform prior
    prior_uniform = np.full(n_cells, PRIOR_UNIFORM_LEVEL, dtype=float)

    # Pre-2010 leak-free prior
    pre_cu_positive = (
        sub_df["pre_2010_cu_positive_n_holes"]
        .to_numpy(dtype=float)
        .reshape(GRID_SIZE, GRID_SIZE)
    )
    smoothed_pre = gaussian_filter(
        pre_cu_positive, sigma=SMOOTH_SIGMA_CELLS, mode="reflect",
    )
    if smoothed_pre.max() > 1e-9:
        prior_pre2010 = (
            smoothed_pre / smoothed_pre.max() * PRIOR_INFORMATIVE_PEAK
        ).ravel()
    else:
        prior_pre2010 = np.full(n_cells, PRIOR_UNIFORM_LEVEL, dtype=float)

    # Shared arrays
    post_positive_field = (
        sub_df["post_2010_cu_positive_n_holes"] > 0
    ).to_numpy(dtype=int)
    post_grade_field = (
        sub_df["post_2010_max_cu_ppm"] / 10000.0
    ).to_numpy(dtype=float)
    pre_drilled_field = (sub_df["pre_2010_n_holes"] > 0).to_numpy(dtype=int)
    coords = sub_df[["x", "y"]].to_numpy(dtype=float)

    print("\n[stats]")
    print(f"  prior_informative: min={prior_informative.min():.3f}  "
          f"max={prior_informative.max():.3f}  "
          f"mean={prior_informative.mean():.3f}")
    print(f"  prior_uniform:     constant {PRIOR_UNIFORM_LEVEL} over {n_cells} cells")
    print(f"  prior_pre2010:     min={prior_pre2010.min():.3f}  "
          f"max={prior_pre2010.max():.3f}  "
          f"mean={prior_pre2010.mean():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common = dict(
        cell_coords_m=coords,
        post_2010_positive=post_positive_field,
        post_2010_grade=post_grade_field,
        pre_2010_drilled=pre_drilled_field,
        row_min=r0, col_min=c0,
        n_rows=GRID_SIZE, n_cols=GRID_SIZE,
    )
    out_informative, out_uniform, out_pre2010 = output_paths(
        district, use_legacy_paths=use_legacy_paths,
    )
    np.savez(out_informative, prior_mean=prior_informative, **common)
    np.savez(out_uniform, prior_mean=prior_uniform, **common)
    np.savez(out_pre2010, prior_mean=prior_pre2010, **common)
    print(f"\n[wrote] {out_informative}")
    print(f"[wrote] {out_uniform}")
    print(f"[wrote] {out_pre2010}")

    return dict(
        district=district,
        n_cells=n_cells,
        n_pre_drilled=n_pre_drilled,
        n_post_drilled=n_post_drilled,
        n_post_positive=n_post_positive,
        prior_informative_mean=float(prior_informative.mean()),
        prior_pre2010_max=float(prior_pre2010.max()),
        out_informative=str(out_informative),
        out_uniform=str(out_uniform),
        out_pre2010=str(out_pre2010),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--district", type=str, default=None,
        help=f"District name. Default: {LEGACY_DEFAULT_DISTRICT} "
             "(written to legacy un-suffixed paths). "
             f"Choices: {list(BCGT_B2_CLUSTERS.keys())}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate inputs for all districts in BCGT_B2_CLUSTERS.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        for district in BCGT_B2_CLUSTERS:
            use_legacy = (district == LEGACY_DEFAULT_DISTRICT)
            build_subarea_inputs(district, use_legacy_paths=use_legacy)
            print()
            # In `--all` mode, also write KSM under the suffixed path
            # so downstream batch scripts can address all districts
            # uniformly.
            if use_legacy:
                build_subarea_inputs(district, use_legacy_paths=False)
                print()
        return 0
    district = args.district or LEGACY_DEFAULT_DISTRICT
    use_legacy = (args.district is None)
    build_subarea_inputs(district, use_legacy_paths=use_legacy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
