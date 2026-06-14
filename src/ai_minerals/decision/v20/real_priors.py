"""Per-deposit-type prior surfaces over the BCGT 30x30 D.1 grid.

The D.1 SARSOP setup uses a 30x30 synthetic grid at 500 m spacing. The
synthetic NW/SE-blob priors live on a (0, 0)-anchored coordinate grid.
For D.1.D we replace the synthetic blobs with prior surfaces aggregated
from the real BCGS MINFILE per-deposit-type cell labels.

Pipeline:
  1. Load the BCGT 500 m feature parquet (108k cells, 460x270 at 500 m).
  2. For each deposit type (`is_porphyry`, `is_skarn`, `is_epithermal`,
     `is_vms`), build a full-resolution binary occurrence map.
  3. Downsample by block-averaging to the target n_side x n_side.
  4. Gaussian-smooth at sigma cells (default 1.5).
  5. Renormalize each surface to a fixed peak (default 0.4, matching
     the B.2 informative-prior convention).

Returns one (n_side ** 2,) flattened surface per type plus a shared
(0, 0)-anchored coordinate grid that matches the synthetic D.1 setup,
so the SARSOP / particle-filter / greedy stack does not need to know
the priors came from real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

__all__ = [
    "bcgs_deposit_type_prior_surfaces",
    "DEFAULT_TYPES",
    "DEFAULT_SMOOTH_SIGMA_CELLS",
    "DEFAULT_PEAK",
]


DEFAULT_TYPES: tuple[str, ...] = (
    "is_porphyry",
    "is_skarn",
    "is_epithermal",
    "is_vms",
)
DEFAULT_SMOOTH_SIGMA_CELLS = 1.5
DEFAULT_PEAK = 0.4


def bcgs_deposit_type_prior_surfaces(
    features_path: str | Path,
    n_side: int = 30,
    spacing_m: float = 500.0,
    types: tuple[str, ...] = DEFAULT_TYPES,
    smooth_sigma_cells: float = DEFAULT_SMOOTH_SIGMA_CELLS,
    peak: float = DEFAULT_PEAK,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build per-deposit-type prior surfaces on an n_side x n_side grid.

    Parameters
    ----------
    features_path : str or Path
        Path to a BCGT 500 m feature parquet with ``row``, ``col``, and
        per-type binary label columns. Default consumer is
        ``data/derived/features_bcgt_500m.parquet``.
    n_side : int, default 30
        Side length of the output grid in cells. Matches the D.1 setup.
    spacing_m : float, default 500
        Physical spacing between output cell centers, in meters. The
        coordinate grid is (0, 0)-anchored to match the D.1 synthetic
        setup; absolute georeferencing is dropped since the GP kernel
        only cares about pairwise distances.
    types : tuple of str
        Binary label column names to aggregate. Default is the four
        BCGS deposit-class columns.
    smooth_sigma_cells : float, default 1.5
        Gaussian smoothing sigma in output-grid cells.
    peak : float, default 0.4
        Per-surface renormalization peak. Matches the B.2 informative-
        prior convention so the SARSOP regime parameters apply.

    Returns
    -------
    surfaces : dict[str, np.ndarray]
        Mapping from type name (without the ``is_`` prefix) to a flat
        length-(n_side ** 2) prior surface.
    coords : np.ndarray
        (n_side ** 2, 2) cell coordinates in meters, (0, 0)-anchored.

    Notes
    -----
    The full-resolution BCGT grid is 460x270 cells. Block-averaging
    onto n_side x n_side aggregates roughly (460/30) * (270/30) ~ 138
    source cells per output cell, which compresses 15 km x 7.5 km of
    BCGT into each output cell when n_side=30. The resulting surfaces
    are coarser than the source data; for finer resolution use a
    larger n_side.
    """
    df = pd.read_parquet(features_path)
    for t in types:
        if t not in df.columns:
            raise ValueError(f"Missing label column {t!r} in {features_path}")
    row_max = int(df["row"].max())
    col_max = int(df["col"].max())

    bin_r = row_max // n_side
    bin_c = col_max // n_side
    if bin_r < 1 or bin_c < 1:
        raise ValueError(
            f"n_side={n_side} too large for source grid {row_max}x{col_max}"
        )

    surfaces: dict[str, np.ndarray] = {}
    for t in types:
        full = np.zeros((row_max, col_max), dtype=np.float64)
        hits = df[df[t] == 1]
        rows = hits["row"].astype(int).values - 1
        cols = hits["col"].astype(int).values - 1
        np.add.at(full, (rows, cols), 1.0)

        cropped = full[: bin_r * n_side, : bin_c * n_side]
        ds = cropped.reshape(n_side, bin_r, n_side, bin_c).mean(axis=(1, 3))
        smoothed = gaussian_filter(ds, sigma=smooth_sigma_cells)
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max() * peak

        short_name = t.removeprefix("is_")
        surfaces[short_name] = smoothed.flatten()

    x = np.arange(n_side) * spacing_m
    y = np.arange(n_side) * spacing_m
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    return surfaces, coords
