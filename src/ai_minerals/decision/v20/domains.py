"""Structured-prior domain generation for Mern 2024 multi-hypothesis priors.

Ports the geometric idea behind Corso 2024's
`HierarchicalMineralExploration.jl/domains.jl` (~133 LOC of Julia
Turing.jl probabilistic programs) to deterministic-given-seed Python.

Each Mern 2024 hypothesis is parameterized by a (n_grabens, n_domains)
pair from the 2x2 grid {(1,1), (1,2), (2,1), (2,2)}. Under each
hypothesis the per-cell GP mean is conditional on which structural
(graben) and geochemical (domain) features the realization happens to
sample. This module:

- samples those structural and geochemical polygons given a seed,
- converts polygon vertices to binary cell masks (1 inside any polygon,
  0 outside) on the 32x32 working grid the paper uses,
- exposes a helper that combines graben + domain masks into a single
  per-cell prior-mean field for the Hypothesis class to consume.

The Julia code samples graben configurations via Turing models with
priors on the graben bottom location and width. The port here uses
explicit `np.random.Generator` uniform / Gaussian draws so the
configuration is fully reproducible given a seed. Loses the Bayesian
formalism but preserves the geometry and the Mern p.28 parameter
values.

References
----------
- Mern et al. 2024 "Intelligent Prospector v2.0",
  arXiv 2410.10610, Appendix p.28.
- Corso 2024 "HierarchicalMineralExploration.jl",
  https://github.com/ancorso/HierarchicalMineralExploration.jl
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

# Default sampling parameters chosen to roughly match the geometric scale
# of the paper's 32x32 grid: grabens are narrow strips (5-12 cells wide
# in their narrow dimension, 16-30 cells long), domains are blobby
# regions (radii 4-9 cells).
DEFAULT_GRABEN_WIDTH_RANGE_CELLS = (5, 12)
DEFAULT_GRABEN_LENGTH_RANGE_CELLS = (16, 30)
DEFAULT_DOMAIN_RADIUS_MEAN_CELLS = 6.0
DEFAULT_DOMAIN_RADIUS_STD_CELLS = 2.0
DEFAULT_DOMAIN_VERTEX_COUNT = 8

# Paper-derived mean boosts for the GP-prior mean field.
# Normalized to the ai_minerals 0-1 range used elsewhere in v20.
# Mern p.28 quotes mean_thickness = 7.5 inside graben (vs 1.0 outside)
# and mean_grade = 0.085 inside altered domain (vs 0.0 outside) in
# normalized units. We collapse the two GP fields to a single
# prior-mean field for the v2.0 single-field Hypothesis class.
GRABEN_MEAN_BOOST = 0.075
DOMAIN_MEAN_BOOST = 0.085
BASE_MEAN = 0.0


def sample_graben_polygons(
    n_grabens: int,
    grid_n: int,
    rng: np.random.Generator,
    width_range_cells: tuple[int, int] = DEFAULT_GRABEN_WIDTH_RANGE_CELLS,
    length_range_cells: tuple[int, int] = DEFAULT_GRABEN_LENGTH_RANGE_CELLS,
    centerline_margin_cells: int = 4,
) -> list[list[tuple[float, float]]]:
    """Sample N graben polygons on a `grid_n` x `grid_n` grid.

    Each graben is a rotated rectangular strip with random center,
    orientation, width, and length. Returns a list of polygons, each a
    list of (row, col) vertex pairs in grid-cell coordinates. Vertex
    coordinates may be slightly outside [0, grid_n] when the graben
    extends past the grid boundary; the mask construction clips them.

    Parameters
    ----------
    n_grabens : int
        Number of grabens to sample. Mern 2024 uses 1 or 2.
    grid_n : int
        Side length of the working grid in cells. The paper uses 32.
    rng : np.random.Generator
        Random state. Determines the realization exactly.
    width_range_cells, length_range_cells : tuple[int, int]
        Inclusive integer ranges for graben width and length.
    centerline_margin_cells : int
        Minimum distance from grid edge for the graben center, so the
        graben mostly lies inside the working area.

    Returns
    -------
    list of polygons, each a list of (row, col) tuples (4 vertices per graben).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> polys = sample_graben_polygons(n_grabens=2, grid_n=32, rng=rng)
    >>> len(polys), len(polys[0])
    (2, 4)
    """
    if n_grabens < 0:
        raise ValueError(f"n_grabens must be >= 0; got {n_grabens}")
    if grid_n < 4:
        raise ValueError(f"grid_n must be >= 4; got {grid_n}")
    polygons: list[list[tuple[float, float]]] = []
    margin = float(centerline_margin_cells)
    lo, hi = margin, float(grid_n) - margin
    for _ in range(n_grabens):
        center_row = float(rng.uniform(lo, hi))
        center_col = float(rng.uniform(lo, hi))
        theta = float(rng.uniform(0.0, np.pi))
        width_cells = int(rng.integers(
            width_range_cells[0], width_range_cells[1] + 1,
        ))
        length_cells = int(rng.integers(
            length_range_cells[0], length_range_cells[1] + 1,
        ))
        half_w = width_cells / 2.0
        half_l = length_cells / 2.0
        local_corners = [
            (-half_w, -half_l),
            (+half_w, -half_l),
            (+half_w, +half_l),
            (-half_w, +half_l),
        ]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        polygon: list[tuple[float, float]] = []
        for delta_row_local, delta_col_local in local_corners:
            row_rot = delta_row_local * cos_t - delta_col_local * sin_t
            col_rot = delta_row_local * sin_t + delta_col_local * cos_t
            polygon.append((center_row + row_rot, center_col + col_rot))
        polygons.append(polygon)
    return polygons


def sample_geochem_domain_polygons(
    n_domains: int,
    grid_n: int,
    rng: np.random.Generator,
    n_vertices: int = DEFAULT_DOMAIN_VERTEX_COUNT,
    radius_mean_cells: float = DEFAULT_DOMAIN_RADIUS_MEAN_CELLS,
    radius_std_cells: float = DEFAULT_DOMAIN_RADIUS_STD_CELLS,
    centerline_margin_cells: int = 6,
) -> list[list[tuple[float, float]]]:
    """Sample N geochemical-domain polygons on a `grid_n` x `grid_n` grid.

    Each domain is an irregular closed blob around a random center. The
    blob shape is generated by sampling per-angle radii from a Gaussian.
    Vertex angles are evenly spaced around the center.

    Parameters
    ----------
    n_domains : int
        Number of domains to sample. Mern 2024 uses 1 or 2.
    grid_n : int
        Side length of the working grid in cells.
    rng : np.random.Generator
        Random state.
    n_vertices : int
        Number of polygon vertices.
    radius_mean_cells, radius_std_cells : float
        Per-vertex radius is sampled from N(radius_mean, radius_std^2),
        clipped to [1.0, grid_n / 2].
    centerline_margin_cells : int
        Minimum distance from grid edge for the domain center.

    Returns
    -------
    list of polygons, each a list of (row, col) tuples.
    """
    if n_domains < 0:
        raise ValueError(f"n_domains must be >= 0; got {n_domains}")
    polygons: list[list[tuple[float, float]]] = []
    margin = float(centerline_margin_cells)
    lo, hi = margin, float(grid_n) - margin
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    max_radius = float(grid_n) / 2.0
    for _ in range(n_domains):
        center_row = float(rng.uniform(lo, hi))
        center_col = float(rng.uniform(lo, hi))
        radii = np.abs(rng.normal(
            radius_mean_cells, radius_std_cells, size=n_vertices,
        ))
        radii = np.clip(radii, 1.0, max_radius)
        polygon: list[tuple[float, float]] = []
        for angle, radius in zip(angles, radii):
            polygon.append((
                center_row + radius * np.sin(angle),
                center_col + radius * np.cos(angle),
            ))
        polygons.append(polygon)
    return polygons


def domain_mask_from_polygons(
    polygons: list[list[tuple[float, float]]],
    grid_n: int,
) -> np.ndarray:
    """Convert polygon vertices to a binary cell mask on a grid_n x grid_n grid.

    Returns a (grid_n, grid_n) boolean array where True indicates the
    cell center is inside any of the provided polygons. Uses
    matplotlib.path.Path.contains_points.

    Parameters
    ----------
    polygons : list of polygons
        Each polygon is a list of (row, col) vertex tuples. Polygons
        with fewer than 3 vertices are silently ignored.
    grid_n : int
        Side length of the working grid in cells.

    Returns
    -------
    np.ndarray, shape (grid_n, grid_n), dtype bool
        Binary mask: True inside at least one polygon.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> polys = sample_graben_polygons(1, 32, rng)
    >>> mask = domain_mask_from_polygons(polys, 32)
    >>> mask.shape, mask.dtype
    ((32, 32), dtype('bool'))
    >>> bool(mask.any())
    True
    """
    if grid_n < 1:
        raise ValueError(f"grid_n must be >= 1; got {grid_n}")
    row_grid, col_grid = np.meshgrid(
        np.arange(grid_n) + 0.5,
        np.arange(grid_n) + 0.5,
        indexing="ij",
    )
    cell_centers = np.column_stack([row_grid.ravel(), col_grid.ravel()])
    mask_flat = np.zeros(grid_n * grid_n, dtype=bool)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        path = Path(polygon)
        mask_flat |= path.contains_points(cell_centers)
    return mask_flat.reshape(grid_n, grid_n)


def prior_mean_field_from_masks(
    graben_mask: np.ndarray,
    domain_mask: np.ndarray,
    base_mean: float = BASE_MEAN,
    graben_boost: float = GRABEN_MEAN_BOOST,
    domain_boost: float = DOMAIN_MEAN_BOOST,
) -> np.ndarray:
    """Combine graben + domain masks into a per-cell prior-mean field.

    The Mern 2024 paper models TWO separate GP fields (thickness and
    grade) with means conditional on different masks. The v20 Hypothesis
    class currently carries a single prior-mean field; we collapse the
    two contributions additively:

        mean[cell] = base + graben_boost * graben_mask + domain_boost * domain_mask

    Both masks are boolean arrays of shape (grid_n, grid_n); the output
    is flat (grid_n * grid_n,) so it can be passed directly to
    Hypothesis.prior_mean_field.

    Examples
    --------
    >>> import numpy as np
    >>> graben = np.zeros((32, 32), dtype=bool)
    >>> graben[10:20, 14:18] = True
    >>> domain = np.zeros((32, 32), dtype=bool)
    >>> domain[5:15, 10:25] = True
    >>> field = prior_mean_field_from_masks(graben, domain)
    >>> field.shape
    (1024,)
    """
    if graben_mask.shape != domain_mask.shape:
        raise ValueError(
            f"graben_mask and domain_mask must have the same shape; "
            f"got {graben_mask.shape} vs {domain_mask.shape}"
        )
    field = (
        float(base_mean)
        + float(graben_boost) * graben_mask.astype(np.float64)
        + float(domain_boost) * domain_mask.astype(np.float64)
    )
    return field.ravel()
