"""Hypothesis hierarchy + GP machinery for bcgt-v2.0.

Maps to `hypotheses.jl` (633 LOC, Anthony Corso's reference) in
`/tmp/HierarchicalMineralExploration.jl/src/hypotheses.jl`.

Mern 2024 uses a 2-by-2 grid of competing geological hypotheses:
{1, 2 grabens} x {1, 2 geochemical domains} = 4 hypothesis classes,
plus a maximum-entropy mixture as the null h_0. Our BCGT adaptation
collapses this to single-hypothesis correlated draws for B.1; the
multi-hypothesis layer arrives in C.2.

Key parameters (locked 2026-06-10 from arXiv 2410.10610 p.28):
    GP kernel:           Matern v=2.5
    GP marginal stdev:   0.1 (synthetic units)
    Correlation length:  3 grid cells (paper) -> 1500 m (BCGT)
    Sensor noise:        sigma=0.001 (B.1 synthetic, matches paper)
    Hypothesis count:    4 paper + 1 null = 5 total (C.2 only)

LIBRARY CHOICE: sklearn.gaussian_process for B.1 prototype (no new dep).
Upgrade to `george` if performance becomes binding on 30x30 BCGT subarea
(2-3x speed over sklearn for static kernels per pip docs). `tinygp`
remains a fallback if C.2 multi-hypothesis needs JAX speed.

Reference (Julia line ranges):
    Hypothesis class:         hypotheses.jl 1-50
    Domain mask rendering:    hypotheses.jl 55-130
    GP prior + posterior:     hypotheses.jl 166-250
    ESS-based MCMC:           hypotheses.jl 350-450
    Null hypothesis:          hypotheses.jl 47-50

B.1 IMPLEMENTATION STATUS (2026-06-11):
    Hypothesis.sample_realization        DONE - GH issue #2
    Hypothesis.conditional_posterior     NOT YET - GH issue (queued)
    NullHypothesis.sample_realization    NOT YET - C.2 milestone
    HypothesisSet.*                      NOT YET - C.2 milestone
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.gaussian_process.kernels import Matern


# Locked parameters (arXiv 2410.10610 p.28; see pomdp_v20_implementation_plan.md)
KERNEL_NU = 2.5                # Matern v=5/2; NOT v=1.5 (corrected 2026-06-10)
KERNEL_MARGINAL_STD = 0.1
KERNEL_LENGTHSCALE_GRID = 3    # paper grid cells
KERNEL_LENGTHSCALE_M_BCGT = 1500.0  # BCGT subarea: 500 m/cell × 3
SENSOR_NOISE_GAUSSIAN_SIGMA = 0.001   # B.1 synthetic
N_HYPOTHESES_PAPER = 4         # C.2; H_1 through H_4
INCLUDE_NULL_HYPOTHESIS = True # C.2; total = N + 1

# Cholesky jitter: Matern v=2.5 eigenvalues can decay fast at long lengthscales,
# so add a small diagonal term for numerical stability. 1e-6 sigma^2 is the
# scale at which the jitter is invisible against the marginal variance.
_CHOLESKY_JITTER = 1e-6


@dataclass(frozen=True)
class Hypothesis:
    """A single geological hypothesis: GP prior over the property field.

    For B.1 single-hypothesis, only one Hypothesis instance is constructed
    per simulation. C.2 introduces a HypothesisSet with multiple instances
    plus a NullHypothesis.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. "porphyry-Cu", "skarn", "null").
    n_grabens, n_domains : int
        Mern 2024 paper's parameterization. For BCGT we collapse these
        to a single domain in B.1; C.2 may use them if real BCGS
        deposit-type splits are available.
    cell_coords_m : np.ndarray, shape (n_cells, 2)
        Per-cell (x, y) coordinates in the working CRS, in meters.
        Used to compute the kernel matrix from pairwise distances.
    prior_mean_field : np.ndarray, shape (n_cells,)
        Per-cell GP mean. Initialized from the v3 RF posterior surface.
        For BCGT this is the posterior mean of porphyry-Cu probability
        at each cell, optionally centered or logit-transformed.
    gp_kernel_nu : float
        Matern smoothness parameter. Locked at 2.5 per paper p.28.
    gp_marginal_std : float
        sigma in dimensionless units (B.1) or normalized grade units (B.2).
    gp_lengthscale_m : float
        Correlation length in working CRS meters (1500 m for BCGT).
    """
    name: str
    n_grabens: int
    n_domains: int
    cell_coords_m: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float64)
    )
    prior_mean_field: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    gp_kernel_nu: float = KERNEL_NU
    gp_marginal_std: float = KERNEL_MARGINAL_STD
    gp_lengthscale_m: float = KERNEL_LENGTHSCALE_M_BCGT

    def __post_init__(self) -> None:
        cc = np.asarray(self.cell_coords_m, dtype=np.float64)
        pmf = np.asarray(self.prior_mean_field, dtype=np.float64)
        if cc.ndim != 2 or cc.shape[1] != 2:
            raise ValueError(
                f"cell_coords_m must be (n_cells, 2); got shape {cc.shape}"
            )
        if pmf.ndim != 1 or pmf.shape[0] != cc.shape[0]:
            raise ValueError(
                f"prior_mean_field must be (n_cells,) matching cell_coords_m "
                f"({cc.shape[0]} cells); got shape {pmf.shape}"
            )
        if self.gp_kernel_nu not in (0.5, 1.5, 2.5):
            # sklearn's Matern only fast-paths v in {0.5, 1.5, 2.5}; nothing
            # else makes physical sense for our GP-prior choice either.
            raise ValueError(
                f"gp_kernel_nu must be one of {{0.5, 1.5, 2.5}}; "
                f"got {self.gp_kernel_nu}"
            )
        # Reflect dtype normalisations back through the frozen dataclass.
        object.__setattr__(self, "cell_coords_m", cc)
        object.__setattr__(self, "prior_mean_field", pmf)
        object.__setattr__(self, "_cholesky_cache", None)

    @property
    def n_cells(self) -> int:
        return int(self.cell_coords_m.shape[0])

    def _kernel_matrix(self) -> np.ndarray:
        """Matern kernel evaluated on cell_coords_m. (n_cells, n_cells)."""
        kernel = Matern(length_scale=self.gp_lengthscale_m, nu=self.gp_kernel_nu)
        return (self.gp_marginal_std ** 2) * kernel(self.cell_coords_m)

    def _cholesky(self) -> np.ndarray:
        """Cached lower-triangular Cholesky of K + jitter * I."""
        cache = self._cholesky_cache  # type: ignore[attr-defined]
        if cache is not None:
            return cache
        K = self._kernel_matrix()
        jitter = _CHOLESKY_JITTER * (self.gp_marginal_std ** 2)
        K_jittered = K + jitter * np.eye(K.shape[0])
        L = np.linalg.cholesky(K_jittered)
        object.__setattr__(self, "_cholesky_cache", L)
        return L

    def sample_realization(
        self,
        rng: np.random.Generator,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Draw N realizations from the GP prior. Shape (n_samples, n_cells).

        Each realization is `prior_mean_field + L @ z` where `L` is the
        Cholesky factor of the kernel matrix and `z` is a standard-normal
        vector. The Cholesky is cached per-Hypothesis (deterministic given
        coords + kernel params), so repeated calls only pay the matrix-vector
        cost.

        For BCGT we threshold the continuous GP draws at a level chosen
        so the marginal positive rate matches the per-cell RF posterior
        mean. That replaces v1.0's iid Bernoulli draws with spatially
        correlated draws where neighbor cells covary per the kernel.
        Thresholding happens outside this function — `CorrelatedDrillingProblem`
        does it.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1; got {n_samples}")
        L = self._cholesky()
        z = rng.standard_normal(size=(self.n_cells, n_samples))
        return (self.prior_mean_field[:, None] + L @ z).T  # (n_samples, n_cells)

    def conditional_posterior(
        self,
        observed_cells: np.ndarray,
        observed_grades: np.ndarray,
        sensor_noise_sigma: float = SENSOR_NOISE_GAUSSIAN_SIGMA,
    ) -> "Hypothesis":
        """Return a new Hypothesis with the GP conditioned on observations.

        Standard GP conditioning: posterior mean = K_*x K_xx^-1 (y - mu_x)
        + mu_*, posterior covariance = K_** - K_*x K_xx^-1 K_x*.

        TODO B.1: implement; reuse sklearn's predict(return_std=True).
        """
        raise NotImplementedError("B.1 milestone")

    @classmethod
    def from_domain_config(
        cls,
        name: str,
        n_grabens: int,
        n_domains: int,
        grid_n: int,
        rng: np.random.Generator,
        cell_spacing_m: float = 1.0,
        gp_marginal_std: float = KERNEL_MARGINAL_STD,
        gp_lengthscale_m: float | None = None,
        gp_kernel_nu: float = KERNEL_NU,
        base_mean: float | None = None,
        graben_boost: float | None = None,
        domain_boost: float | None = None,
    ) -> "Hypothesis":
        """Build a Hypothesis with a structured Mern 2024-style prior.

        Samples ``n_grabens`` rotated rectangular structural strips and
        ``n_domains`` blob-shaped geochemical regions on a
        ``grid_n`` x ``grid_n`` grid, computes binary inside-outside
        masks per the polygon sample, and combines them additively into
        the per-cell GP prior mean field.

        Cell coordinates are laid out on a regular ``grid_n`` x
        ``grid_n`` lattice with spacing ``cell_spacing_m``. The default
        ``gp_lengthscale_m`` (None) is set to ``3 * cell_spacing_m`` to
        match the paper's 3-grid-cell correlation length on the 32x32
        synthetic grid (p.28).

        Parameters
        ----------
        name : str
            Hypothesis label.
        n_grabens, n_domains : int
            Counts feeding into the 2x2 paper hypothesis grid
            {(1,1), (1,2), (2,1), (2,2)}.
        grid_n : int
            Side length of the working grid. Mern 2024 uses 32.
        rng : np.random.Generator
            Random state controlling polygon sampling.
        cell_spacing_m : float, default 1.0
            Physical spacing between cell centers. 1.0 matches the
            paper's normalized units; for BCGT we use 500.0.
        gp_marginal_std : float, default KERNEL_MARGINAL_STD (0.1)
            GP marginal standard deviation; Mern 2024 p.28 normalized.
        gp_lengthscale_m : float or None, default None
            GP correlation length. When None, defaults to 3 *
            cell_spacing_m (matches paper p.28 normalized).
        gp_kernel_nu : float, default 2.5
            Matern smoothness parameter.
        base_mean, graben_boost, domain_boost : float or None
            Per-cell prior-mean field parameters; pass through to
            ``prior_mean_field_from_masks``. Defaults match
            ``BASE_MEAN``, ``GRABEN_MEAN_BOOST``, ``DOMAIN_MEAN_BOOST``
            from the domains module.

        Returns
        -------
        Hypothesis
            Frozen dataclass with the structured prior baked in.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> h = Hypothesis.from_domain_config(
        ...     name="H_1_1", n_grabens=1, n_domains=1, grid_n=32, rng=rng,
        ... )
        >>> h.n_cells
        1024
        >>> h.n_grabens, h.n_domains
        (1, 1)
        """
        from .domains import (
            BASE_MEAN, DOMAIN_MEAN_BOOST, GRABEN_MEAN_BOOST,
            domain_mask_from_polygons, prior_mean_field_from_masks,
            sample_geochem_domain_polygons, sample_graben_polygons,
        )
        graben_polygons = sample_graben_polygons(
            n_grabens=n_grabens, grid_n=grid_n, rng=rng,
        )
        domain_polygons = sample_geochem_domain_polygons(
            n_domains=n_domains, grid_n=grid_n, rng=rng,
        )
        graben_mask = domain_mask_from_polygons(graben_polygons, grid_n)
        domain_mask = domain_mask_from_polygons(domain_polygons, grid_n)
        prior_mean_field = prior_mean_field_from_masks(
            graben_mask, domain_mask,
            base_mean=BASE_MEAN if base_mean is None else base_mean,
            graben_boost=(
                GRABEN_MEAN_BOOST if graben_boost is None else graben_boost
            ),
            domain_boost=(
                DOMAIN_MEAN_BOOST if domain_boost is None else domain_boost
            ),
        )
        row_grid, col_grid = np.meshgrid(
            np.arange(grid_n) * cell_spacing_m,
            np.arange(grid_n) * cell_spacing_m,
            indexing="ij",
        )
        cell_coords_m = np.column_stack([
            row_grid.ravel(), col_grid.ravel(),
        ]).astype(np.float64)
        gp_ell = (
            3.0 * cell_spacing_m if gp_lengthscale_m is None
            else gp_lengthscale_m
        )
        return cls(
            name=name,
            n_grabens=n_grabens,
            n_domains=n_domains,
            cell_coords_m=cell_coords_m,
            prior_mean_field=prior_mean_field,
            gp_kernel_nu=gp_kernel_nu,
            gp_marginal_std=gp_marginal_std,
            gp_lengthscale_m=gp_ell,
        )


@dataclass(frozen=True)
class NullHypothesis:
    """Mern 2024's h_0: maximum-entropy mixture, no spatial correlation.

    The null sits alongside the paper hypotheses {H_1, ..., H_N} in C.2.
    Its likelihood under observed drilling is tracked separately;
    falsification fires when h_0 becomes more likely than every paper
    hypothesis after K observations.

    Per the audit on 2026-06-10, the paper (p.15-16) does NOT specify
    P(h_0) explicitly; it's emergent from the likelihood ratio. Our
    starting prior: uniform 1/(N+1).
    """
    marginal_std: float = KERNEL_MARGINAL_STD

    def sample_realization(
        self,
        rng: np.random.Generator,
        n_cells: int,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Independent N(0, marginal_std^2) draws; no spatial correlation.

        Shape: (n_samples, n_cells). Each cell is independent of every other,
        which is what "maximum-entropy mixture" means here: the null
        hypothesis assigns equal probability to every possible field
        configuration consistent with the marginal variance.
        """
        if n_cells < 0:
            raise ValueError(f"n_cells must be >= 0; got {n_cells}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1; got {n_samples}")
        return rng.normal(
            loc=0.0, scale=self.marginal_std,
            size=(n_samples, n_cells),
        )


@dataclass(frozen=True)
class HypothesisSet:
    """N paper hypotheses + 1 null. C.2 only.

    Manages the categorical posterior over hypothesis indices given
    observed drilling. Conditional on each h, the property field is
    a GP draw.
    """
    hypotheses: tuple[Hypothesis, ...] = ()
    null: NullHypothesis | None = None
    include_null: bool = INCLUDE_NULL_HYPOTHESIS

    @property
    def n_hypotheses(self) -> int:
        """Count of paper hypotheses plus the null if `include_null`."""
        return len(self.hypotheses) + (1 if (self.include_null and self.null is not None) else 0)

    def initial_prior(self) -> np.ndarray:
        """Uniform categorical prior across {H_1, ..., H_N} + h_0 if present.

        Shape: (n_hypotheses,). The null sits at the final index when present.
        """
        n = self.n_hypotheses
        if n == 0:
            raise ValueError(
                "HypothesisSet has no hypotheses; can't form initial_prior"
            )
        return np.full(n, 1.0 / n, dtype=np.float64)

    def update_posterior(
        self,
        prior: np.ndarray,
        observation: float,
        cell_idx: int,
        sensor_noise_sigma: float = 0.0,
    ) -> np.ndarray:
        """Bayesian categorical update given a single Gaussian-continuous obs.

        For each hypothesis h_i, the marginal likelihood at cell_idx is a
        Gaussian:
            p(obs | h_i) = N(obs; prior_mean_h_i[cell_idx],
                             gp_marginal_std_i^2 + sensor_noise_sigma^2)
        For the null hypothesis the prior mean is zero with the same
        marginal variance contribution.

        Posterior_i proportional to prior_i * p(obs | h_i); normalized to a
        probability vector. The shapes:
            prior        (n_hypotheses,)
            return       (n_hypotheses,)
        """
        n = self.n_hypotheses
        if prior.shape != (n,):
            raise ValueError(
                f"prior must be shape ({n},); got {prior.shape}"
            )
        if not np.isclose(prior.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"prior must sum to 1.0; got sum={prior.sum():.6f}"
            )
        log_lik = np.zeros(n, dtype=np.float64)
        for i, h in enumerate(self.hypotheses):
            mean_i = float(h.prior_mean_field[cell_idx])
            var_i = h.gp_marginal_std ** 2 + sensor_noise_sigma ** 2
            log_lik[i] = (
                -0.5 * np.log(2.0 * np.pi * var_i)
                - 0.5 * (observation - mean_i) ** 2 / var_i
            )
        if self.include_null and self.null is not None:
            var_0 = self.null.marginal_std ** 2 + sensor_noise_sigma ** 2
            log_lik[-1] = (
                -0.5 * np.log(2.0 * np.pi * var_0)
                - 0.5 * (observation ** 2) / var_0
            )
        # Log-sum-exp normalization for stability.
        log_post = np.log(np.maximum(prior, 1e-300)) + log_lik
        m = log_post.max()
        post = np.exp(log_post - m)
        post = post / post.sum()
        return post


def porphyry_cu_hypothesis_from_v3_rf(
    p_prior_surface: np.ndarray,
    cell_coords_m: np.ndarray,
    name: str = "porphyry-Cu",
    gp_marginal_std: float = KERNEL_MARGINAL_STD,
    gp_lengthscale_m: float = KERNEL_LENGTHSCALE_M_BCGT,
    n_grabens: int = 1,
    n_domains: int = 1,
) -> Hypothesis:
    """Build a Hypothesis from a per-cell RF posterior probability surface.

    Centers the input field at its global mean so the GP prior mean has
    zero average; kernel parameters default to the locked module values.

    Parameters
    ----------
    p_prior_surface : np.ndarray
        Per-cell RF posterior probability of porphyry-Cu, length N.
    cell_coords_m : np.ndarray
        Per-cell (x, y) coordinates in the working CRS, shape (N, 2).
    name : str
        Human-readable label for the hypothesis.
    """
    if p_prior_surface.ndim != 1:
        raise ValueError(
            f"p_prior_surface must be 1D; got shape {p_prior_surface.shape}"
        )
    n_cells = p_prior_surface.shape[0]
    if cell_coords_m.shape != (n_cells, 2):
        raise ValueError(
            f"cell_coords_m must be ({n_cells}, 2); got {cell_coords_m.shape}"
        )
    centered = p_prior_surface - float(p_prior_surface.mean())
    return Hypothesis(
        name=name,
        n_grabens=n_grabens, n_domains=n_domains,
        cell_coords_m=cell_coords_m.astype(np.float64),
        prior_mean_field=centered.astype(np.float64),
        gp_marginal_std=gp_marginal_std,
        gp_lengthscale_m=gp_lengthscale_m,
    )


def make_mern_2x2_hypothesis_set(
    grid_n: int = 32,
    rng: np.random.Generator | None = None,
    include_null: bool = True,
    cell_spacing_m: float = 1.0,
    gp_marginal_std: float = KERNEL_MARGINAL_STD,
    gp_lengthscale_m: float | None = None,
    seed: int | None = None,
) -> HypothesisSet:
    """Build the Mern 2024 paper's 2x2 hypothesis grid plus an optional null.

    The paper (arXiv 2410.10610, p.2 and p.20) defines the hypothesis
    set as four geological models indexed by (n_grabens, n_domains)
    in {(1, 1), (1, 2), (2, 1), (2, 2)}, plus a null hypothesis
    representing "none of the above; maximum-entropy mixture with no
    spatial structure."

    Each of the four paper hypotheses is built via
    ``Hypothesis.from_domain_config`` with its own seed so the four
    polygon realizations are distinct and reproducible. The null
    hypothesis (if ``include_null=True``) is a ``NullHypothesis``
    instance with the same marginal variance.

    Parameters
    ----------
    grid_n : int, default 32
        Side length of the working grid in cells. Paper uses 32.
    rng : np.random.Generator or None
        Master random state. When None and ``seed`` is also None, uses
        the default RNG. When None and ``seed`` is set, builds an RNG
        from the seed.
    include_null : bool, default True
        Add the NullHypothesis to the set.
    cell_spacing_m : float, default 1.0
        Physical spacing between cell centers.
    gp_marginal_std : float
        GP marginal standard deviation.
    gp_lengthscale_m : float or None
        GP correlation length. When None, defaults to
        ``3 * cell_spacing_m`` (paper p.28).
    seed : int or None
        Convenience for callers that don't want to build an RNG
        themselves. Ignored when ``rng`` is set.

    Returns
    -------
    HypothesisSet
        Four paper hypotheses indexed H_1_1, H_1_2, H_2_1, H_2_2, plus
        the null when ``include_null=True``.

    Examples
    --------
    >>> hset = make_mern_2x2_hypothesis_set(grid_n=32, seed=0)
    >>> hset.n_hypotheses
    5
    >>> [h.name for h in hset.hypotheses]
    ['H_1_1', 'H_1_2', 'H_2_1', 'H_2_2']
    >>> hset.hypotheses[0].n_grabens, hset.hypotheses[0].n_domains
    (1, 1)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    paper_configs = ((1, 1), (1, 2), (2, 1), (2, 2))
    hypotheses: list[Hypothesis] = []
    for n_g, n_d in paper_configs:
        # Spawn a fresh independent generator per hypothesis so the four
        # realizations are decorrelated yet deterministic given the master rng.
        sub_seed = int(rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        hypotheses.append(Hypothesis.from_domain_config(
            name=f"H_{n_g}_{n_d}",
            n_grabens=n_g, n_domains=n_d,
            grid_n=grid_n, rng=sub_rng,
            cell_spacing_m=cell_spacing_m,
            gp_marginal_std=gp_marginal_std,
            gp_lengthscale_m=gp_lengthscale_m,
        ))
    null = NullHypothesis(marginal_std=gp_marginal_std) if include_null else None
    return HypothesisSet(
        hypotheses=tuple(hypotheses),
        null=null,
        include_null=include_null,
    )


def make_bcgt_deposit_type_hypothesis_set(
    features_path: str | None = None,
    n_side: int = 30,
    spacing_m: float = 500.0,
    include_null: bool = True,
    gp_marginal_std: float = KERNEL_MARGINAL_STD,
    gp_lengthscale_m: float = KERNEL_LENGTHSCALE_M_BCGT,
) -> tuple[HypothesisSet, np.ndarray]:
    """Build a 4-hypothesis BCGT prior set from real BCGS deposit-type labels.

    Each hypothesis carries a prior mean field aggregated from the
    BCGT 500 m feature parquet over one of four deposit classes
    (porphyry, skarn, epithermal, VMS). The four surfaces share the
    same (0, 0)-anchored ``n_side x n_side`` coordinate grid, matching
    the D.1 synthetic setup so the SARSOP / particle-filter / greedy
    stack does not need to know the priors came from real data.

    Parameters
    ----------
    features_path : str or None
        Path to the BCGT 500 m feature parquet. When None, falls back
        to ``data/derived/features_bcgt_500m.parquet`` relative to the
        repository root.
    n_side : int, default 30
        Side length of the output grid. Default 30 matches D.1.
    spacing_m : float, default 500
        Cell spacing in meters.
    include_null : bool, default True
        Append a NullHypothesis with the same GP marginal std.
    gp_marginal_std : float
        GP marginal standard deviation, default KERNEL_MARGINAL_STD.
    gp_lengthscale_m : float
        GP correlation length, default KERNEL_LENGTHSCALE_M_BCGT.

    Returns
    -------
    hypothesis_set : HypothesisSet
        Four real-prior hypotheses (porphyry, skarn, epithermal, VMS)
        plus the null when ``include_null=True``.
    coords : np.ndarray
        (n_side ** 2, 2) cell coordinates in meters.
    """
    from ai_minerals.decision.v20.real_priors import (
        bcgs_deposit_type_prior_surfaces,
    )

    if features_path is None:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[4]
        features_path = str(
            repo_root / "data/derived/features_bcgt_500m.parquet"
        )

    surfaces, coords = bcgs_deposit_type_prior_surfaces(
        features_path=features_path,
        n_side=n_side,
        spacing_m=spacing_m,
    )

    hypotheses: list[Hypothesis] = []
    for type_name in ("porphyry", "skarn", "epithermal", "vms"):
        surf = surfaces[type_name]
        centered = surf - float(surf.mean())
        hypotheses.append(Hypothesis(
            name=f"H_{type_name}",
            n_grabens=1, n_domains=1,
            cell_coords_m=coords,
            prior_mean_field=centered.astype(np.float64),
            gp_marginal_std=gp_marginal_std,
            gp_lengthscale_m=gp_lengthscale_m,
        ))

    null = NullHypothesis(marginal_std=gp_marginal_std) if include_null else None
    return HypothesisSet(
        hypotheses=tuple(hypotheses),
        null=null,
        include_null=include_null,
    ), coords
