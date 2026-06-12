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
