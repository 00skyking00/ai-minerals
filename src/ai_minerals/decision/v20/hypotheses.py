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

NOT IMPLEMENTED YET. Skeletons only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Locked parameters (arXiv 2410.10610 p.28; see pomdp_v20_implementation_plan.md)
KERNEL_NU = 2.5                # Matern v=5/2; NOT v=1.5 (corrected 2026-06-10)
KERNEL_MARGINAL_STD = 0.1
KERNEL_LENGTHSCALE_GRID = 3    # paper grid cells
KERNEL_LENGTHSCALE_M_BCGT = 1500.0  # BCGT subarea: 500 m/cell × 3
SENSOR_NOISE_GAUSSIAN_SIGMA = 0.001   # B.1 synthetic
N_HYPOTHESES_PAPER = 4         # C.2; H_1 through H_4
INCLUDE_NULL_HYPOTHESIS = True # C.2; total = N + 1


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
    gp_kernel_nu : float
        Matern smoothness parameter. Locked at 2.5 per paper p.28.
    gp_marginal_std : float
        sigma in dimensionless units (B.1) or normalized grade units (B.2).
    gp_lengthscale_m : float
        Correlation length in working CRS meters (1500 m for BCGT).
    prior_mean_field : np.ndarray
        Per-cell GP mean. Initialized from the v3 RF posterior surface.
        Shape (n_cells,).
    """
    name: str
    n_grabens: int
    n_domains: int
    gp_kernel_nu: float = KERNEL_NU
    gp_marginal_std: float = KERNEL_MARGINAL_STD
    gp_lengthscale_m: float = KERNEL_LENGTHSCALE_M_BCGT
    prior_mean_field: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        # TODO B.1: validate prior_mean_field shape matches the grid.
        pass

    def sample_realization(
        self,
        rng: np.random.Generator,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Draw N realizations from the GP prior. Shape (n_samples, n_cells).

        For BCGT we threshold the continuous GP draws at a level chosen
        so the marginal positive rate matches the per-cell RF posterior
        mean. That replaces v1.0's iid Bernoulli draws with spatially
        correlated draws where neighbor cells covary per the kernel.

        TODO B.1: implement via sklearn.gaussian_process.GaussianProcessRegressor
        or george.GP, with prior_mean_field as the mean function.
        """
        raise NotImplementedError("B.1 milestone")

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

        TODO C.2: implement; trivial numpy.
        """
        raise NotImplementedError("C.2 milestone")


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

    def initial_prior(self) -> np.ndarray:
        """Uniform across {H_1, ..., H_N} + h_0 if null is present.

        TODO C.2: implement.
        """
        raise NotImplementedError("C.2 milestone")

    def update_posterior(
        self,
        prior: np.ndarray,
        observation: float,
        cell_idx: int,
    ) -> np.ndarray:
        """Bayesian update over hypothesis indices given a drill outcome.

        TODO C.2: implement; per-h likelihood from GP marginal at cell_idx.
        """
        raise NotImplementedError("C.2 milestone")


def porphyry_cu_hypothesis_from_v3_rf(
    p_prior_surface: np.ndarray,
    name: str = "porphyry-Cu",
) -> Hypothesis:
    """Build a single Hypothesis from the v3 RF posterior over BCGT cells.

    Parameters
    ----------
    p_prior_surface : np.ndarray
        Per-cell RF posterior probability of porphyry-Cu, length N.
        Centered + scaled so the GP mean has zero global average.

    TODO B.1: implement; trivial mean-centering. Sets all kernel params
    to module defaults locked above.
    """
    raise NotImplementedError("B.1 milestone")
