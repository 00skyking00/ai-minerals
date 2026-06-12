"""Elliptical Slice Sampling (Murray, Adams, MacKay 2010) for bcgt-v2.0 C.2.

ESS samples from p(f | obs) where the prior p(f) is a multivariate Gaussian
and the likelihood L(f) can be evaluated pointwise. The Mern 2024 paper uses
ESS to draw posterior grade fields per hypothesis when the particle filter
degenerates (sharp likelihoods, sparse positives). For the BCGT single-
hypothesis B.1 case ESS is overkill, but C.2 multi-hypothesis falsification
benefits from a GP-posterior sampler that doesn't require the PF's importance-
resampling assumptions to stay well-conditioned.

Algorithm (Murray, Adams, MacKay 2010, ICML appendix):
  Inputs: current state f, prior covariance K, log-likelihood L(.)
  1. nu ~ N(0, K)                          # auxiliary prior draw
  2. log_y = log L(f) + log U(0, 1)        # likelihood threshold
  3. theta ~ U(0, 2 pi)                    # angle
     theta_min, theta_max = theta - 2 pi, theta
  4. loop:
       f_prop = f cos(theta) + nu sin(theta)
       if log L(f_prop) > log_y:  accept; return f_prop
       else: shrink bracket
              if theta < 0: theta_min = theta
              else:         theta_max = theta
              theta ~ U(theta_min, theta_max)

The ellipse parameterization guarantees the proposal preserves the GP-prior
marginal even when the slice shrinks, which is the central trick of ESS.
We hand-roll the algorithm to avoid a dependency.

Reference: Murray, I., Adams, R. P., MacKay, D. J. C. (2010). Elliptical
Slice Sampling. AISTATS Proceedings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hypotheses import Hypothesis

DEFAULT_N_ITERATIONS = 1000
DEFAULT_BURNIN = 200
DEFAULT_THIN = 10
MAX_SHRINK_ITERS = 200      # numerical safety; tightening should converge fast


def log_gaussian_observation_likelihood(
    f: np.ndarray,
    observations: list[tuple[int, float]],
    sensor_noise_sigma: float,
) -> float:
    """Sum of per-observation log-Gaussian-likelihoods given field f.

    log p(obs | f) = -0.5 * sum_i (obs_i - f[cell_i])^2 / sigma^2
                    (dropping the data-independent log(2 pi sigma^2) prefactor;
                     cancels in ESS's log-ratio threshold check.)
    """
    if not observations:
        return 0.0
    total = 0.0
    inv_var = 1.0 / (sensor_noise_sigma ** 2)
    for cell_idx, obs in observations:
        residual = obs - float(f[cell_idx])
        total += -0.5 * residual * residual * inv_var
    return total


def elliptical_slice_step(
    f: np.ndarray,
    log_lik_fn,
    K_chol: np.ndarray,
    rng: np.random.Generator,
    max_shrink: int = MAX_SHRINK_ITERS,
) -> tuple[np.ndarray, int]:
    """One ESS step. Returns the next sample plus the number of shrink iters.

    Args:
        f: current sample, shape (n_cells,)
        log_lik_fn: callable f -> log L(f), accepts shape-(n_cells,) array
        K_chol: lower-triangular Cholesky factor of the GP prior covariance,
                shape (n_cells, n_cells)
        rng: numpy.random.Generator
        max_shrink: safety cap on inner-loop shrinks; should not trigger in
                    well-conditioned problems.

    Raises:
        RuntimeError if the inner loop fails to find an accepted point within
        max_shrink shrinks.
    """
    n_cells = f.shape[0]
    nu = K_chol @ rng.standard_normal(n_cells)
    log_y = log_lik_fn(f) + np.log(rng.uniform())
    theta = rng.uniform(0.0, 2.0 * np.pi)
    theta_min, theta_max = theta - 2.0 * np.pi, theta

    for shrink in range(max_shrink):
        f_prop = f * np.cos(theta) + nu * np.sin(theta)
        if log_lik_fn(f_prop) > log_y:
            return f_prop, shrink
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)
    raise RuntimeError(
        f"ESS shrink loop did not converge after {max_shrink} iterations; "
        f"check likelihood scale or numerical conditioning of K"
    )


@dataclass
class EllipticalSliceSampler:
    """ESS chain driver for a single Hypothesis.

    Samples (n_iterations - burnin) / thin posterior fields from
    p(f | observations) where p(f) is the GP prior under `hypothesis` and
    L(f) is the Gaussian-observation likelihood (defaults to that, can be
    swapped via log_lik_fn).
    """
    hypothesis: Hypothesis
    n_iterations: int = DEFAULT_N_ITERATIONS
    burnin: int = DEFAULT_BURNIN
    thin: int = DEFAULT_THIN

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1; got {self.n_iterations}")
        if self.burnin < 0:
            raise ValueError(f"burnin must be >= 0; got {self.burnin}")
        if self.thin < 1:
            raise ValueError(f"thin must be >= 1; got {self.thin}")
        if self.burnin >= self.n_iterations:
            raise ValueError(
                f"burnin ({self.burnin}) must be < n_iterations "
                f"({self.n_iterations})"
            )

    def sample_chain(
        self,
        observations: list[tuple[int, float]],
        sensor_noise_sigma: float,
        rng: np.random.Generator,
        f0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the ESS chain; return thinned post-burnin samples.

        Args:
            observations: list of (cell_idx, observation) pairs.
            sensor_noise_sigma: Gaussian sensor noise sigma.
            rng: master RNG; the chain consumes randomness for nu, log_y
                threshold draws, and bracket shrinks.
            f0: initial state. Defaults to a single draw from the GP prior
                (a fresh particle).

        Returns:
            (n_samples, n_cells) array of post-burnin, thinned fields, with
            n_samples = floor((n_iterations - burnin) / thin).
        """
        K_chol = self.hypothesis._cholesky()
        n_cells = self.hypothesis.prior_mean_field.shape[0]
        # Initialize from one prior draw if not provided. Centered at the
        # prior mean field so the ESS chain explores the posterior around it
        # rather than around zero.
        if f0 is None:
            f = self.hypothesis.prior_mean_field + (
                K_chol @ rng.standard_normal(n_cells)
            )
        else:
            f = f0.copy()
            if f.shape != (n_cells,):
                raise ValueError(
                    f"f0 must be ({n_cells},); got {f.shape}"
                )

        # ESS samples deviations from prior mean: redefine f = f - prior_mean.
        # The likelihood applies to (prior_mean + f), so wrap accordingly.
        prior_mean = self.hypothesis.prior_mean_field
        f = f - prior_mean

        def log_lik(deviation: np.ndarray) -> float:
            field = prior_mean + deviation
            return log_gaussian_observation_likelihood(
                field, observations, sensor_noise_sigma,
            )

        samples = []
        for i in range(self.n_iterations):
            f, _ = elliptical_slice_step(f, log_lik, K_chol, rng)
            if i >= self.burnin and (i - self.burnin) % self.thin == 0:
                samples.append(prior_mean + f)
        return np.array(samples)
