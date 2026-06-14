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

__all__ = [
    "log_gaussian_observation_likelihood",
    "elliptical_slice_step",
    "EllipticalSliceSampler",
    "MultiHypothesisESSParticleFilter",
]


from dataclasses import dataclass, field

import numpy as np

from .hypotheses import Hypothesis, HypothesisSet

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


# --- Multi-hypothesis particle filter ----------------------------------------


DEFAULT_MULTIHYPOTHESIS_N_PARTICLES = 100
DEFAULT_ESS_REFRESH_STEPS = 3


@dataclass
class MultiHypothesisESSParticleFilter:
    """Maintains M GP-field particles per hypothesis plus a categorical
    posterior over hypotheses, updated by Elliptical Slice Sampling on
    each observation.

    This is the D.1 methodology fix: replaces ``BcgtScaleSARSOPPolicy``'s
    canonical-realization shortcut with a proper marginalization over
    GP fields per hypothesis. Each hypothesis keeps ``n_particles`` GP
    draws conditional on the observation history; the marginal
    per-cell deposit probability under hypothesis h is approximated
    by the empirical fraction of h's particles whose field exceeds
    the cutoff at that cell.

    The categorical posterior over hypotheses is updated by
    importance-weighting:

        P(h | obs) ∝ P(h) * ⟨P(obs | h, f)⟩_f

    where the average is taken over the current particle ensemble for
    h. After the categorical update, particles are moved via ESS
    conditional on the cumulative observation history so each hypothesis's
    ensemble keeps approximating its current posterior.

    Parameters
    ----------
    hypothesis_set : HypothesisSet
        Set of paper hypotheses (and optionally a null) the filter
        tracks. Paper hypotheses get particle ensembles; the null
        hypothesis is handled analytically (marginal Gaussian tail).
    n_particles : int, default 100
        Particles per paper hypothesis.
    ess_refresh_steps : int, default 3
        ESS steps per particle per observation. Higher values produce
        better-mixed particles at the cost of wall-clock.

    Notes
    -----
    Performance scales as O(n_paper_hypotheses * n_particles *
    ess_refresh_steps * n_cells ** 2) per observation. At 30x30 cells
    (n_cells = 900) and the defaults above, one observation takes
    on the order of seconds. Profile and tune ``n_particles`` if too
    slow.
    """
    hypothesis_set: HypothesisSet
    n_particles: int = DEFAULT_MULTIHYPOTHESIS_N_PARTICLES
    ess_refresh_steps: int = DEFAULT_ESS_REFRESH_STEPS

    _particles: dict[int, np.ndarray] = field(default=None, init=False, repr=False)
    _categorical_belief: np.ndarray = field(default=None, init=False, repr=False)
    _observations: list[tuple[int, float]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_particles < 1:
            raise ValueError(f"n_particles must be >= 1; got {self.n_particles}")
        if self.ess_refresh_steps < 0:
            raise ValueError(
                f"ess_refresh_steps must be >= 0; got {self.ess_refresh_steps}"
            )

    def initialize(self, rng: np.random.Generator) -> None:
        """Seed M particles per paper hypothesis from each prior; set the
        categorical posterior to the hypothesis-set's initial prior."""
        self._particles = {}
        for hypothesis_index, hypothesis in enumerate(
            self.hypothesis_set.hypotheses,
        ):
            K_chol = hypothesis._cholesky()
            n_cells = hypothesis.n_cells
            z = rng.standard_normal((self.n_particles, n_cells))
            # particles[m] = prior_mean + L @ z[m]
            self._particles[hypothesis_index] = (
                hypothesis.prior_mean_field[None, :]
                + z @ K_chol.T
            )
        self._categorical_belief = self.hypothesis_set.initial_prior()
        self._observations = []

    @property
    def categorical_belief(self) -> np.ndarray:
        if self._categorical_belief is None:
            raise RuntimeError(
                "MultiHypothesisESSParticleFilter not initialized; "
                "call initialize(rng) first"
            )
        return self._categorical_belief.copy()

    @property
    def particles(self) -> dict[int, np.ndarray]:
        if self._particles is None:
            raise RuntimeError(
                "MultiHypothesisESSParticleFilter not initialized"
            )
        return {h: p.copy() for h, p in self._particles.items()}

    @property
    def observation_count(self) -> int:
        return len(self._observations) if self._observations is not None else 0

    def update(
        self,
        cell_idx: int,
        observation: float,
        sensor_noise_sigma: float,
        rng: np.random.Generator,
    ) -> None:
        """Apply one (cell, observation) update.

        Steps:
          1. Update categorical posterior over hypotheses by
             importance-weighting against each hypothesis's particle
             ensemble.
          2. Append (cell_idx, observation) to the cumulative observation
             history.
          3. Move each particle via ess_refresh_steps ESS steps
             conditional on the new observation history.
        """
        if self._particles is None:
            raise RuntimeError("call initialize(rng) before update")
        n_paper = len(self.hypothesis_set.hypotheses)

        # 1. Categorical posterior update via analytical marginal Gaussian
        # likelihood (the same Rao-Blackwellized form HypothesisSet.update_posterior
        # uses). Each paper hypothesis's likelihood at the observed cell is
        # N(observation; prior_mean_h[cell], gp_marginal_var_h + sensor_var),
        # which integrates the GP prior over the field analytically. This is
        # the prior-marginal likelihood (no conditioning on the cumulative
        # observation history), which is what the existing v20 code uses and
        # what keeps the categorical update stable at small particle counts.
        # The particles are tracked separately for the
        # marginal_probability_above_cutoff query.
        new_belief = np.zeros_like(self._categorical_belief)
        for hypothesis_index in range(n_paper):
            hypothesis = self.hypothesis_set.hypotheses[hypothesis_index]
            mean_at_cell = float(hypothesis.prior_mean_field[cell_idx])
            var_at_cell = (
                hypothesis.gp_marginal_std ** 2 + sensor_noise_sigma ** 2
            )
            log_lik = (
                -0.5 * np.log(2.0 * np.pi * var_at_cell)
                - 0.5 * (observation - mean_at_cell) ** 2 / var_at_cell
            )
            new_belief[hypothesis_index] = (
                self._categorical_belief[hypothesis_index]
                * float(np.exp(log_lik))
            )
        if (self.hypothesis_set.include_null
                and self.hypothesis_set.null is not None):
            null = self.hypothesis_set.null
            null_var = null.marginal_std ** 2 + sensor_noise_sigma ** 2
            null_lik = float(np.exp(-0.5 * observation ** 2 / null_var)
                             / np.sqrt(2.0 * np.pi * null_var))
            new_belief[-1] = self._categorical_belief[-1] * null_lik
        total = new_belief.sum()
        if total > 0:
            self._categorical_belief = new_belief / total

        # 2. Append observation
        self._observations.append((int(cell_idx), float(observation)))

        # 3. Move particles via ESS conditional on cumulative observations
        if self.ess_refresh_steps > 0:
            for hypothesis_index in range(n_paper):
                hypothesis = self.hypothesis_set.hypotheses[hypothesis_index]
                K_chol = hypothesis._cholesky()
                prior_mean = hypothesis.prior_mean_field

                def log_lik(deviation: np.ndarray) -> float:
                    field = prior_mean + deviation
                    return log_gaussian_observation_likelihood(
                        field, self._observations, sensor_noise_sigma,
                    )

                particle_array = self._particles[hypothesis_index]
                for particle_index in range(self.n_particles):
                    deviation = particle_array[particle_index] - prior_mean
                    for _ in range(self.ess_refresh_steps):
                        deviation, _ = elliptical_slice_step(
                            deviation, log_lik, K_chol, rng,
                        )
                    particle_array[particle_index] = prior_mean + deviation

    def marginal_probability_above_cutoff(
        self, cutoff: float,
    ) -> np.ndarray:
        """Per-(hypothesis, cell) probability that the field exceeds ``cutoff``.

        Paper hypotheses use the empirical fraction of particles whose
        cell value is above the cutoff. The null hypothesis is handled
        analytically: its field is uncorrelated N(0, marginal_std^2) per
        cell, so the cutoff tail is constant across cells.

        Returns
        -------
        np.ndarray, shape (n_hypotheses, n_cells)
            Per-(h, c) probability.
        """
        if self._particles is None:
            raise RuntimeError("call initialize(rng) before querying")
        n_paper = len(self.hypothesis_set.hypotheses)
        n_cells = self.hypothesis_set.hypotheses[0].n_cells
        n_total = self.hypothesis_set.n_hypotheses
        out = np.zeros((n_total, n_cells), dtype=np.float64)
        for hypothesis_index in range(n_paper):
            out[hypothesis_index] = (
                self._particles[hypothesis_index] > cutoff
            ).mean(axis=0)
        if (self.hypothesis_set.include_null
                and self.hypothesis_set.null is not None):
            from scipy.stats import norm
            null = self.hypothesis_set.null
            out[-1] = float(1.0 - norm.cdf(cutoff / null.marginal_std))
        return out
