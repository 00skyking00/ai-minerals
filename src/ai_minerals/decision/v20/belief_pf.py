"""Importance-weighted particle filter for B.1; ESS upgrade path for C.2.

Maps to `beliefs.jl` (166 LOC) + `hypotheses.jl` 350-450 (ESS MCMC) in
the Julia reference.

B.1 (locked 2026-06-10):
    Method:        importance-weighted particle filter
    Particle count: empirically tune at first use (500-2000)
    Resampling:    systematic resampling triggered by ESS_eff < 0.5 * N

C.2 (planned):
    Method:        Elliptical Slice Sampling (Murray et al. 2010)
                   over the GP posterior, conditioned on the categorical
                   hypothesis index from HypothesisSet.update_posterior.
    Implementation: hand-roll in numpy (~200 LOC; well-documented
                    algorithm; no clean Python library wraps it for our
                    use case). Murray's MATLAB code is the reference.

WHY importance-weighted PF for B.1 and not ESS from the start:
    - PF is simpler (no MCMC tuning) and sufficient for single-hypothesis
      correlated draws.
    - The known failure mode (particle degeneracy after ~10-15
      observations) is well-understood and we monitor effective sample
      size adaptively.
    - For B.2 retrospective BCGS with K ~10-30 holes per episode, PF
      should hold up. For C.2 with K ~50+ holes and multi-hypothesis
      indices, we may hit the degeneracy wall and switch to ESS.

B.1 IMPLEMENTATION STATUS (2026-06-11):
    ParticleFilter.initialize / update / ESS / resample / posterior_mean /
    posterior_variance: DONE (GH issue #3).
    elliptical_slice_sample (C.2): still NotImplemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hypotheses import Hypothesis

# Defaults — empirically tunable at B.1 first run
DEFAULT_N_PARTICLES = 1000
ESS_RESAMPLING_THRESHOLD = 0.5  # resample if ESS_eff / N_particles < 0.5


@dataclass
class ParticleFilter:
    """Particle filter over per-cell continuous grade for a single hypothesis.

    Each particle is a full per-cell realization of the property field
    (length n_cells), drawn from the hypothesis's GP prior. The filter
    maintains weighted particles, updates weights given new observations
    via the Gaussian likelihood, and resamples adaptively.

    Attributes
    ----------
    hypothesis : Hypothesis
        The single B.1 hypothesis the particles realize.
    n_particles : int
        Particle count. Larger = lower variance, more compute.
    rng : np.random.Generator
        Reproducible seed.

    State
    -----
    particles : np.ndarray
        Shape (n_particles, n_cells). Each row is a property realization.
    log_weights : np.ndarray
        Shape (n_particles,). Log-weights for stable normalization.
    """
    hypothesis: Hypothesis
    n_particles: int = DEFAULT_N_PARTICLES
    rng: np.random.Generator | None = None

    particles: np.ndarray | None = None  # (n_particles, n_cells)
    log_weights: np.ndarray | None = None  # (n_particles,)

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError(f"n_particles must be >= 2; got {self.n_particles}")
        if self.rng is None:
            self.rng = np.random.default_rng(42)

    # --- state queries -----------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self.particles is not None and self.log_weights is not None

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError(
                "ParticleFilter not initialized. Call initialize() first."
            )

    def _normalized_weights(self) -> np.ndarray:
        """Linear-space normalized weights (sum to 1, length n_particles)."""
        self._require_initialized()
        # Subtract max for numerical stability (log-sum-exp trick).
        lw = self.log_weights
        m = lw.max()
        w = np.exp(lw - m)
        return w / w.sum()

    # --- lifecycle ---------------------------------------------------

    def initialize(self) -> None:
        """Draw n_particles realizations from the GP prior; uniform weights."""
        particles = self.hypothesis.sample_realization(
            self.rng, n_samples=self.n_particles,
        )
        # Uniform log-weights: log(1/N) = -log(N). Stored unnormalized;
        # normalisation happens in _normalized_weights / mean / var / ESS.
        log_weights = np.full(
            self.n_particles, -np.log(self.n_particles), dtype=np.float64,
        )
        self.particles = particles
        self.log_weights = log_weights

    def update(
        self,
        cell_idx: int,
        observation: float,
        sensor_noise_sigma: float,
    ) -> None:
        """Update particle weights given a noisy Gaussian drill observation.

        Per-particle log-likelihood under N(particle[cell_idx], sigma^2):
            log p(obs | particle)
              = -0.5 * log(2 pi sigma^2) - (obs - x_i)^2 / (2 sigma^2)
        We drop the constant prefactor (cancels on normalization) and add
        only the data-dependent term to log_weights.

        After updating, adaptively resamples if effective sample size drops
        below `ESS_RESAMPLING_THRESHOLD * n_particles`.
        """
        self._require_initialized()
        if sensor_noise_sigma <= 0:
            raise ValueError(
                f"sensor_noise_sigma must be > 0; got {sensor_noise_sigma}"
            )
        if not (0 <= cell_idx < self.particles.shape[1]):
            raise IndexError(
                f"cell_idx {cell_idx} out of range for "
                f"{self.particles.shape[1]} cells"
            )

        residuals = observation - self.particles[:, cell_idx]
        log_lik = -0.5 * (residuals * residuals) / (sensor_noise_sigma ** 2)
        self.log_weights = self.log_weights + log_lik

        # Adaptive resampling.
        ess = self.effective_sample_size()
        if ess < ESS_RESAMPLING_THRESHOLD * self.n_particles:
            self.resample()

    def effective_sample_size(self) -> float:
        """Kong et al. 1994 ESS: 1 / sum(w_i^2). Range [1, n_particles]."""
        w = self._normalized_weights()
        return float(1.0 / (w * w).sum())

    def resample(self) -> None:
        """Systematic resampling (Doucet 1998). Resets weights to uniform."""
        self._require_initialized()
        w = self._normalized_weights()
        n = self.n_particles
        # Systematic resampling: one uniform draw u_0 in [0, 1/n), then
        # offsets u_0 + i/n for i in 0..n-1. Pick particles whose CDF
        # crosses each offset. O(n) and lower-variance than multinomial.
        cumsum = np.cumsum(w)
        cumsum[-1] = 1.0  # guard against tiny FP drift
        u0 = self.rng.uniform(0.0, 1.0 / n)
        offsets = u0 + np.arange(n) / n
        # np.searchsorted maps each offset to the right particle index.
        indices = np.searchsorted(cumsum, offsets, side="right")
        # Edge case: if any offset exceeds cumsum[-1] (= 1.0) due to FP,
        # clamp to n-1.
        indices = np.clip(indices, 0, n - 1)
        self.particles = self.particles[indices].copy()
        self.log_weights = np.full(n, -np.log(n), dtype=np.float64)

    # --- posterior queries -------------------------------------------

    def posterior_mean(self) -> np.ndarray:
        """Per-cell weighted mean of particles. Shape (n_cells,)."""
        w = self._normalized_weights()
        return w @ self.particles  # (n_particles,) @ (n_particles, n_cells)

    def posterior_variance(self) -> np.ndarray:
        """Per-cell weighted variance of particles. Shape (n_cells,)."""
        w = self._normalized_weights()
        mean = w @ self.particles
        diffs = self.particles - mean  # (n_particles, n_cells)
        return w @ (diffs * diffs)


# --- C.2 ESS scaffolding (separate from B.1 PF) -------------------------------


def elliptical_slice_sample(
    current_state: np.ndarray,
    log_likelihood: callable,
    rng: np.random.Generator,
    n_iterations: int = 1,
) -> np.ndarray:
    """Murray, Adams, Mackay 2010 ESS step.

    Performs one or more slice-sampling steps over a Gaussian-prior
    posterior. The trick: walk on an ellipse defined by current_state
    and a new prior draw, shrink the slice until the likelihood
    threshold is met.

    Reference algorithm: ~50 LOC of pure numpy. We hand-roll because
    pymc's ESS implementation is wrapped behind Theano/Aesara which
    we don't want as a dependency.

    TODO C.2: implement; cross-check against Murray's MATLAB reference.
    """
    raise NotImplementedError("C.2 milestone")
