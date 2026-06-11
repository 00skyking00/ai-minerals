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

NOT IMPLEMENTED YET. Skeletons only.
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
        if self.rng is None:
            self.rng = np.random.default_rng(42)

    def initialize(self) -> None:
        """Draw n_particles realizations from the GP prior; uniform weights.

        TODO B.1: call hypothesis.sample_realization(rng, n_samples=n).
        """
        raise NotImplementedError("B.1 milestone")

    def update(
        self,
        cell_idx: int,
        observation: float,
        sensor_noise_sigma: float,
    ) -> None:
        """Update particle weights given a noisy drill observation.

        Per-particle log-likelihood:
            log p(obs | particle) = log N(obs; particle[cell_idx], sigma)

        Updates log_weights in place; resamples adaptively if ESS_eff
        drops below threshold.

        TODO B.1: implement; trivial numpy + log-sum-exp for normalization.
        """
        raise NotImplementedError("B.1 milestone")

    def effective_sample_size(self) -> float:
        """Kong et al. 1994 ESS: 1 / sum(w_i^2). Range [1, N_particles]."""
        # TODO B.1: implement
        raise NotImplementedError("B.1 milestone")

    def resample(self) -> None:
        """Systematic resampling; resets weights to uniform.

        TODO B.1: implement; standard systematic resampling per Doucet 1998.
        """
        raise NotImplementedError("B.1 milestone")

    def posterior_mean(self) -> np.ndarray:
        """Per-cell weighted mean of particles. Shape (n_cells,)."""
        raise NotImplementedError("B.1 milestone")

    def posterior_variance(self) -> np.ndarray:
        """Per-cell weighted variance of particles. Shape (n_cells,)."""
        raise NotImplementedError("B.1 milestone")


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
