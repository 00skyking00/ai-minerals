"""Masked Autoregressive Flow (MAF) anomaly detection on geochemical assays.

Direct port of Scheidt, Mathieu, Yin, Wang, Caers (2024), "Masked
Autoregressive Flow for geochemical anomaly detection in Lithium-Cesium-
Tantalum pegmatites in the Superior Craton, Canada," DOI 10.1007/s11053-024-10409-2.

The Scheidt paper trains a MAF normalizing flow on high-dimensional
geochemical assay vectors. At inference, the negative log-likelihood of a
sample under the fitted flow scores its anomaly strength — low-likelihood
samples are flagged as candidates for further investigation.

Applied here to Au pathfinder geochem (As, Sb, Bi, W, Cu, with Au:Sb
ratio derived) for placer prospectivity. The Au pathfinder elements are
the canonical fingerprint of orogenic-gold hydrothermal systems
(Goldfarb 2013); their per-sample co-occurrence pattern is what the MAF
learns and the anomaly score then flags samples that deviate.

Implementation uses ``normflows`` (Stimper et al., JOSS 8:5361, 2023) which
provides MAF + RealNVP coupling-layer normalizing flows in PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import normflows as nf


@dataclass
class MAFAnomalyModel:
    """A fitted MAF + standardization wrapper.

    Attributes
    ----------
    flow : nf.NormalizingFlow
        The fitted PyTorch flow.
    mean : np.ndarray
        Per-feature mean from the training data (for inverse-standardize at
        score time).
    std : np.ndarray
        Per-feature std from the training data.
    feature_names : list[str]
        Order of features, for inference-time row construction.
    """

    flow: nf.NormalizingFlow
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample log-density under the fitted flow.

        Lower values = more anomalous. Returned in nats (natural log).
        """
        X_std = (X - self.mean) / np.where(self.std > 0, self.std, 1.0)
        self.flow.eval()
        with torch.no_grad():
            logp = self.flow.log_prob(torch.from_numpy(X_std.astype(np.float32)))
        return logp.numpy()

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample anomaly score = -log_density.

        Higher = more anomalous. Convenience wrapper around `score_samples`.
        """
        return -self.score_samples(X)


def fit_maf_anomaly(
    X: np.ndarray,
    *,
    feature_names: list[str],
    n_flows: int = 8,
    hidden_units: int = 64,
    n_iter: int = 2000,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    random_state: int = 42,
    verbose: bool = True,
) -> MAFAnomalyModel:
    """Fit a Masked Autoregressive Flow on a feature matrix.

    Parameters
    ----------
    X : (n_samples, n_features) float array
        Training samples. NaN rows are dropped before fitting.
    feature_names : list[str]
        Column names, used for the inference-time API.
    n_flows : int
        Number of MAF transformations stacked in the flow. Each
        transformation is an autoregressive coupling layer.
    hidden_units : int
        Hidden size in each MAF transformation's MADE network.
    n_iter : int
        Optimization steps (mini-batch SGD with Adam).
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    random_state : int
        Torch + NumPy seed.

    Returns
    -------
    MAFAnomalyModel
        Wrapped fitted flow plus standardization parameters.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    # Drop NaN rows
    keep = np.isfinite(X).all(axis=1)
    Xc = X[keep]
    if verbose:
        print(f"[maf] dropped {(~keep).sum():,} NaN rows; training on {len(Xc):,} samples × {Xc.shape[1]} features")
    if len(Xc) < 100:
        raise ValueError(
            f"Only {len(Xc)} non-NaN training samples; MAF needs many more "
            f"(Scheidt 2024 uses ~10k+)"
        )

    # Standardize
    mean = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    Xs = (Xc - mean) / np.where(std > 0, std, 1.0)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    d = Xs.shape[1]
    # Stack n_flows MAF transforms. Each transform has a MADE network with
    # `hidden_units` hidden size. The base distribution is a diagonal Gaussian.
    flows = []
    for _ in range(n_flows):
        flows.append(
            nf.flows.MaskedAffineAutoregressive(
                features=d, hidden_features=hidden_units, num_blocks=2
            )
        )
        flows.append(nf.flows.LULinearPermute(d))
    base = nf.distributions.base.DiagGaussian(d)
    flow = nf.NormalizingFlow(base, flows)

    # Train
    flow.train()
    opt = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    X_t = torch.from_numpy(Xs.astype(np.float32))
    n = X_t.shape[0]
    losses = []
    for it in range(n_iter):
        idx = torch.randint(0, n, (batch_size,))
        x_batch = X_t[idx]
        opt.zero_grad()
        loss = flow.forward_kld(x_batch)
        if not torch.isfinite(loss):
            if verbose:
                print(f"[maf] iter {it}: non-finite loss; stopping early")
            break
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if verbose and (it + 1) % max(1, n_iter // 10) == 0:
            print(f"[maf] iter {it+1:5d}/{n_iter}: loss = {float(loss):.4f}")

    return MAFAnomalyModel(
        flow=flow,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_names=list(feature_names),
    )
