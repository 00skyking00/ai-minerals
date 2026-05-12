"""DevNet — Deviation Networks for semi-supervised anomaly detection.

Port of the method from Luo et al. (2026) "DEEP-SEAM: an explainable
semi-supervised deep learning framework for mineral prospectivity
mapping" (Geosci. Model Dev. 19 2593, doi:10.5194/gmd-19-2593-2026),
which itself ports Pang et al. (2019) "Deep Anomaly Detection with
Deviation Networks" (arXiv:1911.08623).

The idea: treat known mineral occurrences as anomalies, treat the rest
as "normal" samples, and train a neural network to push the anomaly
score for known occurrences far above a Gaussian-prior reference
distribution. Unlike PU-bagging (which assumes random pseudo-negatives
are negatives), DevNet only assumes the unlabeled set is dominantly
non-anomalous, which is a weaker and more defensible assumption for
mineral prospectivity work.

Loss (Eq. 7 in DEEP-SEAM, after Pang 2019):

    L(x, y; θ, μ_R, σ_R) = (1 - y) |dev(x)| + y · max(0, a - dev(x))

where dev(x) = (φ(x; θ) - μ_R) / σ_R, μ_R and σ_R are sampled per-batch
from a standard normal prior, and `a` is a confidence margin (default
5; same as the original Pang 2019 default).

Architecture per DEEP-SEAM: input → 24 → 12 → 1 (ReLU activations,
linear output), Nadam optimizer, lr=0.005, batch=128, 500 epochs.

For Mother Lode the input dimensionality is set by the feature frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


@dataclass
class DevNetConfig:
    hidden: tuple[int, ...] = (24, 12)
    learning_rate: float = 0.005
    batch_size: int = 128
    n_epochs: int = 500
    n_ref: int = 5000
    confidence_margin: float = 5.0
    seed: int = 42
    device: str = "cpu"  # MPS / CUDA detected at fit time if available


def _build_mlp(in_features: int, hidden: tuple[int, ...]) -> "nn.Module":
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def deviation_loss(
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    n_ref: int,
    a: float,
    device: str,
) -> "torch.Tensor":
    """Pang 2019 / DEEP-SEAM deviation loss.

    scores: shape (batch,) — raw output of the MLP for a batch of inputs.
    labels: shape (batch,) in {0, 1} — 0 for "normal" / unlabeled, 1 for
            "anomaly" / known occurrence.
    n_ref:  number of standard-normal samples to estimate the reference mean
            and std per-batch.
    a:      confidence margin; anomaly scores are pushed >= a standard
            deviations above the reference.
    """
    ref = torch.randn(n_ref, device=device)
    mu_r = ref.mean()
    sigma_r = ref.std() + 1e-8
    dev = (scores - mu_r) / sigma_r

    # Normal samples: minimize |dev|.
    # Anomaly samples: enforce dev >= a, penalty = max(0, a - dev).
    loss = (1.0 - labels) * torch.abs(dev) + labels * torch.clamp(a - dev, min=0.0)
    return loss.mean()


def fit_devnet(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    label_col: str,
    config: DevNetConfig | None = None,
) -> tuple[np.ndarray, "DevNetConfig", "nn.Module"]:
    """Train DevNet against `df[label_col]` (0/1) on `df[feat_cols]`.

    Returns (per-cell scores, config, trained model). Per-cell scores are
    the raw MLP output (higher = more anomalous = more prospective).

    Following DEEP-SEAM's training protocol: each batch contains a mix of
    positives and unlabeled samples. We construct batches by stratified
    sampling so each batch has both classes.
    """
    if not _HAVE_TORCH:
        raise ImportError(
            "DevNet requires torch. Add it to project deps: `uv add torch`."
        )

    cfg = config or DevNetConfig()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    if device == "cpu":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    X = df[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.float32)

    # Z-score normalize features (DevNet expects scaled inputs; raw -9999
    # sentinels would dominate gradient).
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sd

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        raise ValueError(f"no positives in df[{label_col!r}]")

    # Stratified batches: half from positives, half from unlabeled. This
    # matches Pang 2019 + DEEP-SEAM training-time class balancing.
    half = cfg.batch_size // 2

    model = _build_mlp(X.shape[1], cfg.hidden).to(device)
    opt = optim.NAdam(model.parameters(), lr=cfg.learning_rate)
    Xn_t = torch.from_numpy(Xn).to(device)

    print(f"DevNet: train n_pos={n_pos}, n_unl={n_neg}, n_features={X.shape[1]}, "
          f"epochs={cfg.n_epochs}, batch={cfg.batch_size}, device={device}")

    model.train()
    for epoch in range(cfg.n_epochs):
        # 5 batches per epoch is what DEEP-SEAM uses (cfg sets epoch length
        # implicitly; we follow their cadence).
        for _ in range(5):
            pi = rng.choice(pos_idx, size=half, replace=True)
            ni = rng.choice(neg_idx, size=cfg.batch_size - half, replace=False)
            idx = np.concatenate([pi, ni])
            yb = torch.from_numpy(np.concatenate([np.ones(half), np.zeros(cfg.batch_size - half)]).astype(np.float32)).to(device)
            xb = Xn_t[idx]
            scores = model(xb).squeeze(-1)
            loss = deviation_loss(scores, yb, n_ref=cfg.n_ref, a=cfg.confidence_margin, device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1}/{cfg.n_epochs}  loss={float(loss):.4f}")

    # Score every cell.
    model.eval()
    with torch.no_grad():
        all_scores = model(Xn_t).squeeze(-1).cpu().numpy()
    return all_scores, cfg, model
