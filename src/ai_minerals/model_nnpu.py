"""nnPU (Kiryo et al. 2017, NeurIPS) for v3 Phase C.2.

Used for the Quaternary branch where the 573 modern-channel placer
positives in 800k cells are exactly the small-positive-rate regime
nnPU was built for. Tertiary uses Mordelet-Vert PU bagging from
model_pu.py because the 158 hydraulic-pit-polygon positives are
effectively fully labeled (every positive cell is a known pit
centroid).

Reference:
    Kiryo, Niu, du Plessis, Sugiyama. "Positive-Unlabeled Learning with
    Non-Negative Risk Estimator." NeurIPS 2017. arXiv:1703.00593.
    github.com/kiryor/nnPUlearning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_nnpu_quaternary(
    df: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: list[str],
    prior: float = 0.001,
    n_epochs: int = 100,
    batch_size: int = 8192,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """Fit nnPU on the (positives, unlabeled) data; return per-row P(positive).

    prior: class prior P(Y=1). For Quaternary placer the empirical positive
           rate (573 / 800k ~ 0.0007) is a reasonable starting estimate;
           tune via KM2/TIcE in a v3.5 follow-up.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    y = df[label_col].to_numpy(dtype=np.float32)
    X = df[feature_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    # Standardize: nnPU loss is more stable on standardized inputs.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    n, d = Xn.shape

    class Net(nn.Module):
        def __init__(self, d: int, h: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

    model = Net(d, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.from_numpy(Xn).to(device)
    y_t = torch.from_numpy(y).to(device)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def nnpu_loss(g: "torch.Tensor", y: "torch.Tensor", prior: float) -> "torch.Tensor":
        # Following Kiryo eq. 6 / 7: max(0, neg_risk) when neg_risk < 0.
        pos = (y == 1)
        neg = ~pos
        # Surrogate: sigmoid loss in expectation form.
        pos_loss = (
            torch.sigmoid(-g[pos]).mean()
            if pos.sum() > 0
            else torch.tensor(0.0, device=g.device)
        )
        # neg_risk = (1 - prior) * E_U[sig(g)] - prior * E_P[sig(g)]
        if neg.sum() > 0:
            eu = torch.sigmoid(g[neg]).mean()
        else:
            eu = torch.tensor(0.0, device=g.device)
        ep = (
            torch.sigmoid(g[pos]).mean()
            if pos.sum() > 0
            else torch.tensor(0.0, device=g.device)
        )
        neg_risk = (1 - prior) * eu - prior * ep
        neg_loss = torch.clamp(neg_risk, min=0.0)
        return prior * pos_loss + neg_loss

    model.train()
    for _ in range(n_epochs):
        for xb, yb in dl:
            opt.zero_grad()
            g = model(xb)
            loss = nnpu_loss(g, yb, prior)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(X_t)).cpu().numpy().astype(np.float64)
    return proba, list(feature_cols)
