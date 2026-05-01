"""POMDP problem definition for BCGT drill planning.

Per-cell deposit indicator is Bernoulli with prior `p_prior` from a
Random-Forest classifier trained on the BCGT 500m feature frame. The
agent picks a cell to drill (or STOP); drilling reveals the cell's true
label noiselessly (matching Mern et al. 2023's v1.0 simplification).
Reward per drill is `−drill_cost`; reward per hit is `+discovery_value`;
STOP yields 0.

The state space is implicitly 2^N (N = number of candidate cells), so we
never enumerate states. POMCP only needs a generative model:

    - `sample_state_from_belief()` — draw a joint Bernoulli realization
    - `step(state, action) → (next_state, observation, reward)` — one
      transition

Both are wrapped via `pomdp_py.{TransitionModel, ObservationModel,
RewardModel, GenerativeDistribution}` in `policies.pomcp_plan`.

This module focuses on the data + simulator. The pomdp_py glue lives
in `policies.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .policies import Policy


# ---------------------------------------------------------------------------
# Problem container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrillingProblem:
    """A working subarea of the BCGT AOI distilled into a POMDP problem.

    Attributes
    ----------
    x_ft, y_ft :
        Per-cell coordinates in the working CRS, length N.
    p_prior :
        Per-cell Bernoulli prior probability that the cell hosts a
        deposit, length N. From the RF posterior.
    true_label :
        Per-cell ground-truth deposit indicator, length N. For the
        portfolio demo this is a synthetic realization sampled from
        ``p_prior`` (so the planner is fairly served the prior it has).
    drill_cost :
        Reward per drill (positive number — applied as ``−drill_cost``).
    discovery_value :
        Reward per discovered positive cell (applied additively when
        ``true_label[i] == 1``).
    """

    x_ft: np.ndarray
    y_ft: np.ndarray
    p_prior: np.ndarray
    true_label: np.ndarray
    drill_cost: float = 1.0
    discovery_value: float = 50.0

    @property
    def n_cells(self) -> int:
        return int(len(self.p_prior))

    def step(
        self,
        cell_idx: int,
        drilled: frozenset[int],
    ) -> tuple[int, float, frozenset[int]]:
        """Apply a drill action.

        Returns
        -------
        observation, reward, next_drilled
        """
        if cell_idx in drilled:
            # Drilling an already-drilled cell wastes a turn.
            return 0, -self.drill_cost, drilled
        label = int(self.true_label[cell_idx])
        reward = -self.drill_cost + (self.discovery_value if label == 1 else 0.0)
        return label, reward, drilled | {cell_idx}


# ---------------------------------------------------------------------------
# Subarea selection + RF prior
# ---------------------------------------------------------------------------


def load_subarea_prior(
    features_parquet: str | Path = "data/derived/features_bcgt_500m.parquet",
    label_col: str = "any_mineral_occurrence",
    *,
    center_xy: tuple[float, float] | None = None,
    n_cells_side: int = 30,
    cell_size_m: int = 500,
    rf_max_depth: int = 8,
    rf_n_estimators: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Restrict the BCGT feature frame to a working subarea and build a
    per-cell prior probability with a quick RF fit.

    Parameters
    ----------
    features_parquet :
        Path to the BCGT 500m feature frame.
    label_col :
        Which column to treat as the binary deposit label for the RF.
        Default ``any_mineral_occurrence`` (1,214 positives in BCGT).
    center_xy :
        ``(x, y)`` center of the subarea in working-CRS feet. Default
        is the centroid of all positive cells (deposit-cluster center).
    n_cells_side :
        Side length of the subarea in cells. 30 → 900 cells (matches
        Mern v1.0 scale).
    cell_size_m :
        Cell size in metres. Used only for sizing the subarea bbox.
    rf_max_depth, rf_n_estimators :
        RF hyperparameters. Conservative defaults; not tuned (the prior
        is the *story*, not the *target* here).

    Returns
    -------
    sub_df :
        DataFrame of subarea cells with columns ``x``, ``y``,
        ``p_prior`` (RF posterior), and ``label`` (the requested binary
        label).
    feature_cols :
        Names of the columns the RF used as features (for reproducibility
        / SHAP hooks).
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(rng)
    df = pd.read_parquet(features_parquet)
    if label_col not in df.columns:
        raise KeyError(f"label_col {label_col!r} not in features parquet")

    if center_xy is None:
        pos = df[df[label_col] > 0]
        center_xy = (float(pos["x"].mean()), float(pos["y"].mean()))
    cx, cy = center_xy
    # BCGT working CRS is EPSG:3005 (BC Albers, metres); cell size is metres
    half = (n_cells_side / 2) * cell_size_m
    sub = df[
        (df["x"] >= cx - half) & (df["x"] <= cx + half)
        & (df["y"] >= cy - half) & (df["y"] <= cy + half)
    ].copy()
    if len(sub) == 0:
        raise ValueError(
            f"No cells in subarea around {center_xy} with half-width {half:.0f} ft"
        )

    # Train RF on the FULL feature frame (so the subarea prior is
    # informed by global geology), predict on subarea.
    feature_cols = [
        c for c in df.columns
        if c not in {"row", "col", "x", "y", label_col,
                     "any_mineral_occurrence", "is_porphyry",
                     "is_epithermal", "is_skarn", "is_vms"}
    ]
    X = df[feature_cols].fillna(-9999).values
    y = (df[label_col] > 0).astype(int).values
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=rng.integers(0, 2**31 - 1),
        n_jobs=-1,
    )
    rf.fit(X, y)
    sub["p_prior"] = rf.predict_proba(sub[feature_cols].fillna(-9999).values)[:, 1]
    sub["label"] = (sub[label_col] > 0).astype(int)
    return sub.reset_index(drop=True), feature_cols


def sample_ground_truth(
    p_prior: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw a single Bernoulli realization of the deposit indicator
    field from the prior. Used to seed the planner's "ground truth"
    in synthetic-truth experiments."""
    rng = np.random.default_rng(rng)
    return (rng.random(size=p_prior.shape) < p_prior).astype(int)


# ---------------------------------------------------------------------------
# Episode simulator (works for any policy)
# ---------------------------------------------------------------------------


def simulate_policy(
    problem: DrillingProblem,
    policy: "Policy",
    *,
    horizon: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run a policy for up to ``horizon`` drills on a single ground-truth
    realization.

    Returns a dict with keys ``actions``, ``observations``, ``rewards``,
    ``cumulative_reward``, ``cumulative_discoveries``.
    """
    rng = np.random.default_rng(rng)
    drilled: frozenset[int] = frozenset()
    # Posterior tracker for the policy: starts at prior, updated on each
    # observation (noiseless ⇒ posterior collapses to {0, 1}).
    posterior = problem.p_prior.copy()
    actions: list[int] = []
    observations: list[int] = []
    rewards: list[float] = []
    discoveries = 0
    for _ in range(horizon):
        action = policy.choose(problem, posterior, drilled, rng=rng)
        if action is None:           # policy chose to STOP
            break
        obs, reward, drilled = problem.step(action, drilled)
        # Noiseless update: the cell's posterior is now exactly the obs.
        posterior[action] = float(obs)
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        if obs == 1:
            discoveries += 1
    return {
        "actions": actions,
        "observations": observations,
        "rewards": rewards,
        "cumulative_reward": float(np.sum(rewards)),
        "cumulative_discoveries": discoveries,
        "drilled_set": drilled,
    }
