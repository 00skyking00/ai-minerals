"""LightGBM binary classifier factory for placer-Au PU labels.

Mirrors `model_rf.make_rf`'s signature and conventions: a default
configuration tuned for the same class-imbalance setup the RF baseline
uses, NaN-safe (LightGBM handles missing values natively), and
deterministic given a random seed.

LightGBM is in the `boost` optional dependency group. The import is
deferred to call time so notebook cells and tests can import this
module without the dependency installed; the error message points the
caller at `uv sync --extra boost` exactly when they try to instantiate
a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier


def make_lgbm(
    *,
    n_estimators: int = 400,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    min_data_in_leaf: int = 20,
    feature_fraction: float = 0.9,
    bagging_fraction: float = 0.9,
    bagging_freq: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> "LGBMClassifier":
    """LightGBM binary classifier configured for placer-Au PU labels.

    - is_unbalance=True for class-imbalance handling (analog of RF's
      class_weight='balanced_subsample').
    - verbose=-1 to silence the per-iteration logging.
    - Deterministic given random_state.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is not installed. Install with "
            "`uv sync --extra boost` (adds lightgbm>=4.5)."
        ) from exc

    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        is_unbalance=True,
        verbose=-1,
        random_state=random_state,
        n_jobs=n_jobs,
    )
