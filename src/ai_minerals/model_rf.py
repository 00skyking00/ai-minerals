"""Random Forest / HistGradientBoosting classifiers + SHAP helpers.

Tree models handle NaN natively (no imputation needed) and capture the
non-linear and interaction effects that logistic regression misses. Used
as Day-4's main model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ai_minerals.model import NON_FEATURE_COLUMNS, SpatialBlockCV


def make_rf(*, n_estimators: int = 400, random_state: int = 42) -> RandomForestClassifier:
    """Default Random Forest configuration.

    class_weight='balanced' handles our 56-vs-1216 imbalance; balanced_subsample
    applies it per-tree. n_jobs=-1 uses all cores.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def make_hgb(random_state: int = 42) -> HistGradientBoostingClassifier:
    """Histogram gradient-boosted trees. Handles NaN natively, no imputation."""
    return HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=None,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=random_state,
    )


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of feature importance sorted by magnitude."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        raise ValueError(f"Model {type(model).__name__} has no feature_importances_")
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


def spatial_block_scores_tree(
    X: pd.DataFrame,
    y: np.ndarray,
    rows: pd.DataFrame,
    *,
    model_factory=make_rf,
    block_size_m: float = 20_000.0,
) -> pd.DataFrame:
    """Spatial block CV using a tree model (handles NaN natively — no imputation)."""
    cv = SpatialBlockCV(block_size_m=block_size_m)
    results = []
    for train_idx, test_idx, block_id in cv.split(rows):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if y_test.sum() == 0 or y_train.sum() == 0:
            continue
        model = model_factory()
        # Tree models want NaN-friendly input; fill NaN with a sentinel for RF;
        # HistGradientBoosting handles NaN natively.
        if isinstance(model, RandomForestClassifier):
            X_train = X.iloc[train_idx].fillna(-9999).to_numpy()
            X_test = X.iloc[test_idx].fillna(-9999).to_numpy()
        else:
            X_train = X.iloc[train_idx].to_numpy()
            X_test = X.iloc[test_idx].to_numpy()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results.append({
            "block": block_id,
            "n_train_pos": int(y_train.sum()),
            "n_test_pos": int(y_test.sum()),
            "n_test": int(len(test_idx)),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
        })
    return pd.DataFrame(results)


def count_feature_columns(feature_names: list[str]) -> list[str]:
    """Columns that count nearby samples (proxies for exploration density)."""
    return [c for c in feature_names if c.endswith("_count_5km")]
