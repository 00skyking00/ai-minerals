"""Stacking ensemble (RF + LightGBM, logistic meta-learner) with spatial-block CV.

The meta-learner is trained on out-of-fold base-model predictions where
the folds are spatial blocks, not random splits. That keeps the
logistic regression from learning the same locally-correlated
geometry the base trees already memorize.

NaN handling matches `model_rf.spatial_block_scores_tree`: feature
matrices are filled with the -9999 sentinel before fitting so the RF
base estimator is happy. LightGBM tolerates the sentinel; with both
trees seeing the same encoded input the meta-learner stays calibrated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ai_minerals.model import SpatialBlockCV
from ai_minerals.model_lgbm import make_lgbm
from ai_minerals.model_rf import make_rf


_NAN_SENTINEL = -9999.0


def _fill_for_trees(X: pd.DataFrame) -> np.ndarray:
    """Replace NaN with the -9999 sentinel RF expects, return a numpy array."""
    return X.fillna(_NAN_SENTINEL).to_numpy()


def _cv_pairs(df: pd.DataFrame, *, block_size_m: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize SpatialBlockCV folds as (train_idx, test_idx) pairs for sklearn."""
    blocks = SpatialBlockCV(block_size_m=block_size_m)
    return [
        (np.asarray(tr), np.asarray(te))
        for (tr, te, _block_id) in blocks.split(df)
    ]


def _make_stacking(
    *, cv_pairs: list[tuple[np.ndarray, np.ndarray]], random_state: int
) -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("rf", make_rf(random_state=random_state)),
            ("lgbm", make_lgbm(random_state=random_state)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=cv_pairs,
        passthrough=False,
        n_jobs=1,  # avoid nesting joblib; RF + LGBM each use n_jobs=-1
    )


def fit_stacking_spatial(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    block_size_m: float = 20_000.0,
    random_state: int = 42,
) -> tuple[StackingClassifier, pd.Series]:
    """Fit a 2-level stacking ensemble (RF + LightGBM, logistic regression meta).

    df must contain x/y columns; SpatialBlockCV groups by spatial block so the
    meta-learner sees OOF predictions, not in-fold leakage.

    Returns (fitted_estimator, per_cell_p_stack_series). The series is the
    positive-class probability from the full-data refit, indexed identically
    to df.
    """
    cv_pairs = _cv_pairs(df, block_size_m=block_size_m)
    estimator = _make_stacking(cv_pairs=cv_pairs, random_state=random_state)
    X_arr = _fill_for_trees(X)
    y_arr = np.asarray(y)
    estimator.fit(X_arr, y_arr)
    p_stack = pd.Series(
        estimator.predict_proba(X_arr)[:, 1],
        index=df.index,
        name="p_stack",
    )
    return estimator, p_stack


def stacking_spatial_block_scores(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    block_size_m: float = 20_000.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Per-fold ROC-AUC + PR-AUC for the stacking estimator under spatial CV.

    Each spatial block is held out as the test fold; the stacking estimator
    (with its own inner spatial CV over the remaining blocks) is fit on the
    rest and scored on the held-out block. Returns the same per-fold shape
    as `model_rf.spatial_block_scores_tree`.
    """
    y_arr = np.asarray(y)
    outer = SpatialBlockCV(block_size_m=block_size_m)
    results: list[dict] = []
    for train_idx, test_idx, block_id in outer.split(df):
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]
        if y_test.sum() == 0 or y_train.sum() == 0:
            continue

        df_train = df.iloc[train_idx]
        X_train = _fill_for_trees(X.iloc[train_idx])
        X_test = _fill_for_trees(X.iloc[test_idx])

        # Inner stacking CV runs over the training subset's own spatial blocks.
        inner_pairs = _cv_pairs(df_train.reset_index(drop=True), block_size_m=block_size_m)
        if len(inner_pairs) < 2:
            # Not enough inner blocks for the meta-learner; skip this fold.
            continue
        model = _make_stacking(cv_pairs=inner_pairs, random_state=random_state)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "block": block_id,
                "n_train_pos": int(y_train.sum()),
                "n_test_pos": int(y_test.sum()),
                "n_test": int(len(test_idx)),
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
            }
        )
    return pd.DataFrame(results)
