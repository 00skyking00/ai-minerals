"""Positive-Unlabeled (PU) learning baseline via bootstrap bagging.

Pseudo-negative sampling assumes cells far from any known occurrence are
non-deposits. PU learning refuses that assumption: it treats unlabeled
cells as unlabeled (some may be undiscovered deposits) and builds an
ensemble by sampling many small "unlabeled-as-negative" subsets.

Implementation: Mordelet-&-Vert-style bagging — fit K classifiers, each
on positives + a random unlabeled draw of equal size; each cell's score
is the mean probability over the classifiers that did NOT include it as
a training "negative" (out-of-bag). Keeps the ensemble honest about what
it has and hasn't seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ai_minerals.model import NON_FEATURE_COLUMNS, add_lithology_onehot


def fit_pu_bagging(
    df: pd.DataFrame,
    top_classes: list[int],
    *,
    n_bags: int = 30,
    random_state: int = 42,
    rf_kwargs: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Fit a bagging PU ensemble and return out-of-bag probabilities for EVERY cell.

    Each bag: positives + random unlabeled draw of equal size -> RF fit.
    Each cell's final score averages predictions from bags where the cell was
    NOT in the unlabeled-as-negative training set (OOB).
    """
    rng = np.random.default_rng(random_state)
    rf_kwargs = rf_kwargs or dict(
        n_estimators=200,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    df_oh = add_lithology_onehot(df, top_classes)
    feat_cols = [c for c in df_oh.columns if c not in NON_FEATURE_COLUMNS and c != "lithology_class"]
    X_all = df_oh[feat_cols].fillna(-9999).to_numpy()

    pos_mask = (df["is_porphyry"] == 1).to_numpy()
    pos_idx = np.where(pos_mask)[0]
    unl_idx = np.where(~pos_mask)[0]
    n_pos = len(pos_idx)

    proba_sum = np.zeros(len(df), dtype=np.float64)
    bag_count = np.zeros(len(df), dtype=np.int32)

    for b in range(n_bags):
        neg_sample = rng.choice(unl_idx, size=n_pos, replace=False)
        train_idx = np.concatenate([pos_idx, neg_sample])
        y_train = np.concatenate([np.ones(n_pos), np.zeros(n_pos)]).astype(np.int64)
        clf = RandomForestClassifier(random_state=b, **rf_kwargs)
        clf.fit(X_all[train_idx], y_train)

        # OOB mask: everything NOT in this bag's training
        oob = np.ones(len(df), dtype=bool)
        oob[train_idx] = False
        # Positives are always in training — score them from every bag
        # (standard in Mordelet-Vert; otherwise they have 0 coverage).
        oob[pos_idx] = True

        proba_sum[oob] += clf.predict_proba(X_all[oob])[:, 1]
        bag_count[oob] += 1

    with np.errstate(invalid="ignore"):
        proba = np.where(bag_count > 0, proba_sum / bag_count, np.nan)
    return proba, feat_cols
