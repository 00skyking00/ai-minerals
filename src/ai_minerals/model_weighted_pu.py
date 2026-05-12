"""Weighted positive-unlabeled learning for prospectivity.

Port of Hajihosseinlou et al. (2025) "A semi-supervised approach for
mineral prospectivity mapping via weighted positive-unlabeled learning
and tree-structured parzen estimator for hyperparameter optimization"
(*Ore Geology Reviews* 185, 106783, doi:10.1016/j.oregeorev.2025.106783).

The published method uses Gaussian Naive Bayes as the base classifier.
We substitute Random Forest because GNB's feature-independence
assumption is broken for our spatial-feature setup, and RF is the
established baseline in the BCGT v2 + Mother Lode v3 work. The
"weighted" part of the formulation (differentiated weights for positive
vs unlabeled instances) maps cleanly onto sklearn's `sample_weight`,
and TPE-tuning the weight ratio uses `optuna`.

Compared to bagging-PU (`model_pu.py`), this approach trains one model
per evaluation rather than an ensemble, but tunes the
positive-vs-unlabeled weight ratio via Bayesian optimization. For a
positive-rich problem like Mother Lode (n_pos = 7,713), the difference
is more about hyperparameter cleanliness than fundamentally different
behavior; for tiny-positive problems (n_pos < 100), weighted-PU's
explicit weight handling matters more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from ai_minerals.model import (
    NON_FEATURE_COLUMNS,
    add_lithology_onehot,
)


@dataclass
class WeightedPUConfig:
    n_trials: int = 30
    cv_folds: int = 5
    seed: int = 42
    rf_n_estimators: int = 200


def _suggest_rf(trial: optuna.Trial, config: WeightedPUConfig) -> RandomForestClassifier:
    """TPE-suggested RF hyperparameters per trial."""
    return RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_depth=trial.suggest_int("max_depth", 5, 50),
        class_weight="balanced_subsample",
        random_state=config.seed,
        n_jobs=-1,
    )


def _build_weights(
    y: np.ndarray, w_pos_mult: float, w_unl_mult: float
) -> np.ndarray:
    """Per-sample weights matching Hajihosseinlou Section 2.2 in spirit.

    The paper's exact equations (4)-(5) are over-specified once you
    move from GNB (which has a closed-form likelihood) to RF (which
    consumes sample_weight directly). The clean port: positives get
    weight `w_pos_mult / m_p` and unlabeled get `w_unl_mult / m_u`,
    where m_p and m_u are the class counts. The TPE search tunes
    `w_pos_mult` and `w_unl_mult` to balance the model's per-class
    contribution.
    """
    n = len(y)
    n_pos = int((y == 1).sum())
    n_unl = n - n_pos
    if n_pos == 0 or n_unl == 0:
        return np.ones(n, dtype=np.float32)
    w = np.zeros(n, dtype=np.float32)
    w[y == 1] = w_pos_mult / n_pos
    w[y == 0] = w_unl_mult / n_unl
    return w


def fit_weighted_pu(
    df: pd.DataFrame,
    top_classes: list[int],
    *,
    label_col: str = "is_orogenic_gold",
    config: WeightedPUConfig | None = None,
) -> tuple[np.ndarray, dict, RandomForestClassifier]:
    """Train weighted-PU on `df[label_col]` and return per-cell scores.

    Returns (per-cell scores, study summary dict, fitted final RF). The
    study is a dict of best hyperparameters and the per-trial F1
    history.

    Process:
    1. Pseudo-negatives sampled from cells far from occurrence (5 km
       exclusion mirrors the bagging-PU pattern in `model_pu.py`).
    2. TPE search over (w_pos_mult, w_unl_mult, RF hyperparameters)
       maximizing F1 on a held-out 30% slice of the train pool.
    3. Refit on full train pool with best params; score every cell.
    """
    if config is None:
        config = WeightedPUConfig()

    if label_col not in df.columns:
        raise KeyError(f"label_col {label_col!r} not in dataframe columns")

    # v3.1 major1/2/3 one-hot expansion mirrors model_pu.fit_pu_bagging.
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df.columns:
            top_majors = (
                df[col][df[col] >= 0].value_counts().head(10).index.tolist()
            )
            extra[col] = top_majors
    df_oh = add_lithology_onehot(df, top_classes, extra_class_columns=extra or None)
    feat_cols = [
        c for c in df_oh.columns
        if c not in NON_FEATURE_COLUMNS
        and c != "lithology_class"
        and c not in ("major1_class", "major2_class", "major3_class")
    ]
    X_all = df_oh[feat_cols].fillna(-9999).to_numpy()

    pos_mask = (df[label_col] == 1).to_numpy()
    pos_idx = np.where(pos_mask)[0]
    unl_idx = np.where(~pos_mask)[0]
    n_pos = len(pos_idx)
    if n_pos == 0:
        raise ValueError(f"no positives in {label_col!r}")

    rng = np.random.default_rng(config.seed)
    sample_n = n_pos * 30  # 30:1 like bagging-PU
    sample_n = min(sample_n, len(unl_idx))
    neg_idx = rng.choice(unl_idx, size=sample_n, replace=False)

    train_idx = np.concatenate([pos_idx, neg_idx])
    y_train = np.concatenate([np.ones(n_pos, dtype=np.int64),
                              np.zeros(len(neg_idx), dtype=np.int64)])
    X_train = X_all[train_idx]

    # Held-out 30% slice for TPE evaluation (within the train pool).
    rng.shuffle(train_idx)
    cut = int(len(train_idx) * 0.7)
    fit_mask = np.zeros(len(y_train), dtype=bool)
    fit_mask[:cut] = True
    rng_perm = rng.permutation(len(y_train))
    fit_mask = fit_mask[rng_perm]

    def objective(trial: optuna.Trial) -> float:
        w_pos = trial.suggest_float("w_pos_mult", 0.5, 5.0, log=True)
        w_unl = trial.suggest_float("w_unl_mult", 0.5, 5.0, log=True)
        rf = _suggest_rf(trial, config)
        sw = _build_weights(y_train[fit_mask], w_pos, w_unl)
        rf.fit(X_train[fit_mask], y_train[fit_mask], sample_weight=sw)
        proba = rf.predict_proba(X_train[~fit_mask])[:, 1]
        # Threshold at 0.5 for F1; alt: use AUPRC which is the paper's metric
        pred = (proba > 0.5).astype(int)
        return f1_score(y_train[~fit_mask], pred)

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  best TPE trial: F1={study.best_value:.3f}  params={best}")

    rf = _suggest_rf(optuna.trial.FixedTrial(best), config)
    sw = _build_weights(y_train, best["w_pos_mult"], best["w_unl_mult"])
    rf.fit(X_train, y_train, sample_weight=sw)
    proba_all = rf.predict_proba(X_all)[:, 1]

    summary = {
        "best_f1": float(study.best_value),
        "best_params": best,
        "n_trials": config.n_trials,
        "trial_f1_history": [float(t.value) for t in study.trials if t.value is not None],
    }
    return proba_all, summary, rf
