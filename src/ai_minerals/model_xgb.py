"""XGBoost classifier factory for the placer stacking pool.

Same role as model_rf.py and model_lgbm.py: provide make_xgb(...) that
returns an XGBClassifier configured for the placer problem shape
(sparse positives, tabular features, NaN-tolerant). v3 Phase C.1 adds
this as a third base learner alongside RF and LightGBM; the stacking
meta-learner becomes a 3-input logistic regression.
"""
from __future__ import annotations


def make_xgb(*, random_state: int = 42):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
        # We pass NaN-replaced -9999.0 sentinels from train_predict, but
        # XGBoost's native missing handling is better; let it route NaNs
        # if any survive.
        # missing kept as XGBoost default.
    )
