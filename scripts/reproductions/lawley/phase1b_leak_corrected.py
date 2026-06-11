"""Phase 1b: Lawley 2022 MVT GBM with label leak removed.

Phase 1 reproduced the published AUC 0.983 → reproduced as 0.9965 on
the test fold. The published `preferred` MVT feature configuration
includes both `Training_MVT_Deposit` and `Training_MVT_Occurrence` as
input features, and the target the GBM is fit on is the OR of those
two columns (`MVT_Deposit` produced inside neighbor_deposits step 1).
That's a direct label leak: the model can learn the trivial identity
"Training_MVT_Occurrence == True → target == True" and never need to
use the geological / geophysical / proximity features.

This script reuses the Phase 1 pipeline and drops both Training_MVT_*
columns before fitting. The resulting AUC is the honest data-driven
prospectivity number.

Output:
  data/derived/lawley/path1b_leak_corrected_metrics.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/home/sky/src/learning/ai-minerals")
SRI_REPO = REPO_ROOT / "third_party" / "sri-ta3-baselines"
sys.path.insert(0, str(SRI_REPO))
import utilities as utils  # noqa: E402

# Reuse Phase 1 helpers + column constants.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lawley_phase1_gbm import (  # noqa: E402
    PREFERRED_MVT_COLS, filter_continent, neighbor_deposits_fast,
)

DATACUBE_PARQUET = REPO_ROOT / "data" / "raw" / "lawley2022" / "datacube.parquet"
OUT_DIR = REPO_ROOT / "data" / "derived" / "lawley"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns to drop before the GBM fit. SRI's notebook drops only 5 of
# these; we add the two Training_MVT_* columns that arithmetically
# define the target.
DROP_BEFORE_FIT = (
    "target", "Latitude_EPSG4326", "group",
    "H3_Geometry", "Continent_Majority",
    "Training_MVT_Deposit", "Training_MVT_Occurrence",  # leak fix
)


def main() -> None:
    print(f"=== Lawley 2022 Phase 1b: MVT GBM, label leak removed ===",
          flush=True)
    print(f"datacube: {DATACUBE_PARQUET}", flush=True)

    t0 = time.time()
    data = pd.read_parquet(DATACUBE_PARQUET, columns=PREFERRED_MVT_COLS)
    print(f"loaded {data.shape[0]:,} rows × {data.shape[1]} cols  "
          f"({time.time()-t0:.1f}s, "
          f"{data.memory_usage(deep=True).sum()/1e6:.0f} MB in RAM)",
          flush=True)

    aus = filter_continent(data, "Oceania")
    uscan = filter_continent(data, "North America")
    del data
    gc.collect()

    aus = neighbor_deposits_fast(aus, deptype="MVT")
    uscan = neighbor_deposits_fast(uscan, deptype="MVT")

    cols_dict = utils.load_features_dict(deptype="MVT", baseline="preferred")
    cols_aus, _ = utils.extract_cols(aus, cols_dict)
    cols_uscan, _ = utils.extract_cols(uscan, cols_dict)
    for name, src, target in [("aus", aus, cols_aus), ("uscan", uscan, cols_uscan)]:
        target["target"] = src["MVT_Deposit"].values
        target["Continent_Majority"] = src["Continent_Majority"].values
        target["Latitude_EPSG4326"] = src["Latitude_EPSG4326"].values
        target["Training_MVT_Deposit"] = src["Training_MVT_Deposit"].values
    del aus, uscan
    gc.collect()

    data_filtered = pd.concat([cols_aus, cols_uscan], ignore_index=True)
    data_filtered.reset_index(drop=True, inplace=True)
    del cols_aus, cols_uscan
    gc.collect()

    # Factorize the four string features. (H3_Geometry is also string
    # but we drop it before fit, so we skip encoding to save 6 sec.)
    string_features_for_model = [
        "Geology_Lithology_Majority", "Geology_Lithology_Minority",
        "Geology_Period_Maximum_Majority", "Geology_Period_Minimum_Majority",
    ]
    for col in string_features_for_model:
        if col in data_filtered.columns:
            codes, _ = pd.factorize(data_filtered[col])
            data_filtered[col] = codes.astype("uint8")
    # H3_Geometry: drop without factorizing.
    if "H3_Geometry" in data_filtered.columns:
        data_filtered = data_filtered.drop(columns=["H3_Geometry"])
    print(f"feature frame: {data_filtered.shape}", flush=True)
    print(f"positives: {int(data_filtered['target'].sum())}", flush=True)

    te_df, tr_df, _ = utils.get_spatial_cross_val_idx(
        data_filtered, test_set=1,
        split_col="Training_MVT_Deposit", nbins=72,
    )
    tr_df = tr_df.drop(columns=["Training_MVT_Deposit"])
    te_df = te_df.drop(columns=["Training_MVT_Deposit"])
    print(f"train: {tr_df.shape[0]:,}  test: {te_df.shape[0]:,}", flush=True)
    print(f"train positives: {int(tr_df['target'].sum())}  "
          f"test positives: {int(te_df['target'].sum())}", flush=True)

    feature_cols = [c for c in tr_df.columns if c not in DROP_BEFORE_FIT]
    print(f"  dropped from features: "
          f"{[c for c in tr_df.columns if c in DROP_BEFORE_FIT]}",
          flush=True)
    cat_mask = np.asarray(
        [tr_df[c].dtype.kind in ("u", "b", "i") for c in feature_cols]
    ).astype(bool)
    print(f"  cat mask: {cat_mask.sum()} cats, {(~cat_mask).sum()} numeric "
          f"({len(feature_cols)} total)", flush=True)

    gain = 400
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=110, max_depth=7,
        min_samples_leaf=48, max_leaf_nodes=64, verbose=0,
        l2_regularization=0, class_weight={0: 1, 1: gain},
        validation_fraction=0.1, random_state=1234,
        categorical_features=cat_mask,
    )
    print(f"\nfitting GBM on {len(feature_cols)} features (leak removed) ...",
          flush=True)
    t1 = time.time()
    clf.fit(tr_df[feature_cols], tr_df["target"])
    print(f"  fit done in {time.time()-t1:.1f}s", flush=True)

    train_auc = roc_auc_score(tr_df["target"],
                              clf.predict_proba(tr_df[feature_cols])[:, 1])
    test_auc = roc_auc_score(te_df["target"],
                             clf.predict_proba(te_df[feature_cols])[:, 1])
    all_df = pd.concat([tr_df, te_df])
    all_auc = roc_auc_score(all_df["target"],
                            clf.predict_proba(all_df[feature_cols])[:, 1])

    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Test  AUC: {test_auc:.4f}")
    print(f"All   AUC: {all_auc:.4f}")
    print(f"Phase 1 leaking baseline: 0.9965 (test)")
    print(f"Lawley 2022 published:    0.983")
    print(f"Delta from Phase 1 -> 1b: {test_auc - 0.9965:+.4f}")

    metrics = {
        "stage": "Phase 1b: Lawley 2022 MVT GBM, label leak removed",
        "n_features": len(feature_cols),
        "n_train": int(tr_df.shape[0]),
        "n_test": int(te_df.shape[0]),
        "n_train_pos": int(tr_df["target"].sum()),
        "n_test_pos": int(te_df["target"].sum()),
        "auc_train": float(train_auc),
        "auc_test": float(test_auc),
        "auc_all": float(all_auc),
        "auc_published_with_leak": 0.983,
        "auc_phase1_with_leak": 0.9965,
        "delta_test_auc_vs_phase1": float(test_auc - 0.9965),
        "model": "HistGradientBoostingClassifier",
        "split": "latitude-band CV (72 bins, 6 folds, test_set=1)",
        "leak_columns_dropped": ["Training_MVT_Deposit", "Training_MVT_Occurrence"],
        "elapsed_total_s": float(time.time() - t0),
    }
    out_path = OUT_DIR / "path1b_leak_corrected_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
