"""Phase 1: reproduce the Lawley et al. 2022 MVT GBM baseline.

Runs the SRI TA3 baseline pipeline (HistGradientBoostingClassifier on
the `preferred` feature set) on the Lawley 2022 H3 datacube and reports
train/test/all AUC matching the notebook at
`third_party/sri-ta3-baselines/H3_MVT_GBM_preferred.ipynb`.

Pass condition: test AUC within ~0.01 of the published 0.983.

Outputs:
  data/derived/lawley/path1_baseline_metrics.json
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

# Columns the `preferred` MVT pipeline reads. Loading the full 97-column
# datacube into RAM blows past WSL's 16 GB budget; this subset is ~30
# columns and fits comfortably.
PREFERRED_MVT_COLS = [
    "H3_Geometry", "Continent_Majority", "Latitude_EPSG4326",
    "Geology_Lithology_Majority", "Geology_Lithology_Minority",
    "Geology_Period_Maximum_Majority", "Geology_Period_Minimum_Majority",
    "Geology_Dictionary_Calcareous", "Geology_Dictionary_Carbonaceous",
    "Geology_Dictionary_FineClastic",
    "Geology_Dictionary_Felsic", "Geology_Dictionary_Intermediate",
    "Geology_Dictionary_UltramaficMafic",
    "Geology_Dictionary_Anatectic", "Geology_Dictionary_Gneissose",
    "Geology_Dictionary_Schistose",
    "Seismic_LAB_Priestley", "Seismic_Moho",
    "Gravity_GOCE_ShapeIndex",
    "Geology_Paleolatitude_Period_Minimum",
    "Terrane_Proximity", "Geology_PassiveMargin_Proximity",
    "Geology_BlackShale_Proximity", "Geology_Fault_Proximity",
    "Gravity_Bouguer", "Gravity_Bouguer_HGM", "Gravity_Bouguer_UpCont30km_HGM",
    "Gravity_Bouguer_HGM_Worms_Proximity",
    "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity",
    "Magnetic_HGM", "Magnetic_LongWavelength_HGM",
    "Magnetic_HGM_Worms_Proximity", "Magnetic_LongWavelength_HGM_Worms_Proximity",
    "Training_MVT_Deposit", "Training_MVT_Occurrence",
]

DATACUBE_PARQUET = REPO_ROOT / "data" / "raw" / "lawley2022" / "datacube.parquet"
OUT_DIR = REPO_ROOT / "data" / "derived" / "lawley"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def filter_continent(df: pd.DataFrame, continent: str) -> pd.DataFrame:
    return df[df["Continent_Majority"] == continent].copy()


def neighbor_deposits_fast(df: pd.DataFrame, deptype: str = "MVT") -> pd.DataFrame:
    """Vectorized replacement for utils.neighbor_deposits with progress prints.

    Equivalent to SRI's implementation but:
    - Replaces row-wise apply(axis=1) with vectorized pandas ops.
    - Replaces the O(N²) list-not-in dedup with set.update.
    - Replaces the final per-row vertex-set intersection with explode +
      isin + groupby.any.
    - Logs each phase so a long run is not silent.

    Runs in seconds for the full continent, vs ~30 min for the Python
    apply version.
    """
    t0 = time.time()
    n = len(df)
    print(f"  [neighbor_deposits {deptype}] start, {n:,} rows", flush=True)

    a = df[f"Training_{deptype}_Deposit"].to_numpy(dtype=bool)
    b = df[f"Training_{deptype}_Occurrence"].to_numpy(dtype=bool)
    df[f"{deptype}_Deposit"] = a | b
    n_pos = int(df[f"{deptype}_Deposit"].sum())
    print(f"  [neighbor_deposits {deptype}] step 1 (OR of train flags): "
          f"{n_pos} positive cells  ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: parse WKT vertex strings into list[str] per row.
    # POLYGON ((x y, x y, x y, x y, x y, x y, x y))
    #  -> ["x y", "x y", "x y", "x y", "x y", "x y"]  (drop closing copy)
    df["H3_Geometry2"] = df["H3_Geometry"].str[10:-2].str.split(", ").str[:-1]
    print(f"  [neighbor_deposits {deptype}] step 2 (parse WKT vertices): "
          f"done  ({time.time()-t0:.1f}s)", flush=True)

    # Step 3: union of vertex sets across all positive rows.
    present_vertices: set[str] = set()
    for vlist in df.loc[df[f"{deptype}_Deposit"], "H3_Geometry2"]:
        if vlist is not None:
            present_vertices.update(vlist)
    print(f"  [neighbor_deposits {deptype}] step 3 (unique positive "
          f"vertices): {len(present_vertices):,} vertices  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Step 4: row-wise "any vertex in present_vertices" via explode + isin.
    exploded = df["H3_Geometry2"].explode()
    hits = exploded.isin(present_vertices)
    flag = hits.groupby(level=0).any().reindex(df.index, fill_value=False)
    df[f"{deptype}_Deposit_wNeighbors"] = flag.astype(bool)
    n_neighbors = int(df[f"{deptype}_Deposit_wNeighbors"].sum())
    print(f"  [neighbor_deposits {deptype}] step 4 (vertex-intersect "
          f"check): {n_neighbors:,} cells flagged as deposit-or-neighbor "
          f"({time.time()-t0:.1f}s)", flush=True)

    df = df.drop(columns=["H3_Geometry2"])
    print(f"  [neighbor_deposits {deptype}] total {time.time()-t0:.1f}s",
          flush=True)
    return df


def main() -> None:
    print(f"=== Lawley 2022 Phase 1: MVT GBM baseline ===")
    print(f"datacube: {DATACUBE_PARQUET}")
    if not DATACUBE_PARQUET.exists():
        raise FileNotFoundError(
            f"Datacube parquet missing. Run scripts/reproductions/lawley/csv_to_parquet.py first."
        )

    t0 = time.time()
    data = pd.read_parquet(DATACUBE_PARQUET, columns=PREFERRED_MVT_COLS)
    print(f"loaded {data.shape[0]:,} rows × {data.shape[1]} cols  "
          f"({time.time()-t0:.1f}s, "
          f"{data.memory_usage(deep=True).sum()/1e6:.0f} MB in RAM)", flush=True)

    # Split by continent (paper trains separately and concatenates).
    aus = filter_continent(data, "Oceania")
    uscan = filter_continent(data, "North America")
    del data
    gc.collect()
    print(f"Australia:    {aus.shape[0]:,} rows", flush=True)
    print(f"US+Canada:    {uscan.shape[0]:,} rows", flush=True)

    # Add neighbor-deposit augmentation per the paper's preferred config.
    aus = neighbor_deposits_fast(aus, deptype="MVT")
    print(f"after neighbor expansion (Aus):   "
          f"{int(aus['MVT_Deposit'].sum())} positives "
          f"(MVT_Deposit), "
          f"{int(aus['Training_MVT_Deposit'].sum())} training-flagged",
          flush=True)
    uscan = neighbor_deposits_fast(uscan, deptype="MVT")
    print(f"after neighbor expansion (USCAN): "
          f"{int(uscan['MVT_Deposit'].sum())} positives "
          f"(MVT_Deposit), "
          f"{int(uscan['Training_MVT_Deposit'].sum())} training-flagged",
          flush=True)

    # Pull `preferred` feature columns. extract_cols returns
    # (DataFrame, list_of_kept_col_names).
    cols_dict = utils.load_features_dict(deptype="MVT", baseline="preferred")
    cols_aus, _ = utils.extract_cols(aus, cols_dict)
    cols_uscan, _ = utils.extract_cols(uscan, cols_dict)

    # Reattach metadata + target for spatial split. Drop H3_Geometry —
    # we don't need it for AUC + we don't rasterize in Phase 1.
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
    print(f"after free: working frame "
          f"{data_filtered.memory_usage(deep=True).sum()/1e6:.0f} MB in RAM",
          flush=True)

    drop_for_dtype_check = [
        "target", "Latitude_EPSG4326",
        "Continent_Majority", "Training_MVT_Deposit",
    ]
    # SRI's loop converts every non-(float64|bool) column to a uint8
    # categorical code. We downcasted floats to float32 in the parquet
    # converter, so the literal `!= "float64"` check also picks up
    # numeric columns — that's a bug we have to avoid. Restrict to
    # actual string-typed columns and use pd.factorize (C-level,
    # ~100× faster than the published replace-with-dict idiom).
    cat_cols = []
    for col in data_filtered.drop(columns=drop_for_dtype_check).columns:
        kind = data_filtered[col].dtype.kind
        if kind in ("O", "U"):  # object or string
            cat_cols.append(col)
    print(f"  [convert_categorical] {len(cat_cols)} string columns to encode: "
          f"{cat_cols}", flush=True)
    for col in cat_cols:
        t_enc = time.time()
        codes, _ = pd.factorize(data_filtered[col])
        data_filtered[col] = codes.astype("uint8")
        print(f"  [convert_categorical] {col}: encoded in "
              f"{time.time()-t_enc:.1f}s", flush=True)

    print(f"feature frame: {data_filtered.shape}")
    print(f"positives: {int(data_filtered['target'].sum())}")

    # Lawley-style latitude-band CV: 72 bins, 6 groups, fold 1 = test.
    te_df, tr_df, _ = utils.get_spatial_cross_val_idx(
        data_filtered, test_set=1,
        split_col="Training_MVT_Deposit", nbins=72,
    )
    tr_df = tr_df.drop(columns=["Training_MVT_Deposit"])
    te_df = te_df.drop(columns=["Training_MVT_Deposit"])
    print(f"train: {tr_df.shape[0]:,}  test: {te_df.shape[0]:,}")
    print(f"train positives: {int(tr_df['target'].sum())}  "
          f"test positives: {int(te_df['target'].sum())}")

    feature_cols = [c for c in tr_df.columns if c not in (
        "target", "Latitude_EPSG4326", "group",
        "H3_Geometry", "Continent_Majority",
    )]
    # Flag categorical / boolean features for HistGBM. SRI's notebook
    # checks `!= "float64"` because pandas loads CSVs as float64; we
    # downcasted to float32, so that test mis-flags every numeric.
    # Use dtype kind: u = uint (from factorize), b = bool (from the
    # extract_cols dictionary aggregates), i = int (defensive).
    cat_mask = np.asarray(
        [tr_df[c].dtype.kind in ("u", "b", "i") for c in feature_cols]
    ).astype(bool)
    print(f"  categorical feature mask: "
          f"{cat_mask.sum()} cats, {(~cat_mask).sum()} numeric "
          f"({len(feature_cols)} total)", flush=True)
    cat_cols_for_fit = [c for c, m in zip(feature_cols, cat_mask) if m]
    print(f"  categorical columns: {cat_cols_for_fit}", flush=True)

    gain = 400
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=110,
        max_depth=7,
        min_samples_leaf=48,
        max_leaf_nodes=64,
        verbose=0,
        l2_regularization=0,
        class_weight={0: 1, 1: gain},
        validation_fraction=0.1,
        random_state=1234,
        categorical_features=cat_mask,
    )

    print(f"\nfitting GBM on {len(feature_cols)} features ...")
    t1 = time.time()
    clf.fit(tr_df[feature_cols], tr_df["target"])
    print(f"  fit done in {time.time()-t1:.1f}s")

    train_auc = roc_auc_score(
        tr_df["target"], clf.predict_proba(tr_df[feature_cols])[:, 1]
    )
    test_auc = roc_auc_score(
        te_df["target"], clf.predict_proba(te_df[feature_cols])[:, 1]
    )
    all_df = pd.concat([tr_df, te_df])
    all_auc = roc_auc_score(
        all_df["target"], clf.predict_proba(all_df[feature_cols])[:, 1]
    )

    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Test  AUC: {test_auc:.4f}")
    print(f"All   AUC: {all_auc:.4f}")
    print(f"Lawley 2022 published: 0.983")

    metrics = {
        "stage": "Phase 1: Lawley 2022 MVT GBM baseline",
        "n_features": len(feature_cols),
        "n_train": int(tr_df.shape[0]),
        "n_test": int(te_df.shape[0]),
        "n_train_pos": int(tr_df["target"].sum()),
        "n_test_pos": int(te_df["target"].sum()),
        "auc_train": float(train_auc),
        "auc_test": float(test_auc),
        "auc_all": float(all_auc),
        "auc_published": 0.983,
        "model": "HistGradientBoostingClassifier",
        "split": "latitude-band CV (72 bins, 6 folds, test_set=1)",
        "elapsed_total_s": float(time.time() - t0),
    }
    out_path = OUT_DIR / "path1_baseline_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
