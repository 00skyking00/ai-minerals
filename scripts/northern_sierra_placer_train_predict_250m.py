"""Phase 2 driver: train + predict placer-Au prospectivity per population.

Runs one classifier per placer population (Tertiary deep-gravel and
Quaternary modern-channel). Each population goes through:

    PU baseline (Mordelet-Vert bagging)
    -> Random Forest (spatial-block CV + full-data refit)
    -> LightGBM   (spatial-block CV + full-data refit)
    -> XGBoost    (spatial-block CV + full-data refit; v3 Phase C.1)
    -> Stacking   (logistic-regression meta on (RF, LGBM, XGB) base scores)
    -> Isotonic calibration (sigmoid fallback when positives are sparse)

Anchor districts (Malakoff, Dutch Flat, etc.) are masked out of every
training fold, every PU bag, and every calibration fold, so the Phase 1
validation gate is never trained on.

Hawkes catchment-geochem features (catchment_au_hawkes,
catchment_as_hawkes, catchment_sb_hawkes) are recomputed per spatial-
block fold using only the samples in the fold's training cells, to keep
the leakage discipline honest. When `--no-hawkes-refold` is passed, the
script skips the per-fold recompute and notes the caveat in
`pop_fold_metrics_*.csv`. Use it only when wallclock budget is tight;
the Phase 1 ship-rule applies (per the plan).

Outputs (per population pop) under data/derived/northern_sierra_placer/:

    pop_predictions_<pop>_250m.parquet
        row, col, x, y, p_pu, p_rf, p_lgbm, p_xgb, p_stack
    pop_calibrated_<pop>_250m.parquet
        row, col, x, y, p_cal
    pop_fold_metrics_<pop>.csv
        model, fold_id, n_train, n_test, roc_auc, pr_auc
    prospectivity_placer_<pop>_250m_calibrated_3310.tif
    prospectivity_placer_<pop>_250m_calibrated_4326.tif
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ai_minerals.config import (
    QUATERNARY_FEATURE_COLUMNS,
    TERTIARY_FEATURE_COLUMNS,
)
from ai_minerals.data.adapters import get_adapter
from ai_minerals.features.placer_geology import hawkes_dual_decay_catchment
from ai_minerals.grid import build_grid
from ai_minerals.io.geotiff import write_geotiff_dual_crs
from ai_minerals.model import (
    SpatialBlockCV,
    add_lithology_onehot,
    non_feature_columns,
)
from ai_minerals.model_lgbm import make_lgbm
from ai_minerals.model_pu import fit_pu_bagging
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.model_xgb import make_xgb
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
RES_M = 250.0
BLOCK_SIZE_M = 20_000.0
N_PU_BAGS = 30
ISOTONIC_MIN_POSITIVES = 30  # below this, fall back to Platt scaling
CALIBRATION_CV = 5

POPULATIONS = ("placer_tertiary", "placer_quaternary")

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
IN_FEATURES = DATA_DERIVED / f"features_{REGION.data_prefix}_250m.parquet"
OUT_DIR = DATA_DERIVED / REGION.data_prefix
CKPT_DIR = OUT_DIR / "_k5_checkpoints"


# --- Checkpoint helpers --------------------------------------------------------
#
# K.5 runs for hours per population (~10 spatial-block folds x per-fold Hawkes
# refold for RF and LGBM CV; another ~10 folds for the stacking OOF pass). If
# the process dies anywhere in that pipeline, the next run picks up where the
# last cached stage / fold left off. Two granularities:
#
#   - per-stage: the wrapper `_run_stage` in train_one_population caches the
#     return value of each major stage (PU, RF CV, LGBM CV, stacking, full
#     refits, calibration). One file per pop x stage.
#   - per-fold: `_spatial_block_scores_with_refold` and
#     `_stacking_oof_predictions` write each completed fold as a separate
#     checkpoint, so a death mid-CV loses at most one fold's work.
#
# Checkpoints live under data/derived/<region>/_k5_checkpoints/ and use joblib
# (numpy arrays, sklearn estimators, dataframes all serialize cleanly). To force
# a fresh run, delete the directory:
#
#   rm -rf data/derived/northern_sierra_placer/_k5_checkpoints
#
# Checkpoints are tied to the population name and stage name only; if you
# change feature columns or the labels parquet, clear them or you'll restore
# stale results silently.

def _ckpt_path(name: str) -> Path:
    return CKPT_DIR / f"{name}.joblib"


def _ckpt_save(name: str, obj) -> None:
    import joblib
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _ckpt_path(name).with_suffix(".joblib.tmp")
    joblib.dump(obj, tmp, compress=3)
    tmp.replace(_ckpt_path(name))


def _ckpt_load(name: str):
    """Return the cached object or None if no checkpoint exists."""
    import joblib
    p = _ckpt_path(name)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as exc:
        # Corrupted checkpoint (e.g. partial write killed mid-rename) — drop it.
        print(f"WARNING: dropping corrupt checkpoint {p.name} ({type(exc).__name__})",
              flush=True)
        p.unlink(missing_ok=True)
        return None


def _run_stage(stage_name: str, compute_fn):
    """Cache-or-compute wrapper. `compute_fn` is a zero-arg callable that
    returns the stage's result; the result is pickled to disk under
    CKPT_DIR/{stage_name}.joblib and returned. On a subsequent run the cached
    value is loaded and `compute_fn` is not called."""
    cached = _ckpt_load(stage_name)
    if cached is not None:
        print(f"  [cache hit] {stage_name}", flush=True)
        return cached
    result = compute_fn()
    _ckpt_save(stage_name, result)
    return result


# --- Hawkes per-fold recompute -------------------------------------------------

# Pathfinder element -> (feature column, geochem-source column on the joined
# samples frame). The assemble script names the catchment columns
# catchment_<el>_hawkes; the underlying element column on the NGDB + NURE
# samples uses the upper-case element symbol (Au_ppm or Au, handled by the
# adapter normalization in the assemble script).
HAWKES_ELEMENTS: dict[str, str] = {
    "catchment_au_hawkes": "Au_ppm",
    "catchment_as_hawkes": "As_ppm",
    "catchment_sb_hawkes": "Sb_ppm",
}


def _anchor_cell_indices(df: pd.DataFrame) -> np.ndarray:
    """Snap each anchor district (lon, lat) to its nearest grid-cell row index.

    Mirrors `scripts/northern_sierra_placer_phase1_index.py::_anchor_cell_indices`
    but returns a plain numpy int64 array (not a Series) for set-membership use.
    """
    transformer = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    idxs: list[int] = []
    for _name, (lon, lat) in ANCHOR_DISTRICTS.items():
        ax, ay = transformer.transform(lon, lat)
        d2 = (xs - ax) ** 2 + (ys - ay) ** 2
        idxs.append(int(np.argmin(d2)))
    return np.unique(np.array(idxs, dtype=np.int64))


def _block_ids(df_xy: pd.DataFrame, block_size_m: float) -> np.ndarray:
    """Compute the same block id assignment as SpatialBlockCV.split for one frame."""
    bx = (df_xy["x"].to_numpy() // block_size_m).astype(int)
    by = (df_xy["y"].to_numpy() // block_size_m).astype(int)
    bx_min = bx.min()
    by_min = by.min()
    by_range = by.max() - by_min + 1
    return (bx - bx_min) * by_range + (by - by_min)


def _load_geochem_samples() -> gpd.GeoDataFrame:
    """Load NGDB + NURE stream-sediment samples, concat into one GeoDataFrame."""
    parts: list[gpd.GeoDataFrame] = []
    ngdb_adapter = get_adapter("geochem", REGION.geochem_source)
    ngdb = ngdb_adapter(
        REGION.raw_paths["geochem"], REGION.aoi,
        elements=REGION.pathfinder_elements,
    )
    parts.append(ngdb)
    nure_path = REGION.raw_paths.get("geochem_nure")
    if nure_path is not None and Path(nure_path).exists():
        nure_adapter = get_adapter("geochem", "nure_iicpms")
        nure = nure_adapter(
            nure_path, REGION.aoi, elements=REGION.pathfinder_elements,
        )
        parts.append(nure)
    # Normalize CRS before concat — NGDB is in working_crs (EPSG:3310 for
    # the placer model), NURE arrives in EPSG:4326. Reproject everything
    # to working_crs so the downstream Hawkes snapping is in meters.
    target_crs = REGION.working_crs
    parts = [p.to_crs(target_crs) if p.crs is not None and str(p.crs) != target_crs else p
             for p in parts]
    samples = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(samples, geometry="geometry", crs=target_crs)


def _load_nhd_network() -> gpd.GeoDataFrame:
    """Load the NHD flowline network for Hawkes upstream walks."""
    nhd_path = REGION.raw_paths["nhd_flowlines"]
    return gpd.read_file(nhd_path)


def _recompute_hawkes_for_fold(
    df: pd.DataFrame,
    train_block_ids: set[int],
    cell_block_ids: np.ndarray,
    samples: gpd.GeoDataFrame,
    sample_block_ids: np.ndarray,
    nhd: gpd.GeoDataFrame,
    grid,
    element_col: str,
    element: str,
) -> pd.Series:
    """Recompute one Hawkes feature using only training-fold samples.

    Returns a Series indexed identically to df with NaN outside the test
    block's cells (caller pastes the result back into the per-fold X frame).

    v3 Phase A.1 optimization: build a cell_mask covering only the AOI-clipped
    cells in df (typically ~800k of grid.n_cells ~1.4M, since the full grid
    bounding rectangle includes outside-AOI cells we'll never use). The hot
    loop in hawkes_dual_decay_catchment skips the ~45% wasted cells. The
    optimization is automatic and applies regardless of fold size; it doesn't
    change per-cell math, only which cells the loop visits.
    """
    sample_mask = np.isin(sample_block_ids, list(train_block_ids))

    # Map AOI-clipped df cells to grid-flat positions for the cell_mask.
    flat_idx = (df["row"].to_numpy() * grid.shape[1]
                + df["col"].to_numpy()).astype(np.int64)
    cell_mask = np.zeros(grid.n_cells, dtype=bool)
    cell_mask[flat_idx] = True

    full_grid_series = hawkes_dual_decay_catchment(
        samples, nhd, grid, element=element,
        fold_mask=sample_mask, cell_mask=cell_mask,
    )
    # hawkes returns one entry per full-grid cell (grid.n_cells); df is
    # AOI-clipped. Map back via row-major (row, col) flat index.
    return pd.Series(full_grid_series.to_numpy()[flat_idx], index=df.index, name=element_col)


# --- Per-model scoring helpers -------------------------------------------------

def _score_proba(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and PR-AUC, returning NaN for degenerate (single-class) folds."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return (float("nan"), float("nan"))
    return (
        float(roc_auc_score(y_true, proba)),
        float(average_precision_score(y_true, proba)),
    )


def _fit_predict_tree(model, X_train, y_train, X_test) -> np.ndarray:
    """Fit a tree model and return positive-class probabilities on X_test."""
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def _spatial_block_scores_with_refold(
    df_train: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    model_factory,
    model_name: str,
    *,
    refold_hawkes: bool,
    samples: gpd.GeoDataFrame | None,
    sample_block_ids: np.ndarray | None,
    nhd: gpd.GeoDataFrame | None,
    grid,
    block_size_m: float = BLOCK_SIZE_M,
    ckpt_prefix: str | None = None,
) -> pd.DataFrame:
    """Spatial-block CV with optional per-fold Hawkes refold.

    Returns a per-fold metrics frame: model, fold_id, n_train, n_test,
    roc_auc, pr_auc.

    If `ckpt_prefix` is set, each completed fold's metric row is written to
    `{ckpt_prefix}__fold_{block_id}.joblib` immediately after it finishes;
    on rerun, folds whose checkpoint exists are loaded and not recomputed.
    A death mid-CV loses at most one fold's work.
    """
    cv = SpatialBlockCV(block_size_m=block_size_m)
    cell_block_ids = _block_ids(df_train, block_size_m)
    y = df_train[label_col].to_numpy(dtype=np.int64)
    X_base = df_train[feat_cols].copy()

    rows: list[dict] = []
    for train_idx, test_idx, block_id in cv.split(df_train):
        fold_ckpt = f"{ckpt_prefix}__fold_{int(block_id)}" if ckpt_prefix else None
        if fold_ckpt is not None:
            cached_row = _ckpt_load(fold_ckpt)
            if cached_row is not None:
                rows.append(cached_row)
                print(f"  [cache hit] {fold_ckpt}", flush=True)
                continue

        y_train = y[train_idx]
        y_test = y[test_idx]
        if y_train.sum() == 0 or y_test.sum() == 0:
            continue
        X_fold = X_base.copy()
        if refold_hawkes and samples is not None and nhd is not None:
            train_blocks = set(cell_block_ids[train_idx].tolist())
            for feat_col, el in HAWKES_ELEMENTS.items():
                if feat_col not in X_fold.columns:
                    continue
                refold = _recompute_hawkes_for_fold(
                    df_train, train_blocks, cell_block_ids,
                    samples, sample_block_ids, nhd, grid,
                    element_col=feat_col, element=el,
                )
                X_fold[feat_col] = refold.to_numpy()

        X_train = X_fold.iloc[train_idx].fillna(-9999.0).to_numpy(dtype=np.float32)
        X_test = X_fold.iloc[test_idx].fillna(-9999.0).to_numpy(dtype=np.float32)
        model = model_factory()
        proba = _fit_predict_tree(model, X_train, y_train, X_test)
        roc, pr = _score_proba(y_test, proba)
        row = {
            "model": model_name,
            "fold_id": int(block_id),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_test_pos": int(y_test.sum()),
            "roc_auc": roc,
            "pr_auc": pr,
        }
        rows.append(row)
        if fold_ckpt is not None:
            _ckpt_save(fold_ckpt, row)
    return pd.DataFrame(rows)


def _stacking_oof_predictions(
    df_train: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    *,
    block_size_m: float = BLOCK_SIZE_M,
    ckpt_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spatial-block OOF predictions for RF + LGBM + XGBoost.

    Returns (p_rf_oof, p_lgbm_oof, p_xgb_oof, fold_seen).

    fold_seen is a bool array of cells that ever appeared in a test fold;
    cells not in any held-out block (rare edge case) stay NaN in the OOF arrays.

    v3 Phase C.1 adds XGBoost as a third base learner alongside RF and LightGBM;
    each fold fits all three and the resulting OOF columns feed a 3-input
    logistic-regression meta-learner.

    If `ckpt_prefix` is set, each completed fold's test-cell predictions are
    cached as `{ckpt_prefix}__fold_{block_id}.joblib`. A death mid-stacking
    loses at most one fold's RF + LGBM + XGB fit.
    """
    cv = SpatialBlockCV(block_size_m=block_size_m)
    y = df_train[label_col].to_numpy(dtype=np.int64)
    X_base = df_train[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)

    p_rf = np.full(len(df_train), np.nan, dtype=np.float64)
    p_lgbm = np.full(len(df_train), np.nan, dtype=np.float64)
    p_xgb = np.full(len(df_train), np.nan, dtype=np.float64)
    seen = np.zeros(len(df_train), dtype=bool)

    for train_idx, test_idx, block_id in cv.split(df_train):
        fold_ckpt = f"{ckpt_prefix}__fold_{int(block_id)}" if ckpt_prefix else None
        if fold_ckpt is not None:
            cached = _ckpt_load(fold_ckpt)
            if cached is not None:
                # cached carries (test_idx, p_rf_test, p_lgbm_test, p_xgb_test).
                cti = cached["test_idx"]
                p_rf[cti] = cached["p_rf"]
                p_lgbm[cti] = cached["p_lgbm"]
                p_xgb[cti] = cached["p_xgb"]
                seen[cti] = True
                print(f"  [cache hit] {fold_ckpt}", flush=True)
                continue

        y_train = y[train_idx]
        if y_train.sum() == 0:
            continue
        rf = make_rf(random_state=42)
        rf.fit(X_base[train_idx], y_train)
        p_rf_test = rf.predict_proba(X_base[test_idx])[:, 1]
        p_rf[test_idx] = p_rf_test
        lgbm = make_lgbm(random_state=42)
        lgbm.fit(X_base[train_idx], y_train)
        p_lgbm_test = lgbm.predict_proba(X_base[test_idx])[:, 1]
        p_lgbm[test_idx] = p_lgbm_test
        xgb = make_xgb(random_state=42)
        xgb.fit(X_base[train_idx], y_train)
        p_xgb_test = xgb.predict_proba(X_base[test_idx])[:, 1]
        p_xgb[test_idx] = p_xgb_test
        seen[test_idx] = True
        if fold_ckpt is not None:
            _ckpt_save(fold_ckpt, {
                "test_idx": np.asarray(test_idx),
                "p_rf": p_rf_test,
                "p_lgbm": p_lgbm_test,
                "p_xgb": p_xgb_test,
            })
    return p_rf, p_lgbm, p_xgb, seen


def _fit_stacking_meta(
    p_rf_oof: np.ndarray,
    p_lgbm_oof: np.ndarray,
    p_xgb_oof: np.ndarray,
    y: np.ndarray,
    seen: np.ndarray,
) -> LogisticRegression:
    """Fit the logistic-regression meta-learner on OOF base scores.

    v3 Phase C.1: meta-learner is a 3-input logistic regression over the
    RF, LightGBM, and XGBoost OOF predictions.
    """
    valid = (
        seen
        & np.isfinite(p_rf_oof)
        & np.isfinite(p_lgbm_oof)
        & np.isfinite(p_xgb_oof)
    )
    X_meta = np.column_stack([p_rf_oof[valid], p_lgbm_oof[valid], p_xgb_oof[valid]])
    y_meta = y[valid]
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta, y_meta)
    return meta


# --- Per-population pipeline ---------------------------------------------------

def train_one_population(
    df: pd.DataFrame,
    pop: str,
    *,
    anchor_cells: np.ndarray,
    refold_hawkes: bool,
    samples: gpd.GeoDataFrame | None,
    sample_block_ids: np.ndarray | None,
    nhd: gpd.GeoDataFrame | None,
    grid,
) -> dict:
    """Run PU -> RF -> LGBM -> stacking -> calibration for one population.

    Returns a dict with per-cell prediction arrays (p_pu, p_rf, p_lgbm,
    p_stack, p_cal), a fold-metric DataFrame, and the feature-column list
    actually used.
    """
    label_col = f"is_{pop}"
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not in features parquet")

    n_pos = int(df[label_col].sum())
    print(f"[{pop}] positives: {n_pos:,} of {len(df):,} cells", flush=True)
    if n_pos == 0:
        raise RuntimeError(
            f"[{pop}] zero positives in feature frame; cannot train. "
            f"Check the assemble script and the region deposit_classes."
        )

    # Mask anchors out of the training rows. Anchors stay in `df` for
    # prediction; they're only excluded as training labels / pseudo-negatives.
    not_anchor = np.ones(len(df), dtype=bool)
    not_anchor[anchor_cells] = False
    df_train = df.loc[not_anchor].reset_index(drop=True)
    n_pos_train = int(df_train[label_col].sum())
    print(f"[{pop}] after anchor mask: {n_pos_train:,} positives / {len(df_train):,} cells",
          flush=True)

    # Lithology one-hot. Compute top classes on the anchor-excluded training rows.
    top_classes = df_train["lithology_class"].value_counts().head(10).index.tolist()
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df_train.columns:
            extra[col] = (
                df_train[col][df_train[col] >= 0].value_counts().head(10).index.tolist()
            )
    df_oh_train = add_lithology_onehot(df_train, top_classes, extra_class_columns=extra or None)
    df_oh_full = add_lithology_onehot(df, top_classes, extra_class_columns=extra or None)

    # Feature columns: drop identity + every is_<pop> label + count features.
    label_cols = tuple(f"is_{p}" for p in POPULATIONS)
    non_feat = non_feature_columns(label_cols=label_cols)
    feat_cols = [c for c in df_oh_train.columns if c not in non_feat]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]
    print(f"[{pop}] feature columns: {len(feat_cols)}", flush=True)

    # v3 per-population feature stack filtering. Each population trains on a
    # subset of the assembled features matched to its geomorphic signature.
    # Features in the per-population tuple that aren't in df_oh_train.columns
    # are skipped (graceful for v3 features not yet assembled).
    pop_features = {
        "placer_tertiary": TERTIARY_FEATURE_COLUMNS,
        "placer_quaternary": QUATERNARY_FEATURE_COLUMNS,
    }.get(pop)
    if pop_features is not None:
        # Keep only the per-pop features, but allow one-hot lithology columns
        # (named lithology_class_<N>) through because add_lithology_onehot
        # expands the single lithology_class into per-class columns.
        allowed_prefixes = ("lithology_class_", "major1_class_", "major2_class_", "major3_class_")
        feat_cols_filtered = [
            c for c in feat_cols
            if c in pop_features
            or any(c.startswith(p) for p in allowed_prefixes)
        ]
        missing = [c for c in pop_features
                   if c not in df_oh_train.columns
                   and not any(c.startswith(p.rstrip("_")) for p in allowed_prefixes)]
        if missing:
            print(f"[{pop}] v3 feature filter: skipping (not in parquet): {missing}",
                  flush=True)
        feat_cols = feat_cols_filtered
        print(f"[{pop}] v3 feature filter: {len(feat_cols)} features",
              flush=True)

    # Assert the leakage guard from the assemble step actually fired.
    if "distance_downstream_from_lode_m" not in df_oh_train.columns:
        print(f"[{pop}] WARNING: distance_downstream_from_lode_m missing from features; "
              f"check assemble script's lode-leakage guard.", flush=True)

    fold_metrics: list[pd.DataFrame] = []

    # --- PU baseline. Cached as a pair (training-set p_pu, full-grid p_pu) so a
    # restart skips the ~2-3 min double-PU fit.
    # v3 Phase C.2: Quaternary uses nnPU (Kiryo 2017); Tertiary keeps
    # Mordelet-Vert PU bagging (pit-polygon positives are effectively
    # fully labeled). The stage checkpoint key is unchanged
    # (`{pop}__pu`) so resumed runs pick up the right cached result. ---
    if pop == "placer_quaternary":
        def _nnpu():
            print(f"[{pop}] nnPU training (Kiryo 2017, prior=0.0007)...", flush=True)
            t0 = time.time()
            from ai_minerals.model_nnpu import fit_nnpu_quaternary
            p_train, _ = fit_nnpu_quaternary(
                df_oh_train, label_col=label_col,
                feature_cols=feat_cols, prior=0.0007,
                random_state=42,
            )
            print(f"[{pop}]   nnPU train done in {(time.time()-t0)/60:.1f} min",
                  flush=True)
            p_full, _ = fit_nnpu_quaternary(
                df_oh_full, label_col=label_col,
                feature_cols=feat_cols, prior=0.0007,
                random_state=42,
            )
            return {"p_pu_train": p_train, "p_pu_full": p_full}
        pu_result = _run_stage(f"{pop}__pu", _nnpu)
    else:
        def _pu():
            print(f"[{pop}] PU bagging (n_bags={N_PU_BAGS})...", flush=True)
            t0 = time.time()
            p_pu_train, _ = fit_pu_bagging(
                df_oh_train, top_classes,
                label_col=label_col, n_bags=N_PU_BAGS, random_state=42,
            )
            print(f"[{pop}]   PU train done in {(time.time()-t0)/60:.1f} min "
                  f"(n_finite={int(np.isfinite(p_pu_train).sum())})", flush=True)
            p_pu_full, _ = fit_pu_bagging(
                df_oh_full, top_classes,
                label_col=label_col, n_bags=N_PU_BAGS, random_state=42,
            )
            return {"p_pu_train": p_pu_train, "p_pu_full": p_pu_full}
        pu_result = _run_stage(f"{pop}__pu", _pu)
    p_pu_full = pu_result["p_pu_full"]

    # --- RF with spatial-block CV (+ per-fold Hawkes refold). Per-fold
    # checkpoints under {pop}__rf_cv__fold_<block>.joblib so a mid-CV death
    # loses one fold at most. ---
    def _rf_cv():
        print(f"[{pop}] RF spatial-block CV "
              f"(refold_hawkes={refold_hawkes})...", flush=True)
        t0 = time.time()
        out = _spatial_block_scores_with_refold(
            df_oh_train, feat_cols, label_col,
            model_factory=make_rf, model_name="rf",
            refold_hawkes=refold_hawkes,
            samples=samples, sample_block_ids=sample_block_ids, nhd=nhd, grid=grid,
            ckpt_prefix=f"{pop}__rf_cv",
        )
        print(f"[{pop}]   RF CV done in {(time.time()-t0)/60:.1f} min  "
              f"folds={len(out)}  AUC mean={out['roc_auc'].mean():.3f}", flush=True)
        return out
    rf_cv = _run_stage(f"{pop}__rf_cv", _rf_cv)
    fold_metrics.append(rf_cv)

    # --- LGBM with spatial-block CV. Same per-fold checkpointing. ---
    def _lgbm_cv():
        print(f"[{pop}] LightGBM spatial-block CV...", flush=True)
        t0 = time.time()
        out = _spatial_block_scores_with_refold(
            df_oh_train, feat_cols, label_col,
            model_factory=make_lgbm, model_name="lgbm",
            refold_hawkes=refold_hawkes,
            samples=samples, sample_block_ids=sample_block_ids, nhd=nhd, grid=grid,
            ckpt_prefix=f"{pop}__lgbm_cv",
        )
        print(f"[{pop}]   LGBM CV done in {(time.time()-t0)/60:.1f} min  "
              f"folds={len(out)}  AUC mean={out['roc_auc'].mean():.3f}", flush=True)
        return out
    lgbm_cv = _run_stage(f"{pop}__lgbm_cv", _lgbm_cv)
    fold_metrics.append(lgbm_cv)

    # --- XGBoost with spatial-block CV. v3 Phase C.1 adds it as a third base
    # learner alongside RF and LightGBM. Same per-fold checkpointing. ---
    def _xgb_cv():
        print(f"[{pop}] XGBoost spatial-block CV...", flush=True)
        t0 = time.time()
        out = _spatial_block_scores_with_refold(
            df_oh_train, feat_cols, label_col,
            model_factory=make_xgb, model_name="xgb",
            refold_hawkes=refold_hawkes,
            samples=samples, sample_block_ids=sample_block_ids, nhd=nhd, grid=grid,
            ckpt_prefix=f"{pop}__xgb_cv",
        )
        print(f"[{pop}]   XGB CV done in {(time.time()-t0)/60:.1f} min  "
              f"folds={len(out)}  AUC mean={out['roc_auc'].mean():.3f}", flush=True)
        return out
    xgb_cv = _run_stage(f"{pop}__xgb_cv", _xgb_cv)
    fold_metrics.append(xgb_cv)

    # --- Stacking: spatial-block OOF base scores, logistic-regression meta ---
    def _stacking_oof():
        print(f"[{pop}] stacking: spatial-block OOF base scores...", flush=True)
        t0 = time.time()
        out = _stacking_oof_predictions(
            df_oh_train, feat_cols, label_col, block_size_m=BLOCK_SIZE_M,
            ckpt_prefix=f"{pop}__stack_oof",
        )
        print(f"[{pop}]   stacking OOF done in {(time.time()-t0)/60:.1f} min", flush=True)
        return out
    p_rf_oof, p_lgbm_oof, p_xgb_oof, seen = _run_stage(f"{pop}__stack_oof", _stacking_oof)
    y_train = df_oh_train[label_col].to_numpy(dtype=np.int64)
    meta = _fit_stacking_meta(p_rf_oof, p_lgbm_oof, p_xgb_oof, y_train, seen)
    valid_meta = (
        seen
        & np.isfinite(p_rf_oof)
        & np.isfinite(p_lgbm_oof)
        & np.isfinite(p_xgb_oof)
    )
    if valid_meta.sum() > 0 and y_train[valid_meta].sum() > 0:
        p_stack_oof_train = meta.predict_proba(
            np.column_stack([
                p_rf_oof[valid_meta],
                p_lgbm_oof[valid_meta],
                p_xgb_oof[valid_meta],
            ])
        )[:, 1]
        stack_roc, stack_pr = _score_proba(y_train[valid_meta], p_stack_oof_train)
        fold_metrics.append(pd.DataFrame([{
            "model": "stack",
            "fold_id": -1,  # global OOF score, not a single fold
            "n_train": int(valid_meta.sum()),
            "n_test": int(valid_meta.sum()),
            "n_test_pos": int(y_train[valid_meta].sum()),
            "roc_auc": stack_roc,
            "pr_auc": stack_pr,
        }]))
        print(f"[{pop}]   stacking OOF AUC={stack_roc:.3f}  PR-AUC={stack_pr:.3f}", flush=True)

    # --- Full-data refits for whole-grid predictions ---
    def _fullfit():
        print(f"[{pop}] full-data refits + grid prediction...", flush=True)
        t0 = time.time()
        X_train_full = df_oh_train[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
        X_grid = df_oh_full[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)

        rf_full = make_rf(random_state=42)
        rf_full.fit(X_train_full, y_train)
        p_rf_grid = rf_full.predict_proba(X_grid)[:, 1]

        lgbm_full = make_lgbm(random_state=42)
        lgbm_full.fit(X_train_full, y_train)
        p_lgbm_grid = lgbm_full.predict_proba(X_grid)[:, 1]

        xgb_full = make_xgb(random_state=42)
        xgb_full.fit(X_train_full, y_train)
        p_xgb_grid = xgb_full.predict_proba(X_grid)[:, 1]

        p_stack_grid = meta.predict_proba(
            np.column_stack([p_rf_grid, p_lgbm_grid, p_xgb_grid])
        )[:, 1]
        print(f"[{pop}]   refit + predict done in {(time.time()-t0)/60:.1f} min",
              flush=True)
        return {
            "rf_full": rf_full,
            "lgbm_full": lgbm_full,
            "xgb_full": xgb_full,
            "p_rf_grid": p_rf_grid,
            "p_lgbm_grid": p_lgbm_grid,
            "p_xgb_grid": p_xgb_grid,
            "p_stack_grid": p_stack_grid,
            "X_train_full": X_train_full,
        }
    full = _run_stage(f"{pop}__fullfit", _fullfit)
    rf_full = full["rf_full"]
    lgbm_full = full["lgbm_full"]
    xgb_full = full["xgb_full"]
    p_rf_grid = full["p_rf_grid"]
    p_lgbm_grid = full["p_lgbm_grid"]
    p_xgb_grid = full["p_xgb_grid"]
    p_stack_grid = full["p_stack_grid"]
    X_train_full = full["X_train_full"]

    # --- Calibration: isotonic on the stacked score. Fall back to Platt
    # (sigmoid) when positives are too sparse for stable isotonic bins. ---
    cal_method = "isotonic" if y_train.sum() >= ISOTONIC_MIN_POSITIVES else "sigmoid"
    def _calibrate():
        print(f"[{pop}] calibration (method={cal_method}, cv={CALIBRATION_CV})...",
              flush=True)
        t0 = time.time()
        # CalibratedClassifierCV needs a base estimator. We feed it the stacked
        # base scores as a 3-column matrix (RF, LGBM, XGB) and let it wrap a
        # fresh LogisticRegression so the calibration's CV folds are the same
        # data we already meta-trained on. That re-fits the meta inside
        # calibration, which is fine: it's the same estimator family.
        base_meta = LogisticRegression(max_iter=1000)
        cv_obj = StratifiedKFold(
            n_splits=min(CALIBRATION_CV, max(2, int(y_train.sum()))),
            shuffle=True, random_state=42,
        )
        cal = CalibratedClassifierCV(base_meta, method=cal_method, cv=cv_obj)
        X_meta_train = np.column_stack([p_rf_oof, p_lgbm_oof, p_xgb_oof])
        # Replace NaN OOF rows (cells that never landed in a held-out block) with
        # the full-data refit prediction; calibration needs no NaN inputs.
        nan_oof = ~(
            np.isfinite(p_rf_oof)
            & np.isfinite(p_lgbm_oof)
            & np.isfinite(p_xgb_oof)
        )
        if nan_oof.any():
            X_meta_train[nan_oof, 0] = rf_full.predict_proba(
                X_train_full[nan_oof]
            )[:, 1]
            X_meta_train[nan_oof, 1] = lgbm_full.predict_proba(
                X_train_full[nan_oof]
            )[:, 1]
            X_meta_train[nan_oof, 2] = xgb_full.predict_proba(
                X_train_full[nan_oof]
            )[:, 1]
        if y_train.sum() < 3 * min(CALIBRATION_CV, int(y_train.sum())):
            warnings.warn(
                f"sparse positives in calibration: {int(y_train.sum())} positives "
                f"for {CALIBRATION_CV}-fold CV"
            )
        cal.fit(X_meta_train, y_train)
        p_cal_grid = cal.predict_proba(
            np.column_stack([p_rf_grid, p_lgbm_grid, p_xgb_grid])
        )[:, 1]
        print(f"[{pop}]   calibration done in {(time.time()-t0)/60:.1f} min",
              flush=True)
        return {"cal": cal, "p_cal_grid": p_cal_grid}
    cal_result = _run_stage(f"{pop}__calibrate", _calibrate)
    cal = cal_result["cal"]
    p_cal_grid = cal_result["p_cal_grid"]

    fold_df = pd.concat(fold_metrics, ignore_index=True) if fold_metrics else pd.DataFrame()

    return {
        "p_pu": p_pu_full,
        "p_rf": p_rf_grid,
        "p_lgbm": p_lgbm_grid,
        "p_xgb": p_xgb_grid,
        "p_stack": p_stack_grid,
        "p_cal": p_cal_grid,
        "fold_metrics": fold_df,
        "calibration_method": cal_method,
        "refold_hawkes": refold_hawkes,
        # Fitted estimators kept for downstream rationale (Phase I.2 SHAP) and
        # Calaveras transfer (Phase J.3). Serialized in _write_outputs via joblib.
        "rf_full": rf_full,
        "lgbm_full": lgbm_full,
        "xgb_full": xgb_full,
        "cal": cal,
        "feature_cols": list(feat_cols),
    }


# --- Output writers ------------------------------------------------------------

def _write_outputs(df: pd.DataFrame, pop: str, result: dict) -> None:
    """Write the four output artifacts for one population."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rowcol = df[["row", "col", "x", "y"]].copy()

    preds = rowcol.copy()
    preds["p_pu"] = result["p_pu"].astype(np.float32)
    preds["p_rf"] = result["p_rf"].astype(np.float32)
    preds["p_lgbm"] = result["p_lgbm"].astype(np.float32)
    preds["p_xgb"] = result["p_xgb"].astype(np.float32)
    preds["p_stack"] = result["p_stack"].astype(np.float32)
    preds_path = OUT_DIR / f"pop_predictions_{pop}_250m.parquet"
    preds.to_parquet(preds_path, index=False)
    print(f"[{pop}] wrote {preds_path}", flush=True)

    cal = rowcol.copy()
    cal["p_cal"] = result["p_cal"].astype(np.float32)
    cal_path = OUT_DIR / f"pop_calibrated_{pop}_250m.parquet"
    cal.to_parquet(cal_path, index=False)
    print(f"[{pop}] wrote {cal_path}", flush=True)

    fm = result["fold_metrics"].copy()
    fm["calibration_method"] = result["calibration_method"]
    fm["refold_hawkes"] = result["refold_hawkes"]
    fm_path = OUT_DIR / f"pop_fold_metrics_{pop}.csv"
    fm.to_csv(fm_path, index=False)
    print(f"[{pop}] wrote {fm_path}", flush=True)

    tif_3310 = OUT_DIR / f"prospectivity_placer_{pop}_250m_calibrated_3310.tif"
    tif_4326 = OUT_DIR / f"prospectivity_placer_{pop}_250m_calibrated_4326.tif"
    write_geotiff_dual_crs(
        result["p_cal"], df[["x", "y"]],
        resolution_m=RES_M, src_crs=REGION.working_crs,
        out_src=tif_3310, out_4326=tif_4326,
    )
    print(f"[{pop}] wrote {tif_3310}", flush=True)
    print(f"[{pop}] wrote {tif_4326}", flush=True)

    # Phase I.2 + Phase J.3: serialize the fitted estimators + feature column
    # list so the SHAP rationale script and the Calaveras transfer test can
    # reuse them without retraining.
    import joblib
    bundle_path = OUT_DIR / f"pop_estimator_{pop}_250m.joblib"
    joblib.dump(
        {
            "rf_full": result["rf_full"],
            "lgbm_full": result["lgbm_full"],
            "xgb_full": result["xgb_full"],
            "cal": result["cal"],
            "feature_cols": list(result["feature_cols"]),
            "population": pop,
            "calibration_method": result["calibration_method"],
            "refold_hawkes": result["refold_hawkes"],
        },
        bundle_path,
        compress=3,
    )
    print(f"[{pop}] wrote {bundle_path}", flush=True)


# --- CLI -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=("both", *POPULATIONS),
        default="both",
        help="Which placer population to train (default: both).",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=IN_FEATURES,
        help=f"Path to features parquet (default: {IN_FEATURES}).",
    )
    parser.add_argument(
        "--no-hawkes-refold",
        action="store_true",
        help="Skip per-fold Hawkes recompute. Noted as a caveat in the fold-"
             "metrics CSV; use only when wallclock is the binding constraint.",
    )
    args = parser.parse_args(argv)

    if not args.features.exists():
        print(f"ERROR: features parquet not found at {args.features}.\n"
              f"Run scripts/northern_sierra_placer_assemble_250m.py first.",
              file=sys.stderr)
        return 2

    print(f"==> Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    print(f"    cells: {len(df):,}  columns: {len(df.columns)}")

    anchor_cells = _anchor_cell_indices(df)
    print(f"==> Anchor cells excluded from training: {len(anchor_cells)}")

    refold_hawkes = not args.no_hawkes_refold
    samples = None
    sample_block_ids = None
    nhd = None
    grid = None
    if refold_hawkes:
        print("==> Loading geochem samples + NHD network for per-fold Hawkes refold")
        try:
            samples = _load_geochem_samples()
            nhd = _load_nhd_network()
            grid = build_grid(REGION.aoi, resolution_m=int(RES_M),
                              working_crs=REGION.working_crs)
            # Project samples to working CRS so block-id binning matches df.x/y.
            samples_proj = samples.to_crs(REGION.working_crs)
            sample_xy = pd.DataFrame({
                "x": samples_proj.geometry.x.to_numpy(),
                "y": samples_proj.geometry.y.to_numpy(),
            })
            sample_block_ids = _block_ids(sample_xy, BLOCK_SIZE_M)
            print(f"    samples: {len(samples):,}  NHD reaches: {len(nhd):,}")
        except FileNotFoundError as e:
            print(f"    WARNING: could not load Hawkes inputs ({e}); "
                  f"falling back to --no-hawkes-refold (caveat noted in CSV).",
                  file=sys.stderr)
            refold_hawkes = False
            samples = None
            sample_block_ids = None
            nhd = None
            grid = None

    pops = POPULATIONS if args.population == "both" else (args.population,)
    for pop in pops:
        print(f"\n========= {pop} =========")
        result = train_one_population(
            df, pop,
            anchor_cells=anchor_cells,
            refold_hawkes=refold_hawkes,
            samples=samples,
            sample_block_ids=sample_block_ids,
            nhd=nhd,
            grid=grid,
        )
        _write_outputs(df, pop, result)

    print("\n==> done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
