"""Phase J: within-AOI N/S geographic-split test for the Sierra placer model.

Trains the Phase 2 stacking ensemble twice — once on the N half of the AOI,
scoring the S half; once on the S half, scoring the N. Reports per-side
capture rates on anchor cells and compares to the in-AOI Phase G baseline.

Decision rule (per `~/.claude/plans/hazy-humming-lynx.md`, Phase L):
  - both deltas within 5pp of baseline                 -> clean
  - 5–10pp                                             -> caveat
  - >10pp                                              -> flag overfit

The N/S threshold is 39.0°N. The northern-Sierra hydraulic-mining
districts cluster between 39.0–39.4°N, so this split is genuinely
hard on the model: training on one side strips out a chunk of the
positive support the other side will then be scored against.

Output: data/derived/northern_sierra_placer/north_south_split_results.csv
columns: training_side, predicted_side, population, n_anchors_in_predicted,
         capture_1pct, capture_5pct, capture_10pct,
         capture_1pct_ci_lo, capture_1pct_ci_hi,
         capture_5pct_ci_lo, capture_5pct_ci_hi,
         capture_10pct_ci_lo, capture_10pct_ci_hi,
         baseline_5pct, delta_5pct,
         anchors_in_top_decile

Usage:
    .venv/bin/python scripts/northern_sierra_placer/north_south_split.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ai_minerals.metrics import bootstrap_capture_ci
from ai_minerals.model import (
    SpatialBlockCV,
    add_lithology_onehot,
    non_feature_columns,
)
from ai_minerals.model_lgbm import make_lgbm
from ai_minerals.model_pu import fit_pu_bagging
from ai_minerals.model_rf import count_feature_columns, make_rf
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
RES_M = 250.0
BLOCK_SIZE_M = 20_000.0
N_PU_BAGS = 30
ISOTONIC_MIN_POSITIVES = 30
CALIBRATION_CV = 5

POPULATIONS = ("placer_tertiary", "placer_quaternary")

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
IN_FEATURES = DATA_DERIVED / f"features_{REGION.data_prefix}_250m.parquet"
OUT_DIR = DATA_DERIVED / REGION.data_prefix
OUT_CSV = OUT_DIR / "north_south_split_results.csv"

# Phase G's per-pop anchor comparison table (used to source the baseline
# capture-at-5%). Absent unless Phase G has been run.
IN_BASELINE_ANCHOR_TABLE = OUT_DIR / "anchor_districts_decile_table.csv"

# Phase G's phase1_vs_phase2_comparison.csv carries Brier / F1 / kappa /
# recall under the top-decile cutoff but not capture-at-k%. We compute the
# baseline capture-at-5% from the Phase E calibrated parquet directly when
# it exists, falling back to a "baseline unavailable" note otherwise.
IN_BASELINE_TERTIARY = OUT_DIR / "pop_calibrated_placer_tertiary_250m.parquet"
IN_BASELINE_QUATERNARY = OUT_DIR / "pop_calibrated_placer_quaternary_250m.parquet"

# Latitude threshold splitting the AOI into N and S halves.
SPLIT_LAT_DEG = 39.0


def _anchor_cell_indices(df: pd.DataFrame) -> np.ndarray:
    """Snap each anchor district (lon, lat) to its nearest grid-cell row index."""
    transformer = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    idxs: list[int] = []
    for _name, (lon, lat) in ANCHOR_DISTRICTS.items():
        ax, ay = transformer.transform(lon, lat)
        d2 = (xs - ax) ** 2 + (ys - ay) ** 2
        idxs.append(int(np.argmin(d2)))
    return np.unique(np.array(idxs, dtype=np.int64))


def _split_y_threshold(df: pd.DataFrame, split_lat_deg: float) -> float:
    """Project `split_lat_deg` at the AOI centroid longitude into the working CRS y.

    Cells with df["y"] >= threshold are in the north half.
    """
    centroid_lon = 0.5 * (REGION.aoi.min_lon + REGION.aoi.max_lon)
    transformer = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    _x_at_split, y_at_split = transformer.transform(centroid_lon, split_lat_deg)
    return float(y_at_split)


def _decile_rank(score: pd.Series) -> pd.Series:
    """Decile of each cell in score's distribution; 0 = top decile."""
    return pd.qcut(
        score.rank(method="first", ascending=False, pct=False, na_option="keep"),
        q=10,
        labels=range(10),
    )


def _train_side_and_predict(
    df: pd.DataFrame,
    side_mask_train: np.ndarray,
    side_mask_predict: np.ndarray,
    pop: str,
    anchor_cells: np.ndarray,
) -> np.ndarray:
    """Train stack on `side_mask_train`, predict on `side_mask_predict`.

    Returns a calibrated-probability array aligned to df.index for the
    predict-side cells only; cells outside `side_mask_predict` get NaN.

    Anchor cells are masked out of training (same discipline as the main
    train_predict_250m driver). Stacking ensemble: PU bag + RF + LGBM
    spatial-block OOF -> logistic-regression meta -> isotonic calibration
    (Platt fallback when positives < 30).
    """
    label_col = f"is_{pop}"

    not_anchor = np.ones(len(df), dtype=bool)
    not_anchor[anchor_cells] = False
    train_mask = side_mask_train & not_anchor

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_predict = df.loc[side_mask_predict].reset_index(drop=True)

    n_pos = int(df_train[label_col].sum())
    print(f"    train side: {len(df_train):,} cells, {n_pos:,} positives "
          f"({pop})", flush=True)
    if n_pos == 0:
        print(f"    WARN: zero positives on training side for {pop}; "
              f"returning all-NaN predictions.", flush=True)
        out = np.full(len(df), np.nan, dtype=np.float64)
        return out

    # Lithology one-hot (same logic as train_predict_250m).
    top_classes = df_train["lithology_class"].value_counts().head(10).index.tolist()
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df_train.columns:
            extra[col] = (
                df_train[col][df_train[col] >= 0].value_counts().head(10).index.tolist()
            )
    df_oh_train = add_lithology_onehot(df_train, top_classes,
                                       extra_class_columns=extra or None)
    df_oh_predict = add_lithology_onehot(df_predict, top_classes,
                                         extra_class_columns=extra or None)

    label_cols = tuple(f"is_{p}" for p in POPULATIONS)
    non_feat = non_feature_columns(label_cols=label_cols)
    feat_cols = [c for c in df_oh_train.columns if c not in non_feat]
    drop = count_feature_columns(feat_cols)
    feat_cols = [c for c in feat_cols if c not in drop]

    # Ensure the predict frame has every training feature column. (Top
    # lithology classes computed from train side may differ from predict
    # side; missing columns fall back to 0 just like Phase 5 transfer.)
    for c in feat_cols:
        if c not in df_oh_predict.columns:
            df_oh_predict[c] = 0.0

    X_train = df_oh_train[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    X_predict = df_oh_predict[feat_cols].fillna(-9999.0).to_numpy(dtype=np.float32)
    y_train = df_oh_train[label_col].to_numpy(dtype=np.int64)

    # --- PU bag (informational only here; not fed into the stack at this
    # scale, to keep the script's wallclock budget bounded).
    t0 = time.time()
    _p_pu_train, _ = fit_pu_bagging(
        df_oh_train, top_classes,
        label_col=label_col, n_bags=N_PU_BAGS, random_state=42,
    )
    print(f"      PU bag done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # --- Spatial-block OOF for RF + LGBM (drives the stacking meta).
    cv = SpatialBlockCV(block_size_m=BLOCK_SIZE_M)
    p_rf_oof = np.full(len(df_oh_train), np.nan, dtype=np.float64)
    p_lgbm_oof = np.full(len(df_oh_train), np.nan, dtype=np.float64)
    seen = np.zeros(len(df_oh_train), dtype=bool)
    t0 = time.time()
    for train_idx, test_idx, _block_id in cv.split(df_oh_train):
        if y_train[train_idx].sum() == 0:
            continue
        rf = make_rf(random_state=42)
        rf.fit(X_train[train_idx], y_train[train_idx])
        p_rf_oof[test_idx] = rf.predict_proba(X_train[test_idx])[:, 1]
        lgbm = make_lgbm(random_state=42)
        lgbm.fit(X_train[train_idx], y_train[train_idx])
        p_lgbm_oof[test_idx] = lgbm.predict_proba(X_train[test_idx])[:, 1]
        seen[test_idx] = True
    print(f"      spatial-block OOF done ({(time.time()-t0)/60:.1f} min)", flush=True)

    valid_meta = seen & np.isfinite(p_rf_oof) & np.isfinite(p_lgbm_oof)
    if valid_meta.sum() == 0 or y_train[valid_meta].sum() == 0:
        print(f"    WARN: no usable OOF rows; falling back to RF-only "
              f"calibrated score for {pop}.", flush=True)

    # --- Full-data refits for predict-side scoring.
    rf_full = make_rf(random_state=42)
    rf_full.fit(X_train, y_train)
    lgbm_full = make_lgbm(random_state=42)
    lgbm_full.fit(X_train, y_train)
    p_rf_predict = rf_full.predict_proba(X_predict)[:, 1]
    p_lgbm_predict = lgbm_full.predict_proba(X_predict)[:, 1]

    # --- Meta + calibration. Platt-fallback when positives are sparse.
    cal_method = "isotonic" if y_train.sum() >= ISOTONIC_MIN_POSITIVES else "sigmoid"
    n_splits = min(CALIBRATION_CV, max(2, int(y_train.sum())))
    cv_obj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    base_meta = LogisticRegression(max_iter=1000)
    cal = CalibratedClassifierCV(base_meta, method=cal_method, cv=cv_obj)
    X_meta_train = np.column_stack([p_rf_oof, p_lgbm_oof])
    nan_oof = ~(np.isfinite(p_rf_oof) & np.isfinite(p_lgbm_oof))
    if nan_oof.any():
        X_meta_train[nan_oof, 0] = rf_full.predict_proba(
            X_train[nan_oof]
        )[:, 1]
        X_meta_train[nan_oof, 1] = lgbm_full.predict_proba(
            X_train[nan_oof]
        )[:, 1]
    cal.fit(X_meta_train, y_train)
    p_cal_predict = cal.predict_proba(
        np.column_stack([p_rf_predict, p_lgbm_predict])
    )[:, 1]

    out = np.full(len(df), np.nan, dtype=np.float64)
    out[side_mask_predict] = p_cal_predict
    return out


def _baseline_capture_5pct(df: pd.DataFrame, pop: str,
                            anchor_cells: np.ndarray) -> float | None:
    """Capture-at-top-5% on the full-AOI Phase E calibrated raster for `pop`.

    Returns None if the baseline parquet is absent (Phase E not yet run).
    """
    path = IN_BASELINE_TERTIARY if pop == "placer_tertiary" else IN_BASELINE_QUATERNARY
    if not path.exists():
        return None
    cal = pd.read_parquet(path)
    merged = df[["row", "col"]].merge(
        cal[["row", "col", "p_cal"]], on=["row", "col"], how="left",
    )
    scores = merged["p_cal"].to_numpy(dtype=np.float64)
    finite = np.isfinite(scores)
    if finite.sum() == 0:
        return None
    pos = np.zeros(len(df), dtype=bool)
    pos[anchor_cells] = True
    result = bootstrap_capture_ci(
        scores[finite], pos[finite], ks_percent=(5.0,), n_resamples=200,
    )
    point, _lo, _hi = result[5.0]
    return float(point)


def _capture_table(
    df: pd.DataFrame,
    scores: np.ndarray,
    predicted_side_mask: np.ndarray,
    anchor_cells: np.ndarray,
    *,
    training_side: str,
    predicted_side: str,
    population: str,
    baseline_5pct: float | None,
) -> dict[str, object]:
    """Compute capture rates + decile-rank counts for one (train, predict) pair."""
    finite = predicted_side_mask & np.isfinite(scores)
    s_finite = scores[finite]
    anchor_side = np.zeros(len(df), dtype=bool)
    anchor_side[anchor_cells] = True
    anchor_side &= predicted_side_mask
    pos_finite = anchor_side[finite]
    n_anchors = int(pos_finite.sum())

    # Capture rates (with bootstrap 95% CIs).
    if n_anchors > 0 and finite.sum() > 0:
        ci = bootstrap_capture_ci(
            s_finite, pos_finite, ks_percent=(1.0, 5.0, 10.0),
            n_resamples=500, seed=42,
        )
        c1, c1_lo, c1_hi = ci[1.0]
        c5, c5_lo, c5_hi = ci[5.0]
        c10, c10_lo, c10_hi = ci[10.0]
    else:
        c1 = c5 = c10 = float("nan")
        c1_lo = c1_hi = c5_lo = c5_hi = c10_lo = c10_hi = float("nan")

    # Anchors-in-top-decile on the predicted side's score distribution.
    s_series = pd.Series(scores, index=df.index)
    deciles = _decile_rank(s_series[predicted_side_mask])
    anchor_idx_on_side = np.flatnonzero(anchor_side)
    n_in_top = 0
    for idx in anchor_idx_on_side:
        d = deciles.get(idx)
        if pd.notna(d) and int(d) == 0:
            n_in_top += 1

    delta_5 = (c5 - baseline_5pct) if (baseline_5pct is not None
                                       and np.isfinite(c5)) else float("nan")

    return {
        "training_side": training_side,
        "predicted_side": predicted_side,
        "population": population,
        "n_anchors_in_predicted": n_anchors,
        "capture_1pct": c1,
        "capture_1pct_ci_lo": c1_lo,
        "capture_1pct_ci_hi": c1_hi,
        "capture_5pct": c5,
        "capture_5pct_ci_lo": c5_lo,
        "capture_5pct_ci_hi": c5_hi,
        "capture_10pct": c10,
        "capture_10pct_ci_lo": c10_lo,
        "capture_10pct_ci_hi": c10_hi,
        "baseline_5pct": (baseline_5pct if baseline_5pct is not None
                          else float("nan")),
        "delta_5pct": delta_5,
        "anchors_in_top_decile": n_in_top,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=IN_FEATURES,
        help=f"Path to features parquet (default: {IN_FEATURES}).",
    )
    parser.add_argument(
        "--split-lat-deg", type=float, default=SPLIT_LAT_DEG,
        help=f"Latitude threshold for the N/S split (default: {SPLIT_LAT_DEG}).",
    )
    parser.add_argument(
        "--populations", nargs="+", default=list(POPULATIONS),
        choices=list(POPULATIONS),
        help="Which populations to score (default: both).",
    )
    args = parser.parse_args(argv)

    if not args.features.exists():
        print(f"ERROR: features parquet not found at {args.features}.\n"
              f"Run scripts/northern_sierra_placer/assemble_250m.py first.",
              file=sys.stderr)
        return 2

    print(f"==> Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    print(f"    cells: {len(df):,}  columns: {len(df.columns)}")

    anchor_cells = _anchor_cell_indices(df)
    print(f"==> Anchor cells (held out of training): {len(anchor_cells)}")

    y_threshold = _split_y_threshold(df, args.split_lat_deg)
    print(f"==> Split: lat={args.split_lat_deg}°N -> y={y_threshold:.1f} m "
          f"({REGION.working_crs})")
    north_mask = df["y"].to_numpy() >= y_threshold
    south_mask = ~north_mask
    print(f"    north: {int(north_mask.sum()):,} cells   "
          f"south: {int(south_mask.sum()):,} cells")

    anchor_north = int(np.isin(anchor_cells, np.flatnonzero(north_mask)).sum())
    anchor_south = int(np.isin(anchor_cells, np.flatnonzero(south_mask)).sum())
    print(f"    anchors in N: {anchor_north}   anchors in S: {anchor_south}")

    rows: list[dict[str, object]] = []
    for pop in args.populations:
        print(f"\n========= {pop} =========")

        baseline_5 = _baseline_capture_5pct(df, pop, anchor_cells)
        if baseline_5 is None:
            print(f"  WARN: baseline {IN_BASELINE_TERTIARY.parent.name}/"
                  f"pop_calibrated_{pop}_250m.parquet not found. "
                  f"Run Phase E first for a baseline capture-at-5%; "
                  f"delta_5pct will be NaN.", flush=True)
        else:
            print(f"  baseline capture@5% (full-AOI Phase E): {baseline_5:.3f}",
                  flush=True)

        n_pos_full = int(df[f"is_{pop}"].sum())
        n_pos_n = int(df.loc[north_mask, f"is_{pop}"].sum())
        n_pos_s = int(df.loc[south_mask, f"is_{pop}"].sum())
        print(f"  positives — full: {n_pos_full:,}  N: {n_pos_n:,}  "
              f"S: {n_pos_s:,}")

        # Train N -> predict S.
        print(f"\n  -- train N, predict S --")
        t0 = time.time()
        s_pred_s = _train_side_and_predict(
            df, north_mask, south_mask, pop, anchor_cells,
        )
        print(f"    finished in {(time.time()-t0)/60:.1f} min", flush=True)
        rows.append(_capture_table(
            df, s_pred_s, south_mask, anchor_cells,
            training_side="N", predicted_side="S",
            population=pop, baseline_5pct=baseline_5,
        ))

        # Train S -> predict N.
        print(f"\n  -- train S, predict N --")
        t0 = time.time()
        s_pred_n = _train_side_and_predict(
            df, south_mask, north_mask, pop, anchor_cells,
        )
        print(f"    finished in {(time.time()-t0)/60:.1f} min", flush=True)
        rows.append(_capture_table(
            df, s_pred_n, north_mask, anchor_cells,
            training_side="S", predicted_side="N",
            population=pop, baseline_5pct=baseline_5,
        ))

    out_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n==> wrote {OUT_CSV}")

    # Summary print.
    print("\n=== Summary ===")
    for _, r in out_df.iterrows():
        c5 = r["capture_5pct"]
        lo = r["capture_5pct_ci_lo"]
        hi = r["capture_5pct_ci_hi"]
        b = r["baseline_5pct"]
        d = r["delta_5pct"]
        if pd.isna(b):
            verdict = "baseline unavailable"
        else:
            abs_d = abs(d) if pd.notna(d) else float("nan")
            if pd.isna(abs_d):
                verdict = "NaN delta"
            elif abs_d <= 0.05:
                verdict = "clean"
            elif abs_d <= 0.10:
                verdict = "caveat"
            else:
                verdict = "OVERFIT FLAG"
        print(f"  train={r['training_side']} -> predict={r['predicted_side']}  "
              f"{r['population']:18s}  cap@5%={c5:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"baseline={b if not pd.isna(b) else 'NA'}  "
              f"delta={d if not pd.isna(d) else 'NA'}  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
