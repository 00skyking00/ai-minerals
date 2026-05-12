"""Pseudo-negative sampling, spatial block cross-validation, baseline model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Identity + helper columns present in every region's feature frame.
IDENTITY_COLUMNS = frozenset({
    "row", "col", "x", "y",
    "any_mineral_occurrence",
    "lithology_class",   # handled specially as one-hot
    # v3.1: SGMC MAJOR1/2/3 fine-grained lithology codes. Same handling
    # as `lithology_class`: integer codes are excluded from raw
    # features, and `add_lithology_onehot` expands them to one-hots.
    "major1_class",
    "major2_class",
    "major3_class",
})


def non_feature_columns(label_cols: tuple[str, ...] = ("is_porphyry", "is_porphyry_strict")) -> frozenset[str]:
    """Return the set of columns to exclude from the feature matrix.

    Identity/helper columns + whatever label columns the region defines.
    Defaults to v1's porphyry labels so existing callers keep working.
    """
    return IDENTITY_COLUMNS | set(label_cols)


# Back-compat: the v1 NON_FEATURE_COLUMNS set. Notebook cells and tests
# that import this name continue to work; new code should call
# `non_feature_columns(label_cols)` instead.
NON_FEATURE_COLUMNS = non_feature_columns()


def feature_columns(df: pd.DataFrame, label_cols: tuple[str, ...] = ("is_porphyry", "is_porphyry_strict")) -> list[str]:
    non_feat = non_feature_columns(label_cols)
    return [c for c in df.columns if c not in non_feat]


def _exclusion_mask(
    df: pd.DataFrame, *, exclusion_radius_m: float, label_col: str = "is_porphyry"
) -> np.ndarray:
    """Return a boolean mask of cells to EXCLUDE from pseudo-negatives.

    Cells within `exclusion_radius_m` of any known mineral occurrence or deposit
    positive are excluded to avoid contaminating the negative class with nearby
    unknown deposits.
    """
    occ = df.loc[df["any_mineral_occurrence"] == 1, ["x", "y"]].to_numpy()
    por = df.loc[df[label_col] == 1, ["x", "y"]].to_numpy()
    exclude_pts = np.vstack([occ, por]) if len(por) else occ
    tree = cKDTree(exclude_pts)
    xy = df[["x", "y"]].to_numpy()
    d, _ = tree.query(xy, k=1)
    return d < exclusion_radius_m


def sample_pseudo_negatives(
    df: pd.DataFrame,
    *,
    n_per_positive: int = 30,
    exclusion_radius_m: float = 5000.0,
    stratify_by: str = "lithology_class",
    random_state: int = 42,
    label_col: str = "is_porphyry",
) -> pd.DataFrame:
    """Return a sub-frame of pseudo-negative cells, stratified by lithology.

    For each lithology class that contains positives, sample pseudo-negatives
    from the same class (proportional to each class's share of positives),
    excluding cells within `exclusion_radius_m` of any known occurrence.
    """
    rng = np.random.default_rng(random_state)
    positives = df[df[label_col] == 1]
    n_target = int(n_per_positive * len(positives))

    exclude = _exclusion_mask(df, exclusion_radius_m=exclusion_radius_m, label_col=label_col)
    candidates = df[~exclude].copy()
    print(f"  [pseudo-neg] {exclude.sum():,} cells excluded (< {exclusion_radius_m/1000:g} km "
          f"from any occurrence); {len(candidates):,} candidates remain")

    # Proportional allocation by lithology class of the positives.
    pos_by_class = positives[stratify_by].value_counts(normalize=True)
    out_parts = []
    for cls, frac in pos_by_class.items():
        n_here = max(1, int(round(n_target * frac)))
        pool = candidates[candidates[stratify_by] == cls]
        if len(pool) == 0:
            continue
        take = min(n_here, len(pool))
        chosen = pool.sample(take, random_state=rng.integers(0, 2**32 - 1))
        out_parts.append(chosen)
    negs = pd.concat(out_parts, ignore_index=True)
    print(f"  [pseudo-neg] drew {len(negs):,} negatives across {len(out_parts)} lithology classes")
    return negs


def add_lithology_onehot(
    df: pd.DataFrame,
    top_classes: list[int],
    *,
    extra_class_columns: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Expand lithology_class into one-hot columns for the top-N classes + 'other'.

    If `extra_class_columns` is given, also one-hot-encode each named
    integer column with its own top-N list. Used by v3.1 to encode
    SGMC MAJOR1/2/3 fine-grained lithology alongside the existing
    `lithology_class` one-hot.
    """
    out = df.copy()
    for c in top_classes:
        out[f"litho_{int(c)}"] = (df["lithology_class"] == c).astype(np.uint8)
    out["litho_other"] = (~df["lithology_class"].isin(top_classes)).astype(np.uint8)
    if extra_class_columns:
        for col, classes in extra_class_columns.items():
            short = col.replace("_class", "")
            for c in classes:
                out[f"{short}_{int(c)}"] = (df[col] == c).astype(np.uint8)
            out[f"{short}_other"] = (~df[col].isin(classes)).astype(np.uint8)
    return out


def build_training_set(
    df: pd.DataFrame,
    top_classes: list[int],
    *,
    n_per_positive: int = 30,
    exclusion_radius_m: float = 5000.0,
    random_state: int = 42,
    label_col: str = "is_porphyry",
    label_cols: tuple[str, ...] = ("is_porphyry", "is_porphyry_strict"),
    extra_class_top_n: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble (X, y) for model training: positives + pseudo-negatives with one-hot lith.

    `label_col` selects which binary label column is the training target
    (one of `label_cols`). `label_cols` is the full set of label columns
    that should be excluded from the feature matrix.
    """
    pos = df[df[label_col] == 1].copy()
    neg = sample_pseudo_negatives(
        df,
        n_per_positive=n_per_positive,
        exclusion_radius_m=exclusion_radius_m,
        random_state=random_state,
        label_col=label_col,
    )
    combined = pd.concat([pos, neg], ignore_index=True)

    # v3.1: auto-detect major1/2/3 class columns and compute their top-N
    # from the combined training rows. If absent, the dict is empty and
    # `add_lithology_onehot` falls back to single-column behavior.
    extra: dict[str, list[int]] = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in combined.columns:
            top_majors = (
                combined[col][combined[col] >= 0]
                .value_counts()
                .head(extra_class_top_n)
                .index.tolist()
            )
            extra[col] = top_majors

    combined = add_lithology_onehot(combined, top_classes, extra_class_columns=extra or None)
    y = combined[label_col].to_numpy(dtype=np.int64)
    non_feat = non_feature_columns(label_cols)
    X = combined.drop(
        columns=[c for c in combined.columns if c in non_feat]
        + ["lithology_class"]
    )
    # NaN will be imputed inside the model pipeline.
    return X, y


@dataclass
class SpatialBlockCV:
    """Simple geographic block CV: split cells into KxK blocks, hold one block out at a time.

    block_size_m : edge length of each spatial block in meters. At 20 km, EastAK
                   (~220 x 300 km) gives ~15 blocks -> usable folds.
    """
    block_size_m: float = 20_000.0

    def split(self, rows: pd.DataFrame):
        bx = (rows["x"].to_numpy() // self.block_size_m).astype(int)
        by = (rows["y"].to_numpy() // self.block_size_m).astype(int)
        # Assign block id as a stable pair.
        block_ids = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
        unique = np.unique(block_ids)
        for held_out in unique:
            train = block_ids != held_out
            test = block_ids == held_out
            if test.sum() == 0 or train.sum() == 0:
                continue
            yield np.where(train)[0], np.where(test)[0], int(held_out)


def make_baseline_pipeline() -> Pipeline:
    """Standardized logistic regression with median imputation for NaN features."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def spatial_block_scores(
    X: pd.DataFrame, y: np.ndarray, rows: pd.DataFrame, *, block_size_m: float = 20_000.0
) -> pd.DataFrame:
    """Run spatial-block CV with logistic regression; return per-fold AUC/AP."""
    cv = SpatialBlockCV(block_size_m=block_size_m)
    results = []
    for train_idx, test_idx, block_id in cv.split(rows):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if y_test.sum() == 0 or y_train.sum() == 0:
            continue  # skip folds with no positives on either side
        pipe = make_baseline_pipeline()
        pipe.fit(X.iloc[train_idx], y_train)
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
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


def success_rate_curve(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative fraction of positives captured vs cumulative area flagged, at
    decreasing score thresholds.

    Returns (frac_area, frac_deposits) — standard MPM success-rate pair.
    """
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    cum_pos = np.cumsum(sorted_y)
    total_pos = max(int(y_true.sum()), 1)
    frac_area = np.arange(1, len(scores) + 1) / len(scores)
    frac_dep = cum_pos / total_pos
    return frac_area, frac_dep
