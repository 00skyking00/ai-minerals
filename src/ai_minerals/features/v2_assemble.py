"""Phase 2 supervised-model feature assembler for the Nome placer.

Loads bearcub's labelled drillholes (per-hole pay_zone_average grade,
pay_binary, bottom_of_pay_ft, surface_mineable, deposit_class) and
joins each hole's row with feature values sampled from the v1.5 v2
prospectivity raster + surface geology + DEM.

The resulting DataFrame is the input to Phase 2 model heads:

- Binary classifier (target: ``pay_binary``).
- Grade regressor (target: ``pay_zone_average_oz_cuyd``).
- Depth-to-base-of-pay regressor (target: ``bottom_of_pay_ft``).
- Surface-mineable binary (target: ``surface_mineable``).

Bearcub 2026-06-16 Janin re-grade is absorbed: grades are pay-zone,
not whole-column. ``pay_zone_grade_wholecol_oz_cuyd`` is the bracket
floor and is included as a separate column for the chapter's reserve
table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# Column order in the v1.5 v2 prospectivity raster
PROSPECTIVITY_BAND_KEYS = ("bl", "ap", "tb", "bc", "qm", "buried_bl", "composite")


@dataclass(frozen=True)
class V2FeatureFrame:
    """Phase 2 feature + label table, one row per drillhole."""
    holes: pd.DataFrame      # all rows, joined features + labels
    feature_columns: list[str]   # column names for X (in order)
    target_columns: dict[str, str]  # head_name -> column_name

    def X(self, mask: pd.Series | None = None) -> pd.DataFrame:
        """Feature matrix; optional mask drops rows."""
        df = self.holes if mask is None else self.holes[mask]
        return df[self.feature_columns]

    def y(self, head: str, mask: pd.Series | None = None) -> pd.Series:
        """Target column for a named head."""
        col = self.target_columns[head]
        df = self.holes if mask is None else self.holes[mask]
        return df[col]


def _sample_raster(
    raster_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    band_keys: tuple[str, ...] = PROSPECTIVITY_BAND_KEYS,
) -> pd.DataFrame:
    """Sample a multi-band raster at lat/lon points; return one column per band."""
    with rasterio.open(raster_path) as src:
        n_bands = src.count
        if n_bands != len(band_keys):
            raise ValueError(
                f"raster has {n_bands} bands, expected {len(band_keys)} "
                f"for keys {band_keys}"
            )
        samples = list(src.sample(list(zip(lons, lats))))
    arr = np.asarray(samples, dtype=np.float32)
    return pd.DataFrame(arr, columns=[f"raster_{k}" for k in band_keys])


def assemble(
    drillholes_path: Path,
    prospectivity_raster_path: Path,
) -> V2FeatureFrame:
    """Assemble Phase 2 X+y table from labelled drillholes + the v1.5 v2 raster.

    Per-hole feature stack:

    - 7 prospectivity bands from the raster (bl / ap / tb / bc / qm /
      buried_bl / composite).
    - The hole's own ``bedrock_depth_ft`` (a real measurement at each
      hole; supervised heads can learn the relationship between
      bedrock depth and pay).
    - The hole's ``surface_class`` (already present from bearcub's
      PDF 94-39 tagging; encoded as a categorical).

    Labels carried through verbatim from bearcub:

    - ``pay_zone_average_oz_cuyd`` (pay-zone grade per 2026-06-16
      Janin regrade).
    - ``pay_zone_grade_wholecol_oz_cuyd`` (bracket floor; nullable).
    - ``pay_binary``.
    - ``bottom_of_pay_ft`` (nullable).
    - ``surface_mineable`` (nullable bool).
    - ``deposit_class``.
    """
    holes = pd.read_parquet(drillholes_path)

    # Sample the prospectivity raster
    raster_features = _sample_raster(
        prospectivity_raster_path,
        holes["lon"].to_numpy(),
        holes["lat"].to_numpy(),
    )
    holes = holes.reset_index(drop=True).join(raster_features)

    # Encode surface_class as an integer code; preserve raw string in
    # surface_class for inspection.
    holes["surface_class_code"] = (
        holes["surface_class"].astype("category").cat.codes.astype(np.int32)
    )

    feature_columns: list[str] = [
        f"raster_{k}" for k in PROSPECTIVITY_BAND_KEYS
    ] + [
        "bedrock_depth_ft",
        "surface_class_code",
    ]

    target_columns = {
        "pay_binary": "pay_binary",
        "grade": "pay_zone_average_oz_cuyd",
        "grade_floor": "pay_zone_grade_wholecol_oz_cuyd",
        "bottom_of_pay": "bottom_of_pay_ft",
        "surface_mineable": "surface_mineable",
        "deposit_class": "deposit_class",
    }

    return V2FeatureFrame(
        holes=holes,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


def claim_ms_kfold_split(
    holes: pd.DataFrame,
    n_folds: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate a leave-one-claim-MS-out (or k-fold by MS) split.

    Spatial-block CV for the Nome placer model is by patented claim
    MS. Holes drilled into the same claim share spatial autocorrelation
    that a random split leaks. Default n_folds = number of distinct
    MS values present (true LOO).

    Yields (train_idx, test_idx) per fold. Holes with null
    ``claim_ms`` get an implicit "unknown" group that becomes one
    fold; supervised heads can choose to drop or weight them.
    """
    ms_series = holes["claim_ms"].fillna("__unknown__").astype("string")
    ms_groups = ms_series.unique()
    if n_folds is None:
        n_folds = len(ms_groups)
    if n_folds <= 0:
        raise ValueError(f"n_folds must be > 0; got {n_folds}")

    # Round-robin assign MS groups to folds for k-fold; LOO when k = n_groups
    fold_of_ms = {ms: i % n_folds for i, ms in enumerate(ms_groups)}
    fold_of_hole = ms_series.map(fold_of_ms).to_numpy()

    splits = []
    indices = np.arange(len(holes))
    for k in range(n_folds):
        test_mask = fold_of_hole == k
        if not test_mask.any():
            continue
        splits.append((indices[~test_mask], indices[test_mask]))
    return splits
