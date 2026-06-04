"""v3 F.3 ablation: drop hydraulic_pit_proximity_m_buffered from Tertiary.

SHAP rationale showed hydraulic_pit_proximity_m_buffered as the dominant
feature (combined mean|SHAP| 2.97M, 2.7x the next feature). The plan flagged
this case explicitly: rerun Tertiary RF + XGB CV without the feature; if
AUCs hold, the feature was incremental signal. If AUCs drop heavily, it was
carrying the leakage shortcut.

Runs RF and XGB only (skips LGBM, stack_oof, fullfit, calibrate) so the
ablation lands in ~2h instead of the 13h full pipeline. AUC comparison
against the v3 Tertiary baseline (RF=0.749, XGB=0.826) gives the leakage
measurement.

Output: data/derived/northern_sierra_placer/ablation_no_pit_proximity.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Hot-patch TERTIARY_FEATURE_COLUMNS BEFORE train_predict module imports it.
from ai_minerals import config as _config  # noqa: E402

_BASELINE_TERTIARY = _config.TERTIARY_FEATURE_COLUMNS
DROP = "hydraulic_pit_proximity_m_buffered"
_config.TERTIARY_FEATURE_COLUMNS = tuple(c for c in _BASELINE_TERTIARY if c != DROP)
print(f"baseline Tertiary features: {len(_BASELINE_TERTIARY)}", flush=True)
print(f"ablation Tertiary features: {len(_config.TERTIARY_FEATURE_COLUMNS)} (dropped {DROP})",
      flush=True)

# Import train_predict module by file path so its module-level references to
# the patched TERTIARY_FEATURE_COLUMNS resolve correctly.
_tp_spec = importlib.util.spec_from_file_location(
    "_tp", str(REPO_ROOT / "scripts" / "northern_sierra_placer_train_predict_250m.py")
)
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)

from ai_minerals.config import BLOCK_SIZE_M  # noqa: E402

# RES_M lives in train_predict_250m.py at module scope, not in config.
RES_M = _tp.RES_M

from ai_minerals.grid import build_grid  # noqa: E402
from ai_minerals.model import add_lithology_onehot, non_feature_columns  # noqa: E402
from ai_minerals.model_rf import count_feature_columns, make_rf  # noqa: E402
from ai_minerals.model_xgb import make_xgb  # noqa: E402
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER as REGION  # noqa: E402

IN_FEATURES = REPO_ROOT / "data" / "derived" / "features_northern_sierra_placer_250m.parquet"
OUT_JSON = REPO_ROOT / "data" / "derived" / "northern_sierra_placer" / "ablation_no_pit_proximity.json"
POP = "placer_tertiary"
LABEL_COL = f"is_{POP}"


def _build_oh_train_and_feats(df):
    anchor_cells = _tp._anchor_cell_indices(df)
    not_anchor = np.ones(len(df), dtype=bool)
    not_anchor[anchor_cells] = False
    df_train = df.loc[not_anchor].reset_index(drop=True)
    print(f"anchors excluded: {len(anchor_cells)}; df_train rows: {len(df_train):,}",
          flush=True)

    top_classes = df_train["lithology_class"].value_counts().head(10).index.tolist()
    extra = {}
    for col in ("major1_class", "major2_class", "major3_class"):
        if col in df_train.columns:
            extra[col] = (
                df_train[col][df_train[col] >= 0].value_counts().head(10).index.tolist()
            )
    df_oh_train = add_lithology_onehot(df_train, top_classes,
                                       extra_class_columns=extra or None)

    label_cols = ("is_placer_tertiary", "is_placer_quaternary")
    non_feat = non_feature_columns(label_cols=label_cols)
    all_feats = [c for c in df_oh_train.columns if c not in non_feat]
    drop_cnt = count_feature_columns(all_feats)
    all_feats = [c for c in all_feats if c not in drop_cnt]

    # Filter to v3 ablation Tertiary feature set
    keep = set()
    for c in _config.TERTIARY_FEATURE_COLUMNS:
        if c == "lithology_class":
            keep.update(col for col in all_feats if col.startswith("lithology_class_"))
        elif c in all_feats:
            keep.add(c)
    feat_cols = [c for c in all_feats if c in keep]
    print(f"ablation feat_cols: {len(feat_cols)}", flush=True)
    assert DROP not in feat_cols, f"{DROP} should not be in ablation feat_cols"
    return df_oh_train, feat_cols


def _run_cv(df_oh_train, feat_cols, model_factory, model_name, samples,
            sample_block_ids, nhd, grid):
    t0 = time.time()
    out = _tp._spatial_block_scores_with_refold(
        df_oh_train, feat_cols, LABEL_COL,
        model_factory=model_factory, model_name=model_name,
        refold_hawkes=True,
        samples=samples, sample_block_ids=sample_block_ids,
        nhd=nhd, grid=grid,
        ckpt_prefix=f"ablation_no_pit__{model_name}_cv",
    )
    dur_min = (time.time() - t0) / 60
    aucs = out["roc_auc"].to_numpy()
    n_pos = out["n_test_pos"].to_numpy() if "n_test_pos" in out.columns else None
    plain = float(np.mean(aucs))
    if n_pos is not None:
        mask = n_pos >= 2
        if mask.any():
            wt = float(np.sum(aucs[mask] * n_pos[mask]) / np.sum(n_pos[mask]))
        else:
            wt = float("nan")
        total_pos = int(np.sum(n_pos))
    else:
        wt = float("nan")
        total_pos = -1
    print(f"  {model_name.upper()} CV done in {dur_min:.1f} min  "
          f"folds={len(out)}  plain={plain:.3f}  pos-wtd(n>=2)={wt:.3f}  "
          f"total_pos={total_pos}", flush=True)
    return {
        "duration_min": dur_min,
        "n_folds": int(len(out)),
        "plain_mean_auc": plain,
        "pos_weighted_auc": wt,
        "total_positives": total_pos,
        "per_fold": out.to_dict(orient="records"),
    }


def main():
    print(f"==> Loading features from {IN_FEATURES}", flush=True)
    df = pd.read_parquet(IN_FEATURES)
    print(f"    cells: {len(df):,}  columns: {len(df.columns)}", flush=True)

    df_oh_train, feat_cols = _build_oh_train_and_feats(df)

    print("==> Loading geochem samples + NHD network for Hawkes refold", flush=True)
    samples = _tp._load_geochem_samples()
    nhd = _tp._load_nhd_network()
    grid = build_grid(REGION.aoi, resolution_m=int(RES_M),
                      working_crs=REGION.working_crs)
    samples_proj = samples.to_crs(REGION.working_crs)
    sample_xy = pd.DataFrame({
        "x": samples_proj.geometry.x.to_numpy(),
        "y": samples_proj.geometry.y.to_numpy(),
    })
    sample_block_ids = _tp._block_ids(sample_xy, BLOCK_SIZE_M)
    print(f"    samples: {len(samples):,}  NHD reaches: {len(nhd):,}", flush=True)

    print(f"==> RF CV (Tertiary, no pit-proximity)", flush=True)
    rf_result = _run_cv(df_oh_train, feat_cols, make_rf, "rf",
                        samples, sample_block_ids, nhd, grid)

    print(f"==> XGB CV (Tertiary, no pit-proximity)", flush=True)
    xgb_result = _run_cv(df_oh_train, feat_cols, make_xgb, "xgb",
                         samples, sample_block_ids, nhd, grid)

    summary = {
        "ablation_label": f"drop {DROP} from Tertiary",
        "ablation_feat_count": len(feat_cols),
        "ablation_feat_cols": feat_cols,
        "baseline": {
            "rf_auc_mean_v3": 0.749,
            "xgb_auc_mean_v3": 0.826,
            "stack_oof_auc_v3": 0.970,
        },
        "rf": rf_result,
        "xgb": xgb_result,
        "leakage_assessment": {
            "rf_drop_pp": 0.749 - rf_result["plain_mean_auc"],
            "xgb_drop_pp": 0.826 - xgb_result["plain_mean_auc"],
            "interpretation": (
                "Each pp of drop is roughly the marginal contribution of "
                "hydraulic_pit_proximity_m_buffered to that model's AUC. A "
                "small drop (<0.05) suggests the feature was incremental "
                "signal; a large drop (>0.15) suggests it was a leakage "
                "shortcut. Intermediate values are case-by-case."
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n==> wrote {OUT_JSON}", flush=True)
    print(f"\nRF  baseline=0.749  ablation={rf_result['plain_mean_auc']:.3f}  "
          f"drop={0.749 - rf_result['plain_mean_auc']:+.3f}", flush=True)
    print(f"XGB baseline=0.826  ablation={xgb_result['plain_mean_auc']:.3f}  "
          f"drop={0.826 - xgb_result['plain_mean_auc']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
