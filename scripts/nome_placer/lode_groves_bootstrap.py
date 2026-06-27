"""Paired bootstrap CI on the round-2 lode GATE (struct_groves, auc_gems 0.712).

The round-2 rebuild (``lode_structure_sharpen_cv``) reports the winning arm as a
point estimate: on the typed 36a labels the ``struct_groves`` arm (base + the two
Groves splay/intersection proximities + fold hinge + graphitic host) scores
auc_gems 0.712, d=+0.170 over base, the only arm past the 0.70 gate. A point
estimate on 48 in-footprint positives needs an interval before the surface is
served. This recomputes the out-of-fold predictions on the SAME fixed folds as
the driver and bootstraps the paired AUC delta (arm minus base) on the clean
GeMS-mapped subset, exactly as the round-1 newlayers bootstrap did for the
+0.106 generic-structure screen.

Reuse, not reimplementation: the positives, base features, structure bands and
fold sizing come straight from ``lode_structure_sharpen_cv`` / ``newlayers_*`` so
the bootstrap point estimate reproduces the driver's 0.712 to the dp. The paired
resampler is the round-1 ``newlayers_bootstrap.boot_delta`` unchanged (2000
resamples, 2.5/97.5 percentile CI, P(delta>0)).

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_groves_bootstrap
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from ai_minerals.data import nome_structure as ns
from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof
from scripts.nome_placer.lode_structure_sharpen_cv import (
    RNG, TEMPLATE, ensure_structure_bands, load_base, lode_positives,
)
from scripts.nome_placer.newlayers_bootstrap import N_BOOT, boot_delta
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, sample_bands

OUT_DIR = Path("data/derived/nome_placer/lode_groves_bootstrap")

# The driver's arm definitions (lode_structure_sharpen_cv.run_label_set), copied
# verbatim so the bootstrap grades the identical feature columns.
ARMS = {
    # the gate: the 0.712 winner
    "struct_groves": ns.GROVES_BANDS + ["dist_fold_hinge", "carbonaceous_host"],
    # the round-1 lineage (generic NE/NW + host), for the typed-label comparison
    "struct_generic": ns.STRUCT_BANDS,
}
# clean subset each arm's headline delta is read on, plus the alarm subset
SUBSETS = ("gems", "placer_core")


def main() -> None:
    sp = ensure_structure_bands()
    with rasterio.open(TEMPLATE) as ds:
        bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    px, py = lode_positives(bounds, "36a")
    base_df, y, coords, in_box = load_base((px, py))
    X_base = base_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    st = sample_bands(sp, ex, ny)
    mapped = (st["gems_extent"] > 0.5).to_numpy()
    add = st.fillna(-999.0)
    masks = {"gems": mapped, "placer_core": in_box}

    cv, cvmeta = make_cv(X_base, y, coords)

    def oof(cols: list[str]) -> np.ndarray:
        XX = X_base if not cols else np.column_stack([X_base, add[cols].to_numpy(np.float32)])
        return spatial_cv_oof(XX.astype(np.float32), y, coords, cv,
                              model_factory=default_rf_factory(seed=RNG))

    oof_base = oof([])
    print(f"36a labels: n={len(y)} pos={int(y.sum())} "
          f"(gems-mapped={int(y[mapped].sum())}, placer-core={int(y[in_box].sum())})  "
          f"block={cvmeta['block_size_m']:.0f}m vrange={cvmeta['variogram_range_m']:.0f}m  "
          f"n_boot={N_BOOT}")

    datasets: dict[str, dict] = {}
    for arm, cols in ARMS.items():
        oof_arm = oof(cols)
        rows = {}
        for sub in SUBSETS:
            rows[sub] = boot_delta(y, oof_base, oof_arm, masks[sub])
            r = rows[sub]
            print(f"  {arm:15s} on {sub:11s}: d={r.get('point')} "
                  f"95%CI={r.get('ci95')} P(d>0)={r.get('p_gt_0')} "
                  f"(n_scored={r.get('n_scored')}, n_pos={r.get('n_pos')})")
        datasets[arm] = rows

    out = {
        "question": "Is the round-2 lode gate (struct_groves auc_gems 0.712, d=+0.170 over "
                    "base on typed 36a labels) distinguishable from zero under a paired "
                    "bootstrap on the 48 GeMS-mapped positives?",
        "method": ("paired class-stratified bootstrap of the OOF AUC delta (arm minus base) "
                   "on identical leak-guarded folds; 2000 resamples, 95% CI = 2.5/97.5 "
                   "percentile, P(delta>0) = fraction of resamples with positive delta. "
                   "Identical to scripts/nome_placer/newlayers_bootstrap.boot_delta."),
        "n_boot": N_BOOT,
        "labels": "ARDF Cox-Singer model_code 36a (low-sulfide Au-quartz vein), typed dispersed set",
        "headline_arm": "struct_groves",
        "headline_subset": "gems",
        "scheme": {**cvmeta, "dead_zone_m": 1000.0, "n_folds": 10,
                   "estimator": "RandomForest(300, balanced, seed=42)", "fold_strategy": "contiguous"},
        "subsets": {"gems": "OOF AUC on GeMS-mapped cells (the clean structure test / gate)",
                    "placer_core": "in-box subset (restriction-of-range alarm)"},
        "datasets": datasets,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lode_groves_bootstrap.json").write_text(json.dumps(out, indent=2))
    with (OUT_DIR / "lode_groves_bootstrap.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "subset", "point_delta", "ci95_lo", "ci95_hi", "p_gt_0", "n_scored", "n_pos"])
        for arm, rows in datasets.items():
            for sub, r in rows.items():
                ci = r.get("ci95") or [None, None]
                w.writerow([arm, sub, r.get("point"), ci[0], ci[1],
                            r.get("p_gt_0"), r.get("n_scored"), r.get("n_pos")])
    print(f"\nwrote {OUT_DIR / 'lode_groves_bootstrap.json'}")
    print(f"wrote {OUT_DIR / 'lode_groves_bootstrap.csv'}")


if __name__ == "__main__":
    main()
