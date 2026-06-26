"""Paired bootstrap intervals for the new-layers lode marginal-AUC deltas.

The marginal re-baseline reports point deltas on small positive counts (lode
in-box 30, district 93, fewer once restricted to the clean mapped/flown subset).
Before anyone acts on the one positive clean-subset finding (district structure
features, +0.106 on mapped cells), it needs an interval. This recomputes the
out-of-fold predictions for the key arms on the SAME fixed folds as the driver
and bootstraps the paired AUC delta (arm minus base) over the clean subset.

Paired resampling: each replicate resamples scored-cell indices once and scores
both base and arm on the same draw, so the interval reflects the delta's own
sampling variance, not the sum of two independent AUC variances.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.newlayers_bootstrap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ai_minerals.data import dighem, nome_structure
from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof
from scripts.nome_placer.f1_leak_guarded_rebaseline import load_lode_district, load_lode_inbox
from scripts.nome_placer.newlayers_geophys_rebaseline import (
    DIGHEM_DIR, OUT_DIR, RNG, STRUCT_DIR, make_cv, sample_bands,
)

N_BOOT = 2000


def _paths(base_dir: Path, tag: str, bands: list[str], prefix: str) -> dict[str, Path]:
    return {b: base_dir / tag / f"{prefix}_{b}.tif" for b in bands}


def oof_for(loader, tag: str):
    X_df, y, coords, names, _ = loader()
    X_base = X_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]
    dig = sample_bands(_paths(DIGHEM_DIR, tag, dighem.DIGHEM_BANDS, "dighem"), ex, ny)
    st = sample_bands(_paths(STRUCT_DIR, tag, nome_structure.STRUCT_BANDS + ["gems_extent"],
                             "struct"), ex, ny)
    covered = (dig["mag_residual"] > -998.0).to_numpy()
    mapped = (st["gems_extent"] > 0.5).to_numpy()
    add = pd.concat([dig, st[nome_structure.STRUCT_BANDS]], axis=1).fillna(-999.0)
    cv, meta = make_cv(X_base, y, coords)

    def run(cols, X=X_base):
        XX = X if not cols else np.column_stack([X, add[cols].to_numpy(np.float32)])
        return spatial_cv_oof(XX.astype(np.float32), y, coords, cv,
                              model_factory=default_rf_factory(seed=RNG))

    oof = {
        "base": run([]),
        "geophys": run(dighem.DIGHEM_BANDS),
        "struct": run(nome_structure.STRUCT_BANDS),
        "geophys_struct": run(dighem.DIGHEM_BANDS + nome_structure.STRUCT_BANDS),
    }
    return y, oof, {"dighem": covered, "gems": mapped, "both": covered & mapped}


def boot_delta(y, oof_base, oof_arm, mask) -> dict:
    ok = mask & ~np.isnan(oof_base) & ~np.isnan(oof_arm)
    yy, ab, aa = y[ok], oof_base[ok], oof_arm[ok]
    if len(np.unique(yy)) < 2:
        return {"point": None}
    point = roc_auc_score(yy, aa) - roc_auc_score(yy, ab)
    rng = np.random.default_rng(RNG)
    idx = np.arange(len(yy))
    deltas = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(yy[s])) < 2:
            continue
        deltas.append(roc_auc_score(yy[s], aa[s]) - roc_auc_score(yy[s], ab[s]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"point": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_gt_0": round(float(np.mean(np.asarray(deltas) > 0)), 4),
            "n_scored": int(ok.sum()), "n_pos": int(yy.sum())}


def main() -> None:
    # arm -> which clean subset its headline delta is read on
    arm_subset = {"geophys": "dighem", "struct": "gems", "geophys_struct": "both"}
    out: dict[str, dict] = {"n_boot": N_BOOT, "datasets": {}}
    for name, loader, tag in (("lode_inbox", load_lode_inbox, "inbox_25m"),
                              ("lode_district", load_lode_district, "district_100m")):
        y, oof, masks = oof_for(loader, tag)
        rows = {}
        for arm, sub in arm_subset.items():
            rows[arm] = {"subset": sub, **boot_delta(y, oof["base"], oof[arm], masks[sub])}
            r = rows[arm]
            ci = r.get("ci95")
            print(f"[{name}] {arm:15s} on {sub:6s}: d={r.get('point')} "
                  f"95%CI={ci} P(d>0)={r.get('p_gt_0')} (n_pos={r.get('n_pos')})")
        out["datasets"][name] = rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "newlayers_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'newlayers_bootstrap.json'}")


if __name__ == "__main__":
    main()
