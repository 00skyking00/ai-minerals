"""Phase 3: DevNet vs GBM on the Lawley H3 cube, same 2-D blocks.

Phase 1 reproduced Lawley's 0.983 AUC. Phase 1b corrected a 2.4-pp
label leak. Phase 2 added 2-D spatial-block CV (0.868) and
cross-continent transfer (0.557-0.709). Phase 3 asks whether DevNet —
the deep-anomaly-detection architecture from Pang 2019 + DEEP-SEAM
Luo 2026 — beats the GBM on the same leak-corrected, 2-D-blocked
benchmark.

Our cross-region work earlier
([research/cross_region_methodology_findings_2026-05.md](../research/cross_region_methodology_findings_2026-05.md))
showed DevNet only beats RF on the Curnamona REE dataset (tiny
positive set + heavy upstream PCA + GLCM preprocessing). On Tanacross
(45 pos) and Arizona (191 pos), DevNet trailed RF on regional MPM
regardless of feature engineering. Phase 3 is the same test on
Lawley's continental scale (2,027 positives across NA + Aus).

Same 4-cell evaluation as Phase 2:
1. 1-D latitude-band CV (Lawley's published scheme)
2. 2-D quantile-block CV (12 × 12 lat × lon → 6 folds)
3. USCAN → Aus transfer
4. Aus → USCAN transfer

Output:
  data/derived/lawley/path3_devnet_metrics.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/sky/src/learning/ai-minerals")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lawley_phase2_eval import (  # noqa: E402
    build_feature_frame, assign_1d_lat_bands, assign_2d_blocks,
    bootstrap_metrics, DROP_BEFORE_FIT, TOP_K_PCT, N_BOOT,
)
from ai_minerals.model_devnet import fit_devnet, DevNetConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "derived" / "lawley"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_devnet_on_split(
    tr_df: pd.DataFrame, te_df: pd.DataFrame, feature_cols: list[str],
    *, n_epochs: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit DevNet on tr_df, return (proba_train, proba_test)."""
    t1 = time.time()
    cfg = DevNetConfig(
        hidden=(24, 12), learning_rate=0.005, batch_size=128,
        n_epochs=n_epochs, n_ref=5000, confidence_margin=5.0, seed=42,
    )
    # fit_devnet operates on a single frame with a 0/1 label column. Build
    # a temporary frame containing only the train rows + the binary target.
    train_view = pd.DataFrame(
        tr_df[feature_cols].to_numpy(dtype=np.float32),
        columns=feature_cols,
    )
    train_view["y"] = tr_df["target"].to_numpy(dtype=np.int8)
    _, _, model = fit_devnet(train_view, feat_cols=feature_cols,
                              label_col="y", config=cfg)
    print(f"    DevNet fit done in {time.time()-t1:.1f}s "
          f"(train {len(tr_df):,}, test {len(te_df):,})", flush=True)

    import torch
    mu = train_view[feature_cols].mean().to_numpy()
    sd = train_view[feature_cols].std().to_numpy() + 1e-8
    model.eval()
    with torch.no_grad():
        Xtr = (tr_df[feature_cols].fillna(-9999.0).to_numpy(dtype=np.float32) - mu) / sd
        Xte = (te_df[feature_cols].fillna(-9999.0).to_numpy(dtype=np.float32) - mu) / sd
        proba_train = model(torch.from_numpy(Xtr.astype(np.float32))).squeeze(-1).numpy()
        proba_test = model(torch.from_numpy(Xte.astype(np.float32))).squeeze(-1).numpy()
    return proba_train, proba_test


def main() -> None:
    print(f"=== Lawley 2022 Phase 3: DevNet on the H3 cube ===", flush=True)
    df = build_feature_frame()
    df = assign_1d_lat_bands(df)
    df = assign_2d_blocks(df)
    feature_cols = [c for c in df.columns if c not in DROP_BEFORE_FIT]
    print(f"  feature_cols ({len(feature_cols)}): {feature_cols}", flush=True)

    results: dict = {
        "stage": "Phase 3: DevNet vs GBM on Lawley H3 cube",
        "n_total": int(len(df)),
        "n_positives_total": int(df["target"].sum()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "model": "DevNet (Pang 2019 / DEEP-SEAM)",
        "n_boot": N_BOOT,
        "top_k_pct": TOP_K_PCT,
    }

    for name, tr_mask, te_mask in [
        ("cv_1d_lat_band",
         (df["fold_1d"] != 0), (df["fold_1d"] == 0)),
        ("cv_2d_blocks",
         (df["fold_2d"] != 0), (df["fold_2d"] == 0)),
        ("transfer_uscan_to_aus",
         (df["Continent_Majority"] == "North America"),
         (df["Continent_Majority"] == "Oceania")),
        ("transfer_aus_to_uscan",
         (df["Continent_Majority"] == "Oceania"),
         (df["Continent_Majority"] == "North America")),
    ]:
        print(f"\n[{name}]", flush=True)
        _, proba_te = fit_devnet_on_split(df[tr_mask], df[te_mask], feature_cols)
        y_te = df.loc[te_mask, "target"].to_numpy().astype(np.int8)
        print(f"    bootstrap CI on test (n={len(y_te):,}, "
              f"pos={int(y_te.sum())})...", flush=True)
        results[name] = bootstrap_metrics(y_te, proba_te)
        gc.collect()

    print(f"\n=== SUMMARY (DevNet) ===", flush=True)
    for name in ("cv_1d_lat_band", "cv_2d_blocks",
                 "transfer_uscan_to_aus", "transfer_aus_to_uscan"):
        r = results[name]
        auc = r["auc"]
        top1 = r["capture"]["top_1_pct"]
        top5 = r["capture"]["top_5_pct"]
        print(f"  {name:30s}  AUC = {auc['mean']:.4f} "
              f"[{auc['ci95'][0]:.3f}, {auc['ci95'][1]:.3f}]  "
              f"top1 = {top1['rate']*100:.1f}% (lift {top1['lift']:.1f}x)  "
              f"top5 = {top5['rate']*100:.1f}% (lift {top5['lift']:.1f}x)",
              flush=True)

    out_path = OUT_DIR / "path3_devnet_metrics.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
