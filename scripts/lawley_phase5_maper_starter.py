"""Lawley Phase 5 — SRI MAPER starter scaffolding.

This script sets up the data conversion + environment checks needed to
run the SRI MAPER (Masked Autoencoder Vision Transformer) experiment
against the Lawley H3 cube. It does NOT run the pretraining or
fine-tuning loops — those require a GPU. See
[research/lawley_phase5_maper_setup_2026-05.md](../research/lawley_phase5_maper_setup_2026-05.md)
for the full execution plan and resource requirements.

What this script does:

1. Validates the sri-ta3 repo is cloned at third_party/sri-ta3/.
2. Validates the Lawley parquet exists.
3. Validates the required environment packages (PyTorch Lightning,
   Hydra) can be installed via `uv add`.
4. Optionally subsamples the Lawley H3 cube to a 10% spatially-
   stratified subset for the PoC.
5. Prints the rasterization steps that need to run via
   sri-ta3-baselines/rasterize_H3_datacube.ipynb to produce the
   raster library the MAPER framework expects.

Run with `uv run python scripts/lawley_phase5_maper_starter.py
--validate` for a check-only mode, or `--subsample-10pct` to write
the 10% subset parquet.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path("/home/sky/src/learning/ai-minerals")
SRI_TA3 = REPO_ROOT / "third_party" / "sri-ta3"
SRI_BASELINES = REPO_ROOT / "third_party" / "sri-ta3-baselines"
DATACUBE = REPO_ROOT / "data" / "raw" / "lawley2022" / "datacube.parquet"
SUBSET_PARQUET = REPO_ROOT / "data" / "raw" / "lawley2022" / "datacube_10pct.parquet"

REQUIRED_FOR_PRETRAIN = [
    "pytorch_lightning",
    "hydra",
    "omegaconf",
    "rasterio",
    "torch",
]


def validate_layout() -> bool:
    """Check the file system is set up for Phase 5 work."""
    ok = True
    print("Phase 5 layout check:")
    for label, path, must_exist in [
        ("sri-ta3 repo",           SRI_TA3,        True),
        ("sri-ta3-baselines repo", SRI_BASELINES,  True),
        ("Lawley parquet",         DATACUBE,       True),
        ("10% subset parquet",     SUBSET_PARQUET, False),
    ]:
        status = "OK   " if path.exists() else ("MISS " if must_exist else "      ")
        print(f"  [{status}] {label:25s} {path}")
        if must_exist and not path.exists():
            ok = False

    print("\nPython package availability:")
    for pkg in REQUIRED_FOR_PRETRAIN:
        try:
            importlib.import_module(pkg)
            print(f"  [OK   ] {pkg}")
        except ImportError:
            print(f"  [MISS ] {pkg}    (uv add {pkg})")

    return ok


def write_10pct_subset(seed: int = 42) -> None:
    """Write a 10% stratified subset of the Lawley cube preserving positive counts."""
    import numpy as np
    import pandas as pd

    print(f"Reading {DATACUBE} ...")
    df = pd.read_parquet(DATACUBE)
    print(f"  full: {df.shape[0]:,} rows x {df.shape[1]} cols")

    rng = np.random.default_rng(seed)
    label_cols = ["Training_MVT_Deposit", "Training_MVT_Occurrence"]
    pos_mask = (df[label_cols[0]] | df[label_cols[1]])
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]
    print(f"  positives: {len(pos_idx):,}    unlabeled: {len(neg_idx):,}")

    # Keep every positive (they're rare); 10% of unlabeled cells.
    keep_neg = rng.choice(neg_idx, size=int(len(neg_idx) * 0.10), replace=False)
    keep = np.concatenate([pos_idx, keep_neg])
    keep.sort()

    sub = df.iloc[keep].reset_index(drop=True)
    print(f"  subset: {sub.shape[0]:,} rows ({sub.shape[0]/len(df)*100:.1f}% of full)")
    print(f"  subset positives: {int(pos_mask.iloc[keep].sum()):,}")

    SUBSET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(SUBSET_PARQUET, compression="snappy")
    print(f"\nwrote {SUBSET_PARQUET} ({SUBSET_PARQUET.stat().st_size/1e6:.0f} MB)")


def print_next_steps() -> None:
    print("""
Next steps to run the full Phase 5 PoC:

1. Get GPU access (cloud rental: Vast.ai, RunPod, Lambda Cloud, GCP).
   Recommended: A100 (40-80 GB) or L4 (24 GB). Total compute budget
   estimate: $20-100 for the 10% PoC including fine-tune.

2. Rasterize the H3 cube into a raster library that SRI MAPER's
   TIFFDataModule expects:

       cd third_party/sri-ta3-baselines/
       jupyter nbconvert --to script rasterize_H3_datacube.ipynb
       jupyter nbconvert --to script produce_multiband_raster.ipynb

       # Edit the converted .py to point at the 10% subset parquet,
       # then run them to produce per-feature TIFFs + a multi-band
       # raster.

3. Install Phase 5 deps (one-time, on the GPU host):

       uv add pytorch-lightning hydra-core omegaconf

4. Create the Phase 5 Hydra experiment config:

       cp third_party/sri-ta3/sri_maper/configs/experiment/pretrain_template.yaml \\
          third_party/sri-ta3/sri_maper/configs/experiment/lawley_phase5_pretrain.yaml

       # Edit lawley_phase5_pretrain.yaml to set:
       #   data.tif_dir: <path to the raster library produced in step 2>
       #   data.window_size: 5
       #   trainer.max_epochs: 30 (or whatever fits the GPU budget)

5. Run pretraining:

       cd third_party/sri-ta3
       python sri_maper/src/pretrain.py experiment=lawley_phase5_pretrain

6. Run fine-tuning:

       # Similar pattern, with the classifier_template.yaml
       python sri_maper/src/train.py experiment=lawley_phase5_finetune

7. Evaluate on the held-out 2-D blocked CV fold, write metrics JSON
   in the same format as data/derived/lawley/path1b_leak_corrected_metrics.json.

8. Compare fine-tuned AUC against the Phase 1b GBM baseline (0.972
   under 1-D, 0.868 under 2-D). Document in research/lawley_phase5_maper_findings_2026-05.md.
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="check layout + deps")
    ap.add_argument("--subsample-10pct", action="store_true",
                    help="write 10pct subset parquet")
    ap.add_argument("--next-steps", action="store_true",
                    help="print the next-steps checklist")
    args = ap.parse_args()

    if not any([args.validate, args.subsample_10pct, args.next_steps]):
        args.validate = True
        args.next_steps = True

    if args.validate:
        ok = validate_layout()
        print()
        print(f"  layout {'OK' if ok else 'MISSING REQUIRED FILES'}\n")

    if args.subsample_10pct:
        write_10pct_subset()
        print()

    if args.next_steps:
        print_next_steps()


if __name__ == "__main__":
    main()
