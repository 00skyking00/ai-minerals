"""H2 redesign step 5: does coarseness decline with distance from the contact?

Spearman rank correlation (a negative rho = coarser gold nearer the contact =
local sourcing) and Kruskal-Wallis across the ordinal coarseness classes, for
three distance measures, each on the placers that have both a coarseness tag and
that distance:
  reach_head_dch : down-channel distance at the reach's upstream end (the spec's
                   primary predictor; sparsest, many reaches have no upstream
                   contact crossing)
  snap_dch       : down-channel distance at the snapped confined-stream cell
  straight_contact : straight-line distance to the contact (round-5-style
                     baseline; the fullest sample)

Writes h2_results.json with the per-measure numbers.

Run: uv run python -m scripts.nome_placer.h2_confined_reach.run_test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

OUT = Path("data/derived/nome_placer/h2_confined_reach")
MEASURES = ["reach_head_dch_m", "snap_dch_m", "straight_contact_m"]


def test_measure(df: pd.DataFrame, col: str) -> dict:
    sub = df[df.coarseness_rank.notna() & df[col].notna()].copy()
    sub["cls"] = sub.coarseness_rank.astype(int)
    n = len(sub)
    if n < 4:
        return {"n": n, "note": "too few to test"}
    rho, p = spearmanr(sub.cls, sub[col])
    groups = [sub.loc[sub.cls == k, col].to_numpy() for k in sorted(sub.cls.unique())]
    kw = kruskal(*groups) if len(groups) > 1 and all(len(g) > 0 for g in groups) else None
    med = {int(k): round(float(sub.loc[sub.cls == k, col].median()), 0)
           for k in sorted(sub.cls.unique())}
    return {
        "n": n,
        "class_counts": {int(k): int((sub.cls == k).sum()) for k in sorted(sub.cls.unique())},
        "median_dist_by_class_m": med,
        "spearman_rho_cls_vs_dist": round(float(rho), 3),
        "spearman_p": round(float(p), 4),
        "kruskal_p": round(float(kw.pvalue), 4) if kw is not None else None,
        "interpretation": ("negative rho = coarser gold (higher class) sits at SHORTER "
                           "distance = downstream fining / local sourcing supported"),
    }


def main() -> None:
    df = pd.read_csv(OUT / "reach_features.csv")
    results = {m: test_measure(df, m) for m in MEASURES}
    results["_summary"] = {
        "n_alluvial_in_map": int(len(df)),
        "n_coarseness_tagged": int(df.coarseness_rank.notna().sum()),
        "coarseness_class_counts": {int(k): int(v) for k, v in
                                    df.coarseness_rank.value_counts().items()},
    }
    (OUT / "h2_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
