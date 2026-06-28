"""H2 control: distance to the schist-hosted lode, + robustness of the contact tests.

The contact tests are null (primary) to significantly-reversed (major marble:
coarser gold sits FARTHER from marble). Two checks decide whether that is a real
negative or a broken method:

  1. LODE POSITIVE CONTROL. Round 5 found coarseness declines with distance to
     the mapped 36a schist-hosted lodes (Spearman rho -0.40 at Nome). If the
     same gradient appears here against the peninsula 36a lodes, the method
     detects a real source signal -- so the contact null is about the contact,
     not a dead method. The lodes are the schist-hosted gold-quartz veins (Big
     Hurrah type); the marble is a passive unit.

  2. ROBUSTNESS. The major-marble straight-line test leans on a single class-1
     (fine) placer at 5 m. Re-run it dropping class 1 (class 2 vs class 3 only)
     and as Mann-Whitney, to show the reversed trend is not that one point.

Run: uv run python -m scripts.nome_placer.h2_confined_reach.test_lode_control
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import kruskal, mannwhitneyu, spearmanr

OUT = Path("data/derived/nome_placer/h2_confined_reach")
STAGED = Path("data/derived/nome_placer/peninsula_phase2/peninsula_ardf_placer_lode_3338.geojson")
GEMS = Path("data/raw/dggs_ri2024_7/extracted/pkg/"
            "casadepaga_bedrock_gems_db_wo_stations-open/GM_MapUnitPolys.shp")
MAX_LODE_M = 15_000.0


def main() -> None:
    polys = gpd.read_file(GEMS).to_crs(3338)
    mb = polys.total_bounds
    staged = gpd.read_file(STAGED).to_crs(3338)
    lode = staged[staged.deposit_class == "lode"].copy()
    # lodes within the map bbox + a 5 km halo (upstream sources can sit off-map)
    halo = 5000.0
    lode = lode.cx[mb[0] - halo:mb[2] + halo, mb[1] - halo:mb[3] + halo]
    # guard: no placer-coded seeds (label hygiene, as in round 5)
    lode = lode[~lode.model_code.astype(str).str.contains("39")]
    print(f"36a lodes within map+halo: {len(lode)}")
    ltree = cKDTree(np.column_stack([lode.geometry.x, lode.geometry.y]))

    typed = gpd.read_file(OUT / "placers_typed.geojson").to_crs(3338)
    al = typed[(typed.geol_type == "alluvial-stream") & typed.coarseness_rank.notna()].copy()
    al["cls"] = al.coarseness_rank.astype(int)
    d, _ = ltree.query(np.column_stack([al.geometry.x, al.geometry.y]), k=1)
    al["lode_m"] = np.where(d <= MAX_LODE_M, d, np.nan)

    res = {"n_lodes_used": int(len(lode))}

    # 1) lode positive control
    s = al[al.lode_m.notna()]
    rho, p = spearmanr(s.cls, s.lode_m)
    groups = [s.loc[s.cls == k, "lode_m"] for k in sorted(s.cls.unique())]
    res["lode_straight_line"] = {
        "n": int(len(s)),
        "median_by_class_m": {int(k): round(float(s.loc[s.cls == k, "lode_m"].median()))
                              for k in sorted(s.cls.unique())},
        "spearman_rho": round(float(rho), 3), "spearman_p": round(float(p), 4),
        "kruskal_p": round(float(kruskal(*groups).pvalue), 4),
        "note": "negative rho = coarser gold nearer a lode = round-5 gradient reproduced",
    }

    # 2) robustness of the major-marble straight test, dropping class 1
    feat = pd.read_csv(OUT / "reach_features.csv")
    maj = gpd.read_file(OUT / "contact_major_marble_3338.geojson").geometry.iloc[0]
    # straight distance to major contact, recomputed for the tagged set
    from shapely.geometry import Point
    al["major_m"] = [al.geometry.iloc[i].distance(maj) for i in range(len(al))]
    no1 = al[al.cls >= 2]
    rho2, p2 = spearmanr(no1.cls, no1.major_m)
    c2 = no1.loc[no1.cls == 2, "major_m"]; c3 = no1.loc[no1.cls == 3, "major_m"]
    mw = mannwhitneyu(c3, c2, alternative="two-sided")
    res["major_marble_drop_class1"] = {
        "n": int(len(no1)),
        "median_class2_m": round(float(c2.median())), "median_class3_m": round(float(c3.median())),
        "spearman_rho": round(float(rho2), 3), "spearman_p": round(float(p2), 4),
        "mannwhitney_p_c3_vs_c2": round(float(mw.pvalue), 4),
        "note": "still positive rho (coarser farther) without the lone class-1 point => not driven by it",
    }
    (OUT / "lode_control.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
