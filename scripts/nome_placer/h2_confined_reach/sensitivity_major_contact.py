"""H2 sensitivity: distance to the MAJOR-marble contact only.

The primary contact (942 km) counts every thin DOm/DOs interbed; the Nome
Complex is so interleaved that "distance to the nearest contact" barely varies.
The literature's mineralizing "schist-limestone contact" plausibly means the
substantial marble belts, not centimetre interbeds. This rebuilds the contact
from DOm/Dm marble bodies >= 1 km^2 (58 bodies = 78% of the marble area) and
re-runs the coarseness-vs-distance test, to check the primary null is not an
interbed-dilution artifact.

Run: uv run python -m scripts.nome_placer.h2_confined_reach.sensitivity_major_contact
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from scipy.stats import kruskal, spearmanr
from shapely import line_merge

from scripts.nome_placer.h2_confined_reach.build_distance_and_reaches import (
    down_channel_distance, SNAP_TOL_M, MAX_STRAIGHT_M)

GEMS = Path("data/raw/dggs_ri2024_7/extracted/pkg/"
            "casadepaga_bedrock_gems_db_wo_stations-open/GM_MapUnitPolys.shp")
OUT = Path("data/derived/nome_placer/h2_confined_reach")
MARBLE = {"DOm", "Dm"}
SCHIST = {"DOs", "DOq", "DOg", "Omg", "DOms", "Ds", "DOsq", "DOqs", "Osg", "DOu", "PzPh", "PzPa"}
MIN_BODY_KM2 = 1.0


def main() -> None:
    polys = gpd.read_file(GEMS).to_crs("EPSG:3338")
    marble = polys[polys.MapUnit.isin(MARBLE)].copy()
    marble["a"] = marble.geometry.area / 1e6
    major = marble[marble.a >= MIN_BODY_KM2]
    schist = polys[polys.MapUnit.isin(SCHIST)]
    contact = major.union_all().boundary.intersection(schist.union_all().boundary)
    contact = line_merge(contact) if contact.geom_type != "LineString" else contact
    km = contact.length / 1000.0
    gpd.GeoDataFrame({"contact_def": ["major_marble"]}, geometry=[contact], crs="EPSG:3338").to_file(
        OUT / "contact_major_marble_3338.geojson", driver="GeoJSON")
    print(f"major-marble contact: {len(major)} bodies >= {MIN_BODY_KM2} km2, {km:.0f} km")

    with rasterio.open(OUT / "filled_dem.tif") as src:
        transform = src.transform; crs = src.crs; H, W = src.height, src.width
        filled = src.read(1).astype(np.float32)
    cs = abs(transform.a)
    rr = rasterio.open(OUT / "recv_row.tif").read(1)
    rc = rasterio.open(OUT / "recv_col.tif").read(1)
    streams = rasterio.open(OUT / "streams.tif").read(1) == 1
    confined = rasterio.open(OUT / "confined_valley.tif").read(1) == 1
    accum = rasterio.open(OUT / "flow_accum.tif").read(1)

    cc = rasterize([(contact, 1)], out_shape=(H, W), transform=transform,
                   fill=0, all_touched=True).astype(bool)
    crossings = streams & binary_dilation(cc, iterations=1)
    dch = down_channel_distance(streams, rr, rc, filled, accum, crossings, cs)
    cy, cx = np.where(cc)
    cxs, cys = rasterio.transform.xy(transform, cy, cx)
    ctree = cKDTree(np.column_stack([np.asarray(cxs), np.asarray(cys)]))
    fy, fx = np.where(confined)
    snap_tree = cKDTree(np.column_stack([fx, fy]).astype(float))

    typed = gpd.read_file(OUT / "placers_typed.geojson").to_crs(crs)
    al = typed[(typed.geol_type == "alluvial-stream") & typed.coarseness_rank.notna()].copy()
    rows = []
    for _, p in al.iterrows():
        prow, pcol = rasterio.transform.rowcol(transform, p.geometry.x, p.geometry.y)
        sl, _ = ctree.query([p.geometry.x, p.geometry.y], k=1)
        dq, idx = snap_tree.query([pcol, prow], k=1)
        snap_dch = None
        if dq <= SNAP_TOL_M / cs:
            v = dch[int(fy[idx]), int(fx[idx])]
            snap_dch = round(float(v), 1) if np.isfinite(v) else None
        rows.append({"ardf_num": p.ardf_num, "cls": int(p.coarseness_rank),
                     "straight_major_m": round(float(sl), 1) if sl <= MAX_STRAIGHT_M else None,
                     "snap_dch_major_m": snap_dch})
    df = pd.DataFrame(rows)

    res = {"contact_km": round(km, 0), "n_major_bodies": int(len(major))}
    for col in ["straight_major_m", "snap_dch_major_m"]:
        s = df[df[col].notna()]
        if len(s) >= 4:
            rho, p = spearmanr(s.cls, s[col])
            groups = [s.loc[s.cls == k, col] for k in sorted(s.cls.unique())]
            kw = kruskal(*groups) if len(groups) > 1 else None
            res[col] = {"n": len(s),
                        "median_by_class_m": {int(k): round(float(s.loc[s.cls == k, col].median()))
                                              for k in sorted(s.cls.unique())},
                        "spearman_rho": round(float(rho), 3), "spearman_p": round(float(p), 4),
                        "kruskal_p": round(float(kw.pvalue), 4) if kw is not None else None}
        else:
            res[col] = {"n": len(s), "note": "too few"}
    (OUT / "sensitivity_major_contact.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
