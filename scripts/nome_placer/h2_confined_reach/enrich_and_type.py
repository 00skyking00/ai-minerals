"""H2 redesign step 3: type + ordinal-coarseness the RI 2024-7 placer occurrences.

Takes the staged peninsula ARDF placers (PR #49), keeps the ones that fall
inside the RI 2024-7 map polygons, pulls each one's full untruncated narrative
from the USGS mrdata ARDF JSON service (the 255-char shapefile stub is too short
to mine reliably; gotcha-free coarseness needs the real text), genetic-types
them with the round-5 name-then-narrative classifier, and mines an ordinal
coarseness class per occurrence.

Coarseness (the target) is mined over the FULL combined narrative -- geology +
workings + production notes + comments -- not just the geology field, because
grain-size mentions ("coarse gold", "nuggets", "fine flour gold") show up in the
production/workings text as often as in the lithology description. Classes per
the spec: 3 rough/quartz-attached/nuggety, 2 coarse, 1 fine/flaky/flour.

Writes placers_typed.geojson (EPSG:3338) + an audit CSV with the per-record
basis and the narrative head, so every type and coarseness call is checkable.

Run: uv run python -m scripts.nome_placer.h2_confined_reach.enrich_and_type
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests

from scripts.nome_placer.enrich_ardf_fulltext import fetch_record, flatten, UA
from scripts.nome_placer.inland_local_source.type_placers import classify
from scripts.nome_placer.h2_confined_reach.coarseness import mine_gold_coarseness

GEMS = Path(
    "data/raw/dggs_ri2024_7/extracted/pkg/"
    "casadepaga_bedrock_gems_db_wo_stations-open/GM_MapUnitPolys.shp"
)
STAGED = Path("data/derived/nome_placer/peninsula_phase2/peninsula_ardf_placer_lode_3338.geojson")
DEM = Path("data/raw/ifsar_dggs/ifsar_dtm_5m_bighurrah_council_bluff_3338.tif")
OUT = Path("data/derived/nome_placer/h2_confined_reach")

# Narrative fields to concatenate for coarseness mining (mrdata JSON keys).
NARRATIVE_KEYS = ["geology", "workings", "production_notes", "comments", "deposit_model"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    polys = gpd.read_file(GEMS).to_crs("EPSG:3338")
    union = polys.union_all()

    staged = gpd.read_file(STAGED)
    placer = staged[staged.deposit_class == "placer"].copy()
    inmap = placer[placer.within(union)].copy().reset_index(drop=True)
    print(f"placers within RI map polygons: {len(inmap)} of {len(placer)} peninsula placers")

    # Elevation from the 5 m DTM (tiebreaker for typing + reporting).
    with rasterio.open(DEM) as ds:
        nod = ds.nodata
        xs = inmap.geometry.x.to_numpy()
        ys = inmap.geometry.y.to_numpy()
        elev = np.array([v[0] for v in ds.sample(zip(xs, ys))], dtype=float)
    elev = np.where(elev == nod, np.nan, elev)
    inmap["elev_m"] = elev

    session = requests.Session()
    session.headers.update(UA)

    types, bases, coarse, full_geol, full_comb, failed = [], [], [], [], [], []
    for i, row in inmap.iterrows():
        num = row["ardf_num"]
        rec = fetch_record(session, num, force=False)
        if rec is None:
            failed.append(num)
            geol = str(row.get("geol_desc") or "")
            comb = geol
        else:
            jp = rec["properties"]
            geol = flatten(jp.get("geology")) or str(row.get("geol_desc") or "")
            comb = "\n".join(flatten(jp.get(k)) for k in NARRATIVE_KEYS if jp.get(k))
        site = str(row.get("site") or "")
        t, b = classify(geol, site, elev[i])
        types.append(t)
        bases.append(b)
        coarse.append(mine_gold_coarseness(comb))
        full_geol.append(geol)
        full_comb.append(comb)

    inmap["geol_type"] = types
    inmap["type_basis"] = bases
    inmap["coarseness_rank"] = coarse

    keep = ["ardf_num", "site", "quad_250", "model_code", "comm_main", "production",
            "elev_m", "geol_type", "type_basis", "coarseness_rank", "geometry"]
    out = inmap[keep].copy()
    out.to_file(OUT / "placers_typed.geojson", driver="GeoJSON")

    audit = out.drop(columns="geometry").copy()
    audit["narrative_head"] = [c[:240].replace("\n", " ") for c in full_comb]
    audit.to_csv(OUT / "placers_typed_audit.csv", index=False)

    summary = {
        "n_inmap_placers": int(len(out)),
        "fetch_failures": failed,
        "type_counts": out.geol_type.value_counts().to_dict(),
        "coarseness_counts_all": {str(k): int(v) for k, v in
                                  out.coarseness_rank.value_counts(dropna=False).items()},
        "alluvial_coarseness": {str(k): int(v) for k, v in
                                out[out.geol_type == "alluvial-stream"]
                                .coarseness_rank.value_counts(dropna=False).items()},
    }
    (OUT / "typed_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
