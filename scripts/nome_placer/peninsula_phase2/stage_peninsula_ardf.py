"""Phase-2 prep: stage Seward Peninsula-wide ARDF placer + lode occurrences.

Extracts the Au placer (Cox-Singer model 39a) and lode (36a) occurrences for the
Seward Peninsula gold quadrangles from the full Alaska ARDF shapefile already on
disk, reprojects NAD27 (EPSG:4267) -> WGS84 + EPSG:3338 (same handling as
export_ardf_fossick_wgs84.py), and writes a staged GeoJSON with provenance and
occurrence counts by quad and class. Acquisition + staging only; no modeling.

Quad coverage: the named Phase-2 places (Solomon, Council/Casadepaga, Bluff, Big
Hurrah, Bendeleben, Teller) fall in the 1:250k quads Nome (NM), Solomon (SO),
Bendeleben (BN), Teller (TE); Council/Casadepaga/Bluff/Big Hurrah are within SO.

Run: uv run python -m scripts.nome_placer.peninsula_phase2.stage_peninsula_ardf
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

SRC = Path("data/raw/ardf/ardf/ardf.shp")          # full Alaska ARDF, EPSG:4267
OUT = Path("data/derived/nome_placer/peninsula_phase2")
QUADS = ["NM", "SO", "BN", "TE"]                     # Seward Peninsula gold quads

KEEP = ["ardf_num", "mrds_num", "site", "quad_250", "quad_63360", "comm_main",
        "comm_other", "site_type", "status", "dep_model", "model_code",
        "production", "geol_desc", "latitude", "longitude"]


def classify(model_code: str, comm: str) -> str:
    mc = str(model_code).lower()
    if "39" in mc:
        return "placer"
    if "36a" in mc:
        return "lode"
    return "other"


def main() -> None:
    g = gpd.read_file(SRC)
    pen = g[g["quad_250"].astype(str).isin(QUADS)].copy()
    pen["deposit_class"] = [classify(m, c) for m, c in
                            zip(pen["model_code"], pen["comm_main"])]
    # Keep placer + lode (the two source/target classes for the contact test).
    sub = pen[pen["deposit_class"].isin(["placer", "lode"])].copy()

    # Reproject NAD27 -> WGS84 (write geometry + refreshed lat/lon) and 3338.
    keep = [c for c in KEEP if c in sub.columns] + ["deposit_class", "geometry"]
    sub = sub[keep]
    sub_wgs = sub.to_crs("EPSG:4326").copy()
    sub_wgs["longitude"] = sub_wgs.geometry.x
    sub_wgs["latitude"] = sub_wgs.geometry.y
    sub_3338 = sub_wgs.to_crs("EPSG:3338")

    OUT.mkdir(parents=True, exist_ok=True)
    sub_wgs.to_file(OUT / "peninsula_ardf_placer_lode_4326.geojson", driver="GeoJSON")
    sub_3338.to_file(OUT / "peninsula_ardf_placer_lode_3338.geojson", driver="GeoJSON")

    # Counts by quad x class, and Au-only breakdown.
    by_quad = (sub_wgs.groupby(["quad_250", "deposit_class"]).size()
               .unstack(fill_value=0).to_dict("index"))
    au = sub_wgs[sub_wgs["comm_main"].astype(str).str.startswith("Au")]
    prov = {
        "source": "USGS ARDF full Alaska shapefile (data/raw/ardf/ardf/ardf.shp), EPSG:4267",
        "quads": QUADS,
        "n_total_in_quads": int(len(pen)),
        "n_placer_lode_staged": int(len(sub)),
        "n_placer": int((sub.deposit_class == "placer").sum()),
        "n_lode": int((sub.deposit_class == "lode").sum()),
        "n_au_main": int(len(au)),
        "by_quad_class": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in by_quad.items()},
        "note": ("geol_desc is the 255-char shapefile stub; full narratives are at "
                 "mrdata's per-record JSON service if Phase-2 typing needs them."),
    }
    (OUT / "peninsula_ardf_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))
    print(f"\nwrote {OUT}/peninsula_ardf_placer_lode_{{4326,3338}}.geojson")


if __name__ == "__main__":
    main()
