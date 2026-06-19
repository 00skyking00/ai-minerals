"""Build the union claim-polygon layer for the Cape Nome district.

No single source is complete, so union three vector sources by MS number:
  1. BLM AK PLSS CadNSDI "Special Survey" (FeatureServer/6), SURVTYPTXT='Mineral Survey'
     — authoritative federal vector; SURVNO = MS number.
  2. Alaska DNR "Survey Boundary Poly" (Mapper/Land_Estate_Layers/49), CASE_TYPE='306'
     (U.S. Mineral Survey); FILENUMBER = MS number. Complementary to BLM.
  3. bearcub curated plat (claims_by_ms.geojson) — the dense Dry Creek family/beach
     block (Bear Cub/Molasses/Early Bear/Honey/Great Western) absent from BOTH BLM+DNR.

Dedup by MS, geometry priority: curated (hand-pinned, trusted for the family block)
> BLM (authoritative federal) > DNR. Output schema: MS, name, source, acres + Polygon
EPSG:4326. Wide district bbox to cover Tuck plates A-E.

Run:  uv run python scripts/nome_placer/build_claim_polygons_union.py
Writes: data/derived/nome_placer/claim_polygons_union.geojson (+ delivers to goldbug)
"""
from __future__ import annotations

import json
import subprocess
import urllib.parse
from pathlib import Path

REPO = Path("/home/sky/src/learning/ai-minerals")
OUT = REPO / "data/derived/nome_placer/claim_polygons_union.geojson"
CURATED = REPO / "data/raw/bearcub_nome/claims_by_ms.geojson"
GOLDBUG = Path("/home/sky/src/learning/gldbg/data/nome/claim_polygons_union.geojson")

# District bbox covering Tuck plates A-E (lon/lat WGS84)
BBOX = {"xmin": -166.3, "ymin": 64.35, "xmax": -164.5, "ymax": 65.05}

BLM = "https://gis.blm.gov/akarcgis/rest/services/PLSS/BLM_AK_PLSS_CadNSDI/FeatureServer/6/query"
DNR = "https://arcgis.dnr.alaska.gov/arcgis/rest/services/Mapper/Land_Estate_Layers/MapServer/49/query"


def _curl(url: str, params: dict) -> dict:
    q = urllib.parse.urlencode(params)
    r = subprocess.run(["curl", "-s", "--max-time", "120", "-G", url, "--data", q],
                       capture_output=True, text=True)
    return json.loads(r.stdout)


def fetch_paged(url: str, where: str) -> list[dict]:
    """Fetch all GeoJSON features for the bbox + where, paging past transfer limits."""
    feats, offset = [], 0
    while True:
        params = {
            "where": where,
            "geometry": json.dumps(BBOX),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*", "returnGeometry": "true",
            "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000,
        }
        d = _curl(url, params)
        batch = d.get("features", [])
        feats.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return feats


def norm(ms, name, source, acres, geom) -> dict:
    return {"type": "Feature",
            "properties": {"MS": str(ms), "name": name, "source": source, "acres": acres},
            "geometry": geom}


def main() -> None:
    by_ms: dict[str, dict] = {}   # MS -> normalized feature (priority fill)

    # 3. curated first (trusted for family block; lowest to be overwritten? no:
    # we want curated to WIN for the family block. Fill curated last as override.)
    print("Fetching DNR (CASE_TYPE=306) ...")
    for f in fetch_paged(DNR, "CASE_TYPE='306'"):
        p = f["properties"]; ms = str(p.get("FILENUMBER") or "").strip()
        if ms and f.get("geometry"):
            by_ms[ms] = norm(ms, p.get("CLAIM_NAME") or f"MS {ms}", "DNR", p.get("NET_ACRES"), f["geometry"])
    print(f"  DNR: {len(by_ms)} MS")

    print("Fetching BLM AK Special Survey (Mineral Survey) ...")
    nblm = 0
    for f in fetch_paged(BLM, "SURVTYPTXT='Mineral Survey'"):
        p = f["properties"]; ms = str(p.get("SURVNO") or "").strip()
        if ms and f.get("geometry"):
            by_ms[ms] = norm(ms, p.get("SURVLAB") or f"MS {ms}", "BLM", p.get("GISACRE"), f["geometry"])  # BLM overrides DNR
            nblm += 1
    print(f"  BLM features: {nblm}; union now {len(by_ms)} MS")

    # curated overrides for the family block (hand-pinned, trusted)
    cur = json.loads(CURATED.read_text())["features"]
    for f in cur:
        ms = str(f["properties"].get("MS") or "").strip()
        if ms and f.get("geometry"):
            by_ms[ms] = norm(ms, f["properties"].get("name") or f"MS {ms}", "curated", None, f["geometry"])
    print(f"  curated overrides applied; union now {len(by_ms)} MS")

    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
          "features": sorted(by_ms.values(), key=lambda x: int(x["properties"]["MS"]) if x["properties"]["MS"].isdigit() else 0)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fc))
    print(f"\nwrote {OUT} : {len(fc['features'])} claim polygons")
    from collections import Counter
    print("  by source:", dict(Counter(f["properties"]["source"] for f in fc["features"])))

    GOLDBUG.parent.mkdir(parents=True, exist_ok=True)
    GOLDBUG.write_text(json.dumps(fc))
    print(f"delivered to goldbug: {GOLDBUG}")


if __name__ == "__main__":
    main()
