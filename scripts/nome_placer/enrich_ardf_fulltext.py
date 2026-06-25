"""Enrich the WGS84 ARDF Nome export with the FULL untruncated narrative text.

Coordinator handoff 2026-06-24 (ARDF description text missing from the KG).

Root cause (confirmed at the source): the ARDF shapefile
`data/raw/ardf/ardf/ardf.shp` caps every text field at 255 characters, the
.dbf text-width limit. The shapefile's own metadata (`ardf.met`) documents it:
the five long fields (Location, Geologic Description, Workings/exploration,
Additional Comments, References) all exceed 255 characters in the source
FileMaker database and were truncated when the shapefile was generated. So the
prior WGS84 re-export (6e99b9a) carried only 255-char stubs of the narrative,
never the full text. The full narrative was never in the shapefile to ship.

This script keeps the validated WGS84 geometry and coordinates from that prior
export EXACTLY (read straight from its GeoJSON, so no datum regression and no
coordinate reformatting), and replaces the truncated narrative fields with the
full text pulled per record from the USGS mrdata ARDF JSON service
(https://mrdata.usgs.gov/ardf/json/<ardf_num>), which renders from the full
database rather than the truncated shapefile. It also adds the complete
reference list (`references`) and the `reporter`, which the shapefile never
carried in full.

Network responses are cached under data/raw/ardf/fulltext_json/ (gitignored),
so re-runs do not refetch. Pass --force to refetch.

Run:  .venv/bin/python scripts/nome_placer/enrich_ardf_fulltext.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PRIOR_GEOJSON = Path(
    "handoff/outbox/done/2026-06-21-ardf-nome-wgs84-wider/nome_ardf_all_wgs84.geojson"
)
OUT_DIR = Path("handoff/outbox/2026-06-24-ardf-nome-fulltext")
CACHE_DIR = Path("data/raw/ardf/fulltext_json")
JSON_URL = "https://mrdata.usgs.gov/ardf/json/{num}"
# mrdata 403s the default urllib/requests User-Agent; a real UA is required.
UA = {"User-Agent": "ai-minerals ARDF re-export (skyking@gmail.com)"}

# mrdata JSON property name -> export column name. These are the narrative
# fields the shapefile truncated; we overwrite the prior 255-char stubs with
# the full text. Column names match the prior export so fossick's reads are
# unchanged.
TEXT_MAP = {
    "location": "location",
    "geology": "geol_desc",
    "workings": "work_expl",
    "comments": "comments",
    "production_notes": "prod_notes",
    "reserves": "reserves",
    "alteration": "alteration",
    "deposit_model": "dep_model",
    "primary_reference": "prime_ref",
    "age": "age",
}

# Column order: a strict SUPERSET of the prior export's columns (so nothing
# fossick already reads disappears), with the narrative fields upgraded in
# place and the full `references` + `reporter` appended. Coords/datum stay last.
COL_ORDER = [
    "ardf_num", "mrds_num", "site", "location",
    "quad_250", "quad_63360",
    "comm_main", "comm_other", "ore", "gangue",
    "site_type", "status", "production",
    "dep_model", "model_code", "alteration",
    "geol_desc", "work_expl", "comments",
    "expand_ref", "rept_names", "rept_date", "age", "rept_aff",
    "prod_notes", "reserves", "prime_ref",
    "references", "reporter",
    "latitude", "longitude", "datum", "source_datum",
]


def flatten(value: object) -> str:
    """Collapse an ARDF JSON value (None / str / list[str]) to one string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [flatten(v) for v in value]
        return "\n\n".join(p for p in parts if p)
    return str(value).strip()


def flatten_bibliography(biblio: object) -> str:
    """Join the full reference list into one newline-separated string."""
    if not isinstance(biblio, list):
        return ""
    refs = [str(d.get("reference", "")).strip() for d in biblio if isinstance(d, dict)]
    return "\n".join(r for r in refs if r)


def flatten_reporter(reporter: object) -> str:
    """Render the reporter as 'Name (date, affiliation)'."""
    if not isinstance(reporter, dict):
        return ""
    name = str(reporter.get("name", "")).strip()
    date = str(reporter.get("date", "")).strip()
    aff = str(reporter.get("affiliation", "")).strip()
    tail = ", ".join(t for t in (date, aff) if t)
    return f"{name} ({tail})" if name and tail else name


def fetch_record(session: requests.Session, num: str, *, force: bool) -> dict | None:
    """Return the mrdata ARDF JSON for one record, caching the raw response."""
    cached = CACHE_DIR / f"{num}.json"
    if cached.exists() and not force:
        return json.loads(cached.read_text())
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(JSON_URL.format(num=num), timeout=30)
            if resp.status_code == 200:
                cached.write_text(resp.text)
                time.sleep(0.15)  # be polite to mrdata
                return resp.json()
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last_err = exc
        time.sleep(1.0 * (attempt + 1))
    print(f"  WARN {num}: fetch failed ({last_err}); keeping prior truncated text.")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refetch even if cached")
    args = parser.parse_args()

    if not PRIOR_GEOJSON.exists():
        raise FileNotFoundError(f"Prior WGS84 export not found: {PRIOR_GEOJSON}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fc = json.loads(PRIOR_GEOJSON.read_text())
    features = fc["features"]
    print(f"Prior export: {len(features)} features (geometry source, unchanged).")

    session = requests.Session()
    session.headers.update(UA)

    failed: list[str] = []
    enriched_count = {col: 0 for col in TEXT_MAP.values()}
    out_features: list[dict] = []

    for i, feat in enumerate(features, 1):
        prior = feat["properties"]
        num = prior["ardf_num"]
        rec = fetch_record(session, num, force=args.force)

        props = dict(prior)  # keep all prior cols incl. short attrs + coords + datum
        if rec is not None:
            jp = rec["properties"]
            if jp.get("ardf_num") != num:
                raise RuntimeError(f"ID mismatch: prior {num} vs json {jp.get('ardf_num')}")
            for src, dst in TEXT_MAP.items():
                full = flatten(jp.get(src))
                if full:  # never overwrite real text with a blank
                    props[dst] = full
                    if len(full) > 255:
                        enriched_count[dst] += 1
            props["references"] = flatten_bibliography(jp.get("bibliography"))
            props["reporter"] = flatten_reporter(jp.get("reporter"))
        else:
            failed.append(num)
            props.setdefault("references", "")
            props.setdefault("reporter", "")

        # Stable column order; geometry carried over verbatim. Blank out None and
        # the literal "nan"/"none" missing-markers the prior geopandas write left
        # behind, so the deliverable has no null/None/nan literals; numbers pass through.
        ordered = {}
        for c in COL_ORDER:
            v = props.get(c)
            if v is None or (isinstance(v, str) and v.strip().lower() in ("nan", "none")):
                ordered[c] = ""
            else:
                ordered[c] = v
        out_features.append({"type": "Feature", "properties": ordered, "geometry": feat["geometry"]})
        if i % 50 == 0:
            print(f"  ...{i}/{len(features)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fc = {"type": "FeatureCollection", "features": out_features}
    geojson_path = OUT_DIR / "nome_ardf_all_wgs84_fulltext.geojson"
    csv_path = OUT_DIR / "nome_ardf_all_wgs84_fulltext.csv"
    geojson_path.write_text(json.dumps(out_fc, ensure_ascii=False, indent=0))

    df = pd.DataFrame([f["properties"] for f in out_features])[COL_ORDER]
    df.to_csv(csv_path, index=False)

    # ---- verification ----
    print(f"\nWrote {geojson_path}")
    print(f"Wrote {csv_path}")
    print(f"records: {len(df)} (NM {sum(df.ardf_num.str.startswith('NM'))} + "
          f"SO {sum(df.ardf_num.str.startswith('SO'))})")
    if failed:
        print(f"FETCH FAILURES ({len(failed)}), kept prior truncated text: {failed}")
    else:
        print("all records enriched from mrdata JSON (no fetch failures)")

    print("\nfull-text field lengths (was 255 max in shapefile):")
    for col in ["location", "geol_desc", "work_expl", "prod_notes", "comments", "references"]:
        vals = df[col].astype(str)
        nonempty = (vals.str.len() > 0).sum()
        print(f"  {col:11s}: nonempty={nonempty:3d}  maxlen={vals.str.len().max():5d}  "
              f"records>255={enriched_count.get(col, 0)}")

    nm101 = df[df.ardf_num == "NM101"]
    if len(nm101):
        geom = next(f["geometry"] for f in out_features if f["properties"]["ardf_num"] == "NM101")
        print(f"\nNM101 coords (must match 6e99b9a WGS84 fix): {geom['coordinates']}")

    nm240 = df[df.ardf_num == "NM240"]
    if len(nm240):
        r = nm240.iloc[0]
        print(f"\nNM240 ({r.site}) geol_desc len={len(str(r.geol_desc))}:")
        print("  " + str(r.geol_desc)[:240] + " ...")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
