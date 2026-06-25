"""fossick Nome knowledge-graph export → canonical occurrences (ADR-017).

Per ADR-017 the occurrence labels are sourced from the fossick KG export
(`~/src/learning/fossick/exports/kg_nome.jsonld`) rather than re-derived from a
local ARDF shapefile pull. This adapter reads the export's `@graph`, pulls the
ARDF occurrence parcels (`/parcel/ardf/NM###`), and emits canonical occurrence
records: point geometry, free-text commodity, `usgs:<code>` deposit codes from
the depositType `cmmiCode`, and the source record id.

Scope and a known coordinate caveat — read before wiring into a model:

- Coverage is the Cape Nome bbox only (139 occurrences, lon[-165.59,-165.01]
  lat[64.44,64.70]). Big Hurrah and the eastern / northern / western district
  lodes are NOT in the export; the district-lode grid is wider than this. Use
  this source for labels that fall inside the Cape Nome AOI; keep the wider
  district lode labels on their own source until fossick widens the export bbox.

- The export's point WKT is bare (no CRS URI), so GeoSPARQL defaults it to
  CRS84 / WGS84. The underlying ARDF coordinates are NAD27 (the source
  shapefile is EPSG:4267), and the export emits the raw NAD27 numbers without a
  datum transform. The result is a systematic ~155 m SE offset from the
  geodetically-correct WGS84 position (~6 cells at 25 m). `source_crs` exists so
  a consumer can name the true datum (`EPSG:4267`) and reproject correctly. If a
  future export CRS-qualifies the WKT (`<...EPSG...4267> POINT(...)`), this
  adapter honors that tag and `source_crs` becomes the fallback only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.schemas import validate_occurrences

_ARDF_PARCEL_RE = re.compile(r"/parcel/ardf/(NM\w+)$")
_WKT_RE = re.compile(r"(?:<([^>]+)>\s*)?POINT\s*\(\s*([-0-9.]+)\s+([-0-9.]+)\s*\)", re.IGNORECASE)
_CODE_RE = re.compile(r"\b(\d{1,2}[a-z]?)\b", re.IGNORECASE)

_HAS_GEOMETRY = "http://www.opengis.net/ont/geosparql#hasGeometry"
_AS_WKT = "http://www.opengis.net/ont/geosparql#asWKT"
_DEPOSIT_TYPE = "https://johnsondevco.com/fossick/deposit#depositType"
_CMMI_CODE = "https://johnsondevco.com/fossick/deposit#cmmiCode"
_PREF_LABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
_CLAIM_NAME = "https://johnsondevco.com/fossick/landparcel#claimName"
_PARCEL_KIND = "https://johnsondevco.com/fossick/landparcel#parcelKind"

# CRS-URI fragment → EPSG code, for honoring a CRS-qualified WKT literal if a
# future export emits one. Bare WKT (no URI) falls through to `source_crs`.
_CRS_URI_RE = re.compile(r"(?:EPSG[:/])(\d{4,5})", re.IGNORECASE)


def _val(node: dict, key: str) -> str | None:
    vs = node.get(key)
    if not vs:
        return None
    v = vs[0]
    return v.get("@value", v.get("@id")) if isinstance(v, dict) else v


def _parse_codes(cmmi: str | None) -> tuple[str, ...]:
    """Extract Cox-&-Singer codes from a free-text cmmiCode, prefix `usgs:`."""
    if not cmmi:
        return ()
    return tuple(f"usgs:{m.lower()}" for m in _CODE_RE.findall(str(cmmi)))


def load(
    path: Path,
    aoi: AOI | None = None,
    *,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Read the fossick KG export and emit canonical ARDF occurrence records.

    `path` points at `kg_nome.jsonld` (or its directory). `source_crs` names the
    datum of bare (CRS-unqualified) point WKT; a CRS-qualified literal overrides
    it per-point. Set `source_crs="EPSG:4267"` to correct the current NAD27 export.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "kg_nome.jsonld"

    doc = json.loads(path.read_text())
    graph = doc if isinstance(doc, list) else doc.get("@graph", [])
    by_id = {n["@id"]: n for n in graph if isinstance(n, dict) and "@id" in n}

    geoms: list[Point] = []
    pt_crs: list[str] = []
    commodities: list[str | None] = []
    codes: list[tuple[str, ...]] = []
    names: list[str | None] = []
    kinds: list[str | None] = []
    rec_ids: list[str] = []

    for node in graph:
        if not isinstance(node, dict):
            continue
        m = _ARDF_PARCEL_RE.search(str(node.get("@id", "")))
        if not m:
            continue
        geom_ref = _val(node, _HAS_GEOMETRY)
        wkt = _val(by_id.get(geom_ref, {}), _AS_WKT) if geom_ref else None
        wm = _WKT_RE.search(wkt) if wkt else None
        if not wm:
            continue
        crs_uri, lon, lat = wm.group(1), float(wm.group(2)), float(wm.group(3))
        crs = source_crs
        if crs_uri:
            cm = _CRS_URI_RE.search(crs_uri)
            if cm:
                crs = f"EPSG:{cm.group(1)}"

        dep_ref = _val(node, _DEPOSIT_TYPE)
        dep = by_id.get(dep_ref, {}) if dep_ref else {}
        geoms.append(Point(lon, lat))
        pt_crs.append(crs)
        commodities.append(_val(dep, _PREF_LABEL))
        codes.append(_parse_codes(_val(dep, _CMMI_CODE)))
        names.append(_val(node, _CLAIM_NAME))
        kinds.append(_val(node, _PARCEL_KIND))
        rec_ids.append(m.group(1))

    if len(set(pt_crs)) > 1:
        raise ValueError(f"Mixed point CRS in KG export: {sorted(set(pt_crs))}; per-CRS reprojection not implemented.")
    in_crs = pt_crs[0] if pt_crs else source_crs

    out = gpd.GeoDataFrame(
        {
            "geometry": geoms,
            "commodity": pd.array(commodities, dtype="string"),
            "deposit_codes": codes,
            "year": pd.array([pd.NA] * len(geoms), dtype="Int64"),
            "source": "fossick-kg/ARDF",
            "raw_record_id": rec_ids,
            "name": pd.array(names, dtype="string"),
            "parcel_kind": pd.array(kinds, dtype="string"),
        },
        crs=in_crs,
    )
    if target_crs and str(out.crs) != target_crs:
        out = out.to_crs(target_crs)
    if aoi is not None:
        aoi_geom = gpd.GeoSeries([aoi.polygon], crs=aoi.crs).to_crs(out.crs).iloc[0]
        out = out[out.within(aoi_geom)].copy()
    return validate_occurrences(out.reset_index(drop=True))
