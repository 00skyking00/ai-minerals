"""Occurrence-level multi-creek power + barren-negatives report for the drill-gold validation.

Companion to ``drillgold_capture_validation.py``. That harness needs positioned
per-hole collars to sample the MPM at each hole; it can only run on Little Creek
(the Janin 1912 drill logs, 47 graded collars, one drainage). The coordinator's
2026-07-08 ML-step-3 dispatch asked to re-run the capture harness on "the NEW
multi-creek Tuck grade/drill holes now folded in" by the Part-2 consolidation.

Those new grades did NOT land as positioned collars. The fossick Part-2 fold-in
(``consolidation-part2-foldin-2026-07-08.md``) folded them at OCCURRENCE and AREA
extent: single occurrence points (Otter ardf-NM296, Nome River ardf-NM101, Anvil
ardf-NM236, Osborn ardf-NM305) and area hulls (Second Beach belt, B1 dredge field,
Anvil-Dexter dredge valley), each carrying a grade RANGE + valued/trace counts, not
individually-positioned holes. Fossick itself flagged that the folded grades live
only in the consolidation derived layer + ``ml_extract_examples.json`` and NOT in
``feature_table.csv`` (that export reads raw ``kg_nome.jsonld``, independent of the
consolidation layer).

So the capture-efficiency harness's power is unchanged (still one positioned
drainage). This script reports the power picture the coordinator asked for:

  1. Independent-drainage accounting: how many drainages carry a gold grade read,
     split by whether they are POSITIONED (collar-ready for the capture harness) vs
     OCCURRENCE / AREA extent (a range/aggregate at one point or hull).
  2. Barren-domain negatives from the Tuck corpus: the real sub-economic / barren
     reads (Otter 38 trace of 105, Nome River 2 of 8, Second Beach 6 barren of 9)
     that a future positioned run can use for the negative class, no fabrication.
  3. A descriptive placer-MPM percentile at each drainage's representative point.
     CAVEAT: these are known, already-explored occurrences that the served MPM was
     effectively fit over, so a high percentile is restriction-of-range, not an
     out-of-sample validation. It is a consistency read, not a G-result.

Grade facts are transcribed from the fossick Part-2 fold-in table (cited per row);
geometries are read live from the consolidated package so they stay reproducible.

Reads (this repo + one cross-repo READ of the fossick consolidation package):
  ~/src/learning/fossick/exports/consolidation/kg_nome_consolidated.jsonld
  data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_3338.tif
  data/derived/nome_placer/drillgold_capture/drillgold_capture_validation.json

Writes (this repo only):
  data/derived/nome_placer/drillgold_capture/drillgold_multicreek_power.json
  docs/drillgold_multicreek_power.md

Run: .venv/bin/python -m scripts.nome_placer.drillgold_multicreek_power
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from shapely import wkt as shapely_wkt

KG = Path("/home/sky/src/learning/fossick/exports/consolidation/kg_nome_consolidated.jsonld")
PLACER_MPM = Path("data/derived/nome_placer/mpm_onshore/mpm_onshore_score_district_3338.tif")
CAPTURE = Path("data/derived/nome_placer/drillgold_capture/drillgold_capture_validation.json")
OUT_JSON = Path("data/derived/nome_placer/drillgold_capture/drillgold_multicreek_power.json")
OUT_MD = Path("docs/drillgold_multicreek_power.md")

WKT_KEYS = ("https://johnsondevco.com/fossick/consolidation#wkt",
            "http://www.opengis.net/ont/geosparql#asWKT")
SOURCE_DOC = "fossick/docs/consolidation-part2-foldin-2026-07-08.md (Part-2 fold-in table)"

# Grade reads folded by the Part-2 consolidation, transcribed from the fossick
# fold-in table. `entity` resolves the geometry live from the consolidated
# package; `positioned` marks whether per-hole collars exist (only Little Creek).
# valued / barren counts are the real sub-economic reads for the negative class.
DRAINAGES = [
    {"drainage": "Little Creek", "entity": None, "extent": "positioned_collars",
     "positioned": True, "grade_read": "block grade 1.4-200 c/yd across 23 MPM blocks",
     "grade_median_c": 14.4, "n_valued": 17, "n_barren": 6, "basis": "$20.67",
     "provenance": "Janin 1912 Pioneer Mining Co. drill logs (bearcub); 47 collars",
     "note": "the one positioned drainage; already in drillgold_capture_validation.py"},
    {"drainage": "Otter Creek", "entity": "ardf-NM296", "extent": "occurrence_point",
     "positioned": False, "grade_read": "Fig 11 drill-section tenor 2.9-117.7 c/yd",
     "grade_median_c": 14.4, "n_valued": 63, "n_barren": 38, "basis": "$20",
     "provenance": "Fig 11 Lines 67/50/62/45/43/55/50A; 832.5 excluded as decimal misprint"},
    {"drainage": "Nome River", "entity": "ardf-NM101", "extent": "occurrence_point",
     "positioned": False, "grade_read": "Fig 11 Line 1 bedrock-channel tenor 1.0-13.4 c/yd",
     "grade_median_c": None, "n_valued": 6, "n_barren": 2, "basis": "$20",
     "provenance": "Fig 11 Line 1 (Nome River bedrock channel)"},
    {"drainage": "Anvil Creek", "entity": "ardf-NM236", "extent": "occurrence_point",
     "positioned": False, "grade_read": "Channel 4 grade 87 c/yd; Plein dredge 32 c/yd",
     "grade_median_c": None, "n_valued": None, "n_barren": None, "basis": "$20.67/mixed",
     "provenance": "Tuck 1942 p.159 (TUCK-005); Fig 10 dredge annotation; production-level"},
    {"drainage": "Osborn Creek", "entity": "ardf-NM305", "extent": "occurrence_point",
     "positioned": False, "grade_read": "production ~$4.96M (no c/yd tenor)",
     "grade_median_c": None, "n_valued": None, "n_barren": None, "basis": "mixed",
     "provenance": "Map E Osborn + Julien-dredge notes; production $ only, no grade tenor"},
    {"drainage": "Second Beach belt", "entity": "tuck_nome_second_beach_belt", "extent": "area_hull",
     "positioned": False, "grade_read": "panning 0-13 c/yd (favorability gradient)",
     "grade_median_c": None, "n_valued": 3, "n_barren": 6, "basis": "panning",
     "provenance": "Map C Sheet 8; 9 panning ticks, 6 barren"},
    {"drainage": "B1 Nome dredge field", "entity": "tuck_b1_nome_dredge_field", "extent": "area_hull",
     "positioned": False, "grade_read": "dredge recovery 3-71 c/yd (~383 cells, bulk 17-34)",
     "grade_median_c": None, "n_valued": None, "n_barren": None, "basis": "dredge recovery",
     "provenance": "Map B1 flat-scan Layer-2 (FlatPositioner georef)"},
    {"drainage": "Anvil-Dexter dredge valley", "entity": "tuck_anvil_dexter_dredge_valley",
     "extent": "area_hull", "positioned": False,
     "grade_read": "dredge recovery 16-68 c/yd (median, N=18)",
     "grade_median_c": None, "n_valued": None, "n_barren": None, "basis": "dredge recovery",
     "provenance": "Map D recovery cells"},
]


def _wkt_of(node: dict) -> str | None:
    for k in WKT_KEYS:
        v = node.get(k)
        if v:
            if isinstance(v, list):
                v = v[0]
            if isinstance(v, dict):
                v = v.get("@value", v)
            return str(v)
    return None


def load_geometries(kg_path: Path) -> dict[str, str]:
    """Map each wanted entity id to its WKT, read live from the consolidated package."""
    graph = json.loads(kg_path.read_text())["@graph"]
    wanted = {d["entity"] for d in DRAINAGES if d["entity"]}
    out: dict[str, str] = {}
    for node in graph:
        nid = node.get("@id", "")
        for ent in wanted:
            if nid.endswith(ent):
                w = _wkt_of(node)
                if w:
                    out[ent] = w
    return out


def _served_distribution(raster: Path) -> np.ndarray:
    with rasterio.open(raster) as ds:
        a = ds.read(1)
        return a[a != ds.nodata]


def sample_percentile(raster: Path, lon: float, lat: float,
                      served: np.ndarray) -> tuple[float | None, float | None]:
    """Placer-MPM value + its percentile on the served surface at a lon/lat point.

    Returns (None, None) when the point falls on nodata / outside the MPM footprint.
    """
    tr = Transformer.from_crs(4326, 3338, always_xy=True)
    x, y = tr.transform(lon, lat)
    with rasterio.open(raster) as ds:
        v = next(ds.sample([(x, y)]))[0]
        nod = ds.nodata
    if v == nod or not np.isfinite(v):
        return None, None
    pct = float((served <= v).mean() * 100.0)
    return float(v), pct


def representative_point(wkt_str: str) -> tuple[float, float]:
    """Occurrence point as-is; area hull -> representative interior point (lon, lat)."""
    geom = shapely_wkt.loads(wkt_str)
    p = geom if geom.geom_type == "Point" else geom.representative_point()
    return float(p.x), float(p.y)


def build_report() -> dict:
    geoms = load_geometries(KG)
    served = _served_distribution(PLACER_MPM)
    capture = json.loads(CAPTURE.read_text())

    rows = []
    for d in DRAINAGES:
        row = {k: d[k] for k in ("drainage", "entity", "extent", "positioned",
                                 "grade_read", "grade_median_c", "n_valued", "n_barren",
                                 "basis", "provenance")}
        if d["entity"] and d["entity"] in geoms:
            lon, lat = representative_point(geoms[d["entity"]])
            v, pct = sample_percentile(PLACER_MPM, lon, lat, served)
            row["repr_lon_lat"] = [round(lon, 5), round(lat, 5)]
            row["placer_mpm_value"] = None if v is None else round(v, 4)
            row["placer_mpm_percentile"] = None if pct is None else round(pct, 1)
        else:
            row["repr_lon_lat"] = None
            row["placer_mpm_value"] = None
            row["placer_mpm_percentile"] = None
        rows.append(row)

    positioned = [r for r in rows if r["positioned"]]
    occ_area_grade = [r for r in rows if not r["positioned"]
                      and r["basis"] != "mixed"]  # has a c/yd grade read
    negatives = [{"drainage": r["drainage"], "extent": r["extent"],
                  "n_barren_or_trace": r["n_barren"], "n_valued": r["n_valued"],
                  "provenance": r["provenance"]}
                 for r in rows if r["n_barren"]]
    total_neg = sum(r["n_barren"] for r in rows if r["n_barren"])

    return {
        "task": "coordinator 2026-07-08 ML-step-3 (a): drill-gold on newly-consolidated multi-creek Tuck grades",
        "headline": (
            "The Part-2 consolidation added ZERO positioned drill collars. The new "
            "multi-creek Tuck grades were folded at occurrence / area / claim-stub "
            "extent (grade ranges + valued/trace counts at one point or hull), which "
            "the per-hole capture-efficiency harness cannot consume. So the capture "
            "test's power is unchanged: one positioned drainage (Little Creek)."),
        "capture_harness_state": {
            "script": "scripts/nome_placer/drillgold_capture_validation.py",
            "n_positioned_holes": capture["ground_truth"]["n_holes"],
            "n_positioned_drainages": capture["ground_truth"]["n_drainages"],
            "placer_auc_at_10c_cutoff": capture["placer_mpm"]["capture_by_cutoff"]
                ["cutoff_10c"]["auc_mpm_as_econ_classifier"],
            "verdict": ("unchanged vs the committed 2026-07-05 run: one drainage, "
                        "underpowered, AUC ~ chance with a CI spanning 0.19-0.75; a "
                        "LABELED methods-validation baseline, not a G-result"),
        },
        "power_accounting": {
            "positioned_collar_drainages_capture_ready": len(positioned),
            "occurrence_or_area_drainages_with_grade_read": len(occ_area_grade),
            "occurrence_or_area_drainages_production_only": sum(
                1 for r in rows if not r["positioned"] and r["basis"] == "mixed"),
            "total_independent_drainages_with_any_gold_grade": len(positioned) + len(occ_area_grade),
            "interpretation": (
                "positioned power stays at 1. The consolidation raises the CEILING for "
                "a future positioned multi-creek capture run to ~7 drainages, but only "
                "once fossick positions the Fig-10/11 drill lines (Otter 7 lines, Nome "
                "River Line 1) and the Map A Bay/Odin lines as collar points. Until "
                "then these are ranges at a point/hull, not test power."),
        },
        "barren_domain_negatives": {
            "note": ("real sub-economic / barren reads from the Tuck corpus (not "
                     "fabricated pseudo-absences); usable for the negative class in a "
                     "future positioned run"),
            "total_barren_or_trace_reads": int(total_neg),
            "by_drainage": negatives,
        },
        "placer_mpm_consistency_read": {
            "caveat": (
                "known, already-explored occurrences the served placer MPM was fit over; "
                "a high percentile is restriction of range, NOT out-of-sample validation. "
                "Descriptive consistency only."),
            "by_drainage": [{"drainage": r["drainage"], "extent": r["extent"],
                             "placer_mpm_percentile": r["placer_mpm_percentile"]}
                            for r in rows],
        },
        "drainages": rows,
        "provenance": {"grade_facts": SOURCE_DOC,
                       "geometries": "kg_nome_consolidated.jsonld (@graph wkt), read live"},
    }


def write_markdown(report: dict) -> None:
    r = report
    pa = r["power_accounting"]
    ch = r["capture_harness_state"]
    lines = [
        "# Drill-gold multi-creek power + barren negatives (ML step-3 (a))",
        "",
        "*Occurrence-level power report companion to the positioned capture harness.*",
        "",
        "## The short answer",
        "",
        r["headline"],
        "",
        "## Power accounting: independent drainages",
        "",
        "| class | count |",
        "|---|--:|",
        f"| positioned-collar drainages (capture-harness-ready) | {pa['positioned_collar_drainages_capture_ready']} |",
        f"| occurrence/area drainages with a c/yd grade read | {pa['occurrence_or_area_drainages_with_grade_read']} |",
        f"| occurrence drainages production-$ only (no tenor) | {pa['occurrence_or_area_drainages_production_only']} |",
        f"| total independent drainages with any gold grade | {pa['total_independent_drainages_with_any_gold_grade']} |",
        "",
        pa["interpretation"],
        "",
        f"The positioned capture harness is unchanged: {ch['n_positioned_holes']} holes, "
        f"{ch['n_positioned_drainages']} drainage (Little Creek), placer AUC at the 10 c/yd "
        f"cutoff = {ch['placer_auc_at_10c_cutoff']:.2f}. {ch['verdict']}",
        "",
        "## Barren-domain negatives from the Tuck corpus",
        "",
        f"{r['barren_domain_negatives']['note'].capitalize()}. "
        f"Total real barren/trace reads: **{r['barren_domain_negatives']['total_barren_or_trace_reads']}**.",
        "",
        "| drainage | extent | barren/trace | valued |",
        "|---|---|--:|--:|",
    ]
    for n in r["barren_domain_negatives"]["by_drainage"]:
        lines.append(f"| {n['drainage']} | {n['extent']} | {n['n_barren_or_trace']} | "
                     f"{n['n_valued'] if n['n_valued'] is not None else '-'} |")
    lines += [
        "",
        "## Placer-MPM consistency read (descriptive, restriction-of-range)",
        "",
        r["placer_mpm_consistency_read"]["caveat"],
        "",
        "| drainage | extent | placer MPM percentile |",
        "|---|---|--:|",
    ]
    for c in r["placer_mpm_consistency_read"]["by_drainage"]:
        pct = c["placer_mpm_percentile"]
        lines.append(f"| {c['drainage']} | {c['extent']} | "
                     f"{pct if pct is not None else 'outside MPM footprint'} |")
    lines += [
        "",
        "## What a positioned multi-creek run needs from fossick",
        "",
        "1. The Fig-10/11 drill-section holes positioned as collar points (Otter 7 lines, "
        "Nome River Line 1) with per-hole grade, not an occurrence-level range.",
        "2. The Map A Bay / Odin drill lines de-stubbed from `stub_pending_mtp` points to "
        "surveyed collar positions (pending BLM MTP).",
        "3. Either of the above written into a positioned drill export the capture harness "
        "reads (`exports/phase4/drill_gold_points.geojson`), not only the occurrence/area "
        "consolidation layer.",
        "",
        f"*Grade facts: {r['provenance']['grade_facts']}. "
        f"Geometries: {r['provenance']['geometries']}.*",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print(json.dumps(report["power_accounting"], indent=2))
    print(json.dumps(report["barren_domain_negatives"], indent=2))
    print("\nMPM consistency read:")
    for c in report["placer_mpm_consistency_read"]["by_drainage"]:
        print(f"  {c['drainage']:28s} {c['extent']:18s} pct={c['placer_mpm_percentile']}")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
