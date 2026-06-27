"""Join fossick's typed Tuck-1942 placer areas to authoritative MS-survey coords.

Round-4B powered retest, step 1 (the consumer + map-georef lane, ADR-002).
fossick OCR'd Tuck 1942 and extracted 181 typed ``TuckPlacerArea`` records (41
true-beach, 31 abrasion-platform, 87 upland-residual, 22 unknown) carrying 315
distinct claim names. None of the records carry an MS number directly, so the
join key is the claim NAME, matched against the patented mineral-survey layer.

Coordinate authority (handoff): an MS-numbered patented claim gets the BLM survey
polygon centroid, which is independent of the Tuck plate overlay's 200-500 m
georef offset. So for any area whose claim joins to a survey we use the survey
centroid, never the Tuck raster position. Areas that do not join (buried
strandlines, offshore paystreaks with no patent) are flagged pinpoint-needed and
left for the goldbug pinpoint tool / coordinator tile-composite to place.

Join sources, highest authority first:
  1. bearcub curated crosswalk (``tuck1942_claim_crosswalk.csv``): human-vetted
     Tuck-name -> MS links, confidence-graded. The family + neighbourhood claims.
  2. goldbug published surveys (``nome_mineral_surveys.geojson``, 372 surveys /
     340 named): BLM CadNSDI + DNR + taxroll + curated, MS + name + polygon.
  3. bearcub cadastre draft polygons (``tuck1942_claim_cadastre_draft.geojson``):
     geometry for the curated set, used as a centroid fallback when a crosswalk
     MS is absent from the published-survey layer (the family beach-line block).

Name matching is deliberately precision-biased: a false coordinate corrupts the
downstream per-subtype CV worse than a miss does. Only exact normalised matches
and whole-phrase substring matches on a distinctive (non-generic) claim name are
accepted; everything else is left un-joined and reported as pinpoint-needed.

Run (audit): PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.tuck_placer_join_coords --audit
Run (write): PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.tuck_placer_join_coords
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

LEARNING = Path(__file__).resolve().parents[3]          # /home/sky/src/learning
FOSSICK_AREAS = LEARNING / "fossick/samples/tuck/tuck1942_areas.json"
GOLDBUG_SURVEYS = LEARNING / "gldbg/data/published/nome_mineral_surveys.geojson"
BEARCUB_CADASTRE = LEARNING / "bearcub/research/tuck1942_cadastre/tuck1942_claim_cadastre_draft.geojson"
BEARCUB_CROSSWALK = LEARNING / "bearcub/research/tuck1942_cadastre/tuck1942_claim_crosswalk.csv"

OUT_DIR = Path("data/derived/nome_placer/subtype_local_source")
COASTAL = {"true-beach", "abrasion-platform"}
# Canonical type vocab shared with placer_subtype_local_source.py. Tuck's
# "true-beach" is Hudson's strandline-beach (winnowed marine drift); kept under
# one canonical label so the KG-8 and the Tuck set merge cleanly.
TYPE_CANON = {"true-beach": "strandline_beach", "abrasion-platform": "abrasion_platform",
              "upland-residual": "upland_residual", "unknown": "broad_ambiguous"}

# Decorator tokens that carry no claim identity; dropped before matching.
_DECOR = {"PLCR", "PLACER", "CLAIM", "CLM", "CLAIMS", "MINING", "MS", "USMS",
          "ASSOCIATION", "ASSN", "ASSOC", "MINERAL", "SURVEY", "PLAT"}
# Semantically generic claim tokens: ordinals, beach descriptors, compass and
# size words, common mining nouns. A single-token claim that is one of these
# never identifies a survey on its own ("Second" must not match "No. 2 Bench
# Second Tier"; "Paystreak" must not match "Paystreak Bench Placer"). Multi-token
# claims that merely contain one of these are unaffected.
_STOP = {"FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "PRESENT", "OFFSHORE",
         "SUBMARINE", "BEACH", "BEACHLINE", "BENCH", "PAYSTREAK", "DISCOVERY",
         "NORTH", "SOUTH", "EAST", "WEST", "UPPER", "LOWER", "MIDDLE", "MAIN",
         "NEW", "OLD", "BIG", "LITTLE", "GULCH", "CREEK", "RIVER", "FRACTION",
         "LOCALITY", "AREA", "INTERMEDIATE", "CENTER", "FREECOURT"}
SPREAD_TRUST_M = 1000.0      # multi-claim area coord trusted only if claims cluster tighter


def normalize(name: str) -> str:
    """Uppercase, strip the MS suffix and placer/claim decorators, collapse to a
    canonical token string for matching."""
    s = str(name or "").upper()
    s = re.sub(r"\b(?:US)?MS\.?\s*0*\d+\b", " ", s)        # drop ", MS 1209" etc.
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    toks = [t for t in s.split() if t and t not in _DECOR]
    return " ".join(toks)


def generic_tokens(survey_norms: list[str], frac: float = 0.03) -> set[str]:
    """Tokens appearing in >=frac of survey names: too common to identify a claim
    on their own (BENCH, FRACTION, ABOVE, creek words, ...)."""
    df = Counter()
    for n in survey_norms:
        df.update(set(n.split()))
    thr = max(2, int(frac * len(survey_norms)))
    return {t for t, c in df.items() if c >= thr}


def load_surveys() -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Published mineral surveys -> (DataFrame[MS, name, norm, x, y], {MS: (x,y)}),
    centroids in EPSG:3338."""
    g = gpd.read_file(GOLDBUG_SURVEYS).to_crs("EPSG:3338")
    cen = g.geometry.centroid
    g = g.assign(x=cen.x.to_numpy(), y=cen.y.to_numpy())
    g["MS"] = g["MS"].astype(str).str.strip().str.lstrip("0")
    g["norm"] = [normalize(n) for n in g["name"]]
    by_ms = {ms: (float(x), float(y)) for ms, x, y in zip(g["MS"], g["x"], g["y"])}
    return g[["MS", "name", "norm", "x", "y"]].copy(), by_ms


def load_cadastre_centroids() -> dict[str, tuple[float, float]]:
    """bearcub cadastre draft polygons -> {MS: centroid_3338} (family beach block
    fallback for MS absent from the published-survey layer)."""
    if not BEARCUB_CADASTRE.exists():
        return {}
    g = gpd.read_file(BEARCUB_CADASTRE).to_crs("EPSG:3338")
    out: dict[str, tuple[float, float]] = {}
    cen = g.geometry.centroid
    for ms, x, y in zip(g.get("ms", pd.Series([None] * len(g))), cen.x, cen.y):
        m = str(ms).strip().lstrip("0")
        if m and m not in ("None", "nan"):
            out.setdefault(m, (float(x), float(y)))
    return out


def load_crosswalk() -> list[tuple[str, str, str]]:
    """bearcub curated crosswalk -> [(normalized tuck_name, MS, confidence)] for
    rows that resolved to a survey number."""
    if not BEARCUB_CROSSWALK.exists():
        return []
    out = []
    for r in csv.DictReader(BEARCUB_CROSSWALK.open()):
        ms = str(r.get("matched_ms") or "").strip().lstrip("0")
        if ms and ms.isdigit():
            out.append((normalize(r.get("tuck_name", "")), ms, r.get("match_confidence", "")))
    return out


def candidate_names(area: dict) -> list[str]:
    """Claim names for an area, plus a name distilled from the area title (so a
    'Jupiter claim, Third Beach' area joins on 'Jupiter' even if claim_names is
    sparse). Parentheticals and the beach descriptor are dropped from the title."""
    names = list(area.get("claim_names") or [])
    title = re.sub(r"\(.*?\)", " ", area.get("name", ""))
    title = re.split(r",| near | between | and adjoining | beachline| beach\b",
                     title, flags=re.I)[0]
    title = re.sub(r"\b(claim|claims|area|locality|deposit|concentration|shafts?|"
                   r"open ?cuts?|drift[- ]?mine|past|production)\b", " ", title, flags=re.I)
    title = title.strip()
    if title and len(title.split()) <= 4:
        names.append(title)
    return names


def build_matcher(surveys: pd.DataFrame, crosswalk: list[tuple[str, str, str]]):
    """Return match(name) -> (MS, confidence, matched_label) or None."""
    generic = generic_tokens(surveys["norm"].tolist())
    exact: dict[str, tuple[str, str]] = {}                # norm -> (MS, survey name)
    for ms, nm, nn in zip(surveys["MS"], surveys["name"], surveys["norm"]):
        if nn:
            exact.setdefault(nn, (ms, nm))
    cross = {n: ms for n, ms, _ in crosswalk if n}
    survey_rows = list(zip(surveys["MS"], surveys["name"], surveys["norm"]))

    def distinctive(nn: str) -> bool:
        toks = nn.split()
        if not toks:
            return False
        if len(toks) == 1:                # a lone token must be rare, long, non-generic
            t = toks[0]
            return t not in generic and t not in _STOP and len(t) >= 6
        return any(t not in generic and t not in _STOP for t in toks) and len(nn) >= 4

    def match(name: str):
        nn = normalize(name)
        if not nn:
            return None
        if nn in cross:                                   # 1. curated crosswalk
            return cross[nn], "high_curated", f"crosswalk:{name}"
        if nn in exact:                                   # 2. exact survey-name match
            ms, lab = exact[nn]
            return ms, "high_exact", lab
        if distinctive(nn):                               # 3. whole-phrase substring
            hits = [(ms, lab) for ms, lab, snn in survey_rows
                    if re.search(rf"(?:^| ){re.escape(nn)}(?: |$)", snn)]
            if len(hits) == 1:
                return hits[0][0], "medium_phrase", hits[0][1]
        return None

    return match, generic


def join_areas(audit: bool = False) -> tuple[list[dict], dict]:
    areas = json.loads(FOSSICK_AREAS.read_text())["areas"]
    surveys, ms_xy = load_surveys()
    cad_xy = load_cadastre_centroids()
    crosswalk = load_crosswalk()
    match, _ = build_matcher(surveys, crosswalk)

    def ms_coord(ms: str):
        return ms_xy.get(ms) or cad_xy.get(ms)

    rows: list[dict] = []
    for a in areas:
        ctype = a["deposit_type"]
        hits: list[tuple[str, str, str, tuple[float, float]]] = []  # claim, MS, conf, xy
        seen_ms: set[str] = set()
        for cand in candidate_names(a):
            m = match(cand)
            if not m:
                continue
            ms, conf, lab = m
            xy = ms_coord(ms)
            if xy is None or ms in seen_ms:
                continue
            seen_ms.add(ms)
            hits.append((cand, ms, conf, xy))
        curated = [h for h in hits if h[2] == "high_curated"]
        xs = np.array([h[3][0] for h in hits]); ys = np.array([h[3][1] for h in hits])
        spread = (float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
                  if len(hits) > 1 else 0.0)
        # Trust gate: a curated crosswalk link is authoritative on its own; otherwise
        # require >=2 independent claim matches that agree on location (cluster < 1 km).
        # A lone non-curated match, or matches that scatter, is left for pinpoint.
        if curated:
            cx = np.array([h[3][0] for h in curated]); cy = np.array([h[3][1] for h in curated])
            x, y, conf = float(cx.mean()), float(cy.mean()), "high_curated"
            accept = True
        elif len(hits) >= 2 and spread <= SPREAD_TRUST_M:
            x, y, conf = float(xs.mean()), float(ys.mean()), "medium_cluster"
            accept = True
        else:
            accept = False
        if accept:
            rows.append({
                "slug": a["slug"], "name": a["name"], "deposit_type": ctype,
                "type_canon": TYPE_CANON[ctype], "coordinate_source": "ms_join",
                "x_3338": round(x, 1), "y_3338": round(y, 1),
                "n_ms_matched": len(hits), "match_spread_m": round(spread, 1),
                "best_confidence": conf, "matched_ms": ";".join(h[1] for h in hits),
                "matched_claims": ";".join(h[0] for h in hits),
                "source_pages": ";".join(str(p) for p in (a.get("source_scan_pages") or [])),
            })
        else:
            # Record any weak/scattered matches that failed the trust gate, so the
            # pinpoint operator sees what survey names brushed near this area.
            rows.append({
                "slug": a["slug"], "name": a["name"], "deposit_type": ctype,
                "type_canon": TYPE_CANON[ctype], "coordinate_source": "pinpoint_needed",
                "x_3338": "", "y_3338": "", "n_ms_matched": len(hits),
                "match_spread_m": round(spread, 1) if len(hits) > 1 else "",
                "best_confidence": "rejected" if hits else "",
                "matched_ms": ";".join(h[1] for h in hits),
                "matched_claims": ";".join(h[0] for h in hits),
                "source_pages": ";".join(str(p) for p in (a.get("source_scan_pages") or [])),
            })

    coverage = coverage_report(rows)
    if audit:
        audit_dump(rows, areas, match)
    return rows, coverage


def coverage_report(rows: list[dict]) -> dict:
    def tally(subset):
        joined = sum(1 for r in subset if r["coordinate_source"] == "ms_join")
        return {"total": len(subset), "ms_joined": joined,
                "pinpoint_needed": len(subset) - joined}
    by_type = {t: tally([r for r in rows if r["deposit_type"] == t])
               for t in ("true-beach", "abrasion-platform", "upland-residual", "unknown")}
    coastal = [r for r in rows if r["deposit_type"] in COASTAL]
    return {"all": tally(rows), "coastal": tally(coastal), "by_type": by_type}


def audit_dump(rows, areas, match) -> None:
    by_slug = {a["slug"]: a for a in areas}
    print("\n=== COASTAL MATCH AUDIT (every accepted match) ===")
    for r in rows:
        if r["deposit_type"] not in COASTAL:
            continue
        if r["coordinate_source"] == "ms_join":
            a = by_slug[r["slug"]]
            details = []
            for cand in candidate_names(a):
                m = match(cand)
                if m:
                    details.append(f"{cand!r}->MS{m[0]}({m[1]}) [{m[2]}]")
            print(f"[{r['deposit_type'][:5]}] {r['name'][:46]:46s} sprd={r['match_spread_m']}m")
            for d in details:
                print(f"        {d}")
        else:
            print(f"[{r['deposit_type'][:5]}] {r['name'][:46]:46s} -- PINPOINT NEEDED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="dump every match for inspection, no write")
    args = ap.parse_args()
    rows, coverage = join_areas(audit=args.audit)

    print("\n=== COVERAGE ===")
    print(json.dumps(coverage, indent=2))
    if args.audit:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["slug", "name", "deposit_type", "type_canon", "coordinate_source",
            "x_3338", "y_3338", "n_ms_matched", "match_spread_m", "best_confidence",
            "matched_ms", "matched_claims", "source_pages"]
    out_csv = OUT_DIR / "tuck_placer_positives.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "tuck_placer_join_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(f"\nwrote {out_csv}")
    print(f"wrote {OUT_DIR / 'tuck_placer_join_coverage.json'}")


if __name__ == "__main__":
    main()
