"""Round 5 step 1: genetic typing of Nome-district placer occurrences.

Types every ARDF placer occurrence inside the IfSAR DEM extent into one of
{alluvial-stream, residual-eluvial, marine-beach, glacial, ambiguous}. The
site NAME is the primary signal (Nome beaches are named "X Beach"; the inland
stream placers "X Creek/Gulch/River" and the stream-terrace placers "X
Bench"); a keyword score over the untruncated ARDF ``geol_desc`` narrative is
the fallback for names without a geomorphic noun. Occurrence elevation is a
tiebreaker only. Emits a typed GeoJSON in EPSG:3338 with the per-record basis
so the typing is auditable.

The strictly-alluvial subset is the H1 positive set. Residual/eluvial is
FILTERED OUT (distance-zero tautology, per the experiment spec); marine-beach
and glacial are retained only as the negative control. Ambiguous records are
kept in the file but flagged and used in no set.

Coarseness ("fine/flaky" -> 1, "coarse" -> 2, "rough/nuggety/quartz-attached"
-> 3) is mined from the same narrative for H2; null where no tag is found.

Run: uv run python -m scripts.nome_placer.inland_local_source.type_placers
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

ARDF = Path("/home/sky/src/learning/fossick/samples/ardf/nome_ardf_all.geojson")
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
OUT = Path("data/derived/nome_placer/inland_local_source/placers_typed.geojson")
AUDIT = Path("data/derived/nome_placer/inland_local_source/placers_typed_audit.csv")

# The site name carries the genetic type most reliably.
NAME_MARINE = r"\bbeach\b"
NAME_ALLUVIAL = r"\bcreek\b|\bgulch\b|\briver\b|\bbench\b|\bbar\b|\bcut\b"
# District-wide / coastal-plain summaries are not point occurrences.
NAME_AGGREGATE = r"\bdistrict\b|placer field|coastal plain|mining region|\bregion\b"

# Narrative cues describing the DEPOSIT itself, kept strict so that the
# regional beach/glacial discussion present in nearly every Nome narrative
# does not fire. Bare "drift" is excluded (it means drift mining here).
MARINE = [
    r"beach placer", r"marine placer", r"\bstrandline\b", r"wave[- ]?cut",
    r"stillstand", r"raised beach", r"submarine beach", r"ruby sand",
    r"\bmonroeville\b", r"old beach line", r"marine reworking",
]
GLACIAL = [
    r"glacial gravel", r"glaciofluvial", r"\bmorain", r"\boutwash\b",
    r"glacial deposit", r"in (the )?glacial", r"glacially derived",
    r"ice[- ]?transport", r"derived from .{0,20}glaci",
]
RESIDUAL = [
    r"residual placer", r"\beluvial\b", r"eluvium", r"\bin[- ]?place\b",
    r"bedrock concentration", r"decomposed bedrock", r"weathered bedrock",
    r"saprolite", r"\bin situ\b", r"residual concentration", r"\bautochthon",
]
ALLUVIAL = [
    r"\bcreek\b", r"\bgulch\b", r"\bstream\b", r"\briver\b", r"alluvi",
    r"\bbench\b", r"paystreak", r"\bchannel\b", r"tributary", r"flood ?plain",
    r"\bdrainage\b", r"stream gravel", r"creek gravel", r"bench gravel",
]

COARSE_CUES = {
    3: [r"nugget", r"rough gold", r"quartz[- ]?attached", r"angular gold",
        r"country rock adher", r"coarse and rough", r"rough and coarse"],
    2: [r"\bcoarse\b", r"coarse gold", r"\bcoarser\b"],
    1: [r"\bfine\b", r"\bflaky\b", r"\bflour", r"fine[- ]?grained gold",
        r"thin flakes", r"\bfine gold\b", r"flat gold"],
}


def _score(pats: list[str], text: str) -> int:
    return sum(1 for p in pats if re.search(p, text, flags=re.IGNORECASE))


def classify(geol: str, site: str, elev_m: float) -> tuple[str, str]:
    """Return (genetic_type, basis). Name first, then narrative score."""
    name = site.lower()
    # 0) District-wide / coastal-plain aggregates are not point occurrences.
    if re.search(NAME_AGGREGATE, name):
        return "aggregate", f"name-aggregate:{re.search(NAME_AGGREGATE, name).group()}"
    # 1) Name-based: a "beach" name is marine; a stream/bench noun is alluvial.
    if re.search(NAME_MARINE, name):
        return "marine-beach", f"name:{re.search(NAME_MARINE, name).group()}"
    if re.search(NAME_ALLUVIAL, name):
        # honor an explicit residual deposit description over the stream name
        if _score(RESIDUAL, geol) >= 2 and _score(ALLUVIAL, geol) == 0:
            return "residual-eluvial", "name-stream-but-residual-narrative"
        return "alluvial-stream", f"name:{re.search(NAME_ALLUVIAL, name).group()}"

    # 2) Narrative scoring for names without a geomorphic noun.
    s = {
        "alluvial-stream": _score(ALLUVIAL, geol),
        "marine-beach": _score(MARINE, geol),
        "glacial": _score(GLACIAL, geol),
        "residual-eluvial": _score(RESIDUAL, geol),
    }
    basis = "narrative:" + ",".join(f"{k.split('-')[0]}={v}" for k, v in s.items())
    best = max(s, key=s.get)
    top = s[best]
    if top == 0:
        return "ambiguous", basis + "|no-cues"
    if list(s.values()).count(top) > 1:
        if np.isfinite(elev_m) and elev_m < 15 and s["marine-beach"] == top:
            return "marine-beach", basis + "|tie->coastal"
        return "ambiguous", basis + "|tie"
    return best, basis


def mine_coarseness(geol: str) -> int | None:
    for rank in (3, 2, 1):
        for cue in COARSE_CUES[rank]:
            if re.search(cue, geol, flags=re.IGNORECASE):
                return rank
    return None


def main() -> None:
    ardf = gpd.read_file(ARDF).to_crs("EPSG:3338")
    with rasterio.open(DEM) as ds:
        bl, bb, br, bt = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
        nod = ds.nodata
    aoi = ardf.cx[bl:br, bb:bt].copy()
    placer = aoi[aoi["model_code"].astype(str).str.contains("39")].copy()

    xs = placer.geometry.x.to_numpy()
    ys = placer.geometry.y.to_numpy()
    with rasterio.open(DEM) as ds:
        elev = np.array([v[0] for v in ds.sample(zip(xs, ys))], dtype=float)
    elev = np.where(elev == nod, np.nan, elev)
    placer["elev_m"] = elev

    types, bases, coarse = [], [], []
    for (_, row), e in zip(placer.iterrows(), elev):
        geol = str(row.get("geol_desc") or "")
        site = str(row.get("site") or "")
        t, b = classify(geol, site, e)
        types.append(t)
        bases.append(b)
        coarse.append(mine_coarseness(geol))
    placer["geol_type"] = types
    placer["type_basis"] = bases
    placer["coarseness_rank"] = coarse

    keep = ["ardf_num", "site", "model_code", "comm_main", "elev_m",
            "geol_type", "type_basis", "coarseness_rank", "geometry"]
    out = placer[keep].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT, driver="GeoJSON")

    audit = out.drop(columns="geometry").copy()
    audit["geol_desc_head"] = placer["geol_desc"].astype(str).str.slice(0, 160).values
    audit.to_csv(AUDIT, index=False)

    counts = out["geol_type"].value_counts().to_dict()
    print("typed placer counts:", json.dumps(counts, indent=2))
    al = out[out.geol_type == "alluvial-stream"]
    print("alluvial coarseness ranks:",
          al["coarseness_rank"].value_counts(dropna=False).to_dict())
    print(f"wrote {OUT}  ({len(out)} records)")


if __name__ == "__main__":
    main()
