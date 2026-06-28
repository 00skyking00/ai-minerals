"""H2 redesign step 1: schist-limestone (marble) contact from DGGS RI 2024-7.

The redesigned local-source test needs a clean schist-vs-marble lithologic
contact as the predictor. At Nome the only carbonate is the mixed DOx unit, so
no clean contact exists in-district (the round-5 null). RI 2024-7 (Werdon et
al. 2024, Big Hurrah-Council-Bluff, 1:50,000) subdivides that mixed package
into clean marble (DOm/Dm) vs schist/quartzite (DOs/DOq + the other Nome Group
schists), so the schist-limestone contact is mappable here.

Method (same as the round-5 derive_schist_limestone_contact.py): classify the
GeMS MapUnitPolys by the published DescriptionOfMapUnits lithology, dissolve the
marble set and the schist set, and take the shared boundary as the contact line.
The contact is written as a polyline layer in EPSG:3338; the down-channel
distance feature (build_terrain -> build_distance) intersects it with the
DEM-derived stream network later.

Three contact definitions are produced so the test's sensitivity to the unit
call is explicit:
  primary  : marble {DOm,Dm} vs schist {DOs,DOq,DOg,Omg,DOms,Ds,DOsq,DOqs,Osg,DOu}
  literal  : DOm vs {DOs,DOq}            (the coordinator's named pair)
  inclusive: {DOm,Dm,DOi,Oi} vs schist  (impure marbles count as carbonate)

Run: uv run python -m scripts.nome_placer.h2_confined_reach.build_contact
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely import line_merge

GEMS = Path(
    "data/raw/dggs_ri2024_7/extracted/pkg/"
    "casadepaga_bedrock_gems_db_wo_stations-open/GM_MapUnitPolys.shp"
)
OUT = Path("data/derived/nome_placer/h2_confined_reach")

# Lithology from RI 2024-7 DescriptionOfMapUnits.csv:
#   DOm Marble, Dm Marble, DOi Impure Marble, Oi Impure Marble
#   DOs Schist, DOq Quartzite, DOg/Omg mafic schist+granofels, DOms Chlorite
#   Schist, Ds Qtz-mica schist, DOsq Graphitic schist+qtzite, DOqs Qtzite+schist,
#   Osg metased+metagraywacke, DOu undiff Nome Group, PzPh/PzPa schist
#   DOx Mixed Metasedimentary (ambiguous -- excluded), Qb basalt (excluded)
MARBLE_PURE = {"DOm", "Dm"}
MARBLE_IMPURE = {"DOi", "Oi"}
SCHIST_PACKAGE = {"DOs", "DOq", "DOg", "Omg", "DOms", "Ds", "DOsq", "DOqs",
                  "Osg", "DOu", "PzPh", "PzPa"}
SCHIST_LITERAL = {"DOs", "DOq"}


def contact_line(polys: gpd.GeoDataFrame, marble: set, schist: set):
    """Shared boundary between the dissolved marble set and schist set."""
    m = polys[polys.MapUnit.isin(marble)]
    s = polys[polys.MapUnit.isin(schist)]
    if len(m) == 0 or len(s) == 0:
        return None, 0.0
    contact = m.union_all().boundary.intersection(s.union_all().boundary)
    if contact.is_empty:
        return None, 0.0
    merged = line_merge(contact) if contact.geom_type != "LineString" else contact
    return merged, contact.length / 1000.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    polys = gpd.read_file(GEMS).to_crs("EPSG:3338")
    present = set(polys.MapUnit.unique())

    defs = {
        "primary": (MARBLE_PURE, SCHIST_PACKAGE),
        "literal": ({"DOm"}, SCHIST_LITERAL),
        "inclusive": (MARBLE_PURE | MARBLE_IMPURE, SCHIST_PACKAGE),
    }
    lengths = {}
    for name, (marble, schist) in defs.items():
        line, km = contact_line(polys, marble, schist)
        lengths[name] = round(km, 1)
        if line is not None:
            gpd.GeoDataFrame(
                {"contact_def": [name]}, geometry=[line], crs="EPSG:3338"
            ).to_file(OUT / f"contact_{name}_3338.geojson", driver="GeoJSON")

    report = {
        "source": "DGGS RI 2024-7 (Werdon et al. 2024), GM_MapUnitPolys, "
                  "DOI 10.14509/31308; reprojected EPSG:26703 -> EPSG:3338",
        "n_polys": int(len(polys)),
        "units_present": sorted(present),
        "classification": {
            "marble_pure": sorted(MARBLE_PURE),
            "marble_impure": sorted(MARBLE_IMPURE),
            "schist_package": sorted(SCHIST_PACKAGE),
            "excluded_mixed_other": sorted(present - MARBLE_PURE - MARBLE_IMPURE - SCHIST_PACKAGE),
        },
        "contact_length_km": lengths,
        "primary_used_for_distance": "primary (marble DOm/Dm vs full schist package)",
    }
    (OUT / "contact_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
