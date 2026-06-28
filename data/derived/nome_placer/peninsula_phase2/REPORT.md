# Phase-2 prep: Seward Peninsula data acquisition + staging

Acquisition and staging only, per the coordinator's 2026-06-27 dispatch. No
modeling. Two deliverables: the peninsula-wide ARDF placer + lode occurrences,
and a schist-limestone contact feasibility assessment on the geology that is on
disk.

## 1. Peninsula ARDF (done, offline)

The full Alaska ARDF shapefile is already on disk (`data/raw/ardf/ardf/ardf.shp`,
EPSG:4267), so no fetch was needed. Extracted the four Seward Peninsula gold
quadrangles and kept the placer (Cox-Singer 39a) and lode (36a) occurrences,
reprojected to WGS84 and EPSG:3338. Council, Casadepaga, Bluff, and Big Hurrah
all fall inside the Solomon quad.

| quad (1:250k) | placer | lode |
|---|---|---|
| NM (Nome) | 150 | 52 |
| SO (Solomon) | 113 | 14 |
| BN (Bendeleben) | 74 | 3 |
| TE (Teller) | 65 | 9 |
| **total** | **402** | **78** |

480 occurrences staged (421 with Au as the main commodity). The `geol_desc`
field here is the 255-char shapefile stub; full narratives (needed if Phase-2
does the genetic typing the round-5 Nome test did) are at mrdata's per-record
JSON service, which is reachable.

## 2. Schist-limestone contact: feasibility (Nome quad only)

The peninsula-scale bedrock geology is NOT on disk. Only the Nome quadrangle
(Bundtzen et al. 1994, `nmgeol_dd`) is present, so the feasibility question is
answered there and the method is built for scale-up.

Classifying the 105 Nome-quad polygons by map-unit LABEL:

- **schist (53 polygons):** PzZh (15), Ocs (19), Zn (9), Zo (7), Dcs (3) -
  pelitic and calc-schist of the Nome Group.
- **carbonate / marble (25 polygons):** DOx (21), Oim (3), Pzmm (1).
- **other (27):** surficial, Cretaceous intrusives, water.

The two are distinct map units, so a schist-carbonate contact line is derivable
(shared boundary of the dissolved schist and carbonate sets). **It is separable.**

The catch is the **DOx unit**. With DOx counted as carbonate the contact is
154.8 km; without it (only the Oim + Pzmm marble) it is 10.3 km. DOx is the
single largest non-schist unit (21 polygons) and carries roughly 144 of the
155 km, so the whole contact length rests on whether DOx is the carbonate
platform or something else. It is classed here as carbonate on its map code and
Devonian-Ordovician age, but that assignment must be confirmed against the unit
legend before any peninsula scale-up. This is the key feasibility caveat.

### Bearing on round 5
The prebuilt `dist_to_contact.tif` that the round-5 Nome test used is not this
schist-carbonate contact: Spearman correlation 0.53, and its median distance
(1389 m) is a third of the schist-carbonate raster's (4214 m), so the prebuilt
raster is a denser, different contact set. Round 5's contact-distance null is
specifically about that prebuilt layer. Re-running the round-5 contact feature
against the schist-carbonate contact derived here is a cheap follow-up worth
doing before concluding the contact is null at Nome.

## 3. Peninsula geology gap (what to pull for the full Phase-2)

The Solomon, Bendeleben, and Teller bedrock geology is needed to extend the
contact peninsula-wide. The on-disk `nmgeol_dd` is Nome only. Candidates to pull
(network is available): USGS SIM 3131 if it is the peninsula-wide compilation,
otherwise the AK DGGS bedrock coverages for those quads or the Till 2011 Seward
Peninsula bedrock-geology compilation, picking units with comparable schist and
marble classification so the DOx-type ambiguity is resolved consistently across
sheets.

## Files

- `peninsula_ardf_placer_lode_4326.geojson`, `..._3338.geojson` - staged ARDF
- `peninsula_ardf_provenance.json` - source + counts by quad and class
- `schist_carbonate_contact_nome_3338.geojson` - derived contact line (Nome quad)
- `dist_to_schist_carbonate_contact_3338.tif` - distance raster (Nome DEM grid)
- `schist_carbonate_contact_report.json` - classification, lengths, comparison

Pipeline: `scripts/nome_placer/peninsula_phase2/stage_peninsula_ardf.py`,
`derive_schist_limestone_contact.py`.
