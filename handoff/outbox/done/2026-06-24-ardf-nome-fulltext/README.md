# ARDF Nome extract — full narrative text, WGS84 (for fossick re-ingest)

Re-export of the Nome ARDF occurrences with the **full untruncated narrative
text** restored (coordinator handoff 2026-06-24, "ARDF description text missing
from the KG"). Supersedes the 2026-06-21 WGS84 extract, which carried only
255-character stubs of each description.

## Root cause: the shapefile truncates all narrative to 255 characters

The export source up to now was the ARDF shapefile (`data/raw/ardf/ardf/ardf.shp`).
A shapefile stores attributes in a `.dbf` table whose text fields cap at 255
characters. The shapefile's own metadata (`ardf.met`) documents the loss in its
process history:

> Imported ARDFcomp.fp5 into Filemaker Pro 6.0 and exported all records as a
> DBF file. Left out the following 5 fields because field length record exceeded
> 255 characters: Location, Geologic Description, Workings/exploration,
> Additional Comments, References.

So the long narrative was never present in the shapefile to begin with. The
2026-06-21 export did include the description **columns** (`geol_desc`,
`work_expl`, `comments`, `prod_notes`, `location`, …) and they were populated,
but every value was clipped at 255 chars. NM240 (Mattie) `geol_desc` ended
mid-word at "…gravels that ranged f". The full record is five paragraphs.

This export pulls the full text from the USGS mrdata ARDF per-record JSON
service (`https://mrdata.usgs.gov/ardf/json/<ardf_num>`), which renders from the
full database rather than the truncated shapefile.

## Coordinates are unchanged — this is a text-only enrichment

Geometry and the `latitude`/`longitude` columns are copied verbatim from the
2026-06-21 WGS84 export. Verified: 0 of 286 geometries differ. The NAD27→WGS84
datum fix is preserved (NM101 still at `(-165.286858, 64.645544)`, the validated
155.8 m correction). The mrdata service ships its own WGS84 reprojection; it is
**not** used here, so nothing downstream sees a coordinate shift.

## What changed vs the 2026-06-21 extract

Same 286 records (NM 238 + Solomon SO 48), same geometry, same datum. The
narrative columns now hold the full text instead of 255-char stubs, in the same
column names fossick already reads:

| field | shapefile max | full-text max | records now > 255 |
|-------|--------------:|--------------:|------------------:|
| `location`   | 255 | 1158 | 247 |
| `geol_desc`  | 255 | 6701 | 265 |
| `work_expl`  | 255 | 3480 |  84 |
| `prod_notes` | 255 |  752 |  13 |
| `comments`   | 255 |  266 |   1 |

Two columns are added (the shapefile never carried these in full):

- `references` — the complete bibliography for the occurrence (max 6766 chars;
  278/286 records have one, 8 genuinely have none).
- `reporter` — reporter name, date, affiliation.

`comments`, `work_expl`, `prod_notes`, `reserves` are genuinely sparse in ARDF
(many records have none); empty values are written as `""`, not `null`/`nan`.

## Files

- `nome_ardf_all_wgs84_fulltext.geojson` — 286 features, WGS84 point geometry,
  full descriptive record per point. Schema is a strict **superset** of the
  2026-06-21 export (no prior column dropped; `references` + `reporter` added),
  so `build_ardf_all_nome.py` reads and `_raw_text()` keep working and now see
  the full text.
- `nome_ardf_all_wgs84_fulltext.csv` — same records, no geometry (convenience).

## Reproduce

`.venv/bin/python scripts/nome_placer/enrich_ardf_fulltext.py`

(Per-record JSON is cached under `data/raw/ardf/fulltext_json/`, gitignored;
pass `--force` to refetch.)

## Note for fossick: the prior stubs were dropped downstream too

The coordinator found **zero** long-text on the ARDF occurrence nodes in
`kg_nome.jsonld`. But the 2026-06-21 export I handed over did contain the
narrative columns (truncated to 255 chars). So the re-ingest / KG build at
`b5dafba` dropped even the stubs that were there. This export gives you the full
text to attach, but the ingest still needs to carry the narrative fields through
to the occurrence nodes — re-exporting alone will not surface text in `kg.html`
if the ingest path does not read those columns. Worth a look on the fossick side
while wiring this up.

## NM240 (Mattie) before / after

- shapefile stub (255 chars): "…where high-level gravels were placer mined for
  gold. These deposits were in gravels that ranged f"
- full text (1916 chars): the five-paragraph geologic description, including the
  high-level-gravel origin debate (alluvial vs glacial outwash), the Anvil fault
  setting, grades (~0.45 oz/yd³ in 1900), and the full 11-entry reference list.
