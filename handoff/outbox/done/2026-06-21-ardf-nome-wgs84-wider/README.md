# ARDF Nome extract — WGS84, wider district-lode bbox (for fossick re-ingest)

Datum-corrected, wider-bbox ARDF extract produced for fossick to re-ingest and
regenerate its KG export (coordinator handoff 2026-06-21).

## What changed vs the prior `nome_ardf_all.geojson` (139 Cape Nome records)

1. **WGS84, not NAD27.** The earlier extract carried the raw EPSG:4267 (NAD27)
   shapefile coordinates as bare geometry. fossick's KG then emitted those as
   WGS84 WKT, producing a fixed ~155 m SE offset. This extract reprojects each
   point NAD27 -> WGS84 (EPSG:4326) before writing, so the geometry *and* the
   `latitude`/`longitude` properties are already WGS84. Verified: NM101 moves
   from the old NAD27 `(-165.2841, 64.6463)` to `(-165.286858, 64.645544)`,
   a 155.8 m correction.
2. **Wider bbox.** District-lode grid extent `lon[-166.0, -164.0]
   lat[64.3, 64.8]`. 286 records (NM 238 + Solomon SO 48), up from 139. The
   added Solomon records include the Big Hurrah lode cluster (SO021–SO023,
   SO144, SO172) that the old Cape Nome bbox dropped.

## Files

- `nome_ardf_all_wgs84.geojson` — 286 features, WGS84 point geometry, full
  descriptive ARDF record per point (the fields fossick's `build_ardf_all_nome.py`
  reads: `ardf_num`, `mrds_num`, `site`, `comm_main`/`comm_other`, `dep_model`,
  `model_code`, `status`, `production`, `site_type`, plus `geol_desc`,
  `comments`, `location`, etc. for `_raw_text()`).
- `nome_ardf_all_wgs84.csv` — same records, no geometry (convenience).

`latitude`/`longitude` hold the WGS84 values; `datum` = `WGS84`,
`source_datum` records the NAD27 source. The raw NAD27 lat/lon columns are not
carried — the geometry is authoritative.

## Reproduce

`.venv/bin/python scripts/nome_placer/export_ardf_fossick_wgs84.py`

## Note on Big Hurrah identity

ARDF's Big Hurrah records carry their own `mrds_num` values (e.g. SO023 ->
`A012593; D002598; A010654`), not the MRDS dep_id `10309012` the program uses
as the lode anchor. Keep using `10309012` from MRDS for Big Hurrah regardless,
per the coordinator.
