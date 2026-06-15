# Nome placer build: phase status + integration points

Roll-up across ai-minerals, bearcub, goldbug. Updated 2026-06-14 at
end of Phase 1.5 v1 autonomous run (post-Phase-1-v3).

## Phase status

| Phase | Owner | Status | Closed | Deliverables |
|---|---|---|---|---|
| 0 | ai-minerals + bearcub | **CLOSED** | 2026-06-14 | Region scaffold + IfSAR adapter + ADGGS adapter + bearcub-delivered drillholes / hole_layers / bedrock_topo / variance / claims |
| 1 v0 | ai-minerals | CLOSED | 2026-06-14 | Coastal feature stack (per-stand contours, elevation-relative, distance-to-shoreline); BL / AP / TB scorers; 5/5 mandated gate landmarks PASS |
| 1 v1 | ai-minerals | CLOSED | 2026-06-14 | Stream-distance feature; BC / QM scorers; beach-contour GeoJSONs exported to bearcub |
| 1 v2 | ai-minerals | CLOSED | 2026-06-14 | Tuck 1942 Map B1 dredge-cleanup polygons vectorized (3,383 polygons, rough-georeferenced); v1 auxiliary positive labels |
| 1 v3 | ai-minerals | CLOSED | 2026-06-14 | Tuck 1942 Nome District bedrock-contour lines vectorized (2,358 segments); elevation-label OCR queued for Phase 2 |
| 1 wrap-up | ai-minerals | CLOSED | 2026-06-14 | Integration tests + retrospective |
| 1.5 v1.1 | ai-minerals | CLOSED | 2026-06-14 | features/lithology.py buried-BL detection; Bear Cub MS 1178 passes BC top decile (6/6 gate); Phase 1 prospectivity raster (7-band GeoTIFF) delivered to goldbug; Tuck overlays re-delivered with rotation in raster header; stream-distance KDTree (100x speedup) |
| 1 sibling | bearcub | not started | -- | Janin 1912 cross-sections check (per Sky 2026-06-14: bearcub already has them; needs confirmation) |
| 1 sibling | goldbug | not started | -- | Historical-map overlay viewer (transparency slider; per ai-minerals 2026-06-14 handoff) |
| 1 sibling | bearcub (deferred) | queued | -- | Chapter jargon polish (deferred; bearcub kept focus on Nome thread) |
| 1.5 | ai-minerals | not started | -- | Production georeferencing for Tuck Map B1 + bedrock contours (replace rough-affine with QGIS-survey-grade or published GCPs) |
| 1.5 | Fossick | queued | -- | Hopkins / Kaufman drift-boundary digitization for QM in-drift mask |
| 1.5 | ai-minerals | queued | -- | Re-run Phase 1 scoring with the Tuck Map B1 / bedrock-contour features added to the stack |
| 2 | ai-minerals | not started | -- | Multi-head supervised model on 95 hole + Tuck Map B1 polygons (BL/BC/QM heads; TB stays knowledge-only) |
| 2 | ai-minerals | not started | -- | Elevation-label OCR or manual annotation for Tuck bedrock contours |
| 3 (optional) | ai-minerals | not started | -- | District drill-planner |
| Future | ai-minerals | gated | -- | AGC drillhole ingest re-run (if AGC archive lands) |

## Integration points

### bearcub -> ai-minerals (delivered Phase 0)

Paths on ai-minerals side, populated by bearcub via
`tools/export_nome_placer_labels.py --publish` (bearcub side; per
their 2026-06-14 phase0-reply):

```
data/raw/bearcub_nome/drillholes_nome_placer.parquet         (104 holes)
data/raw/bearcub_nome/hole_layers_nome_placer.parquet        (1,390 intervals)
data/raw/bearcub_nome/bedrock_topo_nome_placer.tif           (35 x 32 cells, 25 m, EPSG:3338)
data/raw/bearcub_nome/bedrock_topo_variance_nome_placer.tif  (same grid)
data/raw/bearcub_nome/claims_by_ms.geojson                   (16 polygons, EPSG:4326)
```

bearcub regenerates on each ingest tick. Files are gitignored on both
sides.

### ai-minerals -> bearcub (delivered Phase 1 v1)

```
~/src/learning/bearcub/handoff/inbox/2026-06-14-from-ai-minerals-nome-beach-contours/
  README.md
  <stand>_contour_3338.geojson  (8 stands)
  <stand>_contour_4326.geojson  (8 stands)
  manifest.json
```

Use case: bearcub's single-claim GP recommender consumes per-stand
distance-to-shoreline as a covariate; missing this is what produces
the negative leave-one-out R^2 bearcub flagged.

### ai-minerals -> goldbug (delivered Phase 1.5 v1.1)

Three packages dropped in goldbug's inbox:

```
gldbg/handoff/inbox/2026-06-14-from-ai-minerals-tuck1942-overlay-v1p5/
  8 sheets x (.tif full affine embedded + 300 DPI)
  tuck1942_overlay_metadata_v1p5.json
gldbg/handoff/inbox/2026-06-14-from-ai-minerals-nome-prospectivity-v1p5/
  nome_placer_prospectivity_v1p5_3338.tif  (7-band working CRS)
  nome_placer_prospectivity_v1p5_4326.tif  (7-band delivery CRS)
  bands.json                                (Nome-specific schema)
gldbg/handoff/inbox/2026-06-14-from-ai-minerals-tuck-v1p5-v1p1-rotation-embedded.md
gldbg/handoff/inbox/2026-06-14-from-ai-minerals-overlay-ack-confirm.md
```

The prospectivity raster carries 7 bands:
1. bl  -- true beach
2. ap  -- abrasion-platform / sloughover
3. tb  -- Tertiary buried high-bench
4. bc  -- beach-stream confluence
5. qm  -- off-beach modern creek
6. buried_bl  -- buried true beach (Phase 1.5 addition)
7. composite  -- per-cell max across populations

Goldbug samples band 7 (composite) for the v1 prospectivity overlay
and bands 1-6 for per-population analytic drill-down. The bands.json
sidecar carries the v1.5 v1 validation-gate results (6/6 PASS).

Future v1.5 v2 / Phase 2 packages (still to deliver):

```
data/derived/nome_placer/
  prospectivity_nome_placer_25m_calibrated_4326.tif  (Phase 2)
  bands.json                                          (Phase 2, Nome-specific)
  coverage_threshold.json                             (Phase 2, Nome-specific)
  prospectivity_depth_band_sidecar_4326.tif           (Phase 2 2.5D)
data/derived/tuck1942/
  map_b1_dredge_cleanups_4326.geojson                 (Phase 1 v2 v1; refine 1.5)
  bedrock_contours_4326.geojson                       (Phase 1 v3 v1; refine 1.5)
  tuck1942_nome_district_*.tif                        (overlay rasters, Phase 1.5)
```

Goldbug handoff already filed:
`handoff/outbox/2026-06-14-goldbug-historical-map-overlay.md` and
copy in `~/src/learning/gldbg/handoff/inbox/`.

### ai-minerals internal: Phase 1 prospectivity pipeline

```
features/coastal.py
  paleo_shoreline_contours_all_stands(dem)
  elevation_relative_to_stand_m(dem, grid, stand)
  distance_to_paleo_shoreline_m(contours, grid, stand)

features/hydrology.py  (existing + Phase 1 v1 additions)
  flow_accumulation(dem_array, transform)
  streams_from_flow_accumulation(flow_acc, transform, crs)
  distance_to_stream_m(streams, grid)

features/coastal_scorer.py
  score_bl(dem, grid)        ; true-beach (BL)
  score_ap(dem, grid)        ; abrasion-platform / sloughover (AP)
  score_tb(dem, grid)        ; Tertiary buried high-bench (TB)
  score_bc(bl, ap, d_stream) ; beach-stream confluence (BC)
  score_qm(dem, grid, d_stream)  ; off-beach modern creek (QM)
  score_all_populations(dem, grid, distance_to_stream_m=...)

data/tuck1942_map_b1.py
  extract(pdf_path, cache_dir) -> MapB1Extraction (3,383 polygons)

data/tuck1942_bedrock_contours.py
  extract(pdf_path, cache_dir) -> BedrockContoursExtraction (2,358 segments)
```

## Decisions snapshot

- v1 model grid: 25 m EPSG:3338. 10 m reserved as a config switch for
  the AGC-gated re-run.
- v1 geology: Wilson 2015 SIM 3340 statewide. ADGGS pubs go to
  `surficial_seward` slot for v2 east-expansion.
- BL stands: Present + Second + Third + Fourth (Tuck p.28: Fourth
  is a planed-off marine bench, not a true beach but close enough).
- AP stands: Submarine outer / inner / deep + Intermediate +
  Monroeville.
- TB stands: tuck_high_300 + tuck_high_600 (Honey 13 m from +600 ft).
- Stream threshold: 500 flow-accumulation cells at IfSAR 7 m
  (~52,000 stream cells, ~362 km total length at Cape Nome).
- BC sigma_stream: 500 m (Bear Cub at 529 m from streams).
- QM sigma_stream: 400 m. QM elevation band: 30 to 250 m.
- District drill-planner: deferred to Phase 3 (bearcub greenlit
  this in 2026-06-14 reply).
- Bear Cub BC: documented partial (BC = 0.249, top quintile not top
  decile). Direct signal from the product structure.
- Tuck map georeferencing: rough-affine v1 (~+/-200-500 m error)
  using 4 visually estimated GCPs. Survey-grade georeferencing
  queued for Phase 1.5.
