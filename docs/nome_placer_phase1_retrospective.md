# End-of-Phase-1 retrospective

Cross-repo retrospective at the close of the Phase 1 autonomous run on
2026-06-14. Compares what landed against the master plan
(`~/.claude/plans/hazy-humming-lynx.md` as it stood at Phase 0 close)
and across ai-minerals, bearcub, and goldbug.

## Headline

Phase 1 closed with five of five mandated landmarks passing the
validation gate, all on real IfSAR + bearcub data, with 53 tests in
the suite. Three deviations from the plan, all of them additive (Tuck
1942 enrichment) rather than scope-cuts. Carrying one documented
partial (Bear Cub BC = 0.249, top quintile not top decile) and one
documented dependency for Phase 1.5 / Phase 2 (Tuck map georeferencing
needs survey-grade GCPs).

## What landed vs the plan

### Plan items: delivered as scoped

| Plan item | Status |
|---|---|
| `regions/nome_placer.py` four-population rewrite | done in Phase 0 (commit `5eb6d64`) |
| IfSAR Alaska adapter | done in Phase 0 (commit `5eb6d64`, fixed `f814bed` / `5175c6b`) |
| ADGGS surficial adapter (with documented Cape Nome gap) | done in Phase 0 (commit `f814bed`) |
| `features/coastal.py` per-stand contours + elevation-relative + distance | done in Phase 1 v0 (commit `f3fe73a`) |
| `features/coastal_scorer.py` fuzzy-overlay BL / AP / TB | done in Phase 1 v0 (commit `b40bfb2`) |
| Stream-distance feature via Whitebox flow accumulation | done in Phase 1 v1 (commit `65c480b`) |
| BC + QM scorers | done in Phase 1 v1 (commit `65c480b`) |
| Beach-contour GeoJSON loopback to bearcub | done in Phase 1 v1 (commit `bef66f8`) |
| Phase 1 four-population (now five) fuzzy-overlay knowledge-driven index | done; coastal_scorer.score_all_populations is the scorer |
| Validation gate (Anvil / Third / Submarine / Molasses / Honey in top deciles, no labels) | **5/5 PASS** |

### Plan items: deferred to Phase 1.5 / Phase 2

| Plan item | Status |
|---|---|
| ADGGS Iron Creek / Nome River drift mask | deferred to Fossick per bearcub 2026-06-14 reply; v1 uses sim3340 statewide fallback |
| `features/lithology.py` for false-bedrock detection | not started; depends on hole_layers schema (bearcub delivered 1,390 intervals on 80 holes) |
| Tuck Map B1 georeferencing (survey-grade) | rough-affine v1 in hand (commit `666973c`); survey-grade queued for Phase 1.5 |
| Tuck bedrock-contours elevations | red-line vectorization in hand (commit `4a623ed`); elevation labels need OCR / manual, queued for Phase 2 |
| District drill-planner | deferred to Phase 3 per bearcub 2026-06-14 reply |

### Plan items: added beyond the original plan (Tuck enrichment)

Triggered by Sky's instruction on 2026-06-14 to pre-pass the Tuck 1942
report. Three new sub-tasks were added to Phase 1 mid-stream:

| Added item | Status |
|---|---|
| Tuck 1942 stand-list extension (+200 / +300 / +600 ft) | done in Phase 1 (commit `33dd07c`) |
| Tuck Map B1 dredge-cleanup polygon vectorizer (3,383 polygons) | done in Phase 1 v2 (commit `666973c`) |
| Tuck Nome District bedrock-contour line extractor (2,358 segments) | done in Phase 1 v3 (commit `4a623ed`) |

## Deviations from the plan: did they break the plan or strengthen it?

**Deviation 1: ADGGS Cape Nome surficial pubs do not exist digitally.**
Original Phase 0 plan assumed ADGGS would cover Cape Nome. Reality:
AOF 125 covers Tolstoi Point (east Seward), USGS OF 72-321/322 covers
Nome C-2/C-3 (east Seward), neither covers Nome C-6 where Bear Cub /
Molasses / Honey sit. **Recovery**: fall back to Wilson 2015 SIM 3340
statewide (same as eastak); queue Hopkins / Kaufman drift digitization
to Fossick. **Can we live with it**: yes. The drift mask is a QM / TB
refinement, not the BL / BC core; v1 scoring still passes the gate.

**Deviation 2: Bear Cub BC scores top quintile not top decile.**
Original gate framing implied all listed landmarks would land in the
top decile of their natural populations. Reality: Bear Cub sits +8 m
above Fourth Beach AND ~530 m from the nearest stream cell extracted
at default flow-accumulation thresholds; BC = max(BL, AP) x
Gaussian(stream) gives 0.25 directly (top 20%). **Recovery options
considered**: (a) per-stand BL sigma (Fourth wider than Second/Third);
(b) BC sigma > 500 m; (c) accept as documented partial. **Picked**:
(c), because the product structure is the unsoftened signal -- Bear Cub
genuinely is neither perfectly on a beach stand nor perfectly on a
stream, and inflating either sigma to force a PASS would obscure that.
**Can we live with it**: yes, documented in commit `65c480b` and in
`tests/nome_placer/test_streams_and_bc_qm.py::test_bear_cub_partial_bc_score_documented`.
Bearcub was notified in the Phase 1 v1 ack.

**Deviation 3: Tuck digitization was not in the Phase 0 plan.** Sky
pulled `/tmp/tuck/` into scope on 2026-06-14 mid-Phase-1. Originally
that was a Phase 2 enrichment. **Recovery**: lifted three Tuck sub-tasks
(stand-list extension; Map B1 polygons; bedrock contours) into Phase 1
as additive items. Took roughly one substantive session of focused
work; no plan item was cut to make room. **Can we live with it**: yes.
The Tuck additions strengthen Phase 1 (richer feature stack) and pre-
position Phase 2 (Tuck Map B1 dredge polygons become auxiliary positive
labels for supervised training).

## Cross-repo status snapshot

### ai-minerals

13 commits over the Phase 1 autonomous run (5eb6d64 through f418eee).
53 tests across 6 test files. All passing on real IfSAR + bearcub
data. Phase 1 scoring artefact + Tuck-derived GeoJSONs in
`data/derived/`. Beach-contour GeoJSONs delivered to bearcub's inbox.

### bearcub

Phase 0 contracts delivered (5 files, all verified). Bearcub also
landed the chapter jargon polish (commit `cac1181`) despite saying
they'd defer; net positive. Janin 1912 cross-section status: Sky said
they're already with bearcub but the bearcub repo's recent commits do
not show an explicit extraction. Recommend the next bearcub session
confirm whether the Janin cross-sections are in `hole_layers` or sit
separately.

### goldbug

Goldbug acked the historical-map overlay ask (commit not yet visible
on the ai-minerals side; ack in inbox at
`2026-06-14-from-goldbug-historical-overlay-ack.md`). Target version
v1.3.0 (the Nome serving milestone). Two design corrections from
goldbug:

- Reuse target is the DEM / hillshade ImageOverlay path, not a
  Lindgren PP73 overlay (which doesn't exist).
- Web-optimised PNGs (transparent background) rather than raw
  fine-resolution TIFFs, to avoid blowing up the Folium base64-
  embedded HTML to >> 160 MB. ai-minerals should ship the PNGs
  (or accept goldbug doing the downsample on ingest).

Goldbug also progressed Nome land-status independently (commits
`94e2262` G0 spike, `c6a2011` G3 Alaska land-status branch).

## Recommendations for Phase 1.5 + Phase 2 kickoff

1. **Survey-grade georeferencing for Tuck Map B1 + Nome District
   (Contours).** QGIS pass with published Nome District survey
   control points; refines the rough-affine to sub-cell accuracy.
   The pixel-space polygons / lines we already have are authoritative;
   only the affine changes.
2. **Web-optimised PNG export for goldbug.** Render the seven planned
   plan-view sheets at goldbug-viewable resolution with transparent
   backgrounds; write `tuck1942_overlay_metadata.json` per goldbug's
   schema ask. Delivered as ai-minerals -> goldbug artefact.
3. **Janin 1912 cross-section confirmation.** bearcub session to
   confirm the Janin side-elevation cross-sections made it into
   `hole_layers_nome_placer.parquet` or to deliver them as a separate
   sidecar.
4. **Hopkins / Kaufman drift digitization (Fossick).** QM in-drift
   mask refinement; v1 uses sim3340 statewide fallback, which is
   coarse for the QM population scoring.
5. **Tuck bedrock-contour elevation labels.** OCR pass or manual
   annotation. Once labelled, TIN / spline interpolation produces a
   district bedrock-topo raster that merges with bearcub's
   Bear-Cub-claim GP surface (gated by GP variance).
6. **Phase 2 supervised model on the 95-hole baseline + Tuck Map B1
   polygons.** Multi-head trainer (BL, BC, QM classifier heads;
   grade regressor; depth-to-pay regressor). Per the master plan.

## What I'd change in the plan

Nothing structural. The plan as written by Phase 0 close held up
through the autonomous Phase 1 run with two additive expansions
(Tuck content addition; goldbug PNG ingest correction). Suggested edits
to the plan file:

- Promote the Tuck pre-pass + Tuck Map B1 / bedrock contours from
  Phase 2 to Phase 1 explicitly (already lifted; capture in the
  plan).
- Note the Bear Cub BC documented partial as a known Phase 1 v1
  state, not a defect.
- Add goldbug's web-PNG correction as the canonical delivery
  contract for the historical-map overlay artefact.
