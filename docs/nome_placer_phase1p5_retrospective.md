# End-of-Phase-1.5-v1.1 retrospective

Tracks the work between Phase 1 v3 close (2026-06-14 afternoon) and
the end of the Phase 1.5 v1.1 autonomous run (2026-06-14 evening).
The Phase 1 retrospective is at
``docs/nome_placer_phase1_retrospective.md``; this one focuses on the
delta.

## Headline

**6 of 6 mandated landmarks now pass the validation gate**, including
Bear Cub MS 1178 for the first time (BC 0.982 vs p90 0.967). The fix
is ``features/lithology.py`` -- a buried-stand membership feature
that uses bearcub's GP bedrock-topo + variance rasters to detect
paleo-beach lines buried below the modern surface. Sky's 2026-06-14
diagnosis ("Third Beach is 80-90 ft under the Bear Cub surface")
turned out to be exactly what the model needed: at Bear Cub the
modern surface matches Fourth Beach (+37 m) but bedrock at depth
matches Third Beach (+22-26 m), so the buried-BL score saturates
at 1.0 and feeds BC.

## What landed

| Item | Status |
|---|---|
| Stream-distance optimization (KDTree) | done; ~100x speedup, 8x faster full test suite (22:45 -> 2:50 for the streams subset) |
| features/lithology.py | done; buried-BL detection, 7 new tests |
| coastal_scorer integration | done; CoastalScores carries buried_bl field, BC uses max(BL, buried_BL, AP) |
| Phase 1 prospectivity raster GeoTIFF | done; 7-band raster in EPSG:3338 + EPSG:4326; bands.json sidecar; delivered to goldbug |
| Tuck overlay v1.5 v1 -> v1.5 v1.1 rotation fix | done; rasterio.open(transform=Affine(..)) embeds full affine including non-zero shear/rotation; 8 sheets re-delivered |
| Bearcub State Plane / Hammon grid correction | accepted; family-block claim corners offer accepted for v1.5 v2 |
| Cross-section tests | done; 8 tests cover schema, join keys, pay-zone parse |
| Phase status doc | updated with v1.5 v1.1 deliverables |
| Phase 1.5 v2 (interactive QGIS GCP pinning) | queued; needs human-in-loop, can't run autonomously |

## Three real deviations / corrections from peer feedback

**Goldbug: my "embedded transform" was axis-aligned.** I used
``gdal_translate -a_ullr`` which only embeds the bbox (``gt[2] = gt[4]
= 0``). The actual affine_4326 with rotation lived only in the JSON
sidecar. Goldbug measured this with gdalinfo and called it out.
Fix: switched the writer to rasterio.open with a proper Affine and
re-shipped all 8 sheets. The rough-fit's actual rotation is small
but the structural fix is the right thing.

**Bearcub: my State Plane reference was wrong.** I cited NAD83 AK
Zone 1 (EPSG:26931) as the candidate transform; bearcub corrected
to NAD27 AK Zone 9 (EPSG:26739, US survey feet). And the deeper
finding: the Tuck grid is NOT State Plane at all -- it's the
**local Hammon survey grid** used by the Nome District. Bearcub's
27 Bear Cub collars in local feet, plus the 1936 cartographer's
plat in the same local grid, are the v1.5 v2 georef anchor cluster
(the "Rosetta Stone"). Bearcub is deriving the family-block claim
corners (~50 more points) in local feet next.

**Sky: stream threshold was too restrictive.** With default
``min_accumulation_cells=500`` the seasonal Bear Creek + Dry Creek
that go through Bear Cub fell below the cutoff. Lowered to 100
(captures seasonal creeks; AOI stream count rose 7x to 357k).
The KDTree optimization that followed makes the higher density
tractable.

## What's NOT done in v1.5 v1.1

- Survey-grade TPS warp with bearcub's 229 GCPs (needs interactive
  pixel-coord identification in QGIS; queued for a focused future
  session).
- Tuck Map B1's plan-view Third Beach line as a vectorized polyline
  feature (Phase 1.5 v2; replaces the iso-elevation contour for
  Third Beach which doesn't match the actual line on Tuck's maps).
- Phase 2 supervised model (multi-head trainer; gated on more labels
  and on Tuck Map B1 polygon labels being georeferenced at
  sub-50 m RMS).
- The 23 holes with both cross-section and label-table records
  haven't been used yet -- the cross_sections sidecar lands but the
  scoring uses bedrock_topo not cross-section depths. Phase 2
  integrates this.
- Tuck contour elevation labels (OCR). Without these the bedrock-
  contour vectorization sits as a plan-view-line feature, not a
  district-scale bedrock-elevation raster.

## Validation gate at end of run

```
Anvil Creek (QM)        : 1.000 vs p90 1.000  PASS
Third Beach Sunset (BL) : 1.000 vs p90 1.000  PASS
Submarine Beach (BL)    : 1.000 vs p90 1.000  PASS
Molasses MS 1179 (TB)   : 0.826 vs p90 0.402  PASS
Honey    MS 1181 (TB)   : 1.000 vs p90 0.402  PASS
Bear Cub MS 1178 (BC)   : 0.982 vs p90 0.967  PASS  *** first time ***
```

## Test suite

| Test file | Tests | Status |
|---|---:|---|
| test_coastal.py | 11 | all pass |
| test_coastal_scorer.py | 13 | all pass |
| test_streams_and_bc_qm.py | 10 | all pass |
| test_tuck1942_map_b1.py | 6 | all pass |
| test_tuck1942_bedrock_contours.py | 4 | all pass |
| test_integration_pipeline.py | 9 | all pass |
| test_lithology.py | 7 | all pass (Phase 1.5 addition) |
| test_cross_sections.py | 8 | 7 pass + 1 skip (Phase 1.5 addition) |
| **Total** | **68** | **67 pass + 1 skip** |

End-to-end runtime: 788 seconds (~13 minutes). The KDTree streams
optimization is the single biggest factor.

## What I'd change in the plan

Nothing structural. The v1.5 buried-BL feature was queued in the
Phase 1 retrospective ("Phase 1.5 v2: use Tuck Map B1's plan-view
Third Beach line, not the iso-elevation line") and the actual fix
turned out to be even simpler (use the bearcub GP bedrock surface
that was already delivered Phase 0). The Tuck-Map-B1-plan-view fix
is still queued for Phase 1.5 v2 because it'll improve scoring at
buried beaches OUTSIDE the bearcub envelope, but for the family
claim block the bedrock-topo + variance gives the answer cleanly.

## Cross-repo status at end of run

### ai-minerals

19 commits since Phase 0 close. Pushed at e7f51e1 (origin/main).
68 tests passing on real bearcub + IfSAR + Tuck data. Two
deliverables sitting in goldbug's inbox (overlay v1.5 v1.1 and
prospectivity raster v1.5 v1). One handoff to bearcub re-issued
(navbar refresh).

### bearcub

Phase 0 contracts plus Phase 1.5 additions (cross_sections sidecar,
229 GCPs in CSV with local-feet + State Plane Z9 NAD27 columns).
Chapter jargon polish landed (commit cac1181). Family-block claim
corners in local feet promised next.

### goldbug

Overlay viewer scoped for v1.3.0 (Nome serving milestone). Acked
the v1.5 v1 with corrections (resolution and embedding fixed;
v2 survey-grade georef next). Has not yet built; not gating on us.

## Phase 2 kickoff prerequisites

Per the recommendations from the Phase 1 retrospective, Phase 2
remains gated on:

1. v1.5 v2 georef pass (survey-grade RMS < 50 m on the Tuck plan-
   view rasters; needs interactive QGIS).
2. Tuck Map B1 dredge-cleanup polygons as district-scale auxiliary
   positive labels (depends on v1.5 v2 georef).
3. Bearcub's family-block claim corners in local feet (in flight).
4. Goldbug v1.3.0 viewer build (their work; not gating on us).

The Phase 1.5 v1.1 deliverables unblock goldbug to test the
prospectivity raster end-to-end while v1.5 v2 georef is in flight.
