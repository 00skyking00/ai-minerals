# SOURCE: Tuck-powered placer retest (MS-join + per-subtype CV)

Extends the round-4B subtype run (`SOURCE.md`) with fossick's Tuck-1942 typed
areas, joined to authoritative MS-survey coordinates and added to the KG positives.

## Typed-area source (private)

`fossick/samples/tuck/tuck1942_areas.json`: 181 `TuckPlacerArea` records from
fossick's OCR + structured extraction of Metcalfe & Tuck 1942, *Some aspects of
the Nome district placer deposits*. Each record carries `deposit_type` (true-beach
/ abrasion-platform / upland-residual / unknown, by fossick's §1d rule),
`claim_names`, `section`, and source pages. Tuck 1942 is a private report; only the
deposit type, claim names, MS join, and coordinate cross into this repo. Raw
paystreak data (production figures, grades, yardages, overburden, verbatim quotes)
stays in the fossick tree and is not carried here.

The type vocabulary is reconciled to the round-4B scheme: Tuck's **true-beach**
(winnowed marine drift) is Hudson's **strandline_beach**; **abrasion-platform** and
**upland-residual** carry across unchanged. The canonical labels are
`abrasion_platform`, `strandline_beach`, `upland_residual`.

## Coordinate authority (the join)

Built by `scripts/nome_placer/tuck_placer_join_coords.py`. The fossick records
carry no MS number, so the join key is the claim name. Sources, highest authority
first:

1. **bearcub curated crosswalk** (`bearcub/research/tuck1942_cadastre/tuck1942_claim_crosswalk.csv`):
   human-vetted Tuck-name -> MS links, confidence-graded. Supplies the family and
   neighbourhood claims (Bear Cub MS 1178, Jupiter 1217, Golden Bull 1209, Happy
   New Year 1113, No. 9 Otter 327, ...).
2. **goldbug published surveys** (`gldbg/data/published/nome_mineral_surveys.geojson`,
   372 surveys / 340 named): BLM CadNSDI + DNR + Nome taxroll + curated plat, each
   carrying MS, name, and polygon. Centroid reprojected to EPSG:3338 is the
   coordinate. Independent of the Tuck plate overlay's 200-500 m georef offset, per
   the handoff.
3. **bearcub cadastre draft polygons** (`tuck1942_claim_cadastre_draft.geojson`):
   centroid fallback for an MS absent from the published-survey layer (the family
   beach-line block).

Matching is precision-biased. A claim name is normalised (uppercase, MS suffix and
placer/claim/association decorators stripped) and accepted only by (a) an exact
curated-crosswalk hit, (b) an exact survey-name match, or (c) a whole-phrase
substring match on a distinctive name (single tokens must be non-generic,
non-ordinal, length >= 6 and resolve to exactly one survey). Generic and ordinal
single tokens (SECOND, THIRD, PAYSTREAK, BENCH, ...) are blocked so "Second Beach"
cannot match "No. 2 Bench Second Tier". An area's coordinate is trusted only with a
curated link or two independent claim matches agreeing within 1 km; otherwise the
area is flagged `pinpoint_needed`. Coverage: 15 of 181 areas join (9 of 72
coastal); see `tuck_placer_join_coverage.json`.

## Positive-set assembly

`scripts/nome_placer/placer_subtype_local_source_tuck.py`. The KG positive block
is loaded by `placer_subtype_local_source.load_typed()` (the exact round-4B 65
positives + 2000 DEM background, seed 42), and the Tuck coastal joins are inserted
after it so positives stay contiguous. A Tuck join within 250 m of a KG positive,
or within 150 m of an already-kept Tuck join, is dropped as the same ground. Net
added: 6 distinct true-beach positives, 0 abrasion-platform (the single
abrasion-platform join collapses into a co-located Golden Bull true-beach claim).
Base features (V3P1 population priors `bl/ap/tb/ss/bc/qm/buried_bl` + DEM / slope /
TPI) are raster-sampled at the new coordinates exactly as `load_placer()` builds
them. Covariates and both tests are the round-4B functions, reused unchanged.

## Files

- `tuck_placer_positives.csv`: all 181 areas with type, coordinate source
  (`ms_join` / `pinpoint_needed`), coordinate, matched MS and claims, source pages.
- `tuck_placer_join_coverage.json`: join coverage counts by type.
- `tuck_coastal_pinpoint_needed.csv`: the 63 coastal areas (30 abrasion-platform,
  33 true-beach) that need a pinpoint coordinate. The unblock list.
- `placer_typing_tuck.csv`: the combined typed positive set used (KG + Tuck).
- `placer_subtype_local_source_tuck.json`: baseline (KG) and powered (KG + Tuck)
  results, both tests.
- `placer_subtype_local_source_tuck.csv`: flat per-(covariate, group) summary.
- `REPORT_tuck_powered_retest.md`: the narrative + verdict.

Build:
```
PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.tuck_placer_join_coords
PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_subtype_local_source_tuck
```
