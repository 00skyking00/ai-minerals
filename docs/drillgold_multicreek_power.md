# Drill-gold multi-creek power + barren negatives (ML step-3 (a))

*Occurrence-level power report companion to the positioned capture harness.*

## The short answer

The Part-2 consolidation added ZERO positioned drill collars. The new multi-creek Tuck grades were folded at occurrence / area / claim-stub extent (grade ranges + valued/trace counts at one point or hull), which the per-hole capture-efficiency harness cannot consume. So the capture test's power is unchanged: one positioned drainage (Little Creek).

## Power accounting: independent drainages

| class | count |
|---|--:|
| positioned-collar drainages (capture-harness-ready) | 1 |
| occurrence/area drainages with a c/yd grade read | 6 |
| occurrence drainages production-$ only (no tenor) | 1 |
| total independent drainages with any gold grade | 7 |

positioned power stays at 1. The consolidation raises the CEILING for a future positioned multi-creek capture run to ~7 drainages, but only once fossick positions the Fig-10/11 drill lines (Otter 7 lines, Nome River Line 1) and the Map A Bay/Odin lines as collar points. Until then these are ranges at a point/hull, not test power.

The positioned capture harness is unchanged: 47 holes, 1 drainage (Little Creek), placer AUC at the 10 c/yd cutoff = 0.48. unchanged vs the committed 2026-07-05 run: one drainage, underpowered, AUC ~ chance with a CI spanning 0.19-0.75; a LABELED methods-validation baseline, not a G-result

## Barren-domain negatives from the Tuck corpus

Real sub-economic / barren reads from the tuck corpus (not fabricated pseudo-absences); usable for the negative class in a future positioned run. Total real barren/trace reads: **52**.

| drainage | extent | barren/trace | valued |
|---|---|--:|--:|
| Little Creek | positioned_collars | 6 | 17 |
| Otter Creek | occurrence_point | 38 | 63 |
| Nome River | occurrence_point | 2 | 6 |
| Second Beach belt | area_hull | 6 | 3 |

## Placer-MPM consistency read (descriptive, restriction-of-range)

known, already-explored occurrences the served placer MPM was fit over; a high percentile is restriction of range, NOT out-of-sample validation. Descriptive consistency only.

| drainage | extent | placer MPM percentile |
|---|---|--:|
| Little Creek | positioned_collars | outside MPM footprint |
| Otter Creek | occurrence_point | 100.0 |
| Nome River | occurrence_point | 46.3 |
| Anvil Creek | occurrence_point | 79.6 |
| Osborn Creek | occurrence_point | outside MPM footprint |
| Second Beach belt | area_hull | 38.2 |
| B1 Nome dredge field | area_hull | outside MPM footprint |
| Anvil-Dexter dredge valley | area_hull | 98.3 |

## What a positioned multi-creek run needs from fossick

1. The Fig-10/11 drill-section holes positioned as collar points (Otter 7 lines, Nome River Line 1) with per-hole grade, not an occurrence-level range.
2. The Map A Bay / Odin drill lines de-stubbed from `stub_pending_mtp` points to surveyed collar positions (pending BLM MTP).
3. Either of the above written into a positioned drill export the capture harness reads (`exports/phase4/drill_gold_points.geojson`), not only the occurrence/area consolidation layer.

*Grade facts: fossick/docs/consolidation-part2-foldin-2026-07-08.md (Part-2 fold-in table). Geometries: kg_nome_consolidated.jsonld (@graph wkt), read live.*
