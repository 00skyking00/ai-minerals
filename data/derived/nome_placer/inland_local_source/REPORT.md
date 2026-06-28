# Round 5: inland-alluvial local-source test (Nome district)

Does down-channel distance to a mapped lode, and distance to the schist-limestone
contact, predict where the inland *stream* placers sit, with coarser gold nearer
the source? Run on the Nome district only, on the strictly-alluvial population,
under the leak-guarded spatial CV.

## Plain verdict

1. **Down-channel distance to a mapped 36a lode could not be tested.** The mapped
   discrete-lode inventory is too sparse and too disconnected from the confined
   stream network: only 6 of 51 strictly-alluvial occurrences sit in a confined
   upland valley downstream of any mapped lode within 30 km (1 of 51 falls exactly
   on a confined channel cell). The intended primary feature is not computable at
   Nome from the mapped lodes. This is a data limitation, reported rather than
   forced. It is also the expected one: the Nome gold is described as eroded from
   sources "disseminated over a wide area," not from a handful of mapped veins.

2. **Distance to the schist-limestone contact is null at Nome.** Alluvial placers
   are no closer to the mapped contact than random background (median 1510 m vs
   1408 m). Its single-feature rank AUC (0.649) barely clears what distance to the
   same number of *random* points scores under spatial autocorrelation (null 95th
   percentile 0.640), and under the leave-one-block-out RF it inverts (0.34). This
   matches the experiment spec's own note that the contact is sparse at Nome and a
   peninsula-scale (Phase-2) feature, not a district one.

3. **Straight-line distance to a lode shows a proximity association that does not
   survive spatial cross-validation.** Placers rank closer to lodes than random
   points (rank AUC 0.742 full-AOI, 0.712 within the uplands), well above the
   random-point null (0.640). But once the background is drawn from the same
   upland terrain as the positives and the model is cross-validated by leaving out
   spatial blocks, it falls to chance (RF 0.52). The association is real but local:
   placers and lodes share the same sub-basins, yet a held-out basin's placers are
   not predictable from its lode distances. By the leak-guarded discipline this
   project holds to, that is a non-result for occurrence.

4. **Coarseness falls with distance to the lode (the clearest positive).** Across
   the 26 alluvial occurrences with a mined grain-size tag, median straight-line
   distance to a lode is 911 m for rough/nuggety gold, 4354 m for coarse, and
   5008 m for fine/flaky. Spearman rho = -0.40 (p = 0.042); Kruskal-Wallis
   p = 0.086. The downstream-fining gradient that Collier (1908) and Tuck (1942)
   describe is visible in the data, carried by straight-line distance because the
   down-channel feature is too sparse to use. Distance to the contact shows no
   such gradient (rho = -0.13, p = 0.53).

5. **Negative control is clean.** On the six marine-beach occurrences, both
   distance features are null (contact AUC 0.524, CI 0.342-0.723; straight 0.546,
   CI 0.394-0.694; both CIs span 0.5). No occurrence in the AOI types as
   genetically glacial-drift, so that arm of the control is empty.

**One line:** the local-source signal at Nome is real but shows up in gold
*coarseness vs distance-to-lode*, not in occurrence-vs-distance, and not at all
through the schist-limestone contact or the mapped-lode down-channel feature,
which the district data cannot support.

## Population (genetic typing)

`type_placers.py` types the 60 ARDF placer occurrences inside the DEM extent by
the site name first (Nome beaches are "X Beach"; inland stream placers are "X
Creek/Gulch/River" and stream-terrace placers "X Bench"), with a strict keyword
score over the full ARDF narrative as the fallback, and elevation as a tiebreaker.
Result: **51 alluvial-stream**, **6 marine-beach**, **3 district aggregates**
dropped (Nome Mining District, Nome placer field, Nome Coastal Plain are not point
occurrences). Residual/eluvial is filtered out at source to remove the
distance-zero tautology; none of the 60 typed as residual once stream-named sites
were honored. Every record's type and basis is in `placers_typed_audit.csv`.

Of the 51 alluvial occurrences, 29 fall in confined upland terrain (local relief
>= 45 m over a 525 m window), 9 in the foothill dead-zone (25-45 m), 13 in the
coastal plain (<= 25 m). The marine beaches sit in the coastal plain as expected
(median relief 22 m).

## Features

- **Down-channel distance to lode** (`down_channel_dist_to_lode.tif`): D8 receiver
  network from the depression-filled 25 m IfSAR DEM; each 36a lode snapped to the
  nearest stream cell within 300 m seeds a downstream walk; every downstream cell
  is tagged with the along-channel distance to its nearest upstream lode. Clipped
  to the confined-upland mask (where the modern channel coincides with the
  ancestral one). This is the DEM adaptation of
  `hydrology.py:distance_downstream_from_lode`, whose NHD/hydroseq walk does not
  apply to a DEM-derived network. The same placer-leakage guard is carried: lode
  seeds with a 39* (placer) model code are refused (2 dropped).
- **Straight-line distance to lode** (`straight_line_dist_to_lode.tif`): per-cell
  Euclidean distance to the nearest lode point, the comparison baseline (mirrors
  `hydrology.py:distance_to_lode_m`).
- **Schist-limestone contact distance**: the prebuilt
  `bedrock_contact/dist_to_contact.tif`, reused as-is.

The D8 pointer convention was verified empirically before use (100.0% of cells
flow to a not-higher neighbor on the filled DEM).

## Method

Positives are the strictly-alluvial occurrences; background is the existing
2000-random on-land pattern (seed 42). Single distance features are scored by
their rank AUC (a monotone feature needs no training, so the spatial CV cannot
leak through it) with a stratified bootstrap CI; multi-feature models use the RF
leave-one-block-out spatial CV (KMeans-10 blocks + 300 m buffer leak-guard)
reused from `mpm_onshore_presence_cv.py`, with the bootstrap CI taken over the
pooled out-of-fold predictions. The spatial null is the AUC of distance to N
random points (N = lode count), repeated 300 times, so a real feature must beat
what any clustered point set scores under spatial autocorrelation.

## H1: occurrence

### H1a, full AOI (51 alluvial vs 2000 background)

| model | AUC | 95% CI |
|---|---|---|
| distance-to-contact (single, rank) | 0.649 | 0.586-0.710 |
| distance-to-lode straight-line (single, rank) | 0.742 | 0.674-0.808 |
| spatial null: distance to random points | mean 0.527 | p95 0.640 |
| RF contact only (spatial CV) | 0.344 | 0.298-0.392 |
| RF contact + straight-line (spatial CV) | 0.613 | 0.539-0.685 |
| RF contact + down-channel (spatial CV) | 0.346 | 0.300-0.394 |
| RF contact + straight-line + down-channel (spatial CV) | 0.613 | 0.541-0.690 |
| RF all distance + terrain (spatial CV) | 0.755 | 0.692-0.818 |

Reading: straight-line distance to lode beats the random-point null and survives
the leak-guarded CV; contact distance does not (it inverts under the CV). The
down-channel feature adds nothing because it is absent at almost every point. The
0.76 of the full terrain model is carried by elevation/slope/TPI (upland-valley
membership), not by the distance features.

### H1b, upland-matched (29 alluvial-in-upland vs 1000 random upland background)

Background drawn from the same upland terrain (relief >= 45 m) as the inland
positives, which removes the upland-vs-coastal-plain confound that inflates H1a.

| model | AUC | 95% CI |
|---|---|---|
| distance-to-contact (single, rank) | 0.514 | 0.410-0.619 |
| distance-to-lode straight-line (single, rank) | 0.712 | 0.618-0.800 |
| down-channel where defined (single, rank; 6 pos / 33 bg) | 0.376 | 0.149-0.626 |
| RF contact only (spatial CV) | 0.497 | 0.438-0.571 |
| RF contact + straight-line (spatial CV) | 0.518 | 0.433-0.611 |

Reading: within the uplands, distance to the contact is null (0.51). Straight-line
distance keeps a within-sample rank signal (0.71) but it does not survive the
leave-one-block-out CV (0.52). The down-channel feature, where its six positives
allow a look, is null with a CI spanning 0.5 (0.15-0.63). This is the cleaner
control, and it says distance-to-lode does not predict alluvial occurrence at Nome
once spatial structure is held out.

## H2: coarseness (ordinal)

26 alluvial occurrences carry a grain-size tag mined from the ARDF narrative
(fine/flaky = 1: 9; coarse = 2: 7; rough/nuggety = 3: 10).

| feature | n | median dist, fine / coarse / rough | Spearman rho | p | Kruskal p |
|---|---|---|---|---|---|
| straight-line to lode | 26 | 5008 / 4354 / 911 m | -0.40 | 0.042 | 0.086 |
| schist-limestone contact | 25 | 2997 / 1184 / 1457 m | -0.13 | 0.53 | 0.56 |
| down-channel to lode (snapped) | 5 | 291 / 752 / 2042 m | n/a | n/a | too few |

Coarser gold sits nearer the lode by straight-line distance, the documented
downstream-fining gradient. The contact distance shows no gradient. The
down-channel feature cannot be tested for coarseness: only 5 of the 26 tagged
occurrences fall near a confined channel cell with a value, and across those five
the order even runs the wrong way (rough gold farther), which at n = 5 means
nothing beyond confirming the feature is too sparse to use here.

## What this does not establish

- It does not test the down-channel hypothesis the round was designed around. The
  mapped 36a lode inventory is the wrong instrument for it at Nome; a dispersed
  source layer (mineralized-schist polygons, or the areal lode-distance feature
  `hydrology.py:distance_to_lode_areal_m`) would be the next thing to try.
- The H2 coarseness signal rests on 26 occurrences with text-mined ordinal tags;
  it is suggestive, not conclusive. The Kruskal-Wallis test is not significant at
  0.05, and the Spearman correlation is not spatially cross-validated, so part of
  it could be the same sub-basin co-occurrence that sinks the H1 occurrence signal.
- The coarseness tags are mined by keyword from the ARDF narratives, so they
  inherit whatever the original reporters chose to write down; they are an ordinal
  proxy, not measured grain-size.
- The 25 m DEM resolves the larger confined valleys but not the narrowest gulches;
  the 5 m IfSAR DTM was not fetched for this run.
- 31 of 59 mapped lodes lie outside the DEM extent and cannot seed the walk; some
  true upstream sources just past the north edge are missed.

## Files

- `placers_typed.geojson`, `placers_typed_audit.csv`: typed occurrences + basis
- `down_channel_dist_to_lode.tif`, `straight_line_dist_to_lode.tif`: feature rasters
- `confined_valley.tif`, `zone.tif`, `streams.tif`: spatial-clip masks
- the bulkier terrain intermediates (`filled_dem.tif`, `flow_accum.tif`, `relief.tif`,
  `recv_row.tif`, `recv_col.tif`) are regenerable via `build_terrain.py`, not committed
- `h1_h2_results.json`: all H1/H2 numbers
- `alluvial_points_features.csv`: per-occurrence feature table
- `terrain_meta.json`, `distance_meta.json`: parameters + counts

Pipeline (run in order, from the repo root):
`type_placers.py` -> `build_terrain.py` -> `build_distance_features.py` -> `run_h1_h2.py`
(all under `scripts/nome_placer/inland_local_source/`).
