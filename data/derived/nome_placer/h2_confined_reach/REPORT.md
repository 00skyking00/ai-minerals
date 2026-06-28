# Redesigned H2: confined-reach coarseness vs the schist-limestone contact

Does gold coarseness decline down-drainage from the schist-limestone (marble)
contact in the confined upland placers of the southern Seward Peninsula? This
is the redesigned local-source test: the coarseness gradient (the literature's
diagnostic of local sourcing), with the predictor as distance to the
DOm-marble-vs-DOs/DOq-schist contact, run where clean contacts exist
(Solomon/Bluff/Council, the RI 2024-7 map area), not at Nome, where the marble
is mixed into the DOx unit and no clean in-district contact exists (the round-5
null).

## Plain verdict

**No. Coarseness does not decline with distance from the schist-limestone
contact.** The signal that round 5 found at Nome (coarser gold nearer the
mapped lode) reproduces here against the lode, and is significant; against the
marble contact it is absent, and against the major marble belts it runs the
other way (coarser gold sits farther from the marble). The local source is the
schist-hosted gold-quartz lode system (Big Hurrah type), not the marble
contact. The marble is a passive unit.

The four numbers that carry this:

| predictor | n | Spearman rho (coarseness vs distance) | p | reading |
|---|---|---|---|---|
| down-channel distance to contact (the spec's predictor) | 10 | -0.04 | 0.92 | null, and computable for only 10 of 24 |
| straight-line distance to contact (all interbeds) | 24 | +0.29 | 0.18 | no gradient; if anything reversed |
| straight-line distance to **major marble belts** (>= 1 km^2) | 24 | +0.42 | 0.040 | **reversed**: coarser gold farther from marble |
| straight-line distance to the **36a lode** (positive control) | 20 | **-0.53** | **0.016** | round-5 gradient reproduced: coarser nearer the lode |

A negative rho is the local-source signature (coarser gold at shorter distance).
Only the lode shows it. The contact does not, and the major-marble contact
inverts it.

## Why this is a real negative and not a dead method

The lode positive control is the check. Round 5 found gold coarseness declines
with distance to the mapped 36a schist-hosted lodes at Nome (Spearman
rho = -0.40). Run here against the 15 36a lodes in and around the RI 2024-7 map,
the same gradient appears and is stronger:

- median straight-line distance to the nearest lode: **4,094 m** for
  rough/nuggety gold (class 3), **9,932 m** for coarse (class 2), **13,459 m**
  for fine/flaky (class 1). Spearman rho = -0.53 (p = 0.016); Kruskal-Wallis
  p = 0.066.

So the method detects a downstream-fining gradient when the predictor is the
schist-hosted lode. The contact null is therefore about the contact, not a
broken pipeline. And it coheres: the famous local lode, Big Hurrah, is
gold-quartz veins in graphitic schist, not a marble-contact deposit. Coarse gold
sits near those schist-hosted veins, which lie away from the major marble belts,
which is exactly why the marble-distance trend inverts.

The reversal against the major marble belts (rho = +0.42, p = 0.040) leans
partly on the single class-1 (fine) placer, which sits 5 m from marble. Dropping
it, the trend stays positive but loses significance (rho = +0.33, p = 0.12;
Mann-Whitney class 3 vs class 2 p = 0.13). So the safe statement is the weaker
one: there is no marble-contact gradient in the local-source direction, and the
only direction the data lean is the opposite one.

## The four gotchas (stated as limitations, per the spec)

1. **ARDF point-proxy mirage.** A point on a 3,000-ft paystreak is hundreds of
   metres of along-channel noise. Mitigated by snapping each placer to the
   nearest confined-stream cell and tracing a reach polyline, then measuring the
   down-channel distance at the reach's upstream end rather than at the raw
   point. It is only a mitigation: the snap offset is up to 250 m and the ARDF
   coordinate still anchors the reach.

2. **Topobathy vs paleo-channel divergence.** Outside confined upland gulches
   the modern DEM routes paths the gold never travelled (terraces, fans,
   reworked coastal plain). The test is clipped strictly to confined upland
   V-valleys (local relief >= 45 m over a 525 m window). This clip is also why
   the sample is small: most of the well-documented producers sit on the
   terraces and benches of the Solomon, Casadepaga, and Niukluk valleys, which
   the clip removes. Of 24 coarseness-tagged placers, only 10 have a computable
   down-channel distance, because the rest are either off the confined network
   or sit at drainage heads with no contact crossing upstream.

3. **Survivorship bias in legacy coarseness text.** The coarseness index is an
   ordinal mined from century-old narratives (Brooks 1901, Collier 1908, Smith
   1910, Moffit 1913) that trumpet nuggets and under-report fines. The index is
   "what the reporter found notable," not a volumetric grain-size average. The
   class is the coarsest evidence present (a reach with both fine gold and
   nuggets is class 3), because nuggets cannot have travelled far, so they are
   the proximity diagnostic; this rule deliberately favours the proximal end and
   inherits the reporters' nugget bias.

4. **Blind buried source.** Mapped contacts are interpolated under tundra; a
   placer "2 km from a contact" may sit on an unmapped buried lode or a contact
   that bends under cover. The distance carries an unknown margin, and several
   class-3 placers sit near a contact in straight-line terms while having no
   contact upstream on their channel, which is consistent with a lateral or
   blind source feeding the channel.

## Extra limitations specific to this run

- **Small n.** 24 placers carry a coarseness tag; the down-channel predictor is
  computable for 7 to 10 of them. None of the contact tests would survive a
  multiple-comparison correction, and the lode control, while significant at
  0.016, is not spatially cross-validated, so part of it could be the
  same-sub-basin co-occurrence that round 5 flagged.
- **Class 1 is nearly empty.** Only one confined reach is described as
  fine-only. The distal/fine end of the gradient lives in the coastal plain and
  the wide valleys, which the confinement clip (gotcha 2) removes, so the
  ordinal target has compressed range (mostly class 2 vs class 3).
- **10 m working grid.** The native 5 m IFSAR DTM was fetched and is on disk,
  but the in-process Whitebox hydrology over the 136-Mcell 5 m grid did not fit
  the RAM cap, so the terrain ran at 10 m (a 2x downsample, still 2.5x finer than
  the round-5 25 m grid that missed the gulches).
- **Coarseness mining.** The round-5 cues over-fired on the richer mrdata
  narratives ("485 fine ounces", "fine micaceous sand", "fine-grained graphitic
  schist" all matched "fine" without describing fine gold). The miner used here
  (`coarseness.py`) binds every size word to gold and excludes the
  sediment/bedrock/assay contexts; every tag was verified in context.

## Data and method

- **Predictor (contact).** DGGS RI 2024-7 (Werdon and others, 2024, "Bedrock
  geologic map of the Big Hurrah-Council-Bluff area," 1:50,000, DOI
  10.14509/31308). The GeMS MapUnitPolys are classified by the published
  DescriptionOfMapUnits lithology: marble = DOm + Dm; schist package = DOs, DOq,
  DOg, Omg, DOms, Ds, DOsq, DOqs, Osg, DOu; the mixed DOx is excluded. The
  schist-marble contact is the shared boundary of the dissolved sets: 942 km for
  all interbeds, 439 km for the major belts (58 DOm/Dm bodies >= 1 km^2, 78% of
  the marble area). This is the unit subdivision the round-5 DOx ambiguity
  needed: at Nome the only carbonate is the mixed DOx, so the clean contact
  is a peninsula-scale feature, and here it is mapped.
- **DEM.** Native 5 m IFSAR DTM (Alaska DGGS ImageServer) over the map area,
  98.9% coverage; hydrology run at a 10 m working grid. Whitebox fill ->
  D8 pointer -> flow accumulation; D8 convention verified empirically (1.0000 of
  cells flow to a not-higher neighbour). Streams at 0.05 km^2 contributing area;
  confined-upland mask at relief >= 45 m over a 525 m window.
- **Down-channel distance.** The contact is rasterized and intersected with the
  stream network (6,458 crossings). A single topological pass over the stream
  cells (upstream to downstream by filled elevation) tags each cell with the
  along-channel distance to its nearest upstream contact crossing. On the
  confined streams this distance has real spread (median 456 m, p90 3,284 m),
  so the null is not a no-variance artifact.
- **Target (coarseness).** 75 ARDF placer occurrences fall inside the map
  polygons; all type as alluvial-stream. Full narratives pulled from the USGS
  mrdata ARDF JSON service. Ordinal coarseness (3 rough/quartz-attached/nuggety,
  2 coarse, 1 fine/flaky/flour) mined gold-bound from geology + workings +
  production + comments; 24 tagged (13/10/1).
- **Test.** Spearman rank correlation and Kruskal-Wallis of coarseness class
  against each distance, on the placers that carry both.

## What this does not establish

- It does not test occurrence (where placers sit), only coarseness (the
  downstream-fining gradient), which is the diagnostic the redesign chose.
- It does not prove the marble is barren, only that placer coarseness is not
  organised around the marble contact. A lode-scale or geochemical test would be
  needed to characterise the marble itself.
- The lode gradient is a straight-line association at n = 20 and is not spatially
  cross-validated; it reproduces round 5 but inherits round 5's caveat that
  shared sub-basins can carry such a correlation.

## Files

- `contact_primary_3338.geojson`, `contact_literal_3338.geojson`,
  `contact_inclusive_3338.geojson`, `contact_major_marble_3338.geojson` - the
  schist-marble contact under four unit definitions
- `reaches.geojson` - traced confined reach polylines per snapped placer
- `placers_typed.geojson`, `placers_typed_audit.csv` - typed + coarseness-tagged
  placers with the per-record basis and narrative head
- `down_channel_dist_to_contact.tif` - the predictor raster (confined-clipped)
- `reach_features.csv` - per-placer coarseness + all distances
- `h2_results.json`, `sensitivity_major_contact.json`, `lode_control.json` - all
  test numbers
- `h2_confined_reach_map.png` - the map figure
- `contact_report.json`, `terrain_meta.json`, `distance_meta.json` - parameters

Pipeline (run in order, from the repo root, under
`scripts/nome_placer/h2_confined_reach/`):
`fetch_dem.py` -> `build_contact.py` -> `build_terrain.py` (run_capped) ->
`enrich_and_type.py` -> `build_distance_and_reaches.py` -> `run_test.py` ->
`sensitivity_major_contact.py` -> `test_lode_control.py` -> `make_figure.py`.
