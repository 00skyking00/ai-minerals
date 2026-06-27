>>> [HANDOFF] 2026-06-27 coordinator → ai-minerals  (SUPERSEDES the Hummel-contact version of this file)

# Round 4B — digitize the discrete Nome River till (Qnt) polygon and build the TRUE drift-on-beach feature

Sky chose to pursue the drift-on-beach lever, then ran NotebookLM over the loaded geological corpus and
found a map that separates the Nome River drift from the beach as its own polygon. I pulled it and
confirmed it visually. **This replaces my earlier instruction to digitize the Hummel bedrock/Quaternary
contact** (the Hummel map lumps drift + beach; this one does not). Use the source below.

## The source (pulled, on disk, verified) — PRIMARY
`ai-minerals/research/nome_debate_library/TaggGreene_1973_PP759A_offshore_Nome.pdf`
USGS **Professional Paper 759-A**, Tagg & Greene 1973, *High-Resolution Seismic Survey of an Offshore
Area Near Nome*. The map is **Figure 2, "Generalized geologic map and cross section of the coastal plain
at Nome (after Hopkins, in Hopkins and others, 1960)"** — render **PDF page index 6** (printed page A3).
This single georeferenced sheet carries everything the feature needs:
- **Qnt** = *Till and outwash of Nome River Glaciation* — the **discrete drift polygon** over the coastal
  plain (the drift we want).
- **The raised-beach units as their own polygons**: **Qtb** (marine sand/gravel of Third + Intermediate
  Beach), **Qsb** (marine sand/gravel of Second Beach), **Qse** (estuarine silt behind Second Beach
  shoreline) — i.e. the strandlines as mapped polygons, independent of the round-3 REM derivation.
- **Buried gold pay-streak lines**, labeled **T** (Third), **M** (Monroeville), **I** (Intermediate),
  **IS** (Inner Submarine), **OS** (Outer Submarine) — the actual surveyed gold geometry.
- **Pz** schist bedrock uplands/inliers; minor Qal/Qg/Qit/QTs.
- **Cross-section A–A′** with beach elevations (Submarine → Second → Intermediate → Monroeville → Third →
  Fourth, sea level to ~+300 ft) and the Qnt-till relationships.
- Graticule **165°00′ / 165°30′ W, 64°30′ N**; scale bar 1-0-4 miles; north arrow; Nome town + Cape Nome
  marked; named creeks (Snake R., Nome R., Dry/Anvil/Dexter/Osborn/Hastings Creek) as control points.

Secondary / cross-check (also on disk): **Greene 1970 Bull 1312-B Fig 3** (same map, 0–5 mi bar, page
index 9) and **Mulligan 1967 USBM** (`Mulligan_1967_USBM_Nome_coastalplain_placers.pdf`, the independent
"ancient beach lines" survey of the same paystreaks) — use either to confirm the registration / paystreak
positions. PP689 Fig 4 maps the regional + offshore glaciation extent if you need the seaward Qnt edge.

## Task
1. **Georeference** Figure 3 to EPSG:3338 (the placer grid CRS) using the graticule intersections
   (165°00′/165°30′ W × 64°30′ N) plus the coastline and named-creek mouths as control points. Report a
   small registration RMS.
2. **Verify overlap FIRST** (the RI 2024-6 lesson): the map's footprint (64°30′N coast, 165°00′–165°30′W)
   should contain essentially all 65 onshore placer positives. Report the fraction inside before building.
3. **Digitize the Qnt polygon** (Nome River till + outwash) as a vector. Optionally also capture the Qsb
   beach polygon — but the **strandlines you already built in round 3** (the +38/+70/+120 ft raised
   beaches on the IfSAR REM) are the beach layer; you do NOT need to re-derive them.
4. **Build the TRUE strandline × Qnt-drift intersection feature** (round-3 `placer_drift_on_beach_cv.py`
   / `build_strandline_on_dem()` machinery, now with the real drift polygon in place of the absent one),
   and **test its marginal AUC under F1 CV** against the current placer feature set. Handle coverage
   explicitly. **A null is still a fine, final result** — but this is the genuine test the genesis
   synthesis intended, not the bedrock-contact proxy.
5. Optional cross-check: PP689 (Nelson & Hopkins 1972, also on disk) Figure 4 maps the regional Nome
   River glaciation maximum extent incl. the offshore lobe; use it only to sanity-check the Qnt boundary
   if the Greene registration is ambiguous near the coast.

## Output
One PR off `origin/main`: the georeferenced Qnt vector (+ SOURCE.md provenance: Greene 1970 Bull 1312-B
Fig 3, after Hopkins 1960), the feature builder, the F1 CV numbers (JSON/CSV) + a short report stating
the marginal lift + CI and whether the real drift polygon adds placer signal. No prod promotion. Voice
rules on prose; ML terms-of-art in the deep notebook only.

— coordinator
