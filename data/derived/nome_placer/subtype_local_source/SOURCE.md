# SOURCE: placer subtype typing + local-source covariates

## Typing scheme

Travis L. Hudson, 2006, *New studies of the Quaternary geology of the Nome
coastal plain* (USGS proposal; `research/PleistoceneNomeCoastalPlainProposal2006.docx`),
synthesizing Metcalfe, R.W., and Tuck, R., 1942, *Some aspects of the Nome
district placer deposits*. The type characterization is in the proposal text:

- L50: "Several placer gold deposits including the Monroeville, Intermediate, and
  Center 'beaches' are on a marine abrasion platform cutting metamorphic bedrock."
- L51: "The abrasion platform placer deposits on bedrock are characterized by
  coarser gold and abundant pyrite and arsenopyrite in heavy mineral concentrates
  compared to those at strandline beach deposits such as Second and Third beaches
  which are garnet and black sand-rich."
- L52: "The abrasion platform deposits have fairly abrupt western limits."
- L53: "Lode and residual alluvial placer deposits define a belt that trends
  southeast to the edge (and probably onto) the coastal plain (Hudson, 1979;
  Figure 2)."
- L54: "Most of the production is from within 4 miles of Newton Peak (Figure 1)."

The handoff adds the Inner/Outer Submarine paystreaks to the abrasion-platform
type. Cross-check map (not digitized here): USGS Professional Paper 759-A, Tagg &
Greene 1973, Fig 2 (`research/nome_debate_library/TaggGreene_1973_PP759A_offshore_Nome.pdf`,
page idx 6) carries the labeled T/M/I/IS/OS paystreak lines and beach polygons.
Typing was done from occurrence names rather than by tracing that scanned figure,
which an unattended run cannot trace reliably; the named deposits agree with
Hudson's named types one-for-one.

## Placer positives

`data/raw/fossick_kg/kg_nome.jsonld` (fossick knowledge graph export), filtered to
placer commodity or USGS deposit code 39, clipped to the v3p1 placer grid, in the
canonical lexsort order. 65 positives; the typing rows align to the positive block
of `f1_leak_guarded_rebaseline.load_placer()` (asserted on coordinates).

## Local-source covariates

- **36a lodes**: `data/raw/nome_mpm/ardf_nome.geojson`, records whose `model_code`
  contains 36a (low-sulfide Au-quartz vein), 63 occurrences.
- **Bedrock/Quaternary contact**: GeMS SIM 3131 Seward Peninsula geologic map
  (1:500,000; `data/raw/sim3131_seward/nmgeol/nmgeolp.shp`), inland boundary of the
  largest Qs (coastal-plain) polygon against non-water bedrock units. Same line as
  the round-4B bedrock-contact run, here kept district-wide (no Nome C-1 clip) so
  upland positives get a defined value. Densified to 50 m and queried by nearest
  vertex.
- **Newton Peak**: USGS Geographic Names Information System feature 1406988
  (Summit, Nome AK), 64.55891231556038 N, -165.31865641368893 W, from
  carto.nationalmap.gov geonames service. Reprojected to EPSG:3338.
- **Belt axis**: principal eigenvector of the 36a lode point cloud, line forced
  through Newton Peak. Azimuth 88 deg (E-W, the lode strike). See REPORT.md sec. 2
  for why this is the lode strike and not Hudson's SE projection.

## Cross-validation + statistics

F1 leak-guarded spatial CV (`ai_minerals.spatial_cv`): contiguous folds sized
from the base-model residual variogram (2x range), 1 km dead zone,
RandomForest(300, balanced, seed 42). Marginal AUC deltas with 2000-resample
paired bootstrap (`newlayers_bootstrap.boot_delta`). Distributional contrast:
one-sided Mann-Whitney U, Cliff's delta, 2000-resample bootstrap CI on median
differences, Kruskal-Wallis across placer groups (`scipy.stats`).

## Files

- `placer_typing.csv`: the 65 positives with type, basis, coordinates, and each
  covariate value.
- `placer_subtype_local_source.json`: full results (typing counts, both tests).
- `placer_subtype_local_source.csv`: flat summary, one row per (covariate, group).
- `REPORT.md`: the narrative + verdict.

Build: `PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_subtype_local_source`
