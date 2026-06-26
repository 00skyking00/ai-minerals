# Do the 400 m DIGHEM magnetics and the targeted GeMS structure features move the Nome lode model?

*Self-contained report. Two acquired layer families were aligned to the Nome
prospectivity grids and added to the placer and lode presence/background models,
and the cross-validated AUC change was measured under the project's leak-guarded
spatial cross-validation (F1). The first family is the 1993 Nome DIGHEM-V
airborne survey (400 m lines: residual magnetics, derivatives, EM apparent
resistivity). The second is a set of structure and lithology covariates built to
target the documented orogenic-gold control at Nome (distance to NE- and
NW-trending faults, distance to fold hinges, a graphitic-host flag). The
question for both: once the footprint-coverage confound is removed, is there
real, transferable signal that the existing ~1 km composite features could not
see?*

## The result in one line

The coordinator's headline layer, the 400 m DIGHEM magnetics, did **not** move
the lode model. On the cells the survey actually flew, the magnetic bands are flat
to negative (district −0.049, in-box −0.018); the eye-catching +0.106 the
magnetics shows on the full extent is the survey outline tracking central Nome,
not signal. The full geophysics stack (magnetics plus EM resistivity) is
suggestive in the small box (+0.058, 95% CI −0.010 to +0.128) and cannot be told
from zero on the district (+0.031, −0.047 to +0.108).

The layer that did move the number is the addendum's. The targeted structure and
lithology features (distance to NE- and NW-trending faults, fold-hinge proximity,
graphitic-host flag) raise the collapsed district lode model by **+0.106** on
mapped cells (AUC 0.540 to 0.646, 95% CI +0.050 to +0.164, the one interval here
that clears zero). That is real, coverage-stripped, ground-property signal
pointing at the documented Rock Creek control. Two qualifiers sit on it: it
moves a near-random district model only to weakly predictive, and it rests on 82
positives, so it is a screen and not yet a verdict. And the test the addendum
expected to corroborate it independently, graphitic host equals EM conductor, does
not hold at Nome: the mapped graphitic units read as resistivity highs, not lows.

## 1. The two questions, and why they are separate

The lode model at Nome is a documented near-miss. In a small box it scores AUC
near 0.80; widened to the full district it falls to roughly 0.58 to 0.63, the
restriction-of-range artifact analysed in the F1 re-baseline. The model has been
running on a ~1 km statewide aeromagnetic composite (`akmag_*`) and an
undifferentiated distance-to-fault from the regional SIM 3131 bedrock map. The
coordinator's two handoffs asked whether higher-resolution, better-targeted
layers change that.

1. **Geophysics (handoff 1).** The 1993 Nome DIGHEM-V survey flew 400 m lines, a
   four-fold tighter spacing than the statewide composite, and gridded magnetics
   plus frequency-domain EM apparent resistivity at 25 m. Does the 400 m
   magnetics give the lode model structural signal the 1 km blur could not?

2. **Structure and lithology (handoff 2 addendum).** The gold-source literature
   names a specific, mappable control: orogenic quartz-vein gold on high-angle
   faults in two regional sets (NE- and NW-trending), hosted in the most
   carbonaceous (graphitic) part of the Nome Group, concentrated near fold
   hinges (USGS Bull. 1786; Piekenbrock & Odden 2009, *Econ. Geol.* v.104; the
   global model is Groves et al. 2018). These are *ground-property* features:
   they exist whether or not anyone ever found gold there, the opposite of the
   F3 knowledge-graph covariates whose apparent signal was the occurrence
   inventory marking its own labels.

The two are kept on separate arms because they answer separate questions and
carry separate coverage footprints.

## 2. The coverage confound, and how it is handled

Both new layer families were collected where the mines are. The DIGHEM survey
flew central Nome; the Nome mining district GeMS maps the district core and
little of its periphery. The lode occurrences cluster in exactly that core. So a
covariate that is merely *present* in the surveyed area, and sentinel-filled to
`-999` elsewhere, will separate occurrences from background through coverage
alone, not geology.

The check on the lode positives makes the confound concrete:

| feature | near positives | near background | also: P(in footprint) |
|---|---|---|---|
| distance to NE fault (district) | ~720 m | ~16 km | positives 88% mapped, background 29% |
| distance to NW fault (district) | ~1.1 km | ~15 km | |
| graphitic host (district) | P=0.086 | P=0.010 | |

The separation is real, but so is the coverage gap: the positives sit almost
entirely inside the mapped footprint. To strip the confound, every arm is graded
not only over the whole modelled extent but also restricted to the cells where
the layer actually exists:

- `auc_full` over the whole extent (mixes signal with the coverage proxy),
- `auc_dighem` restricted to cells the DIGHEM survey flew,
- `auc_gems` restricted to cells inside the GeMS mapped footprint,
- `auc_both` restricted to cells that are flown and mapped.

The marginal AUC change that answers each headline is taken on the clean subset,
not on `auc_full`. For the magnetics that is `auc_dighem(geophys) -
auc_dighem(base)`; for the structure features it is `auc_gems(struct) -
auc_gems(base)`.

## 3. The cross-validation design

Unchanged from F1: presence/background random forests
(`RandomForestClassifier(n_estimators=300, class_weight="balanced",
random_state=42)`), graded by pooled out-of-fold ROC-AUC under leak-guarded
spatial cross-validation. Folds are contiguous spatial blocks; a 1 km dead zone
around each test block exceeds the 400 to 800 m proximity buffers in the
features; the block edge is twice the residual-variogram range.

One change sharpens the marginal reading: the block size is computed **once per
dataset from the base model** and reused for every arm. Adding a covariate
therefore cannot move the fold geometry, so an arm's AUC delta is attributable
to the features, not to a shifted partition.

## 4. The new covariates: provenance and construction

**Geophysics: 1993 Nome DIGHEM-V** (Alaska DGGS GPR 2019-11, DOI
10.14509/30189). ER Mapper grids in NAD27 / UTM 3N, reprojected to EPSG:3338 via
the NADCON5 Alaska grid (the datum shift is ~160 m at Nome, several in-box cells,
so it is applied explicitly rather than left to a ballpark transform). Bands:
residual total-field magnetics (4 km Gaussian regional removed), first vertical
derivative, analytic signal, tilt angle, and log10 apparent resistivity at 900
Hz, 7200 Hz, and 56 kHz. Derivatives are computed on the native 25 m grid before
reprojection so the model-grid boundary never enters the gradient stencils.

**Structure and lithology: Nome mining district GeMS** (Alaska DGGS PDF 94-39),
the most detailed public structural map of the district, reprojected from NAD27 /
UTM 3N to EPSG:3338 through the same NADCON5 grid. Four covariates:

- `dist_ne_fault`, `dist_nw_fault`: distance to the nearest fault or lineament
  segment trending NE (strike azimuth 22.5 to 67.5 degrees) or NW (112.5 to
  157.5 degrees). The 615 mapped faults and 200 mapped lineaments were exploded
  to segments and each segment classified by its own azimuth, so a fault that
  bends contributes its NE and NW reaches to the correct set. The two sets are
  kept separate because the orientation, not bare fault proximity, is the
  control. The length-weighted azimuth distribution confirms the documented
  pattern: the NE set dominates (197 km of NE-trending trace versus 96 km NW).
- `dist_fold_hinge`: distance to the nearest mapped fold axis (10 anticline /
  overturned-anticline traces in the GeMS `StructureLines`).
- `carbonaceous_host`: a flag set where the cell falls in a graphitic Nome Group
  schist unit. Three units carry "graphitic" explicitly in the GeMS unit
  descriptions (`pCPzsg` graphitic schist and quartzite; `pCPzspm` graphitic
  "lumpy" schist; `pCPzsgb` biotite graphitic schist of the Solomon Schist).

A companion `gems_extent` mask records the GeMS bedrock footprint and defines the
`auc_gems` subset above.

## 5. The carbonaceous-host EM cross-check (independent, and it does not corroborate)

The addendum proposed a clean, non-leaky test: graphite is conductive, so the
carbonaceous host should read as a low in the airborne EM apparent resistivity,
independent of any occurrence label. It does not. On the in-box cells that are
both mapped and flown, the graphitic-host cells read **more resistive**, not
less, at every frequency:

| EM frequency | median log10 ohm-m, graphitic host | median, non-host | direction |
|---|--:|--:|---|
| 900 Hz (deepest) | 3.00 | 2.70 | host +0.30 (more resistive) |
| 7200 Hz | 3.47 | 2.65 | host +0.82 |
| 56 kHz (shallowest) | 3.67 | 2.85 | host +0.82 |

The most likely reason is depth of investigation. These airborne frequencies
sense the top ~30 to 150 m, which at Nome is dominated by resistive material:
discontinuous permafrost and dry weathered bedrock on the uplands where the
graphitic schist outcrops. A connected-graphite conductor, if present, sits below
that and at a scale the 400 m line spacing does not resolve. The practical
consequence: the carbonaceous-host flag can still help the model as a lithologic
covariate, but the 1993 EM does not independently confirm the mapped units as
conductors, so the hoped-for clean cross-check is not available here.

## 6. Experiment arms

Each arm is the base features plus one covariate group. Placer carries the
geophysics arms only; structure is a lode-control hypothesis and is not expected
to bear on alluvial gold.

| arm | added covariates |
|---|---|
| base | the existing F1 features (unchanged) |
| mag | 4 DIGHEM magnetic bands |
| res | 3 DIGHEM resistivity bands |
| geophys | all 7 DIGHEM bands |
| struct (lode only) | dist NE/NW fault, dist fold hinge, carbonaceous host |
| geophys_struct (lode only) | all DIGHEM + all structure |

## 7. Results

Read the clean-subset delta, not the full-extent one. The contrast between them
is the whole point: on the full extent every new layer looks like a win, and
most of that win is the survey footprint tracking the occurrence cluster.

**Lode in-box** (30 positives; base AUC 0.767 on GeMS-mapped cells; 900 m blocks):

| arm | clean AUC | clean Δ |
|---|--:|--:|
| mag (4 magnetic bands) | 0.749 | −0.018 |
| res (3 resistivity bands) | 0.766 | −0.001 |
| geophys (mag + res) | 0.825 | **+0.058** |
| struct (NE/NW fault, fold, host) | 0.711 | −0.056 |
| geophys + struct | 0.797 | +0.030 |

**Lode district** (82 positives in-subset; base AUC 0.540 on mapped cells, the
collapsed model; 3800 m blocks). The last column is the full-extent delta, shown
only to expose how much of it is coverage:

| arm | clean AUC | clean Δ | full-extent Δ |
|---|--:|--:|--:|
| mag | 0.491 | −0.049 | +0.106 |
| res | 0.573 | +0.033 | +0.175 |
| geophys | 0.581 | +0.041 | +0.180 |
| struct | 0.646 | **+0.106** | +0.215 |
| geophys + struct | 0.634 | +0.094 | +0.185 |

The magnetics row is the cleanest illustration of the confound. On the full
extent the 400 m magnetics looks like the biggest single-band win on the district
(+0.106); restricted to the cells the survey actually flew, it is **negative**
(−0.049). The entire apparent magnetic gain was the survey outline tracking
central Nome, where the occurrences are. The same inflation, smaller, sits under
every other arm (struct +0.215 full versus +0.106 clean).

**Placer** (65 positives; geophysics arms only): every geophysical arm is flat to
negative on flown cells (mag −0.086, res −0.001, geophys −0.039). Magnetics is
not a placer signal, which is expected; the placer lever is the deferred
beach-line LiDAR, not these layers.

**Intervals on the clean deltas.** Because the clean subsets carry few positives
(in-box 30, district 82), each headline delta was given a paired bootstrap (2000
resamples of the scored cells, base and arm scored on the same draw). Only the
district structure arm clears zero:

| dataset | arm | clean Δ | 95% CI | P(Δ>0) |
|---|---|--:|:-:|--:|
| lode in-box | geophys | +0.058 | −0.010 to +0.128 | 0.96 |
| lode in-box | struct | −0.056 | −0.135 to +0.019 | 0.07 |
| lode district | geophys | +0.031 | −0.047 to +0.108 | 0.79 |
| lode district | **struct** | **+0.106** | **+0.050 to +0.164** | **1.00** |
| lode district | geophys + struct | +0.093 | +0.031 to +0.160 | 1.00 |

## 8. Interpretation

**The magnetics is not the lode lever.** The handoff expected the 400 m magnetics
to be the highest-value new layer because it is four times tighter than the
statewide composite. It is not. Restricted to flown cells the magnetic bands are
negative on both the in-box and the district. The reason is that the lode control
at Nome is the *orientation* of structure and the *lithology* of the host, and
neither is what residual total-field magnetics at 400 m line spacing separates
cleanly here. Whatever coarse magnetic contrast does matter, the existing 1 km
`akmag` composite already carried; the finer survey adds noise, not discrimination.

**The structure features carry the signal, and only at district scale.** The
addendum reasoned from the gold-source literature to a specific, mappable control
and asked for features that target it. Those features are the only arm that moves
a clean number that clears zero: +0.106 on the district, moving the collapsed
model from 0.540 to 0.646. They do the opposite in the small box (−0.056), which
is consistent rather than contradictory: the in-box model already scores 0.77 from
its existing geology and fault features and has 30 positives, so four more
correlated covariates overfit the folds. The district model is the one that was
broken (near-random at 0.540 on mapped cells), and it is the one the targeted
structure repairs. The direction of the gain matches the geology: lode positives
sit a median ~720 m from a NE-trending fault against ~16 km for background, and
are nine times more likely to fall on a graphitic host.

**The full-extent numbers are a coverage trap, the same trap F3 fell into.** Every
arm looks like a large win on the whole extent (+0.18 to +0.215 on the district).
Restricting to the footprint where the layer exists halves the structure gain
(+0.215 to +0.106) and flips the magnetics from +0.106 to −0.049. The sentinel
`-999` outside the survey is a coverage flag, and coverage correlates with the
occurrence cluster. The clean-subset AUC is the only number to act on, and the
gap between it and the full-extent number is a running measure of how much of any
"new layer win" at Nome is just the survey outline.

## 9. What was deferred, and why

This run integrated the two layer families that target the **lode** headline.
Several handoff items were left for a follow-up, each for a stated reason, not
silently dropped:

- **Placer beach-line LiDAR (REM / detrended DEM).** The coordinator's strongest
  placer recommendation: lead the placer model with sea-stand and raised/buried
  beach-line features from the 2021 USACE topobathy LiDAR. The point tiles are
  staged, but the ready-made 1 m bare-earth DEM (NOAA dataset 10207) is not yet
  pulled, and a relative-elevation-model build is its own piece of work. The
  placer arms here therefore test only the geophysics, which is not expected to
  help placer (and, below, does not). The beach-line LiDAR is the right next
  placer step.
- **Detailed-geology one-hot to replace SIM 3340 / the coarse in-box codes.**
  The GeMS units used here for the carbonaceous flag could also replace the
  coarse geology one-hot; that is a larger re-feature of the base model and is
  held separate so it does not confound this marginal test.
- **AGDB4 / Seward soil geochemistry.** Carries an Au-pathfinder circularity
  caveat (a geochemical anomaly is half-way to an occurrence), so it is a
  diagnostic layer, not a headline prospectivity feature; deferred.
- **2024 Seward SkyTEM resistivity.** The survey block sits north of the
  modelled extent (zero overlap with the placer-core in-box, only a thin
  northern district strip), so it adds nothing to these grids. Held for a
  northern-district extension.

## 10. What this does and does not establish

**Establishes.** Under leak-guarded spatial CV, on the cells where each layer
actually exists: the 400 m DIGHEM magnetics does not separate Nome lode from
background (negative on both models); the EM resistivity is weakly positive; and a
small set of structure/lithology covariates built to target the documented
orogenic-gold control gives the district lode model a marginal gain whose 95%
interval clears zero (+0.106, +0.050 to +0.164). The direction matches the
geology, and the features are ground properties that exist independent of the
occurrence labels, so this is not the F3 self-marking confound.

**Does not establish.** That the district lode model is now usable: 0.646 is
weakly predictive, not a working classifier, and the structure features point a
direction rather than solve the district problem. That the +0.106 survives a
proper lode rebuild: this is a screen on 82 positives with a single random-forest
fit, not the leave-one-out, dispersed-label test the addendum flagged as the real
next step. That the graphitic host is an EM-confirmed conductor: at Nome's shallow
airborne frequencies it reads resistive, so the independent cross-check the
addendum wanted is not available, and the host flag earns its place only as a
map-derived lithology covariate. And nothing here speaks to placer: the
geophysics does not help it, and the beach-line LiDAR that should was deferred.

The next steps, in order: the dispersed/typed-label lode rebuild (so the 82
positives become a larger, less restriction-prone target), then the beach-line
LiDAR for placer, then the detailed-geology one-hot and geochem augmentation.

## Reproduce

```
# build the DIGHEM and GeMS-structure covariate grids, then re-grade every arm
PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.newlayers_geophys_rebaseline
# numbers -> data/derived/nome_placer/newlayers_rebaseline/{json,csv}
# EM cross-check -> .../carbonaceous_em_crosscheck.json
```

Modules: `src/ai_minerals/data/dighem.py` (geophysics),
`src/ai_minerals/data/nome_structure.py` (structure/lithology). Both cite their
DGGS sources and DOIs in the module docstring.
