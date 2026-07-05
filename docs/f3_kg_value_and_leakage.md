# Does the knowledge graph help the Nome models? The leakage, and the design that could answer it (G2)

*Self-contained writeup. The fossick lane built a knowledge graph (KG) from
historical Nome mining records: mineral-occurrence reports, patent-claim cards,
assay and bulletin citations. The program-level question G2 asks whether feeding
that KG back into the two Nome prospectivity models as covariates measurably
improves them. The experiment that tests it is F3. The naive result is a large
cross-validated AUC gain. This document shows that the gain is target leakage,
explains why no cross-validation trick can remove it, and specifies the corrected
leak-free design that could answer G2 for real. The seven-arm baseline table, the
rasterizer guards, and the leak-guarded cross-validation internals live in the
companion report [`f3_kg_marginal_lift.md`](f3_kg_marginal_lift.md); this writeup
is the interpretation and the corrected design.*

## The question, and the short answer

Two mineral-prospectivity models cover the Nome district: one for placer
(alluvial) gold, one for lode (hard-rock) gold. Each ranks ground by favorability
from terrain and geology covariates. The KG is a separate, document-derived
inventory of what the historical record says about the same ground. G2: does the
KG, added as covariates, make the models better under leak-guarded evaluation? A
positive, null, or negative answer all count as the deliverable. We did not need a
win.

The short answer has three parts. The naive answer, a cross-validated AUC gain of
**+0.19 to +0.34**, is leakage: the KG covariates and the model labels are drawn
from the same occurrence inventory, so a covariate that says "a record exists
here" is the positive class wearing a disguise. That leakage cannot be cross-
validated away, because it is arithmetic, not autocorrelation. The KG's real
possible contribution is not proximity to records at all; it is the **geology the
records describe** (host rock, structure, alteration, deposit model), mapped as
independent physical fields and separated from the footprint of past exploration
effort. Whether that contribution is real cannot be settled with the data in hand.
It needs a validation set of independent, newly-discovered blind deposits, which
does not yet exist.

## The demonstration: one bit reproduces almost the whole gain

The F3 driver builds a control arm, `kg_present`: a single binary covariate that
is 1 where any KG record falls in the cell and 0 elsewhere. No lithology, no
structure, no distance, one bit. On its own it reproduces nearly the entire gain
of the full KG stack.

| dataset | base AUC | + `kg_present` (1 bit) | + full KG stack | full-stack gain |
|---|--:|--:|--:|--:|
| placer_onshore | 0.679 | 0.970 | 0.988 | +0.31 |
| lode_inbox | 0.806 | 0.987 | 0.997 | +0.19 |
| lode_district | 0.633 | 0.963 | 0.970 | +0.34 |

Pooled out-of-fold ROC-AUC; contiguous spatial folds; 1 km dead-zone; seed 42;
refreshed against the current fossick export (40,099 KG entities). The full table
with every arm is in the companion report.

A single bit cannot carry geology. It can carry exactly one fact: a record was
filed on this cell. Because the model's positive labels are drawn from that same
record inventory, "a record was filed here" is very close to "this cell is a
positive." The model learns the rule from the training positives and applies it to
a held-out positive whose own feature value is, by construction, 1. That is the
whole gain. The full KG stack also adds distance-to-nearest-occurrence, claim
density, and per-record attributes, and every one of those is non-null or
near-extremal precisely where a record sits, so each restates the same coverage
signal. The lode-district restriction-of-range gap is the tell: at base the model
scores 0.737 inside the placer box and 0.633 across the district (it only works
where it was effectively trained), and once the KG is added both rise to about
0.99 and the gap disappears. The covariates now mark the labels, so the model
looks like it generalizes when it does not.

## Why cross-validation cannot fix a circular feature

The project's cross-validation was built to stop leakage. It fits a spherical
variogram to the model residuals, sets spatial-CV blocks to twice the fitted
range, holds out contiguous regions so the model must extrapolate, and drops every
training point within 1 km of any test point (the Airola 2018 dead-zone). Those
guards defeat spatial autocorrelation: they stop a training positive's signal from
bleeding across a fold boundary into a nearby test cell. They do nothing to a
feature whose value on the test cell is tied to that cell's own label.

The distinction is the crux. Autocorrelation leakage is about **neighbors**: cell
B looks positive because cell A next to it is positive and the feature is smooth.
A dead-zone removes A from training and the leak closes. Coverage leakage is about
the cell **itself**: the test positive's `kg_present` is 1 because the test
positive's own record is what made it a positive in the first place. No amount of
removing other cells changes the test cell's own value. The feature is
mathematically circular, and a circular feature survives every fold split you can
draw, because the circularity is inside each point, not between points.

The one arm that can answer the steelman ("distance-to-known-occurrence is a
covariate real exploration maps do use") is `kg_loo`: recompute the spatial fields
fold-aware and leave-one-out, so that for each fold the visible record set excludes
the held-out blocks and no cell ever sees its own record, then re-measure the
marginal change against a paired bootstrap interval. Once the circularity is
removed, the gain is gone:

| dataset | base | `kg_loo` | change | 95% interval | P(change > 0) |
|---|--:|--:|--:|:--:|--:|
| placer_onshore | 0.679 | 0.642 | **-0.036** | (-0.090, +0.016) | 0.09 |
| lode_inbox | 0.806 | 0.777 | **-0.028** | (-0.077, +0.016) | 0.11 |
| lode_district | 0.633 | 0.562 | **-0.071** | (-0.133, -0.013) | 0.007 |

The two in-box models show no change distinguishable from zero. The district model
gets **worse**, by a margin whose whole interval sits below zero: distance-to-
training-occurrence points the model at where the training positives cluster, and
contiguous folds hold out a different cluster, so the feature aims at the wrong
ground. That is the clustering artifact spatial CV is built to penalize, showing up
as a negative instead of a spurious positive.

There is a second, simpler reason the proximity features are a dead end even if
they had survived: a feature defined as distance-to-known-occurrence cannot be
computed on greenfield ground, because greenfield ground has no known occurrences.
A covariate that only exists where the answer is already known is not a
prospectivity covariate. So the circularity is structural, not a tuning bug, and
the fix is not a better fold scheme. It is a different feature set.

## The corrected leak-free design

The design below is a specification, not a finished build. It is staged to the
fossick geology attributes as they land (see the next section). Six parts.

1. **Purge every self-referential proximity feature.** Drop `kg_present`,
   distance-to-nearest-occurrence, claim density, distance-to-nearest-claim, and
   the per-record attribute fields as currently rasterized (they are non-null only
   where a record exists, so they carry the same coverage confound). Nothing that
   answers "is there a record near here" may enter the model.

2. **Features become independently-mapped physical fields.** The KG's value is the
   geology it extracted, not the location of the paperwork. The covariates become
   distance to mapped structural intersections, distance to intrusive contacts,
   distance to alteration halos, host-rock class, and resolved deposit model:
   fields defined at every cell from mapped geology, not attributes pinned to
   occurrence points. This is the KG's extracted geology re-expressed as geometry a
   greenfield cell also has.

3. **Deconfound exploration effort.** Past discovery follows access as much as
   geology: roads, coastline, claim density, and the density of surviving text all
   mark where people looked, not only where gold is. Split the covariates into a
   geology block z(x) and an effort block e(x) (claims, claim density, text
   density, infrastructure proximity). Fit a propensity surface on the effort block
   alone, predicting the probability that a cell was explored, then inverse-
   propensity-weight the background sample so the presence/background contrast can
   no longer be won by mapping accessibility. This is propensity-weighted
   positive-unlabeled learning; the target-group-background idea from species
   distribution modeling (Phillips et al. 2009) is the same correction for the same
   sampling-bias problem, and Elkan and Noto (2008) is the PU-learning basis.

4. **Spatial block cross-validation sized to the new features.** Refit the
   residual variogram on the new geology features and set the block size and the
   dead-zone to exceed that variogram range, not the old one (mapped geology fields
   are smoother than point-attributed ones, so the range will grow). Every
   transform, the propensity surface, the variogram, and any scaling, is fit on the
   training partition only, inside each fold, never on the full set.

5. **Test association first, then a corrected AUC gain.** Run a likelihood-ratio
   test for association between the geology block and the labels before reading any
   AUC, so a null model is rejected on its own terms. Then report the marginal
   change as a bootstrapped delta-AUC with a Bonferroni correction for the number
   of arms tested, so the multiplicity of trying several feature sets does not
   manufacture a false positive. The primary metric is Top-p% capture efficiency
   (what share of held-out deposits fall in the top p% of ranked ground), because a
   prospectivity map is used as a ranking, and capture efficiency scores the
   ranking directly where AUC averages over operating points a driller never uses.

6. **State the ceiling plainly.** Even with every guard above, statistical
   significance for "the KG improves the model" is unreachable with the present
   data. All labels and all KG geology come from the same historical inventory over
   the same explored ground, so the strongest available test is internal
   consistency, not out-of-sample discovery. The claim becomes testable only when a
   validation set of independent, newly-discovered blind deposits exists: sites
   found after the model was frozen, on ground the historical record did not
   already flag. Until then the corrected design produces an effect size and an
   interval labeled underpowered, not a verdict.

## What the KG can contribute now, and what is pending

The corrected design consumes mapped geology, so its build is gated on the fossick
lane turning extracted facts into mapped geometry. The state today:

**Available now** (in the current fossick feature export,
`exports/features/feature_table`):

- **Deposit model, resolved.** `deposit_class` (placer / lode / unknown) and
  `deposit_model_resolved` with a confidence, source count, and agreement flag,
  produced by the truth-discovery layer. This is the one geology feature ready to
  fold in as-is.
- **Extracted geology attributes, as per-record flags.** Host rock (schist,
  marble, carbonate, gneiss, granite, metavolcanic, clastic, unconsolidated),
  structure type (vein, shear, contact), alteration minerals (silicic, sericite,
  chlorite, carbonate, oxide, sulfide), and workings. These are extracted and
  rasterizable today, but only as attributes attached to occurrence and claim
  grounds, so they are non-null only where a record exists. In that form they still
  carry the coverage confound and cannot enter the model until they are re-mapped
  as independent fields (below).

**Pending** (the geometry work in flight in the fossick lane):

- **Structural controls as geometry.** Distance to mapped structural
  intersections and to fault or shear traces, computed from a mapped structure
  network rather than from occurrence points.
- **Host rock as a mapped field.** Host lithology expressed as domain polygons or a
  distance-to-contact field over the whole grid, not a per-occurrence flag.
- **Intrusive contacts and alteration halos as mapped footprints.**
- **Tuck favorability.** The favorability surface derived from the Tuck 1942
  assessment corpus.

The plan is to fold these in as they arrive and re-run the corrected design each
time, reporting capture efficiency with an underpowered label, rather than to
build the full propensity-weighted harness in one pass now against attributes that
are not yet mapped.

## What this establishes, and what it does not

- It does **not** establish that the KG improves the Nome models. The large
  headline gain is leakage; the leak-free proximity arm is null on the in-box
  models and negative on the district model.
- It does **not** establish that the KG cannot help. The extracted geology has not
  been tested as independently-mapped fields, because those fields do not exist
  yet. The negative result is about proximity-to-records, not about geology.
- It **does** establish the mechanism of the leak (shared inventory, no leave-one-
  out, a coverage bit that reproduces the gain) and why cross-validation cannot
  remove it (the circularity is inside each point, not between points).
- It **does** specify the design under which G2 becomes answerable: purge the
  proximity features, map the geology as physical fields, deconfound exploration
  effort by inverse-propensity-weighting the background, size the spatial CV to the
  new features, test association before AUC, and read capture efficiency with a
  multiplicity correction.
- It states the ceiling: the answer stays underpowered until independent blind
  discoveries provide an out-of-sample validation set. That is a data problem, not
  a modeling one.

## Reproduce

The baseline numbers in this writeup come from the committed F3 outputs:

```
data/derived/nome_placer/f3_kg_marginal_lift/f3_kg_marginal_lift.{json,csv}
data/derived/nome_placer/f3_kg_marginal_lift/f3_kg_loo.{json,csv}
```

regenerated by `scripts/nome_placer/f3_kg_marginal_lift.py` and
`scripts/nome_placer/f3_kg_loo.py` (paths factored to
`scripts/nome_placer/f3_paths.py`). The seven-arm table, the rasterizer's
over-attribution guards, and the leak-guarded cross-validation design are
documented in [`f3_kg_marginal_lift.md`](f3_kg_marginal_lift.md).
