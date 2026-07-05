# Does a document-derived knowledge graph improve the Nome gold models? (F3)

> **Status: pre-validation BASELINE, held (2026-07-05).** The stale-path bug is
> fixed and committed: both F3 drivers now read the fossick KG exports from
> `fossick/exports/features` (factored to `scripts/nome_placer/f3_paths.py`), not
> the deleted `fossick-f2`. The numbers in this report have been reproduced against
> the current fossick export (40,099 KG entities, 133 bands): the base AUCs are
> identical, the headline full-KG arm and the leave-one-out deltas move by less than
> 0.02, and no arm moves by more than 0.03, so the leakage reading is unchanged. They are a
> pre-validation baseline, not a settled G2 finding. The F3 design is under the same
> NotebookLM adversarial review that the drill-gold validation received, and the
> authoritative F3 rerun is on hold pending Sky's design corrections. The precise
> fresh baseline (the seven-arm table and the leave-one-out delta with its bootstrap
> interval) lives in the 2026-07-05 coordinator baseline report; it is deliberately
> not wired into any chapter, notebook, or model card.

*Self-contained report. We rasterized a knowledge graph built from historical
mining records onto the Nome prospectivity grids, added it to the placer and lode
models as covariates, and measured the cross-validated AUC change under the
project's leak-guarded spatial cross-validation. The plan called for reporting
the result straight, positive or negative. The result is negative, and the rest
of this document explains why.*

## The result in one line

Adding the full knowledge-graph (KG) covariate stack raised pooled out-of-fold
ROC-AUC by **+0.19 to +0.34** (placer 0.679 to 0.989, lode-district 0.633 to
0.972). That number is not geological signal. A single binary "a KG record
exists in this cell" flag, on its own, reproduces almost the entire change
(placer 0.679 to 0.970, lode-district 0.633 to 0.963). The KG covariates and the
model labels are drawn from the **same occurrence inventory** and there is **no
leave-one-out**, so the features mark the labels. The decisive test is the
fold-aware, leave-one-out arm (`kg_loo`) in [section 9](#9-the-leak-free-test-kg_loo);
its marginal change is the only number here that bears on whether the KG carries
real prospectivity signal.

That test is now run, and **no leak-free KG signal survived**. Once a cell cannot
see its own record, the change collapses to roughly zero on both in-box models
and turns significantly negative on the district model: placer **-0.018** (95%
interval -0.069 to +0.035), lode in-box **-0.003** (-0.063 to +0.056),
lode-district **-0.083** (-0.142 to -0.027). The whole apparent gain was
leakage.

## 1. Objective

Two mineral-prospectivity models cover the Nome district, Alaska: one for placer
(alluvial) gold, one for lode (hard-rock) gold. They rank ground by favorability
from terrain and geology covariates. Separately, the fossick lane built a
knowledge graph from historical mining records (mineral-occurrence reports,
patent-claim cards, assay and bulletin citations). The question for F3: does
feeding the KG back in as covariates measurably improve the models when they are
evaluated without spatial leakage? A positive, null, or negative answer all count
as the deliverable. We did not need a win.

## 2. The three models under test

All three are presence/background classifiers: positives are known mineral
occurrences, and "background" is a sample of other cells. The estimator for this
test is `RandomForestClassifier(n_estimators=300, class_weight="balanced",
random_state=42)`. The metric is pooled out-of-fold (OOF) ROC-AUC.

| dataset | scope | n | positives | base covariates |
|---|---|--:|--:|---|
| placer_onshore | placer core box | 2065 | 65 | 7 geomorphic-population fuzzy scores (beach, abrasion platform, Tertiary bench, sea-stand, beach-stream confluence, off-beach creek, buried beach) + DEM, slope, TPI |
| lode_inbox | placer core box | 2030 | 30 | 5 surface-geology unit flags + aeromagnetics + distance-to-fault + DEM, slope, TPI |
| lode_district | full district | 3093 | 93 | 9 geology units + aeromag + dist-fault + DEM, slope, TPI |

Labels are mineral-occurrence points from the Alaska Resource Data File (ARDF)
and allied records, placed on the grid by a WGS84 loader.

The backdrop matters. The lode model is a documented failure: in a small box it
scored AUC near 0.80, and widened to the full district it collapses to roughly
0.58 to 0.63, a textbook restriction-of-range artifact (the small-box model had
learned elevation, because every known lode sat in upland schist while background
was coastal flat). The placer model is the working one, near 0.68 to 0.73 under
spatial CV; an earlier non-spatial CV reported 0.733.

## 3. The cross-validation design (built to prevent leakage)

This is the leak-guarded scheme (`ai_minerals.spatial_cv`) the whole project was
built around. Three guards:

1. **Residual-variogram block sizing.** Fit a spherical variogram to the model
   *residuals* (not the raw covariates), then set the spatial-CV block edge to
   2x the fitted range, deliberately larger than the variogram suggests, because
   overfitting flattens residual structure and a raw variogram understates the
   needed block size. Measured ranges: 450 m (placer and lode in-box) gives 900 m
   blocks; 1879 m (lode-district) gives 3800 m blocks.
2. **Contiguous-region folds (the headline).** Each held-out fold is a connected
   piece of ground, which forces the model to extrapolate to unseen terrain. A
   scattered/balanced fold variant runs as a sensitivity; for clustered positives
   it hides the lode collapse, so contiguous is the headline (see the scatter
   rows in [section 6](#6-results)).
3. **Airola-2018 dead-zone.** Drop every training point within r = 1 km of any
   test point. 1 km exceeds the KG proximity buffers (400 to 800 m), so
   distance-to-occurrence features cannot leak via *training* neighbors across a
   fold boundary.

In the results, `auc_district` is AUC over the dataset's own extent.
`auc_in_box` is AUC restricted to the placer-core subset; it is only meaningful
for lode_district, where it is the restriction-of-range alarm (in-box AUC far
above district AUC means the model only works where it was effectively trained).

## 4. The knowledge-graph covariates: provenance and construction

**Provenance is the crux.** The KG feature table is built from the **same
occurrence inventory** (ARDF plus a patent-claim card corpus, KARDEX) that the
model **labels** come from. A placer or lode positive cell and its KG "ground"
are placed by the same WGS84 convention and land in the same cell. There is **no
leave-one-out**: a test positive's own record is present in its KG features.

Construction is extent-aware, with two over-attribution guards
(`ai_minerals.kg_rasterize`):

- Each KG ground (occurrence, claim, or area) is attributed to cells by an
  **extent tag**: `point` writes to one cell; `claim_polygon` writes to every
  covered cell; `area_footprint` spreads across a buffered disc (about 200 m;
  see the gap in [section 8](#8-known-caveats-and-gaps)); `district` is dropped
  from cell attribution (a regional footprint pinned to cells is the ecological
  fallacy).
- A **per-cell cap** (after USGS Alaska OFR 2021-1041): where several grounds
  hit one cell, the content comes from the single highest-scoring ground (score =
  source-grade x mean-confidence) and the occurrence count is capped at one, so
  known-deposit clusters cannot dominate.

Two covariate *kinds* come out of this, and the distinction is the whole game:

- **Spatial fields, defined at every cell:** distance to nearest KG claim, claim
  density within 400 m, distance to nearest KG occurrence. These are computed
  from the full ground set, **not** leave-one-out.
- **Ground-attribute fields, non-null only where a record exists:** source grade,
  mean confidence, and prose-derived rock flags (host lithology, alteration,
  structure, workings) plus a 96-dimension prose-text embedding. By construction,
  "has a KG attribute" is collinear with "is a known site."

The rasterizer flags this coverage confound in code comments and the F3 driver
builds a **control arm** (`kg_present`, a lone coverage bit) and a **leakage-demo
arm** (`kg_label`, which adds explicitly label-restating columns such as
is_placer / is_lode) to *quantify* the trap rather than hide it.

## 5. Experiment arms (each = base features + a delta)

| arm | added features |
|---|---|
| base | F1 base covariates only |
| kg_present | the single "a KG record exists here" coverage bit (the control) |
| kg_spatial | distance-to-claim, claim density, distance-to-occurrence |
| kg | full KG stack (spatial + ground-attribute fields), about 27 to 32 extra columns |
| prose | the 96-d prose embedding only |
| kg_prose | full KG + prose embedding |
| kg_label | full KG + explicitly label-restating columns (is_placer / is_lode / commodity): the leakage demo |
| **kg_loo** | **fold-aware, leave-one-out spatial fields only (the leak-free test, [section 9](#9-the-leak-free-test-kg_loo))** |

## 6. Results

Pooled OOF ROC-AUC; contiguous folds; r = 1 km dead-zone; seed 42.

| dataset | base | kg_present (1 bit) | kg_spatial | kg (full) | prose | kg_prose | kg_label |
|---|--:|--:|--:|--:|--:|--:|--:|
| placer_onshore | **0.679** | 0.970 | 0.955 | 0.989 | 0.981 | 0.988 | 0.988 |
| lode_inbox | **0.806** | 0.987 | 0.885 | 0.997 | 0.984 | 0.992 | 0.998 |
| lode_district | **0.633** | 0.963 | 0.963 | 0.972 | 0.968 | 0.966 | 0.966 |

Marginal change over base (full `kg` arm): placer **+0.31**, lode_inbox
**+0.19**, lode_district **+0.34**.

Sensitivities:

- **Scatter folds** (the less-strict variant): base placer 0.708, lode-district
  0.731; the `kg` arm stays 0.98 to 0.99. The KG change is fold-strategy
  invariant.
- **Sample weighting** (source-grade x confidence on positives): moves AUC by
  less than 0.003. No effect.
- **lode_district in-box subset** under the `kg` arm is 0.995 versus district
  0.972 (both high); the base restriction-of-range gap (in-box 0.737 versus
  district 0.633) essentially disappears once KG is added, which is what
  "features now mark the labels" looks like.

The numbers behind this table are in
`data/derived/nome_placer/f3_kg_marginal_lift/f3_kg_marginal_lift.{json,csv}`.

## 7. Interpretation: target leakage from a coverage confound

We read this as target leakage, not geological signal, for four reasons:

1. **A single coverage bit explains nearly all the change.** `kg_present` alone
   takes lode-district 0.633 to 0.963 and placer 0.679 to 0.970. One bit cannot
   carry lithology or structure; it can only encode "a record was filed here,"
   which, because labels and KG features come from the same occurrence inventory,
   is the positive class itself.
2. **No leave-one-out.** A test positive cell's KG features include its own
   record: `kg_present` = 1, distance-to-occurrence near 0, ground attributes
   populated from its own card. The model learns "kg_present = 1 implies
   positive" from other training positives and applies it to the test positive,
   whose own feature value is by definition 1.
3. **The dead-zone does not fix this.** r = 1 km removes *training points* near a
   test point, which controls the spatial autocorrelation of the training signal.
   It does not, and cannot, remove the *test point's own feature value*, which is
   tied to its label. So even leak-guarded CV passes the leak straight through any
   feature derived from the label inventory.
4. The placer / lode-district near-equality (about 0.97 to 0.99) and the collapse
   of the lode restriction-of-range gap are both consistent with "the features
   now mark the labels," not with "the model learned transferable geology."

The steelman for the other side: the *spatial fields*
(distance-to-known-occurrence, density) are arguably legitimate prospectivity
covariates, because a real exploration map does use proximity to known
mineralization, and the dead-zone exists to keep that usage leak-free. The
question is whether distance-to-nearest-occurrence is a valid covariate here or
is circular because the test positive's own occurrence sits at distance 0. The
only way to answer it is to exclude each test cell's own record from every KG
feature and re-measure. That is [section 9](#9-the-leak-free-test-kg_loo).

## 8. Known caveats and gaps (independent of the leakage question)

- **Grid choice.** KG covariates were rendered on the *served* model grids (25 m
  placer-core, 100 m district), not the nominal 250 m of the modeling plan,
  because 250 m collapses most claim polygons to a single cell and defeats the
  extent-aware guard. The base AUCs come from the same grids, so the marginal
  change is within-grid, but the absolute numbers are grid-specific.
- **area_footprint export gap.** 178 KARDEX "area" grounds carry only a centroid
  in the F2 export, not a surveyed polygon, so "spread across the footprint" is
  approximated by a 200 m disc.
- **CRS offset.** A roughly 155 m NAD27/WGS84 offset is shared by labels and
  covariates, so it cancels for the marginal-AUC comparison, but it is a real
  absolute-position error.
- **Small positive counts** (30 to 93). AUC variance is non-trivial. The
  leak-free arm in section 9 carries a bootstrap interval for that reason; the
  earlier arms are single point estimates at seed 42.

## 9. The leak-free test (kg_loo)

The arms in section 6 cannot answer "does the KG genuinely help," because every
KG feature is collinear with the labels. The `kg_loo` arm is the reformulation
that can.

**Design.**

- **Fold-aware, leave-one-out spatial fields only.** For each CV fold, the
  visible ground set excludes every ground that falls inside the held-out fold's
  blocks. Each cell's `kg_dist_occurrence_m`, `kg_dist_claim_m`, and
  `kg_claim_density` are then computed from that fold's *training-region* grounds
  only, and never from a ground in the cell's own grid cell. A held-out positive
  therefore cannot see its own record, and neither can a training positive see
  its own (the leave-one-out). This is the leak-free version of
  "distance to known mineralization."
- **The coverage-confounded features are dropped.** `kg_present`, the
  ground-attribute fields (source grade, confidence, rock flags), and the prose
  embedding are all non-null only where a record exists, so they cannot be made
  leak-free for a presence model whose positives *are* those records. They remain
  in section 6 only as the labelled leakage-demo arms.
- **Same scheme as everything else.** Contiguous folds, r = 1 km dead-zone,
  residual-variogram block sizes (taken from the base model so the folds are
  identical to the base arm), RandomForest(300, balanced), seed 42. The only
  thing that differs from `base` is the three leak-free spatial columns.
- **Interval.** Because the positive counts are small (30 to 93), the marginal
  change carries a 95% interval from a paired, class-stratified bootstrap of the
  pooled OOF predictions (2000 resamples, seed 42), so a near-zero result can be
  read against its noise.

**Results.** Pooled OOF ROC-AUC; same folds as section 6 (the base AUC recomputed
inside this arm matches the section 6 base to within 0.0000 on all three
datasets, confirming the folds are identical). 95% intervals are from a paired,
class-stratified bootstrap (2000 resamples, seed 42).

| dataset | base | kg_loo | change | 95% interval on the change | P(change > 0) |
|---|--:|--:|--:|:--:|--:|
| placer_onshore | 0.679 | 0.660 | **-0.018** | (-0.069, +0.035) | 0.26 |
| lode_inbox | 0.806 | 0.802 | **-0.003** | (-0.063, +0.056) | 0.46 |
| lode_district | 0.633 | 0.550 | **-0.083** | (-0.142, -0.027) | 0.006 |

The two in-box models show no change distinguishable from zero: the point
estimates are slightly negative and the intervals straddle zero. The district
model gets **worse** by a margin whose whole interval sits below zero. For
lode_district the leak-free in-box subset AUC also collapses, from 0.737 (base)
to 0.532 (`kg_loo`).

The numbers are in `data/derived/nome_placer/f3_kg_marginal_lift/f3_kg_loo.{json,csv}`.

**How to read it.** The section 6 gain was self-leak. Once a cell cannot see its
own record, the KG adds nothing on the in-box models and actively hurts the
district model. The district sign is not a surprise: distance-to-training-
occurrence encodes "near where the training positives cluster," and contiguous
folds hold out a *different* cluster, so the feature points the model at the
wrong ground. That is the clustering artifact spatial CV is built to penalize,
showing up here as a negative rather than as a spurious positive. No leak-free
proximity signal survived.

## 10. What this does and does not establish

- It does **not** establish that the document-derived KG improves the Nome
  models. The large section 6 changes are leakage.
- It **does** establish a clean way to test the claim: drop the
  coverage-confounded features, recompute the spatial fields fold-aware and
  leave-one-out, and read the marginal change against a bootstrap interval. The
  `kg_loo` number in section 9 is that test.
- The `kg_loo` result is that finding, and it is negative: a knowledge graph
  built from the same inventory that defines the labels does not, by itself, add
  prospectivity signal once a cell cannot see its own record. On the district
  model it subtracts signal. This is the negative outcome the plan anticipated,
  not a bug to be tuned away.
- The next experiment that could still turn up a real signal uses **disjoint
  inventories**: label the model on ARDF occurrences and derive the KG proximity
  features from KARDEX claims only (or the reverse), so the feature and the label
  are no longer the same record. That is the one design under which
  distance-to-known-mineralization is not circular.
