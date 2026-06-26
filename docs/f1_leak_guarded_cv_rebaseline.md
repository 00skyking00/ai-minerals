# F1: leak-guarded spatial CV moves the placer to 0.68 and reproduces the lode collapse

This re-grades the three Nome presence/background models under a stricter
cross-validation scheme that closes two leaks naive folds leave open. The placer
regional AUC falls from 0.733 to 0.679. The lode model still scores 0.806 inside
the coastal placer box and 0.633 across the full district, so the restriction-of-
range failure that the district widening first exposed survives the stricter
ruler. A third result is methodological: how blocks are assigned to folds shifts
the district AUC by about 0.10, and the wrong choice hides the collapse.

Machine-readable numbers: `data/derived/nome_placer/f1_rebaseline/f1_leak_guarded_rebaseline.json`
and `.csv`. The reusable harness is `src/ai_minerals/spatial_cv.py`
(tests in `tests/test_spatial_cv.py`); the driver is
`scripts/nome_placer/f1_leak_guarded_rebaseline.py`.

## Why the old AUCs were optimistic

Random K-fold cross-validation on spatially autocorrelated samples grades a model
on points it has effectively already seen: a held-out test cell usually has a
near-twin in the training fold a few hundred metres away. Two specific leaks
matter for these models.

The first is general spatial autocorrelation. Terrain, geology, and magnetics all
vary smoothly, so nearby cells carry nearly the same covariates and the same
label. A fold that scatters test cells among training cells measures
interpolation, not the extrapolation a real prospectivity map has to do.

The second is feature-specific and matters for the next build. The F2 covariates
encode distance to known occurrences with 400 to 800 m buffers. A distance-to-
occurrence value written into a test cell is computed from the same occurrence
that sits in a training cell across the fold boundary, so the label leaks through
the feature regardless of how the folds are drawn.

## The scheme

The harness combines three standard pieces from the spatial-ML literature, cited
in the module.

Block size comes from the autocorrelation range of the model residuals, not the
raw covariates. We fit a non-spatial K-fold first, take its out-of-fold residuals,
and fit a spherical variogram to them (Roberts et al. 2017; Valavi et al. 2019,
blockCV). The block edge is set to twice the fitted range, deliberately above what
the variogram suggests, because any overfitting flattens residual structure and a
residual variogram understates the range.

Blocks are grouped into ten folds as contiguous regions (compact block groups),
so a held-out fold is a connected piece of ground and the model is forced to
predict a region it never trained on. This is what turns the cross-validation into
an extrapolation test.

A dead-zone buffer drops every training point within 1 km of any test point
(Airola et al. 2018). One kilometre is the figure that exceeds the 400 to 800 m
proximity buffers, so the F2 distance features cannot leak across a fold edge.

The estimator is the same RandomForest the existing scripts use (300 trees,
balanced classes, seed 42), so the AUCs are comparable to the old numbers.

## Results

All AUCs are pooled out-of-fold ROC-AUC. "Old scheme" is the prior KMeans-10
spatial CV (no buffer / 300 m buffer). "Leak-guarded" is the new contiguous-fold
scheme with the 1 km dead-zone. "Scatter" is the same scheme with positive-
balanced scattered folds instead of contiguous ones (discussed below).

| Model | n positives | Old scheme | Leak-guarded | Scatter | Variogram range | Block edge |
|-------|------------:|-----------:|-------------:|--------:|----------------:|-----------:|
| Placer onshore (geomorph + terrain) | 65 | 0.733 / 0.741 | **0.679** | 0.708 | 450 m | 900 m |
| Lode in-box (structural, existing feats) | 30 | 0.802 / 0.791 | **0.806** | 0.751 | 450 m | 900 m |
| Lode district (structural, existing feats) | 93 | 0.620 / 0.582 | **0.633** | 0.731 | 1879 m | 3800 m |

Within the district run, the AUC restricted to the placer-core in-box subset is
0.737 against the district-wide 0.633. Every positive was scored in every run
(no fold held out so many positives that training lost the class).

## The lode collapse, reproduced

The acceptance check for this piece was whether the harness still detects the
restriction-of-range failure on the existing structural features. It does. The
lode model scores 0.806 inside the coastal placer box and 0.633 across the
district, a gap of 0.17 under the stricter scheme. The single-run alarm agrees:
from one district model, the in-box subset reads 0.737 against 0.633 district-
wide.

The mechanism is unchanged from the original diagnosis. Inside the box, every
known lode sits on upland schist and the background is coastal flat, so elevation
separates the two classes and the model leans on it (elevation is the top feature
at 0.24 importance). Widen the frame to include the Big Hurrah lode and the
upland hinterland, where background cells are also high elevation, and elevation
stops separating anything. A model that learned "high ground means lode" cannot
generalise, and the district AUC reports that.

The reassuring part is that the leak-guarded contiguous numbers land within noise
of the prior KMeans-10 numbers (0.806 against 0.802 in-box, 0.633 against 0.620
district). The earlier collapse was not an artifact of the old fold scheme. The
new harness confirms it and adds the variogram justification, the dead-zone, and
the in-box-versus-district divergence as a standing alarm.

A point in the harness's favour: the same contiguous scheme leaves the in-box
model at 0.806. It does not deflate every model by reflex. It deflates the one
that fails to generalise and leaves the one that does. That is the property a
calibrated ruler should have, and it is the counterweight to the known tendency
of spatial CV to overstate error on genuinely independent data.

## The placer number drops to 0.68

The placer regional model falls from 0.733 to 0.679. The cause is the 1 km dead-
zone: placer occurrences cluster along drainages, so removing training points
within a kilometre of each test point strips out the correlated near neighbours
the old no-buffer fold kept. The drop of about 0.05 is the size of the
autocorrelation leak in the old number.

This is the result the piece was meant to produce. A regional AUC near 0.68 is
the number to carry forward. It still sits well above the v3.1 rule-based
composite baseline (0.444) and well above chance, so the geomorphology-plus-
terrain map ranks known placer gold above background with real skill. The skill
is just smaller than the prior figure implied.

## Fold assignment is the lever, and it works against one instruction

The largest single methodological finding is that grouping blocks into folds by
spatial contiguity versus by positive-balanced scatter changes the district AUC
by about 0.10, and the two disagree about which model is stricter depending on
the geometry.

For the district, scattered folds read 0.731 against the contiguous 0.633. The
scatter is optimistic: it spreads 3.8 km blocks across all ten folds, so every
test block has training blocks a few kilometres away on every side and the model
interpolates the elevation gradient instead of extrapolating to an unseen region.
The residual variogram range here is 1.9 km, far smaller than the roughly 30 km
scale over which the coastal-to-upland elevation gradient drives the restriction-
of-range bias, so variogram-sized blocks alone do not break it. Only contiguous
regional holdout does.

This is where the build departs from the written instruction. The directive asked
for blocks assigned to folds so that positives are distributed evenly across
folds. For the lode positives, which concentrate in the placer core and at Big
Hurrah, even distribution requires scattering the folds, and scattering hides the
collapse (district reads 0.73 instead of 0.63). Even distribution and reproducing
the collapse are in direct conflict for clustered positives. The harness
implements both and reports both; the headline number is the contiguous one,
because reproducing the collapse was the stated acceptance test and a lower
number under stricter validation is the stated goal. Pooled out-of-fold AUC does
not need per-fold positive balance to be defined, so contiguous folds cost
nothing on that front as long as no single fold holds out the whole positive
class, which none did here.

If the program wants the scattered, positive-balanced number as the headline
instead, it is one argument away in the driver, and the trade is spelled out
above. This is the one decision in the piece worth a second opinion.

## The dead-zone and the F2 proximity features

The current placer and lode models carry no distance-to-occurrence features, so
the 1 km dead-zone mostly guards against general autocorrelation today. Its
reason for being 1 km rather than 300 m is forward-looking: F2 adds proximity
covariates with 400 to 800 m buffers, and a dead-zone narrower than the buffer
would let those features leak. The harness is therefore already set for F2 to be
measured without that leak.

One caveat for the scattered strategy specifically: the district residual range
(1.9 km) exceeds the 1 km dead-zone, so under scatter, correlated training points
in the 1.0 to 1.9 km band still leak. Contiguous folds do not have this problem
because the fold geometry, not the dead-zone, carries the separation. If a future
run needs the scattered number to be strict, set the dead-zone to the larger of
1 km and the fitted residual range.

## What this does not prove

The numbers are pooled out-of-fold AUC with no confidence interval; with 30 to 93
positives the sampling noise is real, and the lode in-box 0.806 against the prior
0.802 should be read as "unchanged," not "improved." The variogram is isotropic
and spherical by assumption; Nome terrain is anisotropic along the coast, so the
fitted range is an average direction. The background is a random on-land sample,
so these AUCs measure ranking of known occurrences against generic ground, not
against carefully matched pseudo-absences. None of this changes the two findings
that matter: the placer skill is near 0.68 rather than 0.73, and the lode model
does not generalise past the box it was built in.

## Reproduce

```
uv run python -m scripts.nome_placer.f1_leak_guarded_rebaseline
uv run python -m pytest tests/test_spatial_cv.py -v
```

The driver writes the JSON and CSV under
`data/derived/nome_placer/f1_rebaseline/`. The run is deterministic at seed 42.
