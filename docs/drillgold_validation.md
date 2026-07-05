# Does the placer MPM predict drill grade? A ground-truth test against the 1912 Janin Little Creek holes

> **Superseded (2026-07-05).** Sky's NotebookLM adversarial design review found that
> this single-creek test cannot answer whether the MPM predicts grade. Correlating
> raw point-support drill grade against the smoothed 25 m block-support MPM, on 47
> holes that collapse to about 23 spatially-autocorrelated pixels in one 1.4 km
> creek at the 77th prospectivity percentile, is driven toward zero by
> change-of-support and regression dilution, the placer nugget effect, restricted
> predictor range, near-zero effective independent n, and testing a continuous grade
> against a presence/background classifier. The 3338-to-4326 sign flip is the tell
> that the evaluated signal sits below resampling noise. So the pixel Spearman near
> zero reported below is most likely a statistical artifact, not a verdict on the
> model, and the proximity result is reinterpreted as observation bias (see the
> guard section). The test is rebuilt on the corrected design (capture-efficiency,
> native CRS, block-upscaled economic-presence, drainage-level power) in
> [drillgold_capture_validation.md](drillgold_capture_validation.md); read that page
> as the current result. This page is kept as the record of the superseded baseline.

*Self-contained report. The served onshore placer MPM is a presence/background
RandomForest over geomorph and terrain features. It was trained on 65 ARDF/KG
placer occurrence points versus 2000 background cells and never saw a gold grade.
The fossick Phase-4 layer gives us 47 metre-scale drill holes from the 1912 Janin
campaign (Pioneer Mining Company, Little Creek) that each carry a measured value
(cents of gold per cubic yard) and grade (ounces per cubic yard). Because those
grades were never a label or a feature, sampling the MPM at the holes and asking
whether prospectivity ranks grade is a genuine held-out test of the surface. The
answer is a null, and the null is the expected one. All numbers here are read
from a single committed, seeded script;
`scripts/nome_placer/validate_mpm_vs_drillgold.py` regenerates them.*

## The result in one line

The placer MPM does not rank drill grade within Little Creek. At the resolution
the 25 m grid can actually resolve, the rank correlation is indistinguishable
from zero (pixel Spearman rho = -0.11, cluster-bootstrap 95% CI -0.51 to +0.32),
and it flips sign
between the two delivery rasters, which is what a correlation buried in resampling
noise looks like. The lode MPM shows a weak positive hint (rho = +0.36, p = 0.12)
that is not significant and sits over a near-zero score range, so it carries no
claim. This is a presence model tested as a grade model; the null says presence
prospectivity and within-deposit grade are different questions, not that the MPM
is broken.

## The ground truth

The fossick Phase-4 export (`drill_gold_points.geojson`, commit `2411b9e`) holds
78 points. The subset that can carry this test is the 1912 Janin drilling on
Little Creek: 50 holes with a measured grade, of which 47 are positioned per hole
on surveyed collars at metre scale (`extent=collar`) and 3 fall back to a
kilometre-scale block centroid. The 47 metre-scale collars are the primary set;
the 3 coarse points are carried only as a sensitivity and do not change the
answer. The 10 Tuck-claim points sit at claim centroids (claim-scale extent, not
a drill fix) and are left out of the correlation.

Value and grade are the same variable up to a constant: `value_c_cuyd` equals
`grade_oz_cuyd` times the gold price, so their Pearson correlation is 1.000 and a
rank test against either is identical. The report uses `value_c_cuyd`. Grade
spans two orders of magnitude across the 47 holes (0.0007 to 0.097 oz per cubic
yard; median 0.0076), so there is real variation for a model to rank.

## Effective sample size: 23 pixels, not 47 holes

All 47 holes sit in one Little Creek cluster about 1.4 km across. The served MPM
is a 25 m grid, so holes closer together than a pixel share one MPM value. The 47
holes fall in 23 distinct 25 m pixels. The unit of analysis has to be the pixel,
not the hole: within a pixel the MPM is constant while grade varies, so the
within-pixel grade spread is noise the model cannot rank by construction. One
pixel alone holds holes spanning 9 to 143 cents per cubic yard at a single MPM
value of 0.163. Every correlation below is reported at both levels, but the
pixel-level number with a cluster bootstrap is the one to read; the hole-level
number is pseudo-replicated and its p-value is not to be trusted. Even 23 pixels
overstates independence, because adjacent 25 m pixels in one creek are spatially
autocorrelated, so the true resolving power is lower still.

## The placer result

| level | n | Spearman rho | p |
|---|---|---|---|
| hole (pseudo-replicated) | 47 | +0.05 | 0.76 |
| distinct 25 m pixel | 23 | **-0.11** | 0.63 |
| pixel, cluster bootstrap 95% CI | 23 | **[-0.51, +0.32]** | frac(rho>0)=0.31 |
| 4326 delivery raster, pixel | 23 | +0.19 | 0.39 |

The pixel-level correlation is -0.11 with a bootstrap interval that straddles
zero symmetrically. Sampling the 4326 delivery raster (a bilinear warp of the
native 3338 grid) gives +0.19, also null but opposite in sign. A real signal
survives reprojection; a sign flip under bilinear resampling is the signature of
noise. The scatter (`mpm_vs_grade_scatter.png`) shows the same thing by eye: the
richest hole (200 cents per cubic yard) sits at a low MPM value, and the
highest-MPM pixels carry low-to-middling grades.

The MPM is not scoring this ground low. The 47 holes sit at a district median of
the 77th percentile of the served surface (range 54th to 98th), so Little Creek
reads as elevated placer ground, which is correct. The failure is specific: the
MPM ranks *where placers are present* (0.733 spatial-CV AUC, 0.741 buffered, at
the district scale) but does not rank *how rich* the gravel is inside one creek.

## Leakage and circularity guard

There is no direct target leak: grade and value are neither a label nor a feature
of the MPM, which trains on presence/background over geomorph and terrain bands
only.

The subtler concern is spatial circularity. Little Creek is a famous placer, so a
training *presence* positive could sit on the drilled ground and light the surface
up there by construction. It does not. Zero of the 65 training positives fall
inside the convex hull of the drill cluster; the nearest is 457 m from the nearest
hole and 693 m from the cluster centroid. The drilled holes are held-out ground,
not memorized label locations. (The training positives carry a documented ~155 m
NAD27-to-WGS84 offset from the KG export; at that scale the "nearest positive is
several hundred metres away" conclusion is unaffected.)

The proximity check turns up something worth stating plainly. At the pixel level,
drill grade *does* increase toward the nearest ARDF occurrence (rho = +0.52, p =
0.011): the occurrences sit on the richer part of the creek, which is what one
would expect of records made where gold was found. The MPM score does *not* track
that same proximity (rho = +0.12, p = 0.59).

The earlier reading of this contrast, that a real within-creek grade gradient
exists and the presence-MPM fails to resolve it, does not survive the design
review. Grade tracking occurrence-proximity is observation bias: the ARDF
occurrences were logged where the old miners found pay, so distance-to-occurrence
encodes historical sampling effort, not an independent grade structure the MPM was
obliged to capture. The 0.52 is therefore not evidence that the MPM missed real
signal. The correct reading is weaker and is the one the corrected harness carries:
at one drainage this test cannot say whether the MPM resolves within-deposit grade.
The zero training positives in the drill hull still hold, so the surface's elevated
level over Little Creek is not label memorization; that part of the guard stands.

## The lode MPM

The served lode MPM (struct_groves RF on ARDF 36a Au-quartz-vein labels, 100 m
grid) was sampled the same way. Over Little Creek its values are near zero (0.0 to
0.01), as expected for placer ground rather than lode ground. The 47 holes fall in
20 distinct 100 m pixels; pixel Spearman is +0.36 (p = 0.12), cluster-bootstrap
95% CI -0.11 to +0.65, with 94% of resamples positive. This is a directional hint
that richer placer holes sit on marginally higher lode prospectivity, which would
be consistent with a local-source reading (placer gold richer nearer its bedrock
source, the thread the H2 confined-reach work pursues). It is not significant, the
interval includes zero, and the score range it rank-orders is a compressed band
just above zero. It is a hint to file, not a result to cite.

## What this shows and what it does not

It shows that the served placer MPM, validated against real drill grades it never
saw, does not predict grade inside a single creek, and that this holds up under a
clean leak guard. A null against grade is a fine answer: the model was built and
cross-validated as a regional presence detector, and presence prospectivity is not
grade. Nothing here weakens the 0.733 spatial-CV presence result; the two measure
different things.

It does not show that the MPM has no grade information anywhere. This is one
clustered campaign in one creek, about 23 independent pixels of near-uniform high
prospectivity, which is a hard place to detect a grade gradient even if one were
encoded. The 71 unpositioned Tuck drill results and any future campaigns on other
creeks would widen the test to across-creek grade contrast, where a regional
surface has more room to discriminate. Until then the read is: the placer MPM
tells you where to expect placer gold, not how rich the pay will be.

## Reproduce

```
.venv/bin/python -m scripts.nome_placer.validate_mpm_vs_drillgold
```

Writes `data/derived/nome_placer/drillgold_validation/`:
`drillgold_validation.json` (all numbers), `holes_sampled.csv` (per-hole audit),
`mpm_vs_grade_scatter.png` (the figure).
