# Does the placer MPM predict drill grade? The corrected capture-efficiency test

*Self-contained report. This rebuilds the drill-gold validation on the design that
Sky's NotebookLM adversarial review prescribed, after that review found the first
attempt (a raw Spearman of point grade against the smoothed block-support MPM in
one creek) could not answer the question and produced a near-zero number that was a
statistical artifact. The corrected harness samples the MPM in its native CRS,
upscales grade to the grid support, converts grade to a binary economic-presence
response at a proposed cutoff, and scores the MPM as an economic-presence
classifier by capture efficiency, with range-restriction disattenuation as a
secondary check and statistical power counted in independent drainages. One
committed, seeded script regenerates every number:
`scripts/nome_placer/drillgold_capture_validation.py`.*

> **This is a labeled methods-validation / underpowered baseline, not a G-result.**
> The only grade-bearing, metre-positioned drill data available today is the 1912
> Janin campaign on Little Creek: 47 holes in a single drainage. One drainage is one
> independent unit, so nothing here can be conclusive. The point of running it now is
> to prove the machinery and set the baseline; the authoritative multi-creek run
> follows the positioned Tuck drill facts that fossick's Map B Track 2 is preparing.

## The result in one line

At one drainage the placer MPM does not separate economic from sub-economic gravel
inside Little Creek (area under the ROC curve 0.47 at the proposed 10 cents-per-yard
cutoff, cluster-bootstrap 95% interval 0.19 to 0.75), and this null survives the
range-restriction correction, so it is a power-and-support limit, not the
attenuation artifact the raw correlation was. The served lode MPM shows a
directional signal at low cutoffs (AUC 0.74 to 0.81) that is consistent with a
local-source reading but rests on 20 blocks with two to five negatives and a range
so compressed that its disattenuated correlation is uninterpretable. Both are hints
at best; the test needs more creeks before it can carry a verdict.

## What the design review corrected

The first version (`drillgold_validation.md`, now superseded) correlated raw
per-hole grade against the MPM value and read the near-zero Spearman as a null. The
review found five reasons that number was driven toward zero regardless of model
quality: change-of-support and regression dilution (point grade against a 25 m
block-average surface), the placer nugget effect, restricted predictor range (every
hole sits at the 54th to 98th percentile of the surface), near-zero effective
independent n (47 holes collapse to about 23 autocorrelated pixels in one creek),
and a continuous-grade response tested against a presence/background classifier. The
sign flip between the native 3338 raster and the 4326 delivery raster was the tell
that the evaluated signal sat below resampling noise. The corrected harness answers
the six design points:

1. **Native CRS only.** The MPM is read by point-to-cell-centre extraction from the
   native EPSG:3338 raster. The 4326 delivery raster is never used for ground truth.
2. **Grade upscaled to the grid support.** Hole grades are aggregated to each grid
   cell by spatial mean. The affine point-to-block variance correction (Isaaks and
   Srivastava 1989) would additionally shrink the block-grade spread; it needs a
   point-support variogram, which one hole per cell in one creek cannot estimate, so
   the block mean is used and that assumption is flagged in the output.
3. **Binary economic-presence response.** The block grade is thresholded to
   economic-presence versus sub-economic at a proposed cutoff in cents of gold per
   cubic yard on the $20.67 per ounce basis. The cutoff is flagged for Sky; a ladder
   is reported so its sensitivity is visible.
4. **Capture efficiency, not raw correlation.** The MPM is scored as an
   economic-presence classifier: the area under its ROC curve, and the share of
   economic blocks captured in its top-ranked blocks, with a
   leave-one-drainage-out structure that is degenerate at one drainage and becomes a
   real held-out measure when more creeks land.
5. **Range-restriction handling.** Any correlation reported is disattenuated for
   restriction of range in the MPM by Thorndike Case II, with the additional
   attenuation from single-assay unreliability noted rather than corrected (no
   duplicate assays exist to estimate reliability). Capture efficiency leads;
   correlation is secondary.
6. **Power in drainages, not pixels or holes.** The true independent n is the number
   of drainages. Little Creek is one, so the power is stated plainly as insufficient
   for a conclusion.

## The economic-presence cutoff (proposed, flagged for Sky)

Converting grade to a binary presence needs a cutoff, and the cutoff is a
geological-economic choice, not a statistical one. The proposed primary is **10
cents of gold per cubic yard** at the $20.67 per ounce basis (about 0.0048 ounces
per cubic yard), the order of magnitude of workable dredge-era placer ground on the
Seward Peninsula. Little Creek was drift-mined for much richer bench gravel, so at
Little Creek 10 cents per yard separates workable from lean rather than pay from
barren. The rationale should be anchored to Moffit 1913 (USGS Bulletin 533) on Nome
placer economics; Tuck 1942 normalizes the pre-1934 gold price.

**Flag for Sky:** confirm the intended cutoff. Three choices each answer a
different question: a dredge-workable threshold near 10 cents per yard, a
drift-mine pay threshold in dollars per yard, or a relative rich-versus-lean split
at the sample median. The harness reports a ladder at 5, 10, 20, and 40 cents per
yard so the answer's dependence on the choice is on the table rather than buried.

## The Little Creek result

The 47 collar-positioned Janin holes carry grades from 1.3 to 200 cents per cubic
yard (median 15.7), so both economic and sub-economic ground is present in the real
data and no negatives are fabricated. They fall in 23 distinct 25 m placer cells and
20 distinct 100 m lode cells, all in one drainage.

**Placer MPM, as an economic-presence classifier (area under ROC curve):**

| cutoff (c/yd) | economic blocks | sub-economic | AUC | cluster-bootstrap 95% |
|---|--:|--:|--:|:--:|
| 5 | 19 | 4 | 0.48 | 0.15 to 0.91 |
| **10 (proposed)** | 17 | 6 | **0.47** | **0.19 to 0.75** |
| 20 | 11 | 12 | 0.44 | 0.20 to 0.70 |
| 40 | 8 | 15 | 0.59 | 0.33 to 0.83 |

Every interval spans 0.5 by a wide margin. Within the drilled set the top 20 percent
of MPM-ranked blocks capture 24 percent of the economic blocks at the 10 cents
cutoff, a capture ratio of 1.18 against the 1.0 a random ranking would give, which
is no better than chance at this n. The placer MPM does not rank economic gravel
inside Little Creek.

The correction that matters here is the one for restricted range, because that was
the review's leading suspect. It is not the explanation for the placer null. The
placer MPM's spread across the drilled blocks (standard deviation 0.048) is close to
its spread across the whole district (0.056), a restriction ratio of only 1.2, so
Thorndike Case II moves the observed Pearson correlation only from -0.10 to -0.12.
The placer surface has nearly its full dynamic range over this creek and still does
not order the grade. The limit is power and support at one drainage, not attenuation.

**Lode MPM, same test (100 m grid):**

| cutoff (c/yd) | economic | sub-economic | AUC | cluster-bootstrap 95% |
|---|--:|--:|--:|:--:|
| 5 | 18 | 2 | 0.81 | 0.69 to 0.92 |
| **10 (proposed)** | 15 | 5 | **0.74** | **0.50 to 0.92** |
| 20 | 10 | 10 | 0.52 | 0.29 to 0.75 |
| 40 | 7 | 13 | 0.60 | 0.36 to 0.82 |

At the two low cutoffs the served lode MPM separates the richer placer blocks from
the leaner ones (AUC 0.74 to 0.81, and every bootstrap resample stays above 0.5).
This is the stronger form of the "richer placer holes sit on marginally higher lode
prospectivity" hint the first report filed, and it points the same way as the H2
confined-reach reading, in which placer gold is richer nearer its bedrock source. It
is still only a hint. The low-cutoff classes are two and five sub-economic blocks,
which is too few to trust an AUC, and the discrimination is gone by the 20 cents
cutoff.

The lode result is also the cautionary case for disattenuation. The lode MPM is near
zero over this placer ground (standard deviation 0.0025) against a district spread of
0.078, a restriction ratio of 31. At that ratio Thorndike Case II amplifies the
non-significant observed correlation (Pearson 0.18, p = 0.44) to 0.985. That number
is an artifact of extreme restriction, not a signal, and the harness marks it
unstable so it cannot be cited. It is the concrete reason the design puts capture
efficiency first and correlation second.

The district-percentile capture (what share of economic blocks sit in the top
percentiles of the whole served surface) is reported in the JSON but reads as
confounded: the entire drilled cluster is high-prospectivity ground at the 54th to
98th district percentile, so both economic and sub-economic blocks land in the
district's upper tail and the metric mostly restates that Little Creek is favorable,
which the presence model already gets right.

## The barren-domain problem and the plan

An economic-presence classifier needs a negative class, and historic placer records
report successes, so true barren ground is scarce. This run avoids fabricating any
negatives: the sub-economic class is the real low-grade tail of the same Janin
holes, so both classes are measured, not invented. That is enough for a
within-drilled-set AUC but not for a district-wide capture curve, which needs a
background of non-economic ground.

Two ways to get that background, neither of them taken as a result here:

- **Pseudo-absences from low-prospectivity non-drilled cells.** Cheap, but circular:
  drawing negatives from low-MPM ground guarantees the MPM appears to separate them.
  Reportable only with that caveat stated, never as a headline.
- **A barren-ground data hunt.** The material read of the problem. Genuine negatives
  live in drill and pit logs that hit no pay: Janin's Dry Creek sub-pay holes and the
  barren property assessments in Tuck 1942 are candidates. They are not positioned
  yet. This is the sound source and it is a data task, not a modeling one.

## The data dependency

The authoritative run is gated on more drainages. The current fossick export
(`phase4/drill_gold_points.geojson`) carries the 47 Janin Little Creek holes plus 10
Tuck points that are claim-centroid only and carry no grade, so they cannot extend
the grade test. fossick's Map B Track 2 is positioning roughly 45 Tuck drill facts
across additional creeks; when those land with collar or block positions and grades,
the harness picks them up with no change (it filters generically for positioned,
grade-bearing holes and clusters them into drainages), the independent n rises above
one, and the leave-one-drainage-out capture becomes a real held-out test.

## What this shows and what it does not

It shows that the corrected metric runs, that it is the right family (the MPM was
built and cross-validated as a classifier, and this scores it as one), and that on
the only drainage available the placer MPM does not resolve within-creek economic
grade even with nearly its full dynamic range present. It shows a directional lode
hint worth keeping.

It does not show that the placer MPM lacks grade information across the district. One
drainage cannot show that. It does not settle the economic cutoff, which is Sky's
call. And it does not replace the presence result: the 0.733 spatial-CV presence AUC
stands, and presence prospectivity and within-deposit grade remain different
questions. The value of this run is the harness and the baseline, waiting on the
drill data that turns one drainage into several.

## Reproduce

```
.venv/bin/python -m scripts.nome_placer.drillgold_capture_validation
```

Writes `data/derived/nome_placer/drillgold_capture/`:
`drillgold_capture_validation.json` (all numbers, all cutoffs, both surfaces),
`drillgold_capture_blocks.csv` (per-block audit), `capture_curve.png` (the placer
capture-efficiency curve at the proposed cutoff).
