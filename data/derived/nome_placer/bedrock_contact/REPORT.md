# Round 4B: does the bedrock-contact drift-on-beach proxy add placer signal?

**Verdict: no. The proxy adds nothing over the beach-line base. This is a final negative.**

## The lever and why it is a contact, not a drift polygon

The Nome raised-beach placer model holds that gold concentrates where a raised
strandline crosses gold-bearing glacial drift: the beach reworks and reconcentrates
the drift's gold. Rounds 3 and 4 could not build the strandline-by-drift overlay
because the two surficial maps that map a discrete Nome River drift unit (AOF 125,
RI 2024-6) both lie far east of the placer ground with zero overlap.

Round 4B uses the precise Nome-town source, USGS MF-247 (Hummel 1962, Nome C-1
quad, 1:63,360). MF-247 maps the whole coastal plain as one undifferentiated
"Unconsolidated deposits" unit, so there is no drift-versus-beach edge to trace.
The one digitizable boundary is the bedrock (Nome Group) / Quaternary-cover
contact, the inland edge of the drift-and-beach plain. That contact is still a
placer covariate: the gold-bearing drift was shed from the bedrock uplands and
reworked seaward, so distance to the contact is a proxy for the drift's inland
extent and source proximity, and the raised strandlines cross it.

## What was done

1. **Georeferenced MF-247 to EPSG:3338** on its four graticule corners. Residual
   about 20 m, below the line width at this scale. See `SOURCE.md`.
2. **Overlap check first** (the RI 2024-6 lesson): **58 of the 65 placer positives
   fall inside the Nome C-1 footprint**. The seven outside are south of the
   coastline, the beach and offshore occurrences, exactly as expected. Unlike
   AOF 125 and RI 2024-6, MF-247 actually covers the placer ground, so the feature
   is worth building.
3. **Contact vector**: the inland edge of the coastal-plain Qs polygon from the
   digital GeMS SIM 3131 (1:500,000), clipped to the C-1 footprint. The scanned
   MF-247 cannot be hand-traced reliably in an unattended run; SIM 3131 supplies
   the same contact as a clean digital line, and near the placer ground it follows
   the MF-247-mapped boundary within the 1:500,000 generalization (a few hundred
   meters). MF-247 is georeferenced and staged for the manual-digitizing precision
   upgrade if it is ever wanted.
4. **Feature**: cells where a raised strandline (Second +11.58 m, Third +21.34 m,
   Fourth +36.58 m NAVD88, ±2.5 m) meets the contact, rasterized at 5 m on the
   IfSAR DEM. The feature is the distance to the nearest such cell, defined inside
   the C-1 footprint only. The crossing self-restricts to the coast: the low
   raised-beach bands only meet the contact near the shoreline (58 crossing cells),
   so the contact's inland convolution never enters the feature. A second arm tests
   plain distance-to-contact.

## Result

Folds, estimator and bootstrap follow the F1 leak-guarded recipe (contiguous-region
folds sized from the base model variogram, 1 km dead zone, RandomForest(300),
2000 bootstrap resamples). `auc` is the out-of-fold ranking score; the marginal is
the change in that score from adding the feature to the base, with a 95% interval.

| feature added to base | marginal (all positives) | 95% interval | marginal (inside C-1) | 95% interval |
|---|---|---|---|---|
| strandline × contact | **−0.030** | [−0.072, +0.011] | **−0.031** | [−0.070, +0.009] |
| distance to contact | −0.016 | [−0.052, +0.020] | −0.010 | [−0.048, +0.029] |

Base out-of-fold auc: 0.679 (all 65 positives), 0.593 (58 inside C-1). Both
features land at or just below zero; every interval spans zero, and the strandline
crossing is slightly negative (adding a feature with no signal costs a random forest
a little). The beach-line backbone already in the base captures whatever the contact
proxy could contribute.

## What this does not prove

- It does not say the strandline-by-drift mechanism is geologically wrong. It says
  the bedrock-contact proxy for that mechanism adds no ranking signal over a base
  that already scores strandline elevation, abrasion platform and buried-beach
  proximity. The drift effect, if present, is already inside the beach-line score.
- A discrete Nome River drift polygon over the central placer ground, if one is
  ever digitized, could carry more than the bedrock contact does. No such vector
  exists for central Nome; both AOF 125 and RI 2024-6 are eastern maps.
- The contact is the 1:500,000 GeMS line. The 1:63,360 MF-247 line would place it a
  few hundred meters more precisely, but the positives sit a median 1.6 km seaward
  of the contact, so that precision cannot move a null this size.
