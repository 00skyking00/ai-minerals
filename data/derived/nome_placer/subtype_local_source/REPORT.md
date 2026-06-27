# Typing the Nome placer positives: no resolvable local-source signal in the abrasion-platform subset

Round-4B addendum. Verdict: **a final null, set by sample size.** Typing the 65
placer positives into Hudson's two coastal populations leaves only 5 abrasion-
platform and 3 strandline-beach occurrences. The two coastal types are
indistinguishable on every local-source covariate tested, and at n = 5 vs 3 no
rank test or per-subset cross-validation can resolve a difference even if one
exists. The pooled distance-to-lode null (round 3) reproduces. The one direction
consistent with the hypothesis (abrasion-platform sits nearer the lode-belt axis
than strandline) is present but not significant.

## 1. The type split

Typing is from the occurrence `name` field, which carries the beach name
("Monroeville Beach", "Second Beach", ...), cross-checked against position. The
scheme follows Hudson (2006) lines 50-52, after Metcalfe & Tuck (1942):

| type | n | what it is |
|---|---|---|
| abrasion_platform | 5 | Monroeville, Intermediate, Center beaches + Inner/Outer Submarine. Coarse gold, pyrite + arsenopyrite, on a wave-cut bedrock platform. The local-source coastal type. |
| strandline_beach | 3 | Second, Third beaches (+ present/First by process). Garnet + black-sand rich, finer gold. The winnowed-drift coastal type. |
| upland_residual | 52 | Named creeks, gulches, and benches (Anvil, Glacier, Rock, Dexter, ...). Hudson's residual-alluvial-placer belt; not a coastal beach. |
| broad_ambiguous | 5 | District-wide or composite records (Nome placer field, Nome Coastal Plain, Nome Mining District, Fourth Beach, the Center-plus-creeks composite). |

Mapped to the brief's three-way scheme, **abrasion-platform = 5, strandline-beach
= 3, ambiguous (upland_residual + broad) = 57 (88%)**. The high ambiguous
fraction is the first result: the knowledge-graph placer positives are dominated
by upland stream-gulch placers, not coastal beaches. Only 8 of 65 are cleanly
typed coastal beaches, and that count, not the choice of covariate, is what caps
the rest of this analysis.

38 of 65 positives sit within 4 miles of Newton Peak (median 3.6 mi), consistent
with Hudson's "most production within 4 miles of Newton Peak."

## 2. The local-source covariates

All four are leak-free and defined at every model cell.

- **dist_to_36a_lode**: nearest ARDF 36a (low-sulfide Au-quartz vein) occurrence
  (63 points).
- **dist_to_contact**: nearest bedrock(Nome Group)/Quaternary contact, the
  abrasion-platform inland edge (SIM 3131 coastal-plain Qs inland boundary; the
  same line the round-4B bedrock-contact run used, here defined district-wide).
- **dist_to_newton_pt**: straight-line distance to Newton Peak (GNIS 1406988,
  64.5589 N, 165.3187 W). The direct reading of Hudson's 4-mile production radius.
- **dist_to_newton_axis**: perpendicular distance to the lode-belt axis. The
  axis is the principal direction of the 36a lode cloud through Newton Peak. That
  direction comes out **E-W (azimuth 88 deg)**, the along-strike spread of the
  Nome lode belt along the foothills. Hudson's "trends southeast onto the coastal
  plain" describes the cross-strike projection toward Norton Sound, a direction
  the proposal states qualitatively and Fig 2 shows as a slide; it is not pinned
  to a numeric azimuth, so the data-driven strike axis is used and labelled as
  such. The point-distance covariate above carries the proximity notion without
  this assumption.

## 3. Test 1: distributional contrast (the powered test)

Median distance (m) to each source, by group, with the one-sided Mann-Whitney p
that the group sits closer than background and Cliff's delta vs background
(negative = closer than background).

| covariate | abrasion (n=5) | strandline (n=3) | upland (n=52) | background |
|---|---|---|---|---|
| dist_to_36a_lode | 5597 (p=0.70, d=+0.14) | 7012 (p=0.70, d=+0.18) | **1446 (p<0.001, d=-0.48)** | 4652 |
| dist_to_contact | 4588 (p=0.87, d=+0.30) | 6212 (p=0.77, d=+0.26) | **1513 (p<0.001, d=-0.30)** | 2170 |
| dist_to_newton_axis | 3673 (p=0.11, d=-0.32) | 5402 (p=0.53, d=+0.03) | 4173 (p=0.014, d=-0.18) | 5328 |
| dist_to_newton_pt | 6989 (p=0.43, d=-0.05) | 8513 (p=0.47, d=-0.03) | **5944 (p<0.001, d=-0.40)** | 7857 |

The direct abrasion-vs-strandline contrast (the actual question) is
indistinguishable on all four covariates: Cliff's delta |d| <= 0.33, every
bootstrap CI on the median difference spans zero, every rank p >= 0.29. At n = 5
vs 3 the smallest two-sided p a rank test can return is about 0.10, so this is a
limit of the sample, not evidence that the two types are the same.

Two things do show up:

- **Upland residual placers sit close to lodes (median 1.4 km), the bedrock
  contact (1.5 km), and Newton Peak (5.9 km)**, all far inside background and
  significant. This is expected and partly circular: a stream-gulch placer is the
  eroded lode directly below it (e.g. "Cooper Gulch (placer and M. Charles lode
  occurrence)" is one record for both). It confirms the local-source mechanism is
  real for the upland placers, but it is not news and, as Test 2 shows, it is
  already captured by terrain.
- **Abrasion-platform trends nearer the lode-belt axis than background** (median
  3.7 km, d=-0.32, p=0.11) while strandline does not (d=+0.03, p=0.53). This is
  the one direction consistent with the brief's hypothesis. It is not significant,
  and the abrasion-vs-strandline difference on this covariate (d=-0.33, p=0.29)
  cannot be resolved at this n.

## 4. Test 2: per-subset leak-guarded CV (the literal deliverable)

Marginal out-of-fold AUC delta from adding each covariate to the geomorph+terrain
base, on the F1 leak-guarded scheme (contiguous folds, 1 km dead zone,
RandomForest(300), 2000-resample paired bootstrap). Read on each subtype's
positives vs background.

| covariate | pooled (65) | upland (52) | abrasion (5) | strandline (3) |
|---|---|---|---|---|
| dist_to_36a_lode | +0.026 [-0.020,+0.071] | +0.001 | +0.084 [+0.004,+0.164] | +0.080 [-0.080,+0.337] |
| dist_to_contact | -0.021 [-0.053,+0.012] | -0.033 [-0.066,-0.001] | +0.027 [-0.144,+0.267] | +0.034 [-0.082,+0.235] |
| dist_to_newton_axis | -0.022 [-0.055,+0.010] | -0.027 [-0.065,+0.011] | -0.005 [-0.086,+0.088] | +0.071 [-0.005,+0.159] |
| dist_to_newton_pt | +0.002 [-0.041,+0.043] | -0.002 | -0.114 [-0.381,+0.139] | +0.005 |

- **Pooled reproduces the round-3 null**: distance-to-lode adds +0.026 with a CI
  spanning zero. None of the four covariates moves the pooled model.
- **Upland adds nothing** despite sitting closest to every source: the geomorph
  base already separates upland placers from background, so the distance features
  are redundant (deltas at or below zero).
- **The abrasion and strandline columns are not trustworthy at n = 5 and 3.** The
  one CI that clears zero (dist_to_36a_lode on abrasion, +0.084 [+0.004,+0.164])
  points the opposite way from Test 1, where abrasion-platform is farther from
  lodes than background, not closer. A 5-positive out-of-fold AUC under spatial
  folds is graded on a handful of points in one or two folds; the bootstrap
  resamples those few. Read it as noise, and read Test 1's direction-aware
  contrast as the better evidence.

## 5. Verdict on the hypothesis

The brief asked whether the pooled distance-to-lode null masks a real local-
source signal in the abrasion-platform subtype. **It does not, as far as this
positive set can show.** Both coastal types sit far from the lodes and the bedrock
contact (farther than background, because they are seaward on the coastal plain);
they are indistinguishable from each other on every covariate; and the per-subtype
CV is too small to resolve a marginal effect. The only nudge toward the
hypothesis is abrasion-platform lying nearer the lode-belt axis than strandline,
and that does not reach significance.

The cleaner finding underneath is structural: the local-source signal is
concentrated in the 52 upland residual placers, where it is both partly circular
(placer co-located with its lode) and already encoded by terrain, which is why
the pooled model gains nothing from a distance-to-lode feature. Typing the
coastal beaches finer does not change that, because there are too few of them.

## 6. What this does not prove

- **n caps everything.** 5 abrasion-platform and 3 strandline positives cannot
  support a per-subtype model or a significant rank test. A different positive set
  (the full Metcalfe & Tuck paystreak inventory, or the NovaGold drill database
  Hudson cites) could carry enough typed coastal beaches to retest this; the KG
  set cannot.
- The abrasion-platform / strandline split was read from occurrence names, not
  from PP759-A Fig 2 paystreak geometry. The names agree with Hudson's named
  deposits, but a positive that the KG names generically ("Nome placer field")
  could belong to either type and was left ambiguous.
- The Newton belt axis is the data-driven lode strike (E-W), not Hudson's
  qualitative SE projection; the point-distance covariate is the assumption-free
  version and shows no coastal-subtype signal either.
- Distance-to-lode at an upland placer is near zero by co-location, so its
  significance there is not independent evidence of a transport-distance control.
