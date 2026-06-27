# Tuck-powered placer retest: the cadastre cannot place the abrasion-platform paystreaks, so the subtype is still ungradeable

Round-4B powered retest, using fossick's Tuck-1942 extraction. Verdict: **the
test still cannot grade the local-source hypothesis, and the reason is the join,
not the model.** Of 72 typed coastal areas, only 9 carry an authoritative
MS-survey coordinate, and 8 of those 9 are true-beach (strandline) claims. The
abrasion-platform paystreaks (the submarine, offshore, and buried-strandline
deposits that the hypothesis names as the local-source type) were never patented
as individual mineral surveys, so the cadastre cannot place them. The
abrasion-platform positive count therefore stays at 5, exactly where round 4B left
it, and the subtype is as ungradeable as before. The added true-beach positives do
move the strandline arm, but in a direction set by which claims hold patents, not
by deposit genesis. The genuine test needs the pinpoint coordinates for the 30
un-joined abrasion-platform features.

## 1. The join is precision-bound and asymmetric

fossick OCR'd Tuck 1942 into 181 typed `TuckPlacerArea` records (41 true-beach, 31
abrasion-platform, 87 upland-residual, 22 unknown) carrying 315 distinct claim
names. None of the records carry an MS number directly, so the join key is the
claim name, matched against the 372 patented mineral surveys in goldbug's
published layer (340 named) and the bearcub curated crosswalk. Matching is
deliberately precision-biased: a false coordinate corrupts the per-subtype CV
worse than a miss does, so only curated crosswalk links and tightly-clustered
multi-claim agreements are accepted, and everything else is left for pinpoint. (An
early permissive pass placed "Second Beach" on the upland claim "No. 2 Bench
Second Tier" and "Paystreak" on a generic "Paystreak Bench"; those are the false
matches the trust gate now rejects.)

| type | areas | MS-joined | pinpoint-needed |
|---|---|---|---|
| true-beach (strandline) | 41 | 8 | 33 |
| abrasion-platform | 31 | **1** | 30 |
| upland-residual | 87 | 5 | 82 |
| unknown | 22 | 1 | 21 |
| **coastal total** | **72** | **9** | **63** |

The asymmetry is the result. Eight true-beach areas join because the Third Beach
claims (Bear Cub MS 1178, Jupiter 1217, Golden Bull 1209, Happy New Years 1113,
No. 9 Otter 327, North Pole 1232, and the Wild Goose ground) were patented and sit
in the cadastre. One abrasion-platform area joins, and it shares the Golden Bull
claim with a true-beach area 130 m away, so de-duplication collapses it. The 30
un-joined abrasion-platform features are the submarine and offshore paystreaks,
Monroeville / Intermediate / Present / Center beaches, and Nome River bedrock
benches: the deposits Tuck describes from drill holes and bedrock cuts, not from
patented surface claims. The cadastre has no polygon for them.

## 2. What the added positives are

After dropping joins within 250 m of an existing KG positive and collapsing areas
that resolve to the same claim, the retest adds **6 distinct true-beach positives
and 0 abrasion-platform positives**:

| subtype | baseline (KG) | powered (KG + Tuck) |
|---|---|---|
| abrasion_platform | 5 | 5 |
| strandline_beach (true-beach) | 3 | **9** |
| upland_residual | 52 | 52 |

The baseline arm reproduces the round-4B numbers to the digit (pooled
distance-to-lode marginal AUC delta +0.0258, abrasion +0.0844), so the powered run
is a clean superset and every change below is attributable to the 6 added
true-beach positives. The 9 strandline positives span a 10 by 8 km box along the
Third Beach line (7 of 36 pairs within 1.5 km), so they are not one degenerate
cluster, but they do share a setting: the foot of the upland, near the
residual-and-lode belt.

## 3. The test: a coastal signal appears, carried by true-beach, not abrasion

### Test 2, per-subset leak-guarded CV (marginal AUC delta, 95% bootstrap CI)

| covariate | pooled (71) | coastal all (14) | abrasion (5) | strandline (9) |
|---|---|---|---|---|
| dist_to_36a_lode | +0.032 [-0.000, +0.066] | **+0.087 [+0.002, +0.169]** | +0.117 [-0.084, +0.284] | +0.070 [-0.017, +0.164] |
| dist_to_contact | +0.006 [-0.029, +0.037] | +0.048 [-0.019, +0.113] | +0.012 [-0.132, +0.137] | +0.068 [-0.005, +0.152] |
| dist_to_newton_pt | -0.021 [-0.053, +0.009] | -0.005 [-0.074, +0.059] | -0.117 [-0.219, -0.020] | +0.057 [+0.004, +0.127] |
| dist_to_newton_axis | +0.003 [-0.041, +0.042] | **+0.125 [+0.056, +0.185]** | +0.139 [-0.004, +0.260] | **+0.116 [+0.034, +0.191]** |

The pooled model still gains nothing (every pooled CI spans zero, as in round 3).
But the 14-positive coastal set now shows a distance-to-belt-axis gain of +0.125
with a CI clear of zero, and a distance-to-lode gain of +0.087, also clear of
zero. Read at face value that is a coastal local-source signal the pooled model
averages away.

The signal is carried by the strandline (true-beach) arm. Its distance-to-axis
delta is +0.116 [+0.034, +0.191] and its distance-to-Newton-Peak delta is +0.057
[+0.004, +0.127], both clear of zero at n = 9. The abrasion arm is still n = 5 and
still untrustworthy: its two CIs that exclude zero point in opposite directions
(distance-to-lode +0.117 wide-positive, distance-to-Newton-Peak -0.117
narrow-negative), the signature of a five-point out-of-fold AUC graded on one or
two folds, not a coherent effect.

### Test 1, distributional contrast (the direction-aware read)

The added true-beach positives move the strandline medians sharply toward every
local source:

| covariate (median m) | abrasion (n=5) | strandline 3 -> 9 | background |
|---|---|---|---|
| dist_to_36a_lode | 5597 | 7012 -> **2718** (cliffs -0.26) | 4652 |
| dist_to_contact | 4588 | 6212 -> **1793** | 2170 |
| dist_to_newton_pt | 6989 | 8513 -> **4284** (cliffs -0.50, p=0.005) | 7857 |

This is where the result turns against the hypothesis. The hypothesis holds that
abrasion-platform is the local-source type and should sit closest to the lode. In
the powered data the newly-placed true-beach claims sit closer to the lode, the
bedrock contact, and Newton Peak than the abrasion-platform positives do. The
direct abrasion-minus-strandline Cliff's delta is +0.6 on distance-to-lode, +0.6
on distance-to-contact, and +0.69 on distance-to-Newton-Peak (positive meaning
abrasion is farther from the source). The only covariate that leans the
hypothesis's way is distance-to-belt-axis (abrasion delta -0.33), and it does not
reach significance.

## 4. Verdict on the coordinator's question

*Does distance-to-lode predict the abrasion-platform subtype while staying null
for true-beach, now that the test has power?*

**No, and the test still does not have the power to grade it.** Two things are
true at once and they pull apart cleanly:

1. The abrasion-platform subtype gained no positives, because its paystreaks are
   unpatented. Its CV column is the same n = 5 noise as round 4B. The pooled null
   is not masking an abrasion-platform local-source signal that this join can
   reveal, because the join cannot place a single new abrasion-platform deposit.

2. The coastal signal that does appear (distance-to-axis +0.125, distance-to-lode
   +0.087 on the 14-positive coastal set) is carried by true-beach claims, and the
   abrasion-minus-true-beach contrast runs the wrong way for the hypothesis. Both
   the signal and its direction track which claims hold patents: the joinable
   true-beach claims are the inland Third Beach claims near the residual-and-lode
   belt, while the abrasion-platform deposits that would test the local-source
   idea are the seaward submarine and offshore paystreaks that never joined. The
   contrast measures the cadastre's patent geography, not deposit genesis.

So the powered retest sharpens round 4B's null rather than overturning it. The
local-source proximity that is real and significant remains the one round 4B
already found, in the 52 upland-residual placers (distance-to-lode median 1.4 km,
p < 0.001), where it is partly circular (a stream-gulch placer sits on its own
lode) and already encoded by terrain. fossick's Anvil Creek datum (a 170 oz nugget
with quartz and schist still attached, Tuck: "gold must be from local sources")
anchors that upland mechanism physically, but Anvil Creek is upland, not coastal,
so it does not bear on the abrasion-platform question.

## 5. What this does not prove, and the unblock

- **The hypothesis is untested, not refuted.** Nothing here shows abrasion-platform
  gold is *not* locally sourced. It shows the cadastre cannot place the
  abrasion-platform deposits, so the per-subtype test cannot be run. The 30
  abrasion-platform features in `tuck_coastal_pinpoint_needed.csv` are the
  unblock: placed by goldbug's pinpoint tool or the coordinator's tile-composite,
  they would take the abrasion arm from n = 5 to a number that can carry the test.
- **The coastal signal is real but not type-discriminating at this n.** With 14
  coastal positives and a patent-biased sample, "the gold here is locally sourced"
  and "these claims sit near the hills" are not separable. The signal does not
  isolate to the abrasion-platform type, which was the testable claim.
- **The n = 5 abrasion CV column is noise.** Its two zero-excluding CIs disagree on
  sign across covariates; neither is evidence.
- **The join is precision-first by design.** Recall on the coastal set is low (9 of
  72) on purpose. A looser matcher would place more areas at the cost of false
  coordinates that would corrupt the very contrast the retest exists to measure.
