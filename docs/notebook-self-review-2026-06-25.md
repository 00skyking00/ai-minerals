# ai-minerals self-review proposal (2026-06-25)

Module self-review of the ai-minerals chapters against the shared review
criteria (`portfolio/docs/notebook-review-pass-2026-06-24.md`), run per the
coordinator dispatch `2026-06-25-self-review-chapters-propose-rewrites.md`.

**Status: PROPOSAL. No chapter edited. No deploy.** This is the artifact the
coordinator QAs for voice / consistency / cross-chapter flow before anything is
staged to `/ai-minerals-beta/` and approved by Sky.

Scope reviewed (my lane): `index.qmd` (intro), `regional.qmd`, `placer.qmd`,
`reproductions.qmd`, `cross_region.qmd`, `drill_planning.qmd`, `nome_placer.qmd`.
Deeper notebooks under `portfolio/notebooks/`: lighter pass, flag-only (last
section). All paths below are under `portfolio/`.

Two chapters in the criteria-doc reading order are NOT mine and are not reviewed
here: Bear Cub (Ch1, bearcub lane) and goldbug (Ch3, goldbug lane). Their
index-card text lives in my `index.qmd`; where I touch it I flag any number for
the owning lane to confirm rather than asserting a change.

## How to read this

Each chapter has a one-line verdict, then concrete **BEFORE → AFTER** blocks
grouped by criterion, then **flags** (things I am not proposing to fix here,
with why). AFTER text is written to the voice rules (no banned words, no
em-dashes, no banned idioms). Line numbers are from the current files at review
time.

The two correctness bugs the dispatch named are both confirmed against the data;
they lead the regional section. Three more correctness items I found on my own
(an intro misattribution that inverts a chapter finding, an intro range that
matches nothing, and a wrong grid size) are called out in their chapter sections
and again in **Cross-chapter continuity**.

---

## Verdict summary

| Chapter | Verdict | Must-fix (correctness) | Voice / over-def / level | Flags |
|---|---|---|---|---|
| index.qmd | needs-work | 2 (leak-free misattribution; 7-to-37 range) + the named over-def trim | over-def ×1 | 1 |
| regional.qmd | needs-work | 2 named bugs (AGDB4→NGDB; 56→45) + "eight" count | voice ×2 | 2 |
| placer.qmd | mostly-keep | 1 (292 vs 158 count) | level-of-detail (version density) | 2 |
| reproductions.qmd | mostly-keep | 1 (waterfall caption arithmetic) | clean | 0 |
| cross_region.qmd | mostly-keep | 0 | idiom ×1 | 0 |
| drill_planning.qmd | mostly-keep | 1 (250 m → 500 m) | dash ×2, idiom ×1 | 1 |
| nome_placer.qmd | keep | 0 | process-residue cut ×1 | synthesis deferred per dispatch |

Voice lint (`portfolio/scripts/voice_lint.py`) on the seven chapters: **0 banned
words**, 2 dash hits (both citation page-ranges in `drill_planning.qmd`), 5 soft
idiom warnings (4 are acceptable technical "not just X" usage; 1 real idiom in
`cross_region.qmd`). Detail in **Lint status**.

---

## regional.qmd: the two named bugs (both confirmed against data)

### Correctness

**Bug 1. Arizona geochemistry database (line 297). CONFIRMED.**

AGDB4 is the Alaska Geochemical Database; it is Alaska-only and feeds Tanacross.
Arizona is configured `geochem_source="ngdb"` (`src/ai_minerals/regions/arizona.py:75`),
its fetch pulls NGDB (`scripts/arizona/fetch_all.py:42`), and the Arizona
notebook says NGDB throughout with zero occurrences of "AGDB". The chapter even
contradicts itself: line 66-67 lists "AGDB4 v4 **for Tanacross**" as the
Alaska-specific feed while line 297 attributes AGDB4 to Arizona.

> BEFORE (line 297-299): On Arizona the top features are AGDB4 Cu and Mo
> stream-sediment, porphyry-host intrusive lithology, and distance-to-fault,
> which is the canonical porphyry-Cu recipe.

> AFTER (minimal, required): On Arizona the top features are NGDB Cu and Mo
> stream-sediment, porphyry-host intrusive lithology, and distance-to-fault,
> which is the canonical porphyry-Cu recipe.

**Caveat that affects the AFTER (decision needed).** There is no saved Arizona
SHAP ranking on disk. The only computed SHAP artifact in this chapter is the
Tanacross one (the figure at line 302). The Arizona notebook explicitly defers
the per-feature SHAP writeup (`notebooks/arizona/main.qmd:378-382`). So the
clause "the top features are NGDB Cu and Mo..." asserts a ranking that is not
backed by a committed computation; it reads as a geological inference (a correct
one for porphyry-Cu, but inferred). The bug very likely originated by carrying
the wording over from the adjacent Tanacross SHAP figure, which legitimately
uses AGDB4.

> AFTER (recommended, claim-safe): On Arizona the model leans on NGDB Cu and Mo
> stream-sediment, porphyry-host intrusive lithology, and distance-to-fault, the
> canonical porphyry-Cu recipe.

"leans on" does not assert a saved SHAP ranking the way "the top features are"
does. If Sky wants the strong "top features are" phrasing kept, the right fix is
to generate and commit an Arizona SHAP figure to back it (see Figures below).
The one non-negotiable correction either way: **AGDB4 must become NGDB.**

The California half of the same paragraph ("NGDB arsenic and antimony, plus the
Calaveras Complex and Mariposa Slate") is correct as written.

**Bug 2. Tanacross positive count (line 28). CONFIRMED. Correct number is 45.**

The map caption says 56; the results table (line 37) says 45; `cross_region.qmd`
(line 38) also says 45. 56 is the old raw `is_porphyry` count (v1, includes 11
polymetallic/composite records). 45 is the current `is_porphyry_clean` count
(Cox-Singer Cu-specific filter). Every current metrics file reports 45
(`data/derived/eastak/path2_stage1_metrics.json` and three siblings), and the
table's own numbers tie to 45 arithmetically: top-1% 22.2% = 10/45, top-5%
64% = 29/45 (10/56 and 29/56 do not match).

> BEFORE (line 28): ...red crosses are the 56 porphyry positives from the Alaska
> Resource Data File (ARDF), held out for evaluation.

> AFTER: ...red crosses are the 45 porphyry positives from the Alaska Resource
> Data File (ARDF), held out for evaluation.

**Figure caveat.** `fig_prospectivity_tanacross.png` is dated 2026-05-29 and
predates the clean-label work, so the crosses physically plotted on it are most
likely the 56 old-family positives. Changing the caption to 45 makes the chapter
internally consistent (table + metrics + cross_region all say 45); the thorough
fix is to regenerate the figure against the `is_porphyry_clean` set so the
plotted crosses and the caption both read 45. Flagged under Figures.

**Bug 3 (found here). "eight v3.1 experiments" (line 224). Count is wrong.**

The motherlode v3.1 notebook (`notebooks/motherlode/v3p1_improvements.qmd`)
contains nine experiments: A1-A4 (diagnostics/measurements), B1, B2, C1, C2, D1
(the changes). The chapter says "eight" and then lists only five (B1, B2, C1,
C2, D1). "Eight" matches neither the nine total nor the five listed. The same
sentence also carries a voice issue ("shipped as v3 in April 2026"): "shipped"
as a release verb plus a dated parenthetical (kill-list: process residue).

> BEFORE (line 224-226): Mother Lode shipped as v3 in April 2026 with an AUC of
> 0.85 under spatial-block CV, then iterated through eight v3.1 experiments that
> either improved or tightened the result:

> AFTER: The Mother Lode model reached v3 at an AUC of 0.85 under spatial-block
> CV, then went through a series of v3.1 experiments that either improved or
> tightened the result. The five that changed the result:

(Drops the wrong count, the release verb, and the date. If a count is wanted,
"nine v3.1 experiments (the five that moved the result are below)" is accurate
to the notebook. Coordinator's call.)

### Tone / voice

> BEFORE (line 17-18): Random Forest classification (a tree-ensemble model
> that's the working horse for tabular geoscience data)

> AFTER: Random Forest classification (a tree-ensemble model, the standard
> workhorse for tabular geoscience data)

("working horse" → "workhorse".)

### Flags (regional, not fixing here)

- **SHAP defined twice.** SHAP is glossed in the body (line 234) and re-glossed
  in the figure caption (line 302: "SHAP attributes each prediction back to the
  inputs that drove it"). Minor redundancy. Captions are semi-standalone, so I
  am not proposing the cut, but the caption parenthetical could go since the body
  defines it just above. Low priority.
- **Glossing density.** The Setup and pipeline sections gloss many genuinely
  technical terms (aeromagnetic, Bouguer gravity, magnetic-field derivatives,
  spatial-block CV). These are HIGHLY technical, so the new over-definition rule
  does not require cutting them. Left as-is.

### Figures (regional)

- Tanacross prospectivity map: caption number fix above; recommend regenerating
  so plotted crosses = 45.
- Arizona prospectivity map (line 30): correct and well-captioned (191 positives,
  23.0% top-1%, 23x). Keep.
- Tanacross SHAP (line 302): real artifact, correct. Keep.
- **Gap:** the text asserts Arizona top features but the chapter shows no Arizona
  SHAP figure (only Tanacross). Either generate an Arizona SHAP figure to back
  the claim, or use the softened "leans on" wording above.

---

## index.qmd (intro)

**Verdict: needs-work.** The named over-definition trim, plus two correctness
items that are more serious than cosmetic: one inverts a chapter's central
finding on the front page, one states a range that matches nothing. Per the
dispatch, the intro through-line stays understated and I am not adding a
synthesis spine; the items below are the over-definition trim, voice, and
correctness only.

### Over-definition (the named example)

> BEFORE (line 156-159): Placer modeling (hunting for gold loose in streambeds
> and old gravel beds, rather than locked in solid rock) sits about a decade
> behind lode prospectivity in the published literature.

> AFTER: Placer modeling sits about a decade behind lode prospectivity in the
> published literature.

This is the only common-mining-vocab over-definition in the whole portfolio. A
sweep of all seven chapters for parenthetical glosses of placer / lode / claim /
prospect / assay / ore / vein / drill log returned no other hits (the other
near-matches are deposit-type classifiers like "(orogenic Au)" and coordinates,
which are fine).

### Correctness (these go beyond the dispatch's "over-def + voice + level"
guidance for the intro; flagging for explicit coordinator/Sky sign-off)

**A. The drill-planning card misattributes a number and inverts the finding
(line 233-236).** The card says a *leak-free* prior captures 86% and therefore
"the planner cannot improve on a good prior." In the chapter, the 86% top-25% is
on the MINFILE-informative, *temporally contaminated* prior, the one the chapter
says is "partly an artifact" because "the prior knows the answer"
(`drill_planning.qmd:228-239`). The actual leak-free finding is the opposite:
the Bayesian-updating policies start to beat the static map on the better-sampled
districts (Red Chris 54% vs 31%, KSM 50% vs 43%). So the front page currently
reports the inverse of the chapter's central result.

> BEFORE (line 233-240): On the real BC Golden Triangle data a leak-free
> starting prior already captures 86% of post-2010 copper hits in its top
> quarter, so the planner cannot improve on a good prior; the planning advantage
> shows up against weak priors, not strong ones. Scaled up to the full BCGT grid
> with real BCGS deposit-type priors, four policies tie within experimental noise
> on immediate reward, but SARSOP ends each episode about seven points more
> confident in the true geological hypothesis than greedy does (0.71 vs 0.64).

> AFTER: On the real BC Golden Triangle data the picture is mixed. On a clean
> pre-2010 prior the Bayesian-updating policies start to beat a static
> recommendation on the districts with enough post-2010 hits to measure (at Red
> Chris, 54% of the later copper captured against the static prior's 31%). Where
> the prior is already strong, the static map is hard to beat. Scaled up to the
> full BCGT grid with real BCGS deposit-type priors, the policies tie within
> experimental noise on immediate discovery, but SARSOP ends each episode about
> seven points more confident in the true geological hypothesis than greedy does
> (0.71 vs 0.64).

(The second half of the card, D.1.D and the 0.71 vs 0.64, is correct and kept.)

**B. The cross-region card's range matches nothing (line 211-214).** The card
says the tree-vs-DEEP-SEAM gap "widens to 7 to 37 percentage points." The
chapter says 9 to 28 (`cross_region.qmd:24`), which is what its own scorecard
table supports (per-region gaps of 8.9, 14.1, 28.4). The deeper notebook
`four_cell.qmd:321` says a third thing, "15 to 37," for its 2x2 controlled
experiment. The index's "7 to 37" matches neither the chapter table nor the
notebook.

> BEFORE (line 211-214): On three other regions covering porphyry-Cu and
> sediment-hosted Zn-Pb under spatially-blocked cross-validation, the gap widens
> to 7 to 37 percentage points in favor of tree-based methods.

> AFTER: On three other regions covering porphyry-Cu and sediment-hosted Zn-Pb
> under spatially-blocked cross-validation, the gap widens to 9 to 28 percentage
> points in favor of tree-based methods.

(Aligns the index to the chapter's table. The notebook's "15 to 37" is a
separate experiment; flagged in the deeper-notebook section to confirm it is
labeled as distinct.)

### Flags (index)

- **Bear Cub / goldbug card numbers** ("last drilled in the 1910s-30s",
  AUC 0.733 / 0.741, "more than 20 times") describe other lanes' work. The
  0.733 / 0.741 match `nome_placer.qmd` exactly, so those are consistent. Bear
  Cub specifics belong to the bearcub lane to confirm; I am not changing them.
- **Goldbug card density** (line 123-131) carries two AUC numbers plus the
  buffer detail. Slightly number-heavy for a Tier-1 card, but it is the
  established per-card style and the dispatch keeps the intro understated rather
  than restructured. Left as-is.

---

## placer.qmd

**Verdict: mostly-keep.** Internally consistent on its metrics (I traced the
anchor table, the base-learner AUCs, the v3.7.0 PR-AUC and coverage numbers, and
they reconcile). Two items: one internal count contradiction, and a
level-of-detail issue (version-number and dated-parenthetical density).

### Correctness

**The v3 Tertiary positive count is stated two ways.** Line 135 says "Final v3
positive counts: 292 Tertiary cells and 437 Quaternary cells"; line 256 says
"The v3 Tertiary label set was 158 cells." The v3.6 narrative then builds on 158
("grows from 158 binary cells to 2,709 weighted cells"). Most likely 158 = pit-
polygon centroids and 292 = total v3 Tertiary including MRDS-reclassified points,
but the prose never reconciles them, and a reader hits the contradiction. I am
not proposing a number here because I cannot tell from the prose which is the
operative count; recommend the author state which is the v3 Tertiary positive
count and make the two lines agree (and confirm the v3.6 "from 158" baseline is
the right one).

### Level of detail

The chapter carries a dense version ladder (v2, v3, v3.5, v3.6, v3.7.0,
v3.7.0.1) and several dated parentheticals that read as process residue to an
external reader. Concrete cuts:

> BEFORE (line 152-153): The Quaternary classifier finished its full pipeline on
> 2026-06-05 with a stacking out-of-fold AUC of 0.832

> AFTER: The Quaternary classifier reaches a stacking out-of-fold AUC of 0.832

> BEFORE (line 609): rerun `scripts/...v3_ablation_no_pit_proximity.py` against
> the v3.6 stacking ensemble, not just the v3 RF + XGB pair from the June 3
> ablation.

> AFTER: rerun `scripts/...v3_ablation_no_pit_proximity.py` against the v3.6
> stacking ensemble, not only the v3 RF + XGB pair from the earlier ablation.

> BEFORE (line 498-499): The goldbug live tool re-banded against the v3.7.0
> raster on the day it shipped (2026-06-11) and ran every gap parcel...

> AFTER: The goldbug live tool re-banded against the v3.7.0 raster and ran every
> gap parcel...

**Broader flag (not fixing unilaterally).** A full de-versioning pass (folding
v2/v3/v3.5/v3.6 into "an earlier version did X, the current model does Y") would
help an external reader, but the version progression is also part of the
chapter's leakage-audit story (v2 leak → v3 buffer → v3.6 rasterization →
v3.7.0 expansion), so collapsing it is an editorial judgment for the coordinator
to scope rather than something I should cut silently.

### Flags (placer)

- **Title vs index naming.** The chapter title is "(northern Sierra)"; the index
  card and the v3.7.0 content say full California Mother Lode. Continuity item,
  carried in the Cross-chapter section below.
- **"ship gate" / "right thing to ship"** (lines 161, 182, 318). Internal
  release-criterion jargon in user-facing prose. Fine to keep as
  product-shipping usage (not the banned self-referential "the chapter ships
  X"), but "release threshold" / "the 30% top-quintile bar" would read cleaner.
  Low priority; coordinator's call.

### Figures (placer)

All figures are data-backed and captioned to the right message. The headline
v3.6-vs-v3.7.0 Quaternary comparison (line 415) is the right lead. No changes.

---

## reproductions.qmd

**Verdict: mostly-keep.** Clean voice, clear arc, strong chapter. One figure-
caption arithmetic item.

### Correctness / figures

**Lawley waterfall caption (line 24).** The caption frames the waterfall as
starting from the published 0.983: "Lawley 2022's published continental Zn-Pb
AUC of 0.983 comes from. Removing a label leak takes it to 0.972." But 0.983 to
0.972 is a 1.1 pp drop, while the audit table (line 50) states the label leak is
2.4 pp. The 0.972 is consistent with starting from the *reproduced* 0.9965
(0.9965 − 0.024 = 0.9725 ≈ 0.972), not the published 0.983. So the caption's
start value and its first step disagree with the audit table by about 1.3 pp.

I could not locate the chart's source values quickly, so this needs a look at the
figure's first bar to finalize. If the first bar is 0.9965 (which the math
implies), the caption should say so:

> AFTER (pending confirmation the first bar is 0.9965): Where Lawley 2022's
> continental Zn-Pb AUC comes apart under proper validation. The reproduction
> scores 0.9965 (the published number is 0.983, reproduced within 0.014).
> Removing a label leak takes it to 0.972; switching from 1-D to 2-D
> spatial-block cross-validation takes it to 0.868; training on Australia +
> Canada and testing on the US takes it to 0.709.

The externally-quoted headline numbers (published 0.983, corrected range
0.71-0.87, reverse transfer 0.557) are consistent across index, this chapter,
and the waterfall endpoint. Only the caption's leak step is off.

### Figures / other

No other changes. The chapter is the one I would point a reader at first, and it
reads that way.

---

## cross_region.qmd

**Verdict: mostly-keep.** Internally consistent (the 9-to-28 range matches the
scorecard table; the 14 carbonatite positives, 7 Curnamona positives reconcile
across chapters). One banned idiom.

### Tone / idiom

> BEFORE (line 110-111): The architecture isn't broken; it's matched to the
> wrong problem shape for general regional MPM.

> AFTER: The architecture does real work; it is matched to the wrong problem
> shape for general regional MPM.

("isn't X; it's Y" is the banned contrastive-reframe idiom. The replacement
keeps the meaning and loses the tell.)

### Flags (cross_region)

- The index card range ("7 to 37") disagrees with this chapter ("9 to 28"). The
  fix lands in `index.qmd` (above); this chapter's number is the correct one.

---

## drill_planning.qmd

**Verdict: mostly-keep.** Long and didactic by design (it explains POMDP, SARSOP,
alpha vectors, belief updates), which is appropriate: every glossed term is
highly technical, so the new over-definition rule does not touch it. Metrics are
internally consistent across the body, captions, Findings, and the index card (I
traced the D.1 and D.1.D reward and truth-belief numbers; they reconcile). One
correctness item, two citation dashes, one optional idiom.

### Correctness

**Grid size (line 13).** The opening defines the prospectivity map as "per-250 m
grid cell." The BCGT / porphyry work is on a 500 m grid everywhere else in this
chapter (line 182 "500 m BCGT grid", line 549 "500 m per cell") and in
`regional.qmd:71` ("500-m grid"). 250 m is the placer grid, not the porphyry
grid.

> BEFORE (line 11-14): A prospectivity map is a per-cell probability surface, in
> this case "per-250 m grid cell, the model's estimate of how likely an unmined
> porphyry-copper deposit is to be there."

> AFTER: A prospectivity map is a per-cell probability surface, in this case
> "per-500 m grid cell, the model's estimate of how likely an unmined
> porphyry-copper deposit is to be there."

### Tone / dashes

Two citation page-ranges use en-dashes (the only dash hits in the seven
chapters). Sky's rule flags en-dashes as well as em-dashes; converting to
hyphens makes the lint green and matches the citation style elsewhere.

> BEFORE (line 1153): *Geosci. Model Dev.* 16:289–307.
> AFTER: *Geosci. Model Dev.* 16:289-307.

> BEFORE (line 1168): *Artificial Intelligence* 101:99–134.
> AFTER: *Artificial Intelligence* 101:99-134.

### Tone / idiom (optional)

> BEFORE (line 158-160): POMCP weighs not just "what is the next hole worth in
> immediate reward" but "what does the next hole's observation buy me for the
> holes after that."

> AFTER: POMCP weighs the next hole's immediate reward against what its
> observation buys for the holes after that.

(Optional. The other two "not just" lint hits, lines 632 and 1137, are ordinary
"more than merely X" technical usage and read fine; no change.)

### Flags (drill_planning)

- **"Headline test" used twice for two grids.** Line 47 calls the 32x32
  reproduction "the 2024 paper's headline experiment"; line 116 calls the 30x30
  stress test "the 2024 paper's headline test." They are different experiments.
  Minor; line 116 could read "matches the 2024 paper's conditions" (which its
  caption already says). Low priority.

### Figures (drill_planning)

Many figures, all data-backed and captioned to the right message. No changes.

---

## nome_placer.qmd (Cape Nome, Ch8)

**Verdict: keep.** Best voice in the set. Per the dispatch, the synthesis
tightening and the dark-data elevation are a SEPARATE coordinator round, so I am
not rewriting the synthesis structure here. Numbers all reconcile with the
appendix tables (0.444 / 0.733 / 0.741 placer; 0.802 / 0.620 / 0.582 lode) and
with the index goldbug card (0.733 / 0.741). One clean cut.

### Tone / process residue

> BEFORE (line 207-209): You can open the graph three ways and poke at it
> yourself: a tabular SQL browser, a spatial map of claims and occurrences, and
> a linked per-entity wiki where each claim, document, and operator has its own
> page. (The coordinator is assembling the figure that links the three viewers;
> the publish task to give them a reader-facing home is tracked.) On the goldbug
> side, the claim detail pages now carry the attached graph records...

> AFTER: You can open the graph three ways and poke at it yourself: a tabular SQL
> browser, a spatial map of claims and occurrences, and a linked per-entity wiki
> where each claim, document, and operator has its own page. On the goldbug side,
> the claim detail pages now carry the attached graph records...

(The parenthetical about coordinator tasks and a tracked publish task is internal
project-management leakage in published prose. Clean cut, not a synthesis
change.)

### Flags (nome_placer, deferred per dispatch)

- **Synthesis structure and dark-data elevation:** deferred to the separate
  coordinator round as instructed. The synthesis ("doing for a district what we
  did for one claim") reads coherently as-is.
- **Minor:** line 167-170 attributes the 0.805 jump to "adding the pathfinder
  elements," while the appendix distinguishes "full incl. stream-sed geochem"
  (0.805) from "pathfinders only" (0.691). The body is describing the full model
  with geochem added, so it is not wrong, but a precise reader might expect the
  0.691 number. Low priority; leave unless the coordinator wants the distinction
  surfaced.

---

## Cross-chapter continuity (the items that span chapters)

1. **DEEP-SEAM gap stated three ways.** index "7 to 37" / cross_region "9 to 28"
   / four_cell notebook "15 to 37". Fix index to 9-28 (chapter-table supported);
   confirm the notebook's 15-37 is a distinct, labeled experiment (the 2x2
   controlled design), not a third statement of the same number.

2. **Drill-planning leak-free vs contaminated prior.** The index card's 86%
   "leak-free" claim is the contaminated-prior number and inverts the chapter's
   finding. Fix in index (above). This is the highest-impact item because it is
   on the front page and reverses the chapter's central result.

3. **Placer chapter naming.** Chapter title says "(northern Sierra)"; index card
   and v3.7.0 content say full California Mother Lode; `regional.qmd:307` and
   `regional.qmd:266-267` also call it "the northern Sierra placer model." The
   Quaternary model is now Mother-Lode-wide; the Tertiary model is genuinely
   northern-Sierra-only. Recommend standardizing on "California Mother Lode" (to
   match the index and the expanded scope) and updating the two regional.qmd
   cross-references to match. If the coordinator prefers to keep "northern Sierra"
   for the Tertiary emphasis, then the *index card* should change instead so the
   two agree. Either way, the title and the index card must not disagree.

   > Proposed (placer.qmd line 2):
   > BEFORE: title: "Chapter 4: Placer-Au prospectivity (northern Sierra)"
   > AFTER: title: "Chapter 4: Placer-Au prospectivity (California Mother Lode)"

4. **Tanacross 45 everywhere.** With the regional caption fixed to 45, the count
   is consistent across regional table, regional caption, and cross_region table.
   No other chapter quotes a Tanacross positive count.

---

## Deeper notebooks (flag-only, lighter pass)

Voice lint across the deeper notebooks that map to my chapters surfaced:

- **2 banned words:** `notebooks/drill_planning/index.qmd:89` and
  `notebooks/prospectus/motherlode.qmd:125` both contain "defensible". These
  should be rewritten (not word-swapped) when the deeper-notebook pass runs.
- **Many em/en-dash hits**, concentrated in `notebooks/cross_region/four_cell.qmd`
  (12), `notebooks/prospectus/us_carbonatite_ree.qmd` (8),
  `notebooks/prospectus/motherlode.qmd` (9),
  `notebooks/drill_planning/index.qmd` (3), and one in
  `deep_seam_cross_region.qmd`. A sample read suggests a mix of clause em-dashes
  (real fixes) and numeric/page en-dashes (range convention). Worth a dash sweep
  in the deeper-notebook pass.
- **Content flag:** `four_cell.qmd:321` states the tree-vs-DEEP-SEAM gap as "15
  to 37 percentage points," a third value distinct from the chapter (9-28) and
  index (7-37). Confirm it is the 2x2 controlled experiment's gap and labeled as
  such, so the three documents do not read as contradicting each other.

I did not do BEFORE→AFTER on deeper notebooks per the dispatch (flag-only). The
above is the candidate work-list for that pass.

---

## Lint status (seven chapters)

`python3 portfolio/scripts/voice_lint.py` on the seven chapters:

- **Banned words: 0.**
- **Dash hits: 2**, both citation page-ranges in `drill_planning.qmd`
  (1153, 1168). Fixes proposed above.
- **Idiom warnings: 5.** `placer.qmd:609-610` and `drill_planning.qmd:632,1137`
  are ordinary "not just / not only X" technical usage and read fine.
  `drill_planning.qmd:158` is borderline (optional rewrite proposed).
  `cross_region.qmd` carries the one real idiom ("isn't broken; it's...",
  proposed above; the lint missed it because its regex wants "isn't a/an/just").

After the proposed edits, the chapters are lint-clean (0 banned, 0 dash) and the
one real idiom is removed.

---

## Decisions for the coordinator / Sky

1. **Arizona SHAP claim (regional Bug 1):** swap AGDB4→NGDB and soften to "leans
   on" (recommended, since no Arizona SHAP artifact exists), or keep "the top
   features are" and generate an Arizona SHAP figure to back it?
2. **Tanacross figure (regional Bug 2):** caption-only fix to 45, or also
   regenerate `fig_prospectivity_tanacross.png` so the plotted crosses are the 45
   cleaned positives?
3. **Intro correctness edits (index A and B):** the dispatch limited intro edits
   to over-definition + voice + level-of-detail. Items A (leak-free
   misattribution) and B (7-to-37 range) are correctness fixes. Confirm they are
   in scope for this round (I recommend yes; A in particular is a front-page
   statement that reverses a chapter finding).
4. **Placer naming:** standardize on "California Mother Lode" (recommended) or
   keep "northern Sierra" and change the index card instead?
5. **Placer version density:** apply the targeted dated-parenthetical cuts now;
   scope a fuller de-versioning pass separately, or leave the version ladder as
   the leakage-audit narrative?
6. **Lawley waterfall caption:** confirm the chart's first bar (0.9965 vs 0.983)
   so the caption's leak step matches the 2.4 pp audit number.

Nothing in this proposal has been applied to the chapters. On approval, the
edits are mechanical and I can apply them through the staged pipeline.
