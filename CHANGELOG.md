# Changelog

All substantive changes to the ai-minerals portfolio. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are when work landed on `main`; the live site usually goes out
the same day.

## 2026-06-14 — Portfolio polish, Iterations 4-11

Light-asides pass across the remaining ai-minerals chapters,
two sibling-repo HANDOFFs, and one stale-notebook cleanup.

- **Iteration 4 (`regional.qmd`).** Heading renamed to
  "Regional mineral-prospectivity pipelines." Inline definitions
  for USGS, NRCan, ARDF, MRDS, NGDB, RGS, MINFILE, CRS, CONUS,
  Cox-Singer, AUC, out-of-fold scoring, KS / Mann-Whitney tests,
  SHAP, PU learning, RBF interpolator. Magnetic-field derivatives
  (1VD, HGM, analytic signal, tilt) spelled out.
- **Iteration 5 (`reproductions.qmd`).** Inline definitions for
  MVT, H3, AUC, GBM, REE, Pang 2019 deviation network,
  leave-one-out cross-validation. Audit-table prose now gives the
  plain-language reason each spatial-block scheme leaks.
  Methodology checklist item 1 expands 0-D / 1-D / 2-D / LOO into
  operational descriptions.
- **Iteration 6 (`cross_region.qmd`).** Inline definitions for
  REE, DevNet, top-1% capture, ILR, PCA, GLCM, MVT, MPM.
  Scorecard table footnotes now define leave-one-out and
  spatial-block out-of-fold in plain language.
- **Iteration 7 (`placer.qmd`).** Light-asides pass on the chapter
  with the heaviest jargon load (12 figures, 6.6k words). New
  "Reading the metric names" primer at the end of Setup defines
  AUC, AUC-PA, PR-AUC, OOF, SHAP for standalone readers. Inline
  definitions for REM, GMI/geomorphon, TPI, Hawkes catchment, NURE,
  MRDS, CGS PTYPE, ECE, ksn, spi_band, NHD HR, PU learning,
  isotonic calibration.
- **Iteration 8 (`drill_planning.qmd`).** Banned-word fixes:
  "more honest particle filter" replaced with "methodologically
  tighter particle filter"; "regime robustness sweep" replaced
  with "regime-sensitivity sweep" ("robust" and morphological
  variants are banned in user-facing prose). Inline definitions
  for BCGT, BCGS, DBSCAN, MCTS on first use. New Terminology
  entry for particle filter (PF) covering the C.3 methodology
  hardening.
- **Iterations 9 + 10 (sibling repo HANDOFFs).** Bear Cub
  (`~/src/learning/bearcub/`) and goldbug (`~/src/learning/gldbg/`)
  each get a HANDOFF note (`handoff/outbox/2026-06-14-*.md`) with
  the per-chapter jargon list and voice-rule reminders. Each repo
  manages its own beta+prod deploy. Acks expected back in
  `handoff/inbox/`.
- **Iteration 11.** Removed `portfolio/notebooks/motherlode/data_exploration_v3p1.qmd`
  (superseded by `v3p1_improvements.qmd`; already excluded from the
  public render).

## 2026-06-13 — Portfolio polish, Iterations 1-3

- **Iteration 1 (audit).** Generated `research/portfolio_polish_audit.md`
  (gitignored). Surveyed 6 top-level chapters, 30+ deeper notebooks,
  every chapter figure, and the two sibling-repo intros (bear cub +
  goldbug). Identified one stale notebook, one chapter-card thumbnail
  to refresh, four figures appropriate to move to deeper notebooks
  (3 sensitivity sweeps + 1 preview section), one high-value missing
  figure (belief-accuracy chart), and a per-chapter jargon list. The
  audit's GB-vs-MB file-size claim was a misread and was retracted.
- **Iteration 2 (figures).** Moved the C.1 Bernoulli sensor sweep,
  D.1.C K-sweep, and D.1.C POMCP-simulation sweep from
  `drill_planning.qmd` to `notebooks/bcgt/decision_planning.qmd`
  (chapter retains the one-paragraph finding with a link). Kept the
  regime sweep inline as the cleanest SARSOP-wins robustness story.
  Moved the MAF anomaly preview section from `placer.qmd` to
  `notebooks/northern_sierra_placer/data_overview.qmd`. Added the
  new `fig_v20_d1_belief_accuracy.png` chart and prose where the
  truth-belief edge is discussed. Refreshed the ch7 chapter-card
  thumbnail.
- **Iteration 3 (intro polish).** Tightened all six chapter cards on
  `index.qmd`. Defined jargon on first use (Random Forest, AUC, REE,
  DEEP-SEAM, notice of location, placer, ROC). Expanded acronyms
  (MPM -> mineral-prospectivity, BLM -> Bureau of Land Management).
  Added the Chapter 2 headline finding to its card. Trimmed the
  Chapter 4 card heavily (dropped per-version model-stack and data-
  source acronym detail). Trimmed the Chapter 7 card from ~150 to
  ~110 words AND refreshed stale numbers (the old card claimed
  SARSOP final posterior 0.82 vs greedy 0.67, which predated the
  C.3 methodology hardening; new numbers 0.71 vs 0.64 from D.1.D
  real priors). Methodology section now defines Random Forest and
  AUC inline.

## 2026-06-13 — Chapter 7 D.1.D real BCGS deposit-type priors (Chunk E)

Real BCGT deposit-type prior surfaces (`src/ai_minerals/decision/v20/real_priors.py`)
aggregated from BCGT 500m per-cell binary deposit-type labels onto a
30x30 D.1-scale grid. New factory
`make_bcgt_deposit_type_hypothesis_set()` wraps them into a 4-paper
+ null HypothesisSet (porphyry, skarn, epithermal, VMS) that drops
into the existing D.1 SARSOP/PF/greedy stack. C.2 falsification on
real priors runs in seconds and produces three interpretable
scenarios (porphyry truth concentrates, null truth falsifies, the
epithermal/VMS r=0.7 correlation shows up as posterior confusion).
D.1.D 30-episode benchmark holds the same ordering as D.1.C
synthetic (greedy +26.3, POMCP +26.1, SARSOP +25.6, tied within
SEM ~4.1); truth-belief at episode end is where SARSOP earns its
keep (0.71 vs greedy 0.64).

## 2026-06-13 — Chapter 7 v20 module polish (Chunk F)

`__all__` exports across the eight v20 modules that were missing
them. Ten new edge-case tests (`test_real_priors.py`,
`test_v20_edge_cases.py`); total v20 suite now 182 passing (was 172).
`v20/README.md` updated with the two new modules (`domains.py`,
`real_priors.py`) and three new milestone tags. Glossary additions
called out in the original plan (posterior, SEM, grid-drilling, etc.)
were already landed in the chapter Terminology block from the prior
MAX-mode pass.

## 2026-06-13 — Chapter 7 D.1 redesign and hardening (Chunks D + C.3)

D.1 redesign in the porphyry-Cu economics regime: TRUTH_CUTOFF=0.15
+ wrong-commitment penalty=30 restored. Main benchmark at 100
episodes shows the regime narrowed the SARSOP-greedy gap but did
not flip the ordering (all five policies tied within SEM). The D.4
regime-robustness sweep shows SARSOP wins 9/9 cells in the 3x3
(cutoff, penalty) grid by +0.82 to +2.11 pp (mean +1.49). Multi-
hypothesis ESS particle filter (`MultiHypothesisESSParticleFilter`)
replaces the canonical-realization shortcut as the methodology-
honest variant.

## 2026-06-12 — Chapter 7 B.0 Mern 2024 reproduction (Chunk B)

`make_mern_2x2_hypothesis_set()` factory + structured graben /
geochem-domain polygon priors. `GridDrillingPolicy` baseline (6x6 =
36 holes). New B.0 chapter section "Reproducing Mern 2024"
demonstrates POMCP at 9 holes reaches 90-100% discovery rate while
the grid baseline needs ~14-16 holes for the same rate.

## 2026-06-11 — Chapter 7 B.2 BCGT retrospective expansion

B.2 retrospective benchmark expanded from KSM alone to 7 BCGT
mining districts at four drill budgets each (50, 100, 200, 625).
New `capture@N-drills` scorer. Chapter rewritten to lead with the
strongest positive findings (small-budget regime where POMCP and
BayesianGreedy beat the static-prior baseline by 7-23 pp at SEM
separation), with the KSM-at-625-budget tie demoted to a one-
paragraph caveat.

## Earlier history

Earlier chapter work (Chapters 1-6, B.1 single-hypothesis B.2 first
pass, C.1 noise sweep, C.2 multi-hypothesis falsification, Bear Cub
3D model + GP recommender, goldbug live tool, Lawley + DEEP-SEAM
audits, cross-region transferability test, Northern Sierra placer
v3) predates this changelog. See `git log` for the full history.
