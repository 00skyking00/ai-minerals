# ai-minerals changelog

Per-repo changelog for the ai-minerals portfolio. Newest first; dates are when
work landed on `main`. Notable milestones only; see `git log` for the full
commit-level history. Each module keeps its own changelog under `docs/`; the
program-wide unified changelog lives in the portfolio coordinator repo.

This file consolidates and supersedes the earlier root `CHANGELOG.md`, which
covered only the 2026-06-11 to 2026-06-14 chapter-polish window. The entries
below extend coverage to the full project history (2026-04-22 onward).

## 2026-06-25: F1 leak-guarded spatial-CV harness + placer/lode re-baseline

- New reusable spatial-CV module `src/ai_minerals/spatial_cv.py`: residual-variogram
  block sizing, contiguous-region folds, and an Airola (2018) dead-zone buffer that
  drops training points within r of any test point. Reports in-box-vs-district AUC and
  the fitted variogram range per run. 10 unit tests in `tests/test_spatial_cv.py`
  (variogram tracking, dead-zone enforcement, fold compactness, collapse reproduction).
  Documented equivalent of `verde.BlockKFold` / `spacv`; no new dependency added
  (pyproject's deferred `spacv` placeholder stays deferred). (F1, coordinator dispatch)
- Re-baseline under the leak-guarded scheme (1 km dead-zone, above the 400-800 m F2
  proximity buffers; contiguous folds; blocks at 2x the residual-variogram range):
  placer onshore 0.733 -> 0.679; lode in-box 0.802 -> 0.806; lode district 0.620 ->
  0.633. The lode collapse (in-box 0.806 vs district 0.633) survives the stricter
  ruler; the placer drop of ~0.05 is the autocorrelation leak the old no-buffer fold
  kept. Offline modeling only; does not change the served goldbug raster. Report at
  `docs/f1_leak_guarded_cv_rebaseline.md`; numbers under
  `data/derived/nome_placer/f1_rebaseline/`.
- Finding escalated to the coordinator: fold-assignment strategy moves the district AUC
  by ~0.10 (scattered 0.731 vs contiguous 0.633). Scattered variogram-blocks let the
  model interpolate the coastal-to-upland elevation gradient; only contiguous regional
  holdout exposes restriction-of-range. This conflicts with the dispatch's "distribute
  positives evenly across folds" instruction for clustered positives. Both strategies
  are implemented and reported; the headline is contiguous (it meets the acceptance
  test). Driver: `scripts/nome_placer/f1_leak_guarded_rebaseline.py`.

## 2026-06-24: Nome ARDF full-text re-export + production push

- ARDF Nome re-export with the full untruncated narrative text, rebuilt from the
  USGS mrdata ARDF per-record JSON service. The shapefile `.dbf` had capped every
  narrative field at 255 characters; 286 records now carry full text plus new
  `references` (full bibliography, 278/286) and `reporter` columns. WGS84 geometry
  unchanged (0/286 features moved). Handed to fossick for KG re-ingest. (`f60e167`)
- Production push to goldbug: nome_placer chapter reframed around the validated
  terrain model (0.733 / 0.741 spatial CV) now live; Regional card rewrite;
  Chapter 2 porphyry-highlight reframe (enrichment + trained-separately). (`42bd491`)

## 2026-06-22 to 06-23: Nome placer framing flip + portfolio card rewrites

- nome_placer chapter flipped to lead with the validated terrain model now live on
  goldbug; the v3.1 knowledge overlay reframed as the superseded first map. (`a1bc833`)
- Cape Nome moved to the closing synthesis chapter; methodology section moved to the
  front; front-door index card rewritten to lead with the story. (`350fa6c`, `7a58d6f`, `c1b9645`)
- Bear Cub, Regional, and Methodology card edits in Sky's voice. (`4a29f48`, `11e8942`, `d26278f`)
- Nome Placers chapter redraft: journey frame + discipline spine. (`406a226`)

## 2026-06-20 to 06-22: Nome MPM (placer + lode) + program coordination

- Onshore placer MPM: terrain-aware covariate stack with buffered 300 m spatial CV
  (AUC 0.736, holds vs 0.756 unbuffered, so not an autocorrelation artifact).
  Presence/background eval clears the v3.1 knowledge overlay (0.756 vs 0.444 on
  district placers). (`1ff10f6`, `118a427`, `4ac59a4`, `559d031`)
- District-wide lode MPM: structural signal does NOT hold at district scale
  (0.62 debunks the v0 0.80 placer-core result); recorded as a negative result. (`12d51cd`, `f0086fc`, `ed00f66`)
- MTP durable pyramid: BLM Master Title Plats re-fetched at ~1.66 m/px, white-to-transparent
  XYZ tiles z8-16 with COG fallback; goldbug serves BLM-independently (ADR-019). (`f97f276`)
- Placer MPM production scoring: served surface + goldbug raster format proposal (ADR-013). (`11d121e`, `1911e16`)
- Program coordination posture: `.project/status.json`, the `activity_flag.sh`
  live-activity writer, hub-and-spoke handoff convention. (`c5a71e6`, `8ad6b74`, `e15c473`)

## 2026-06-21: ARDF WGS84 re-export + IFSAR fetcher + KG occurrence adapter

- ARDF re-export for fossick: NAD27 to WGS84 datum fix (NM101 corrected by 155.8 m) +
  wider district-lode bbox. (`6e99b9a`, `c39d75e`)
- IFSAR native 5 m DTM + DSM fetcher for the Nome district. (`bf0a199`)
- Knowledge-graph occurrence adapter (ADR-017) consuming fossick KG labels; the label
  switch was held on a NAD27 datum bug, later cleared. (`18949f0`, `33be997`)

## 2026-06-14 to 06-16: Nome placer Phase 0 to Phase 2 baseline

- Phase 0: four-population region scaffold (Tertiary deep-gravel, Quaternary
  modern-channel, beach, offshore); ADGGS surficial + IfSAR Alaska adapters. (`5eb6d64`, `f814bed`)
- Phase 1: coastal fuzzy-overlay scorer with per-stand contours (+200 / +300 / +600 ft
  Tuck 1942 stands), stream features, buried-beach detection; validation gate 5/5. (`f3fe73a`, `65c480b`, `b40bfb2`)
- Tuck 1942 atlas pipeline: dredge-cleanup polygon vectorizer, bedrock-contour
  extractor, 8 plan-view overlays with embedded affine CRS for goldbug. (`666973c`, `4a623ed`, `01ba1be`, `74e66ef`)
- Phase 1.5: KDTree streams (100x faster), cross-section depth integration extending
  the buried beach-line envelope 15x beyond the Bear Cub footprint; 9 end-to-end
  integration tests. (`e7f51e1`, `03682bb`, `f418eee`)
- Phase 2 v0 baseline: RandomForest pay/barren classifier (AUC 0.83 LOO). (`546643a`, `50787ee`)

## 2026-06-13 to 06-14: Chapter 7 decision-planning hardening + portfolio polish

- D.1 redesign in the porphyry-Cu economics regime: the regime narrowed the
  SARSOP-greedy gap but did not flip the ordering (all five policies tied within SEM);
  the D.4 regime-sensitivity sweep shows SARSOP winning 9/9 cells of the 3x3
  (cutoff, penalty) grid. (`0cd17ca`, `db81dcd`, `0078daa`)
- Multi-hypothesis ESS particle filter (`MultiHypothesisESSParticleFilter`) replaces the
  canonical-realization shortcut as the methodologically tighter variant. (`e10ddb6`, `b1390ca`)
- Real BCGS deposit-type priors + `make_bcgt_deposit_type_hypothesis_set()`
  (porphyry / skarn / epithermal / VMS + null); C.2 falsification on real priors. (`38f07fb`, `b6add62`)
- Portfolio polish Iterations 1-11: jargon-definition pass across all six chapters,
  figure consolidation into deeper notebooks, the added belief-accuracy chart,
  sibling-repo HANDOFFs, one stale-notebook removal. (`eb09faf`, `9b2883d`, `662c11f`, `c046b7a`)

## 2026-06-10 to 06-13: BCGT decision-planning Chapter 7 build-out (POMDP)

- B.1: `CorrelatedDrillingProblem` + Gaussian sensor, GP-prior
  `Hypothesis.sample_realization`, importance-weighted particle filter, POMCP
  integration. (`be9c7b9`, `b2d5515`, `f1e538d`, `3442494`)
- B.2: `RetrospectiveBCGSValidator` expanded from KSM alone to 7 BCGT districts at
  four drill budgets each (50 / 100 / 200 / 625); new `capture@N-drills` scorer. (`313ce7c`, `b25c15f`, `b4b9121`)
- C.1 Bernoulli sensor + 3x3 sensitivity sweep. C.2: HypothesisSet + Dirichlet
  posterior, Elliptical Slice Sampling (Murray 2010), SARSOP integration via
  pomdp_py + APPL pomdpsol, falsification check. (`ff95fac`, `4a8fa8d`, `a21d5fe`, `bd5b3d2`)
- B.0: Mern 2024 reproduction. `make_mern_2x2_hypothesis_set()` factory,
  graben / geochem polygon priors, `GridDrillingPolicy` baseline; POMCP reaches
  90-100% discovery at 9 holes where the grid baseline needs ~14-16. (`87ebcad`, `aaeb568`, `820e332`)
- SARSOP path verified end-to-end via pomdp_py + the APPL pomdpsol binary. (`b2990c3`)

## 2026-06-09 to 06-11: Placer v3.7.0 + public-repo prep

- v3.7.0 label work: USMIN audit + channel-aligned Gaussian kernel relabel;
  nnPU prior auto-estimation with a `--nnpu-prior` CLI override. (`fd73578`, `9381823`, `e9a6d5a`)
- v3.7.0 chapter section + Mother Lode framing; placer sidecars (bands.json,
  coverage mask, 2-band refresh). (`85b5960`, `c874032`)
- Public-repo prep: exclude internal research, handoff, and NotebookLM from the
  public render. (`81e7bc8`)

## 2026-06-01 to 06-08: Northern Sierra placer v3 pipeline + deploy infra

- v3 placer pipeline: per-population feature stacks, XGBoost + nnPU + calibrated-logistic
  fusion, leakage-ablation (Hawkes `cell_mask`, distance-downstream seed filter). (`9050eed`, `4795c83`, `dc0578f`)
- v3 Phase D/E: feature-stack growth (NHD VAA, OSM mining, CGS Jennings scaffold),
  six parallel validation upgrades. (`f4f1168`, `b95e993`, `f4140c5`)
- v3.6 polygon-rasterization: Tertiary labels grow from 158 to ~1700 effective. (`3007a0e`)
- Resumable per-stage / per-fold checkpointing; fold watcher with live mean AUC. (`ee7db68`, `351b082`)
- Deploy infra: single-site `/ai-minerals/` with beta toggle, no-cache `.htaccess`,
  visit beacon, `SITE-ARCHITECTURE.md`. (`4512dfc`, `5d94519`, `1ef2334`)
- KoBold easy wins (EW1-EW5) + Chapter 2 BCGS-to-dh2loop bedrock plate. (`0a5f4e9`, `49c4fee`)

## 2026-05-28 to 05-31: Site redesign, sibling-repo extraction, region cleanup

- Unified portfolio redesign: navbar + sidebars + clickable map, 12 deeper notebooks
  dropped from the public render; Sandstone theme + locator map + thumbnail swaps. (`ac0531b`, `0aeb5b6`, `9c9cd54`)
- Bear Cub extracted to `~/src/learning/bearcub`; goldbug/gldbg moved to `../gldbg`;
  the umbrella repo owns `/ai-minerals/`. (`957972f`, `f54972d`, `ab98e8e`)
- Chapter 3 (goldbug) added; Mountain Pass v2 dropped after ground-truth validation. (`4ec5f0c`, `5c7ae29`, `89db5c9`)
- Northern Sierra placer v6 build + K.4 feature-engineering fixes. (`4bdaf16`)
- Privacy scrub: remove personal name / Tweet and Hostinger SSH info from publicly
  deployed pages. (`f860df3`, `17c4522`)

## 2026-05-12 to 05-13: Lawley + DEEP-SEAM audits, cross-region experiment, voice pass

- Post-Lawley + DEEP-SEAM audits + cross-region transferability experiments;
  internal-site reorg into a 5-chapter structure with a drill-planning chapter. (`273088e`, `cd0d1a6`)
- `VOICE-AND-STYLE.md` anti-AI-voice filter; index + chapter qmds rewritten in
  Sky's voice. (`752b992`, `ddf40f0`, `6ba8558`, `303f853`)

## 2026-04-29 to 05-02: Bear Cub dark-data drill-log pilot

- Dark-data ingestion pipeline for 24 family drill logs (pypdfium2 + anthropic OCR). (`bfed4c6`, `3a8ecb5`)
- Per-hole review checklists, bedrock imputation sensitivity, reviewer UX (row
  management, sample linking, suspect queue). (`c9737a4`, `a47f61d`, `cbcc546`)
- 3D drill-hole model (PyVista) + cross-hole fence diagram (4-state gold / barren /
  data-gap / undrilled encoding). (`61a2212`, `8f53410`, `2765edf`)
- BCGT decision_planning: POMDP drill planner end-to-end (Mern v1.0 framing);
  mining-engineer review notebook. (`5a87a33`, `49c4528`)

## 2026-04-22 to 04-25: v1 scaffold + Tanacross / EastAK data pipeline

- Project scaffold with research notes + MVP design. (`b972a29`)
- Tanacross data-acquisition pipeline: Sentinel-2 median composite (AWS Earth Search
  backend, threads-scheduler deadlock workaround), ARDF + AOI filtering, Cox & Singer
  porphyry positives by model code. (`11c620b`, `36798b7`, `cf6918e`)
- Expanded v1 AOI to Eastern Alaska (Tanacross + Mosquito Hills + Nabesna). (`6f6aa1b`)
- EastAK porphyry prospectivity: Random Forest + SHAP, spatial-block CV, with the
  exploration-bias confound surfaced and the v1 limits exposed. (`7640f42`, `55a0ef6`, `cc98499`)
- BCGT BC Golden Triangle v2: adapter layer + Region configs, real NRCan / NOAA
  geophysics (NOAA EMAG2 v3, 200 m magnetic, 2 km isostatic gravity). (`412fbcc`, `cb4ee51`, `ed1dbc3`)

## Earlier history

The 2026-04-22 scaffold is the first commit. There is no project history before it.
See `git log` for commit-level detail behind any entry above.
