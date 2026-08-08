>>> [HANDOFF] 2026-06-27 coordinator → ai-minerals

# Nome model round 4 — confirm the eastern lode signal (A) + the one new placer lever (B)

Round 4 is **refinement, not breakthrough**. Rounds 2-3 already fixed and served the lode model
(leak-free **0.712**, CI [+0.108, +0.234], peninsula generalization 0.82-0.89, live on goldbug +
chapter) and validated the placer model as a sound beach-line model. Round 4 closes the **two honest
limits** the round-3 results named. No new deep-research — the geology is settled by the three prior
runs + `portfolio/docs/reports/nome_genesis_modeling_synthesis_2026-06-26.md`. Two parallel
workstreams, each its own PR off `origin/main`. Report numbers as JSON/CSV + a short report per piece.

---

## Workstream A — confirm the eastern lode splay/intersection signal

Round-3 peninsula generalization was strong overall (0.82-0.89) and the NE/NW **orientation** signal
transferred with significance, but the **Groves splay/intersection refinement specifically** was
underpowered out east: only ~9 eastern 36a positives, gain-over-base CI [-0.07, +0.28] still includes
zero. Close that:

1. **Expand the eastern low-sulfide Au-quartz (Cox-Singer 36a) labels.** Pull more 36a lode
   occurrences from ARDF/MRDS across the **eastern Seward Peninsula** — Council, Casadepaga,
   Solomon-east, and the **Big Hurrah** district (SP11). Use the same typed-label discipline as the
   dispersed central set (no occurrence-derived features — that was the F3 leak). Report the new
   eastern positive count.
2. **Re-test the splay/intersection gain over a lithology+fault base** on the larger eastern set,
   under the F1 leak-guarded CV (`leak_guarded_evaluate`, contiguous-region folds, r=1 km dead-zone)
   with a bootstrap CI. The structure features are already staged from the detailed bedrock GeMS
   (SIM 3131 / RI 2024-7) out east — reuse `nome_structure.py` (splays, NE×NW intersections, fold
   hinges). Report whether the refinement now **clears zero in the east**.
3. **Conditional — do NOT auto-promote.** If the splay refinement confirms, regenerate the
   peninsula-wide lode raster (the lode analogue of `scripts/nome_placer/mpm_lode_score_district.py`)
   and stage it for relay to goldbug, but flag it for Sky's serve decision (central-district vs
   peninsula-wide as the live headline). If it does not confirm, report the refinement as
   Nome-district-specific and the central-district served surface stands. Either result is fine —
   report it straight, the way you reported the per-named-fault negative.

## Workstream B — drift-on-beach placer feature (the one un-built lever)

Round 3 found the placer model already IS a beach-line model; the only genuinely-new lever is the
**strandline × glacial-drift intersection** (gold peaks where the raised beaches cross the Nome River
drift). It was data-walled last round because AOF 125 doesn't cover central Nome and maps no discrete
drift unit.

1. **Pull the RI 2024-6 surficial-geologic vector** (DGGS, DOI 10.14509/31054) — it covers central
   Nome and maps the Nome River drift as discrete polygons. The PDF is already in the research
   library; you need the GeMS vector (geodatabase/shapefile). **If the vector won't download cleanly,
   tell me and I (coordinator) will pull it** — don't burn the round on it.
2. **Build the strandline × drift-polygon intersection feature** on the full-coverage IfSAR 5 m DEM,
   reusing `build_strandline_on_dem()` from PR #33. The feature is only defined where strandlines
   cross mapped drift — handle that coverage explicitly (mask/indicator, don't impute).
3. **Test its marginal AUC** under the F1 CV against the current placer feature set. **A null is a
   perfectly good result** — it means the beach-line backbone already captures the drift effect.
   Report the marginal lift with its CI and say plainly whether it adds anything.

---

## Housekeeping — re-merge PR #39 (trivial, do first)
PR #38 just merged to `origin/main`; **PR #39 (the private walkthrough notebook) now re-conflicts**
on `.project/status.json` — the same recurring overlap. Merge current `origin/main` into the #39
branch `f1/nome-walkthrough-notebook-2026-06-27` (merge, not rebase; union the status.json the way you
did for #38) so it goes back to MERGEABLE/CLEAN, and tell me. The notebook content itself is fine and
already reviewed; this is just the status.json union.

## Sequencing & output
A and B are independent — run in parallel. A's eastern structure is already staged; B gates on the
RI 2024-6 vector (ping me if it's a wall). One PR per workstream off `origin/main`. **No prod
promotion** — the conditional peninsula-lode serve is the only deploy and it's Sky-gated. Reply to my
inbox with: the eastern positive count + splay CI (A), the drift-on-beach marginal AUC + CI (B), and
each report path. Voice rules apply to any prose (banned words/idioms, no em-dashes, humble register);
keep ML terms-of-art (lift, etc.) in the deep technical notebook only.
