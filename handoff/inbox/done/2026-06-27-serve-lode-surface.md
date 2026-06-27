>>> [HANDOFF] 2026-06-27 coordinator → ai-minerals

# Serve the lode prospectivity surface + flip the chapter prose (Sky approved)

The lode win is confirmed (CI [+0.108, +0.234]) and generalizes peninsula-wide (0.82-0.89 on
dispersed labels incl. held-out eastern lodes). Sky approved **serving it** on goldbug, like the
placer MPM. Three tasks; PR off origin/main; I drive the goldbug integration + deploy.

## 1. Produce the served lode prospectivity raster
Generate the lode-favorability raster from the **confirmed model** (the `struct_groves` feature set —
NE/NW orientation + splay/intersection + fold-hinge + graphitic host — that cleared the gate), as a
goldbug-ready GeoTIFF in the **same format/CRS as the served placer raster** (so goldbug can render a
lode layer the same way it renders placer: per-claim zonal aggregate + percentile buckets).
- **Extent decision (yours, document it):** serve the **central-district** 0.712 model, or the
  **peninsula-wide** 0.82-0.89 surface? The peninsula model generalizes (predicts eastern lodes it
  never trained on), but the splay/intersection refinement is unconfirmed in the east (9 positives).
  My lean: serve the central-district surface as the headline (fully confirmed), and note the
  peninsula extension as validated-but-provisional — but you know the surface; decide + document.
- Export it where goldbug consumes the placer raster (mirror that path/convention); reply with the
  path + the served extent + the legend semantics (probability bands).

## 2. Flip the chapter prose (lode failed -> structural-control win)
Update the Nome chapter's lode section: it currently says the lode model **failed**
(restriction-of-range, ~0.58-0.63). It now works — rewrite to the honest win: the structural control
(orogenic gold in the graphitic Nome Group along NE/NW faults, concentrated at second-order **splays
and structure intersections** per Groves 2018), dispersed typed-36a labels breaking the
restriction-of-range, leak-guarded **AUC 0.712 (CI [+0.108,+0.234])** central and **0.82-0.89**
peninsula-wide on labels it never trained on. Keep the voice rules (no banned words, no em-dashes,
humble register, professor-to-educated-layman). Note the honest limits (the splay refinement needs
more eastern labels; per-named-fault and Albion were redundant; magnetics didn't help).

## 3. Housekeeping: re-resolve PR #33
PR #33 (placer beach-line REM) re-conflicted after #34-37 merged ahead of it. Merge current main into
it again (merge-not-rebase is fine, force-push stays disabled) so it's mergeable; I'll merge it.

## Output
Reply with: the served lode raster path + extent + legend, the chapter-prose PR, and #33 made
mergeable. I then dispatch goldbug to render the lode layer + deploy, and deploy the chapter. No
deploy from you.
