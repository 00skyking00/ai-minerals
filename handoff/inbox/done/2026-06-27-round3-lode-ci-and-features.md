>>> [HANDOFF] 2026-06-27 coordinator → ai-minerals

# Round 3 — confirm the 0.712, rebase PR#33, then the next feature wave

The lode 0.712 (PR #32) is **merged to main**. Sky greenlit the full round-3 agenda. Priorities in
order; separate PRs welcome. No deploy.

## P0 — housekeeping (quick)
- **Rebase PR #33** (placer beach-line REM) onto current main (post-#32) and resolve the conflicts
  (#32 touched the same structure/CV files). Sky wants it merged — it carries the reusable
  `build_strandline_on_dem()` + the AOF 125 staging + the negative-result report. Push so it's
  mergeable; I'll merge.

## P1 — confirm the lode win (the gate for serving it)
- **Paired bootstrap CI on the 0.712 Groves arm** (`struct_groves`), same method as the round-1
  +0.106 (2000 resamples, 95% CI + P(Δ>0)), on the 48 in-footprint typed-36a positives. This is the
  confirmation before we treat 0.712 as settled and before serving the surface. Report the interval.

## P2 — extend the structure beyond central Nome (the real "peninsula-wide" test)
- Pull the **detailed Seward bedrock GeMS — SIM 3131** (pubs.usgs.gov/of/2009/1254/database/) **+ RI
  2024-7** (DOI 10.14509/31308) — you already pulled AOF 125 this way. These give structure + the
  graphitic host BEYOND the central-Nome district GeMS.
- Recompute the Groves features (splay / NE×NW-intersection / fold-hinge / graphitic-host) on the
  wider extent, **disperse the 36a lode labels truly peninsula-wide** (past the district footprint),
  and **re-test the 0.712 under F1 CV**. This is what tells us whether the structural control
  generalizes or is central-Nome-specific.

## P3 — the two deferred feature builds
- **Digitize the Albion fault** — a *deliberate, documented* georeference of the fault trace from the
  Otto/Piekenbrock/Odden 2009 figure (`research/nome_debate_library/`). You were right not to
  hand-fabricate it unattended; do it carefully now, record the control points + provenance, then add
  distance-to-Albion to the lode features and report whether it helps.
- **Build the drift-on-beach intersection** placer feature (strandline × the staged AOF 125 surficial
  drift shapefile, `data/raw/nome_surficial_aof125/`) — the one genuinely-new beach-line lever.
  Handle its coastal-coverage limit explicitly (it's only defined where strandlines cross mapped
  drift). Measure marginal AUC under F1 CV; a null is fine.

## Note
"Serve the lode prospectivity surface on goldbug" is gated on the P1 CI confirming 0.712 — I'll drive
that deploy after your CI lands, don't do it here.

## Output
PRs off origin/main per piece, each with numbers (JSON/CSV) + a short report. Reply with: the
bootstrap CI on 0.712; whether the structure generalizes peninsula-wide; whether Albion +
drift-on-beach add anything. No deploy.
