>>> [HANDOFF] 2026-06-22T12:05:00-07:00 coordinator
# 2026-06-22 — ai-minerals: PROMOTE the placer terrain MPM to production (Sky: "put our best model forward")

**From:** portfolio (coordinator). Sky greenlit promoting the validated model to the live goldbug map — this is
the ADR-013 condition met ("v3.1 stays live until an MPM beats it AND the framing is ratified; now done"). The
**placer presence/background terrain MPM (0.733 spatial-CV / 0.741 buffered vs the v3.1 0.444 overlay)** becomes
the served Nome prospectivity surface, replacing the hand-encoded beach overlay.

## Your part (you own the model + the surface)
1. **Score the district with the production placer MPM** and export the prospectivity surface as a served layer
   in the format goldbug's Nome render consumes — **coordinate the exact format with goldbug** (its `render_nome.py`
   currently serves the v3.1 overlay; agree raster/COG/geojson + grid/CRS + the score scaling/legend so the swap
   is clean). Hand goldbug the surface.
2. **Keep it honest:** it's presence/background with the scope-correction reading; carry the caveats into the
   map legend/popup (what the score means, the 0.733/0.741 validation, that it's onshore placer). Don't oversell.
3. **Regenerate Fig 3 + update the chapter prose.** Right now `nome_placer.qmd` §3 says "the live map is the
   0.444 overlay; the terrain MPM is analysis-only." After promotion that flips: **the validated model IS the
   live map.** Update Fig 3 (the prospectivity surface = the served MPM now) and the "placer model that held up"
   + "where we are" sections to say the validated surface is what a reader clicks today. This makes the chapter
   stronger for Sky's KoBold outreach. Re-run voice_lint/claim_trace after.

## Boundaries
- Placer only. The **lode** stays the reported negative — do NOT promote it.
- Validate the served surface actually reflects the 0.733 model (same covariates/scoring), not a stale RF dump.
- This supersedes ADR-011's "v3.1 stays the map; RF is analysis-only" — record it.

Reply with the surface handed to goldbug (format + location), the updated Fig 3, and the chapter-prose diff.
Coordinate the served format directly with goldbug (its inbox). Move with care — this is the public, KoBold-
facing tool.
