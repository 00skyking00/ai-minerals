>>> [HANDOFF] 2026-06-22T12:25:00-07:00 coordinator
# 2026-06-22 — ai-minerals: PRIORITY — render + deploy the rewritten Nome Placers chapter NOW (Sky wants a live review link)

**From:** portfolio (coordinator). Sky wants to review the rewrite live, today. The live
`/ai-minerals/nome_placer.html` is still the OLD pre-rewrite version (deployed HTML is Jun 19); your rewrite
(`406a226`) is source-only. **Render + deploy it now** so Sky has a real link.

- **Immediate priority:** render the rewritten `portfolio/nome_placer.qmd` (use the nbconvert/quarto workaround
  you flagged for the hang risk) and deploy it to `/ai-minerals/` so `nome_placer.html` is the new chapter.
  Wire the 2 new figures (`ch1_nome_placer_headline_auc.png`, `ch1_nome_lode_mirage.png`) into the chart/thumb
  build so they render on deploy (the deploy point you flagged — do it as part of this).
- **Reply with the live URL** the moment it's up; I relay to Sky for review.
- **The model-promotion work continues separately** — when the promoted MPM surface + the Fig 3/framing flip are
  ready, that's a SECOND deploy. Don't block this review deploy on the promotion; deploy the current draft now,
  re-deploy with the promotion when it lands. (If the promotion happens to be ready, fold it in — your call, but
  speed of getting a review link up is the priority.)
- If `quarto render` hangs even with the workaround, say so immediately and we find another path (static export)
  — don't sit on it silently.

Reply: the live `/ai-minerals/nome_placer.html` URL + whether it's the pre- or post-promotion framing.
