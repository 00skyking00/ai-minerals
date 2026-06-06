# ai-minerals

Mineral prospectivity portfolio: two regional Random-Forest+SHAP models with
a published-blind-test validation, a dark-data ingestion pipeline for
1919-1955 placer-gold drill logs, and a decision-theoretic drill planner
(POMDP) on top of the regional prior.

A portfolio artifact, not a production tool — demonstrating competent
engagement with the problem space that [KoBold Metals](https://koboldmetals.com/),
[ExploreTech](https://exploretech.ai/), and [Earth AI](https://earth-ai.com/)
work in.

**Live site:** <https://johnsondevco.com/ai-minerals/>

## Where to start

| If you are… | Read |
|---|---|
| AI/ML hiring manager (KoBold/ExploreTech/Earth AI) | [`index.qmd`](index.qmd) → [`eastak/eastak_porphyry_prospectivity.qmd`](notebooks/eastak/eastak_porphyry_prospectivity.qmd) → [`bcgt/bcgt_porphyry_prospectivity.qmd`](notebooks/bcgt/bcgt_porphyry_prospectivity.qmd) → [`bcgt/decision_planning.qmd`](notebooks/bcgt/decision_planning.qmd) → [`bear_cub/main.qmd`](notebooks/bear_cub/main.qmd) |
| Placer mining engineer (Bear Cub-specific) | [`bear_cub/mining_review.qmd`](notebooks/bear_cub/mining_review.qmd) |
| Reproducing or extending the work | [`bear_cub/internal.qmd`](notebooks/bear_cub/internal.qmd) (full caveats) + this README's *Reproduce* section |

## Headline findings

**v1 — Eastern Alaska porphyry.** On standard metrics (20 km spatial block CV,
SHAP interpretation against mineral-systems theory), the Random Forest
looks good: ROC-AUC 0.88, top-2 % of area captures 80 % of known positives,
top SHAP features match the textbook porphyry pathfinder suite. The
external blind test — Kenorland Minerals `23ETD062`, a 2023 porphyry drill
intersection (174 m @ 0.14 % Cu) that postdates every training label —
scores **P=0.012, 62nd percentile**. Feature diagnostic traces this to
exploration bias in three channels (count / NaN-vs-value /
magnitude-within-explored). None of the v1.1 remediations (drop counts,
`*_has_data` indicators, no-geochem baseline, PU bagging) rescue the blind
test. *Honest message: a standard pseudo-supervised MPM pipeline at 500 m
regional resolution cannot distinguish a real low-grade distal porphyry hit
from background, regardless of feature engineering. The follow-on artifacts
are the v1.2 next-steps.*

**v2 — BC Golden Triangle.** Multi-deposit-class model (porphyry + epithermal
+ skarn + VMS) validated against a *distribution* of 241 post-2015 drill
holes (154 intersected positives + 87 drilled-negatives) — not the v1 N=1
anecdote. Plus a [decision-theoretic drill planner
(POMDP)](notebooks/bcgt/decision_planning.qmd) on top of the BCGT prior:
random / greedy / POMCP baselines on a 30×30 cell subarea, ported from
[Mern et al. *Intelligent Prospector* v1.0 (GMD 2023)](https://gmd.copernicus.org/articles/16/289/2023/)
onto our Python stack via [`pomdp_py`](https://github.com/h2r/pomdp-py).

**Bear Cub pilot — Nome placer-Au dark-data ingestion.** Twenty-four
hand-written 1919-1955 paper drill logs from a family-held archive, ingested
end-to-end with a self-correcting OCR + property-map cross-check + empirical
yield-formula derivation pipeline. The cross-check itself caught a 45°
geomorphology trap in the cartographer's drill-hole map that an OCR
confidence score alone wouldn't have surfaced. The resource estimate (6,251
fine oz pay-zone polygon, 90% MC CI 6,010–6,739) lands 38% below the
published 1936-vintage 10,056-oz figure for the same property; per-hole
agreement is within 6–10% on matching pay-zone windows, so the gap is
methodological (sliding-window vs uniform 8-ft layer), not measurement.
Targets KoBold's "TerraShed" pitch directly.

## Repo layout

```
notebooks/
  eastak/                              # v1: Eastern Alaska porphyry Cu-Mo-Au-Ag
    eastak_porphyry_prospectivity.qmd  # ← v1 integrated report
    data_exploration.qmd
    baseline_model.qmd
    random_forest_and_shap.qmd
    validation.qmd                     # external blind test + sensitivities
  bcgt/                                # v2: BC Golden Triangle multi-deposit
    bcgt_porphyry_prospectivity.qmd    # ← v2 integrated report
    data_exploration.qmd
    baseline_model.qmd
    random_forest_and_shap.qmd
    validation.qmd                     # 241-hole distribution blind test
    decision_planning.qmd              # ← POMDP drill planner on the BCGT prior
  bear_cub/                            # Pilot: Nome placer-Au dark-data ingestion
    main.qmd                           # ← AI/ML jobsearch-facing report
    mining_review.qmd                  # ← placer mining engineer-facing review
    internal.qmd                       # ← full review + caveats + TODOs

src/ai_minerals/
  bear_cub/                            # OCR + georef + grade + 3D model module
  decision/                            # POMDP problem + policies (random/greedy/POMCP)
  data/, features/, regions/           # canonical schemas + adapters per region
  model_rf.py, model_pu.py             # RF + PU learning helpers

tools/                                 # CLI drivers (resource analysis, reviewer apps,
                                       # 3D model generator, redistribution diff lister, etc.)
research/                              # company / state-of-the-art briefs
docs/                                  # scoping doc, v1.1 findings trace, runbooks
data/raw/, data/derived/               # gitignored except small reference inputs
scripts/deploy_to_hostinger.sh         # rsync the rendered _site/ to johnsondevco.com
```

## Reproduce

```bash
git clone <this-repo>
cd ai-minerals
uv sync                                          # Python 3.11+, locked deps
uv run python -m ai_minerals.data.fetch.all      # ~5 GB raw downloads (~1 hr)
uv run python -m ai_minerals.features.assemble   # rebuilds the parquet feature frames
QUARTO_PYTHON=.venv/bin/python3 quarto render    # renders the full Quarto site → _site/
```

For Bear Cub specifically, the OCR pipeline is in [`tools/`](tools/):
`bear_cub_full_log_ocr.py` (Claude Opus pass, ~$14 across all 24 logs),
`bear_cub_aggregate_ocr.py` (canonical schema), `bear_cub_resource_analysis.py`
(Voronoi + triangle + MC volumetrics), `bear_cub_3d_model.py` (PyVista
3D scene). The Streamlit reviewers (`bear_cub_ocr_reviewer.py`,
`bear_cub_suspect_reviewer.py`) are the human-in-the-loop correction layer.

Notebooks have no hidden state between cells; restart-and-run-all works from
a fresh environment. Quarto render workflow if `_freeze` ever gets stale:
bust `_freeze` and `.quarto/_freeze`, run `jupyter nbconvert --execute --inplace`
on each ipynb, then `quarto render --no-execute`.

## Target regions + commodity

- **v1** — eastern Alaska porphyry belt: Tanacross + Mt Hayes + Nabesna
  1:250,000 quadrangles (~62,000 km²), spanning the Wrangellia–Yukon-Tanana
  tectonic boundary. The same regional framework KoBold's Skolai project
  sits inside.
- **v2** — BC Golden Triangle: Brucejack / KSM / Red Chris / Galore Creek
  / Eskay Creek. Multi-deposit-class.
- **Pilot** — Nome placer-Au, 4-claim family property in the Cape Nome
  district. Bear Cub MS 1178 is the first claim done; the Janin Huntington
  archive is the planned next OCR target (separate sub-project).
- **v3 (planned sequel)** — Mother Lode Belt, California — orogenic Au.
  Reuses most of this scaffolding, tests generalization across deposit
  types and commodities.

## License

MIT. See [LICENSE](LICENSE).
