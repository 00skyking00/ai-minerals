# ai-minerals

A mineral-prospectivity ML portfolio: seven chapters covering regional
random-forest models, a live per-claim filtering tool, a placer-Au
classifier, two reproductions of published mineral-prospectivity papers
(with the failure modes I found), a five-region transferability test,
and a POMDP drill planner.

A portfolio artifact, not a production tool. The target audience is
hiring managers at [KoBold Metals](https://koboldmetals.com/),
[ExploreTech](https://exploretech.ai/), and
[Earth AI](https://earth-ai.com/).

**Live site:** <https://johnsondevco.com/ai-minerals/>

**Current release:** `ai-minerals-v1.1.0` (placer-v3.7.0 Mother-Lode-wide
Quaternary retrain). Per-component versions live in
[`data/ml/ml_versions.json`](data/ml/ml_versions.json); release-history
is in [`git tag`](https://github.com/00skyking00/ai-minerals/tags).

## What's in the seven chapters

1. **Bear Cub (Cape Nome placer-Au).** A family-owned 1899 patented
   placer claim. 24 churn-drill logs from 1899-1930s OCR'd via a
   multi-engine pipeline, mapped onto a recent plat, and turned into a
   3D bedrock + pay-zone model with Monte Carlo volumetrics.
   Cross-validated against two independent mining-engineer reports
   within 6%. Lives at the sibling site
   <https://johnsondevco.com/bearcub/>; this repo only carries the
   chapter framing.

2. **Regional MPM pipelines.** Four regions (Eastern Alaska
   porphyry-Cu, BC Golden Triangle multi-deposit, California Mother
   Lode orogenic-Au, Arizona porphyry-Cu) with one shared scaffolding:
   pull public USGS/NRCan/state-survey rasters onto a common 250 m
   grid, train a stacked supervised classifier, hold spatial blocks
   out during CV.

3. **goldbug (live tool).** Mother Lode model output gated against
   federal-land availability for an individual prospector to file a
   notice of location on. Hard part was BLM's dozen disjoint
   endpoints with inconsistent schemas, not the modelling. Lives at
   <https://johnsondevco.com/goldbug/>.

4. **Placer-Au prospectivity (California Mother Lode).** The v3.7.0
   model. Tertiary deep-gravel + Quaternary modern-channel classifiers
   trained per-population, fused per-cell into the goldbug-consumed
   raster. Anisotropic channel-aligned kernel along NHD HR turns
   sparse USMIN point labels into per-cell weighted positives. Audit
   covers Tertiary polygon-rasterization inflation, per-county MRDS
   held-out gates in below-gate counties, and one documented gate
   failure (Mariposa).

5. **Reproducing + auditing two published methods.** Lawley 2022
   (continental Zn-Pb, AUC 0.983) reproduces cleanly but the
   published number includes a 2.4 pp label leak, a 9 pp spatial-
   blocking gap, and a cross-continent transfer collapse; properly
   validated the AUC drops to 0.71-0.87. DEEP-SEAM (Curnamona REE,
   86% top-2% capture) is scored under a 0-D random split with no
   spatial blocking. Both audits trace the gap from headline to
   transferable signal.

6. **Cross-region experiments.** A five-region transferability test
   on the DEEP-SEAM deviation-network architecture. The 2-3× win the
   paper reports holds on its tuning dataset (Curnamona REE, 7
   positives) and nowhere else. Useful negative result.

7. **Drill planning under uncertainty.** POMDP planner with Monte
   Carlo Tree Search (POMCP via `pomdp_py`) on the BCGT prior, ported
   from Mern et al. *Intelligent Prospector* v1.0 (GMD 2023). Random
   + greedy + Efficacy of Information + POMCP side-by-side. v1.0
   assumptions (iid Bernoulli prior, noiseless sensor) make greedy
   one-step optimal; the chapter explains why. The v2.0 closure
   (Mern 2024, correlated draws + multi-hypothesis falsification) is
   the bcgt-v2.0 milestone tracked in this repo's milestones.

## Reproduce

```bash
git clone https://github.com/00skyking00/ai-minerals.git
cd ai-minerals
uv sync                          # Python 3.12+, locked deps
quarto render portfolio          # renders the Quarto site -> portfolio/_site/
```

The bare `quarto render portfolio` reads from the cached Quarto freeze
(`portfolio/_freeze`); no Python execution needed if you only want the
rendered site. To re-execute the notebooks from scratch, drop
`--no-execute` from the Quarto invocation and run the feature-build
pipeline first via the per-region assemble scripts in `scripts/`.

For Chapter 7 (POMDP) and any future bcgt-v2.0 SARSOP work:

```bash
bash scripts/build_pomdpsol.sh   # builds the APPL SARSOP binary
                                 # at vendor/sarsop/pomdpsol (~90 sec)
```

Raw fetch (~10-20 GB across regions; not required to render the site):

```bash
.venv/bin/python -m ai_minerals.data.fetch.all
```

## Repo layout

```
portfolio/                  Quarto site source (rendered output at portfolio/_site/)
  index.qmd                 chapter cards + landing map
  regional.qmd              Chapter 2
  ...                       (one .qmd per chapter; chapter notebooks under notebooks/)

src/ai_minerals/            library: data adapters, features, models, decision
  data/, features/, regions/      canonical schemas + per-region wiring
  decision/                       v1.0 POMDP drill planner (Chapter 7)
  decision/v20/                   v2.0 POMDP skeleton (bcgt-v2.0 milestone)
  uncertainty/                    Monte Carlo bracket for placer raster

scripts/                    pipeline drivers
  northern_sierra_placer_*.py     v3.7.0 placer pipeline (Chapter 4)
  bcgs_to_dh2loop.py              BCGS drillhole-database -> dh2loop converter
  build_pomdpsol.sh               external APPL SARSOP binary build
  v37_*.py + v37_post_training_runbook.sh    v3.7.0 post-training pipeline

data/                       gitignored except small reference inputs
  ml/                       (tracked) goldbug-facing deliverables + sidecars
  raw/, derived/            (gitignored) fetcher outputs + intermediates
  ml/ml_versions.json       canonical per-component version manifest

docs/                       reproducibility runbook + product scoping
SITE-ARCHITECTURE.md        why portfolio/ + how the site builds + deploys
CLAUDE.md                   per-project conventions for Claude Code
```

## License

MIT. See [LICENSE](LICENSE).
