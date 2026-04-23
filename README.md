# ai-minerals

A bare-bones demonstration of AI-powered mineral prospectivity mapping using open data and simple ML. A portfolio artifact, not a production tool.

## What this is

A Jupyter / Quarto notebook that takes publicly available geoscientific data for a specific region and commodity, trains a simple machine-learning model, and produces a prospectivity map with honest validation and interpretation. The writeup — not the code — is the primary artifact.

Target region (v1): **Eastern Alaska porphyry belt** — three contiguous 1:250,000 quadrangles (Tanacross + Mt Hayes + Nabesna) spanning the Wrangellia–Yukon-Tanana tectonic boundary. Target commodity: **porphyry copper (Cu-Mo-Au-Ag)**.

Planned sequel (v2): **Mother Lode Belt, California**. Target commodity: **orogenic gold**.

## What this is not

A working exploration tool, a replacement for professional geoscience judgement, or a claim about specific real-world deposits. It's a demonstration that the author can engage competently with the problem space that companies like [KoBold Metals](https://koboldmetals.com/), [ExploreTech](https://exploretech.ai/), and [Earth AI](https://earth-ai.com/) work in.

## Repo layout

```
research/       # Research briefs on companies, state-of-the-art, datasets
design/         # MVP design doc
src/            # Python package (ai_minerals)
notebooks/      # Quarto (.qmd) notebooks that render to HTML + .ipynb
data/
  raw/          # Raw downloads (gitignored)
  derived/      # Pipeline outputs (gitignored)
```

## Status

Day 1 of 7. Research + design complete; no data fetches or code yet. See [research/](research/) and [design/mvp-proposal.md](design/mvp-proposal.md).

## License

MIT. See [LICENSE](LICENSE).
