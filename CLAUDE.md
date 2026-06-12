# CLAUDE.md — ai-minerals project

Scope rules for Claude Code when working in this repo. Local to this project; don't pull in conventions from other repos.

## Scope

- **Purpose:** Portfolio artifact demonstrating competent engagement with AI-mineral-exploration problem space. Not a production tool.
- **Deliverable:** A Quarto notebook + writeup producing a mineral-prospectivity map, published as a public GitHub repo.
- **The writeup is the artifact**, not the code. Prose quality matters as much as code quality.
- **Public repo from Day 1.** Assume everything in the working tree will be published.
- Day 1 (v1) region: Tanacross, AK — porphyry Cu-Mo-Au-Ag.
- Planned sequel (v2): Mother Lode, CA — orogenic Au.

## Coding rules

- Python 3.11+.
- Typed function signatures (`def foo(x: int) -> str:`) on public functions in `src/ai_minerals/`.
- Docstrings only on public functions. One-line is fine; no multi-paragraph boilerplate.
- Small modules. No premature abstractions. Three similar lines > a wrong abstraction.
- No speculative exception handling. Only catch errors you can meaningfully handle at that level.
- Prefer pure functions. Side-effecting I/O (downloads, disk writes) in clearly named functions.
- No test scaffolding for single-use pipeline code. Smoke-check data loads with visual or assertion-based cells in the notebook.
- Notebook = `.qmd` source in `notebooks/`. Never edit `.ipynb` directly.

## Testing + verification

- The notebook must run end-to-end from a fresh environment (`uv sync && quarto render notebooks/*.qmd`).
- Pin dependencies in `uv.lock`; do not rely on latest.
- Every data-fetch step has a shape/length assertion and a rendered visual sanity check.
- No hidden state between cells. Restart-and-run-all must work.
- Before committing, re-run the notebook if source changed.

## Writing rules (for notebook prose + READMEs)

- **Audience:** senior ML + geoscience reader.
- **Tone:** honest, precise, aware of the limits of a weekend demo. No breathless pitch.
- **Citations:** every non-trivial technical claim links to the primary source (paper, official docs, company page). No citing blog summaries when the primary exists.
- **Limitations are explicit.** Each substantive section includes a "what this doesn't prove" note.
- **No filler.** No "In this section, we will discuss..." scaffolding. Say the thing.
- **Headings carry the arc.** A skim-first reader should get the thesis from headings alone.
- **Mention what doesn't work** as readily as what does. Failure modes are signal, not weakness.

## Data hygiene

- No secrets in the repo (`.env` files, API keys, credentials) — `.gitignore` covers this as belt-and-suspenders.
- No proprietary or commercially-restricted data. Everything must be public-domain or a compatible open license.
- Raw downloads go to `data/raw/` (gitignored — can be large).
- Pipeline outputs go to `data/derived/` (gitignored — regenerable).
- If a small output file *should* be tracked for reproducibility (a derived summary CSV, say), `git add -f` it and justify in the commit message.
- Cross-repo coordination requests (work items going to or coming from `~/src/bearcub/`, `~/src/goldbug/`, `~/src/fossick/`) live under `handoff/outbox/` and `handoff/inbox/`. Use `research/` for substantive findings; `handoff/` for routine "please do X in your tree" notes. See `handoff/README.md`.

## Out of scope (don't drift)

- 3D physics-consistent geophysical inversions (SimPEG, etc.)
- POMDP / decision-theoretic drill planning
- OCR of legacy drill logs for the main demo (separate Bear Cub subproject)
- Production deployment, hosting, Docker, CI orchestration
- Multi-region models in v1
- Prefect/Dagster/Airflow/MLflow/DVC — single notebook + thin `src/` is the target posture

## Other

- Ignore `NotebookLM/` — it's preserved as reference material from the user's pre-project research. Don't modify or extend it.
- The plan file at `~/.claude/plans/portfolio-project-a-bare-bones-logical-wolf.md` is the source of truth for the agreed scope.
