# ai_minerals.decision.v20

POMDP machinery for sequential drill planning under epistemic
geological uncertainty, ported from Mern et al. *Intelligent
Prospector* v2.0 (arXiv 2410.10610, 2024) and extended for the
BC Golden Triangle (BCGT) working area.

## Installation

```bash
git clone https://github.com/00skyking00/ai-minerals.git
cd ai-minerals
uv sync                              # Python 3.12+, pinned deps
bash scripts/build_pomdpsol.sh       # ~90 sec, builds APPL pomdpsol
```

The SARSOP wiring needs the APPL `pomdpsol` binary built once. The
build script writes it to `vendor/sarsop/pomdpsol`.

## Quick start

The four canonical entry points are stable. Anything else in the
module is subject to change.

```python
import numpy as np
from pathlib import Path
from ai_minerals.decision.v20.bcgt_scale import (
    make_bcgt_synthetic_hypothesis_set,
    realize_deposit_sets,
    expected_deposit_per_cell,
    BcgtScaleSARSOPPolicy,
)

# Build a 3-hypothesis prior on the 30x30 BCGT subarea
hset, coords = make_bcgt_synthetic_hypothesis_set(n_side=30)

# Sample one ground-truth realization per hypothesis
deposit_sets = realize_deposit_sets(hset, np.random.default_rng(0))

# Inspect the marginal deposit probability under a uniform belief
belief = hset.initial_prior()
deposit_probability_per_cell = expected_deposit_per_cell(hset, belief)
top_candidates = np.argsort(-deposit_probability_per_cell)[:20]

# Build a per-step SARSOP policy and run one episode
policy = BcgtScaleSARSOPPolicy(
    hypothesis_set=hset,
    deposit_sets=deposit_sets,
    pomdpsol_path=Path("vendor/sarsop/pomdpsol"),
    top_k=20,
)

for drill_step in range(9):
    cell = policy.choose_action()
    # ... carry out the drill in the field, get observation ...
    observation = 1  # placeholder
    policy.observe(cell, observation)
```

## Module layout

```
v20/
  hypotheses.py        Hypothesis, HypothesisSet, NullHypothesis,
                       and the GP-prior primitives. Matern v=2.5
                       kernel; locked parameters.

  pomdp.py             CorrelatedDrillingProblem and
                       MultiHypothesisDrillingProblem: the v2.0
                       extensions to the v1.0 POMDP (correlated
                       draws, noisy Bernoulli sensor, multi-
                       hypothesis state).

  belief_pf.py         Importance-weighted particle filter for the
                       single-hypothesis belief representation
                       (B.1 milestone).

  belief_ess.py        Elliptical Slice Sampling (Murray, Adams,
                       MacKay 2010 AISTATS) for the multi-
                       hypothesis belief (C.2 part 1).

  policies.py          Random, GreedyMean, BayesianGreedy,
                       CorrelatedPriorPOMCPPolicy, and the
                       multi-hypothesis falsification policy.

  simulator.py         SyntheticMonteCarloSimulator (B.1) and
                       RetrospectiveBCGSValidator (B.2).

  sarsop_policy.py     Small-grid multi-hypothesis SARSOP wrapper
                       (C.2 SARSOP integration). Builds a
                       pomdp_py.Agent over a finite enumerable
                       state/action/obs space and runs the APPL
                       pomdpsol binary.

  bcgt_scale.py        BCGT-scale belief-conditioned top-K SARSOP
                       (D.1). Composes the small-grid wrapper onto
                       the real 30 by 30 grid via per-step action
                       pruning.

  domains.py           Graben and geochemical-domain polygon
                       generation for the Mern 2024 structured-prior
                       2-by-2 hypothesis grid (B.0 reproduction).

  real_priors.py       BCGS deposit-type prior surfaces (D.1.D).
                       Aggregates BCGT 500m per-cell binary deposit-
                       type labels onto a target n_side x n_side grid
                       by block-averaging, Gaussian smoothing, and
                       per-surface renormalization. Drop-in replacement
                       for make_bcgt_synthetic_hypothesis_set in the
                       D.1 SARSOP stack.

README.md              You are here.
```

## Milestone history

The module landed in five steps, each its own ai-minerals minor tag.

- **B.0 (v1.2.0).** Reference-code walkthrough. Julia
  *HierarchicalMineralExploration.jl* mapped to Python equivalents.
- **B.1 (v1.3.0).** Correlated-prior single-hypothesis POMCP.
  Replaces the v1.0 iid Bernoulli prior with a thresholded GP. Matern
  kernel parameters locked. Importance-weighted particle filter for
  the belief representation.
- **B.2 (v1.4.0).** Retrospective BCGS validation. The original
  contribution beyond Mern 2024: scores the planner against the BCGS
  pre-2010 drill record at the Cox-Singer porphyry-Cu cutoff.
- **C.1 (v1.5.0).** Bernoulli noisy sensor with parameters
  :math:`\alpha` (false-positive rate) and :math:`\beta` (false-
  negative rate). 3-by-3 sensitivity sweep on (:math:`\alpha`,
  :math:`\beta`).
- **C.2 (v1.6.0).** Multi-hypothesis machinery with null-hypothesis
  falsification check. SARSOP integration via `pomdp_py` and the
  APPL `pomdpsol` binary. Tiger-style demonstration of the planning
  advantage on a 4-cell toy problem.
- **D.1 (v1.7.0).** BCGT-scale wrapper. Belief-conditioned top-K
  pruning so SARSOP stays tractable on the 900-cell action space.
- **B.0 (v1.8.0).** Mern 2024 reproduction. Structured graben +
  geochemical-domain polygon priors over the 2-by-2 hypothesis grid,
  6-by-6 grid-drilling baseline at 36 holes vs the POMDP at 9.
- **B.1 hardening (v1.9.0).** Multi-hypothesis ESS particle filter
  with Rao-Blackwellized categorical update; SARSOP+PF variant in the
  D.1 benchmark stack.
- **D.1.D + C.2 real priors (v1.10.0).** Four BCGS deposit-type
  hypotheses (porphyry, skarn, epithermal, VMS) plus null, aggregated
  from the BCGT 500m feature parquet. Falsification demo on the real-
  prior 5-hypothesis set; D.1 benchmark on real priors.

## Conventions

Public API is reexported from each module's `__all__`. Anything not in
`__all__` is internal and may change.

Type hints on every public function signature. Docstrings in NumPy
style (`Parameters` / `Returns` / `Examples` sections). One module
docstring per file covering the module's role in the chapter the
module supports.

Locked parameter values (kernel choices, sensor calibration values,
cutoffs) live as module-level constants prefixed `DEFAULT_`. Callers
override per call.

Floats are explicit (`float(x)` casts, `np.float64` dtype on arrays
where it matters for numerical stability). Random state is always
explicit (`rng: np.random.Generator`) so episodes are reproducible
end to end.

## Tests

```bash
.venv/bin/python -m pytest tests/decision/v20/
```

End-to-end SARSOP tests require the `pomdpsol` binary. They skip
gracefully if `vendor/sarsop/pomdpsol` is not built.

## References

Per-chapter references and DOIs live in
`portfolio/drill_planning.qmd` ([rendered chapter](https://johnsondevco.com/ai-minerals/drill_planning.html#references))
and the repository-wide list at
[portfolio/references.qmd](../../../../portfolio/references.qmd).

## License

MIT, same as the repository. See `LICENSE` at the repo root.
