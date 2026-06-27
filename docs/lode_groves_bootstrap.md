# The 0.712 lode gate, with an interval

The round-2 rebuild promoted the structure screen to a verdict: on typed 36a labels the
`struct_groves` arm scores **auc_gems 0.712**, +0.170 over the base model, the only arm
past the 0.70 gate (`docs/lode_structure_sharpen.md`). That is a point estimate on 48
positives inside the GeMS-mapped footprint. Before the surface is served, the delta needs
an interval. This is that interval.

**Question:** is the +0.170 gain distinguishable from zero under a paired bootstrap on the
48 GeMS-mapped 36a positives, by the same method that put a 95% CI of [+0.050, +0.164] on
the round-1 generic-structure screen?

**Answer: yes. Δ = +0.170, 95% CI [+0.108, +0.234], P(Δ>0) = 1.000.** The whole interval
sits above zero, and not one of the 2000 resamples produced a negative delta. The gate
holds.

## Numbers

Paired class-stratified bootstrap of the out-of-fold AUC delta (arm minus base) on
identical leak-guarded folds, 2000 resamples, 95% CI = 2.5/97.5 percentile. This is the
round-1 `newlayers_bootstrap.boot_delta` reused unchanged, so the point estimate
reproduces the driver's 0.712 to the decimal place.

| arm | subset | n_pos | Δ (point) | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|
| **struct_groves** | **gems (mapped)** | **48** | **+0.170** | **[+0.108, +0.234]** | **1.000** |
| struct_groves | placer_core (in-box) | 27 | +0.149 | [+0.067, +0.239] | 1.000 |
| struct_generic | gems (mapped) | 48 | +0.138 | [+0.063, +0.209] | 0.9995 |
| struct_generic | placer_core (in-box) | 27 | +0.128 | [+0.038, +0.226] | 0.9985 |

The headline row is `struct_groves` on the mapped subset: the gate. The other three rows
are context.

- **placer_core (in-box):** the round-1 generic structure features *hurt* the small in-box
  model (the box already scored 0.77 on 30 positives and overfit). The typed 36a labels
  plus the Groves intersection/splay proximities reverse that: +0.149 in-box, interval
  clear of zero. The restriction-of-range alarm that flagged round-1 does not fire here.
- **struct_generic on gems:** the typed-label version of the round-1 +0.106 screen. Its
  point estimate is higher (+0.138) and its interval clears zero, so the typing sharpened
  the generic screen as well, not only the Groves arm.

## What the interval establishes, and what it does not

The bootstrap resamples the 48 positives and the 881 background cells in the mapped
footprint. It quantifies how much the +0.170 depends on *which* of those occurrences and
background cells were drawn. The interval is narrow enough, and far enough from zero, that
the structural signal is not an artefact of a few lucky positives. P(Δ>0) = 1.000 means the
ordering (Groves arm beats base) is stable across every resample.

It does **not** establish three things, all carried over from the rebuild report:

1. **Generalization beyond the GeMS footprint.** The 48 positives and the structure bands
   both live inside central-Nome's mapped district. The bootstrap cannot tell whether the
   control is district-specific or peninsula-wide; only re-testing on a wider structure
   map (SIM 3131 / RI 2024-7) and dispersed labels can, and that is the separate
   peninsula-wide test.
2. **Calibration.** AUC is rank-only. A served probability surface needs the band
   thresholds checked against the actual occurrence density, the same open item the placer
   surface carries.
3. **Label cleanliness past the model_code typing.** 36a is the Cox-Singer code for
   low-sulfide Au-quartz veins; a mistyped or relocated ARDF record is still a mistyped
   record. The typing narrowed the set from 93 keyword hits to 54 typed positives, which is
   the main reason the signal sharpened, but it is not a guarantee of zero label error.

## Reproduce

```
PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_groves_bootstrap
```

Numbers: `data/derived/nome_placer/lode_groves_bootstrap/{lode_groves_bootstrap.json,csv}`.
Deterministic (seed 42); the background draw, fold sizing and estimator are all the ones
the rebuild driver uses, so base and arm are graded on identical folds.
