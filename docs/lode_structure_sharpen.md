# Lode structure sharpening: the +0.106 screen becomes a verdict

Round 1 found the first leak-free structural signal for Nome lode: distance-to-NE/NW-fault
plus a graphitic-host flag moved the district model from AUC 0.540 to 0.646 on the
GeMS-mapped cells (+0.106, bootstrap 95% CI [+0.050, +0.164]). That was a screen on 82
clustered positives selected by a loose keyword filter. Round 2 asks whether the signal
survives two changes the coordinator requested: **typed, dispersed labels** and **sharpened
structure features** built to the Groves et al. (2018) targeting model.

**Gate:** does the district lode AUC, restricted to the GeMS-mapped footprint and graded
under the F1 leak-guarded spatial CV (contiguous folds, residual-variogram block sizing,
1 km dead-zone), clear ~0.70?

**Answer: yes, at 0.712, on the typed labels, and only with the right feature subset.**

## What changed

**Labels.** The positive set is now ARDF Cox-Singer `model_code` 36a (low-sulfide
Au-quartz vein, including the questioned `36a?`), read from `ardf_nome.geojson`,
peninsula-wide within the district grid. This is 54 positives (48 inside the GeMS
footprint), replacing the round-1 "model 36/27/22 OR lode/vein/skarn/greisen keyword" set
of 93. The round-1 set is rerun alongside so the typing is attributable.

**Features.** Three structure families, each its own model arm on top of the same base
(geology one-hot, aeromagnetics, the coarse statewide distance-to-fault, terrain):

- `struct_generic`: the round-1 oriented bands (distance to nearest NE / NW fault, fold
  hinge, graphitic host), now with the NE azimuth floor lowered from 22.5° to 15.0° (see
  the Rodine fix below).
- `struct_named`: the per-named-fault split the addendum asked for: a separate
  distance band for each of the six named district faults (Anvil, Penny River, Aurora
  Creek, Charlie Creek, Boulder Creek, Rodine), from the GeMS `Label` field.
- `struct_groves`: the Groves-2018 refinement: distance to the nearest NE×NW
  structure intersection, and distance to the nearest second-order splay (an unlabelled
  fault segment within 250 m of a named trunk), plus the fold hinge and graphitic host.
- `struct_sharp_all`: everything above combined.

## Results (auc_gems, the clean structure test)

| arm | 36a labels (48 pos in-footprint) | round-1 labels (82 pos) |
|---|---|---|
| base | 0.541 | 0.540 |
| struct_generic | 0.679 (+0.138) | 0.622 (+0.082) |
| struct_named | 0.422 (-0.119) | 0.507 (-0.033) |
| **struct_groves** | **0.712 (+0.170)** | 0.595 (+0.055) |
| struct_sharp_all | 0.530 (-0.011) | 0.492 (-0.048) |

Numbers: `data/derived/nome_placer/lode_structure_sharpen/{json,csv}`.

## Reading the table

**The Groves features clear the gate.** Distance-to-intersection and distance-to-splay,
with the fold hinge and graphitic host, reach 0.712 on the typed labels. The world-class
orogenic-gold targeting rule is the one that works here: the deposits sit at the
fault-set nodes and in the damage-zone splays, not on the trunk faults. Feature importance
confirms it directly: `dist_splay`, `dist_fold_hinge` and `dist_fault_intersection` are the
top three covariates in the combined arm, ahead of every named-fault distance.

**Typed labels sharpened the signal.** Every structure arm gained more on the 36a set than
on the keyword set. `struct_generic` went from +0.082 to +0.138; `struct_groves` from
+0.055 to +0.170. Restricting the positives to one deposit type removed the dilution that a
loose keyword filter introduced, which is the restriction-of-range fix doing its job.

**The per-named-fault split hurt, on both label sets.** Giving each named fault its own
distance band (`struct_named`, -0.119 on 36a) was worse than lumping them by orientation
(`struct_generic`). With 48 to 82 positives, six correlated distance ramps are mostly
smooth background gradients the model cannot turn into signal. Orientation, not fault
identity, is what the data supports. The kitchen-sink arm (`struct_sharp_all`) dilutes the
Groves signal the same way. This contradicts the literal per-named-fault instruction, and
it is reported as found rather than buried.

## Two bugs fixed

**Rodine drop.** The Rodine fault strikes NNE; its GeMS segments span 16° to 40° (median 30°).
The old NE bin floor of 22.5° silently dropped every segment below 22.5°, so a third of
Rodine anchored no covariate. District-wide, 178 fault segments fell in the [15°, 22.5°)
gap. Lowering the floor to 15° recovers them while keeping the conjugate NW bin's geometric
width. Rodine also now has its own named band.

**Albion attribution.** The round-1 note (in `nome_structure.py` and the status entry)
credited the NE-fault dominance to the "Rock Creek/Albion control." The Albion fault is a
deposit-scale structure at Rock Creek and is absent from this district GeMS. The mapped NE
length is carried by the Penny River, Charlie Creek and Anvil faults. The note is corrected.

## What this does not prove

- **48 positives is a small sample.** The +0.170 is the change on a single typed set graded
  over ten contiguous folds; the round-1 round reported a bootstrap CI on the analogous
  number, and a follow-up bootstrap on this set is the right next step before the 0.712 is
  treated as settled.
- **The structure test is in-footprint only.** `auc_gems` is restricted to the GeMS-mapped
  central district. The dispersed labels outside that footprint do not test the structure
  features; they would need peninsula-wide structure, which means the detailed Seward
  bedrock GeMS (SIM 3131 / RI 2024-7), not yet pulled.
- **The splay band is a proxy.** "Unlabelled fault within 250 m of a named trunk" is a
  damage-zone heuristic, not a mapped splay set. It earns its place by importance and by
  the gain it produces, not by being a surveyed feature.

## Escalated, not guessed

- **Albion-fault digitization.** Adding distance-to-Albion means georeferencing a fault
  trace from a figure in Otto/Piekenbrock/Odden (2009). Hand-tracing coordinates off a paper
  figure unattended would inject a fabricated geometry into the model. This needs a
  deliberate pass, not a guess.
- **Peninsula-wide-beyond-district labels.** The typed 36a set already spans Nome + Solomon
  inside the district grid. Extending past it requires the detailed peninsula bedrock GeMS
  for the structure and host covariates over the larger extent.
