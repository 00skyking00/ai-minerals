# Does the lode structure control generalize past central Nome?

The 0.712 gate (`struct_groves`, typed 36a labels) was built and graded entirely
inside the central-Nome district GeMS. The labels clustered there and the
structure features existed only there. The open question the coordinator gated on:
is that control district-specific, or do the same NE×NW intersection / splay /
fold-hinge / graphitic-host features predict 36a lode occurrences across the wider
Seward Peninsula?

To answer it, the Groves features were rebuilt on two bedrock maps that reach past
central Nome, the 36a labels were dispersed peninsula-wide, and the model was
re-graded under the same F1 leak-guarded CV.

**Answer: it generalizes. The structure features reach AUC 0.82 to 0.89 on dispersed
peninsula-wide labels, including a held-out eastern cluster the central model never
saw. The struct-over-base delta clears zero on both the central district (+0.24)
and the pooled wider extent (+0.37). The one limit is statistical power in the
east, where the Groves splay/intersection refinement specifically cannot be
separated from a lithology+fault base on nine positives.**

## Sources and design

**Structure beyond central Nome.** Two public bedrock maps:

- **SIM 3131** (Till et al. 2011), Seward Peninsula bedrock, 1:500,000, the Nome
  (`nm`) and Solomon (`so`) quadrangles. Faults are ARC_CODE 4/30/60; graphitic
  host LABELs are DOx and Dcs (the units whose unit description says "graphitic").
  NAD27/UTM-3N, reprojected to EPSG:3338 through the NADCON grid.
- **RI 2024-7** (Werdon et al. 2024), the detailed Big Hurrah-Council-Bluff
  (Casadepaga) map, AK GeMS schema. This supplies the eastern detail SIM 3131
  lacks: 1835 fault features, 223 fold axes, and nine graphitic map units.

**Labels.** ARDF `model_code` 36a from the full statewide ARDF (not the Nome
clip). 65 positives fall in the wider grid (150×111 km), 54 in the central
district. Nine sit east of the central district right edge: the Solomon-quadrangle
Council/Casadepaga cluster (Bunker Hill, Koyana, Saddle, Daniels, the Idaho-Eskimo
group), the held-out generalization target.

**Matched base.** So the central and wider deltas are comparable, the base is built
identically on both grids: SIM 3131 geology one-hot plus distance-to-any-regional-
fault. It is deliberately terrain-free and aeromag-free (those are not available on
both grids without a heavy rebuild). The absolute AUC therefore differs from the
0.712, which used terrain and aeromag. What is being measured is the
**struct_groves delta** and whether it clears zero, not the absolute level.

**Splay caveat.** Regionally there are no named fault trunks (SIM 3131 faults carry
no name), so the splay band uses a top-quintile fault-length trunk proxy rather
than the central module's named-trunk rule. The intersection, fold-hinge and host
bands transfer unchanged.

## Numbers (F1 leak-guarded CV, contiguous folds, 1 km dead-zone, seed 42)

| grid | arm | auc_mapped | Δ vs base | 95% CI (mapped) | auc_east | Δ_east | 95% CI (east) |
|---|---|---|---|---|---|---|---|
| wider | base | 0.513 | n/a | n/a | 0.714 | n/a | n/a |
| wider | struct_generic | 0.813 | +0.300 | [+0.237, +0.367] | 0.887 | +0.173 | [+0.015, +0.341] |
| **wider** | **struct_groves** | **0.883** | **+0.370** | **[+0.306, +0.436]** | **0.818** | +0.104 | [-0.073, +0.282] |
| central | base | 0.598 | n/a | n/a | n/a | n/a | n/a |
| central | struct_generic | 0.814 | +0.216 | [+0.135, +0.295] | n/a | n/a | n/a |
| **central** | **struct_groves** | **0.837** | **+0.239** | **[+0.154, +0.320]** | n/a | n/a | n/a |

`auc_mapped` is the OOF AUC on mapped cells (SIM 3131 covers the whole grid, so it
is nearly the full extent). `auc_east` is the OOF AUC on cells east of the central
district, i.e. the held-out Solomon/Council occurrences. Paired bootstrap, 2000
resamples, the same `boot_delta` as the round-1 and gate CIs.

## Reading the result

**The structure features predict eastern occurrences.** On the nine held-out
eastern positives, `struct_groves` reaches AUC 0.818 and `struct_generic` 0.887.
A model whose training mass is central and western Nome ranks the eastern
Council/Casadepaga lodes well above eastern background. That is the direct test of
generalization, and it passes.

**The within-region delta is real on uniformly-mapped ground.** On the central
district (+0.239) and on the pooled wider extent (+0.370) the struct_groves gain
clears zero with the whole bootstrap interval above zero. The structure control is
not an artefact of the central labels.

**Where the power runs out.** The eastern *marginal* test is the weak spot.
The base (geology + fault distance) already scores 0.714 in the east, so the room
for structure to add is smaller, and on nine positives the `struct_groves` eastern
delta (+0.104) has a CI that crosses zero (P(Δ>0)=0.87). The `struct_generic`
eastern delta (+0.173) does clear zero (P=0.99). Read together: the NE/NW
orientation signal transfers with significance; the Groves splay/intersection
refinement transfers in absolute AUC but its marginal gain over a lithology+fault
base cannot be confirmed on the eastern labels available.

**Two caveats on the pooled +0.37.** First, the wider base is weaker than the
central base (0.513 vs 0.598), because the coarse SIM 3131 geology one-hot carries
less over the wider extent, so part of the larger wider delta is a weaker base, not
a stronger structure. Second, SIM 3131 (1:500,000) maps the west coarsely and
RI 2024-7 maps the east in detail, so a distance-to-structure feature is
systematically smaller in the detailed east. The contiguous-fold leak guard and the
within-east `auc_east` control this, but the pooled +0.37 still mixes a mapping-
density gradient with structural signal. The matched central +0.24 and the eastern
transfer AUCs are the numbers to trust.

## What this does and does not settle

Settles: the structural control is not central-Nome-specific. The same features
predict eastern lode occurrences they were not trained on, and the within-region
delta is positive on two independent footprints.

Does not settle: whether the Groves splay/intersection refinement specifically
(as opposed to plain NE/NW orientation) generalizes, which needs more than nine
eastern positives; and the absolute peninsula-wide AUC, which would need the
matched terrain + aeromag the 0.712 used, rebuilt on the wider grid.

## Reproduce

```
PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_peninsula_generalization
```

Module `src/ai_minerals/data/nome_structure_regional.py`; numbers
`data/derived/nome_placer/lode_peninsula_generalization/{json,csv}`. Sources staged
under `data/raw/sim3131_seward/` and `data/raw/dggs_ri2024_7/` with `SOURCE.md`
provenance.
