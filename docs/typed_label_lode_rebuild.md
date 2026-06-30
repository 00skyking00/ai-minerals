# Does the +0.106 structure signal survive a typed/dispersed lode rebuild?

*Self-contained report. The round-1 new-layers re-baseline found that a small set
of ground-property structure features (distance to NE- and NW-trending faults kept
separate, fold-hinge proximity, a graphitic-host flag) raised the collapsed Nome
district lode model by +0.106 under leak-guarded spatial CV, but flagged it as "a
screen on 82 clustered positives, not a verdict." This rebuilds the lode target
from typed and dispersed labels and re-grades the same features. The question:
does the structure signal survive, and does it promote from screen to a usable
layer? All numbers here are read from committed, seeded outputs of four merged
PRs; no model is re-run.*

## The result in one line

It survives. On a typed 36a target, over the same terrain-plus-aeromag base and
the same clean cross-validated subset as the +0.106 anchor, the four named
features give +0.138 (95% CI +0.063 to +0.209), so the interval clears zero and
the screen becomes a confirmed marginal signal. The four features alone reach
auc_gems 0.679, just short of the 0.70 usable gate; adding the documented
NE×NW fault-intersection and splay control (Groves et al. 2018) clears it at
0.712 (+0.170, CI +0.108 to +0.234). The direction also holds on dispersed
statewide and disciplined-typed labels.

## Why a rebuild was needed

The +0.106 was measured on the round-1 lode target: every ARDF record whose
Cox-Singer model code starts 36/27/22 *or* whose text matches
lode/vein/skarn/greisen, clipped to the district from the Nome ARDF subset. That
net is broad (it folds polymetallic and skarn vein families in with the
low-sulfide gold-quartz veins the control is built for) and the positives cluster
in the district core, where the structure map also has its footprint. A delta on
that target is exposed to two confounds at once: label noise from the mixed vein
families, and the coverage proxy from positives sitting where the map exists. The
round-1 report named the fix as the real next step: type the labels to the
gold-quartz model and disperse them, then re-grade.

## The rebuild design

Three changes, each isolating one variable, every arm under the same F1
leak-guarded CV (RandomForest(300, balanced, seed 42); contiguous blocks at twice
the base-model residual-variogram range; 1 km dead-zone; fold geometry fixed per
dataset so a delta is the features, not a shifted partition; clean delta read on
the mapped subset; 2000-resample paired bootstrap).

1. **Typed labels, matched base (round 2).** Replace the broad net with ARDF
   model_code 36a (low-sulfide Au-quartz vein), keeping the same terrain + aeromag
   + geology base and the same `auc_gems` clean subset as the +0.106 anchor. This
   is the apples-to-apples test: only the label definition moves.
2. **Dispersed statewide labels (round 3).** Rebuild the same feature *names*
   from regional maps (SIM 3131 + RI 2024-7) that reach beyond the district, and
   grade on the full statewide-ARDF 36a set, wider and central grids. The matched
   base here is deliberately terrain-free and aeromag-free so it builds identically
   on both grids; that weakens the base and inflates the structure delta, so this
   arm tests *direction and generalization*, not magnitude.
3. **Dispersed plus disciplined typing (round 4a).** Add a disqualifier-guarded
   recovery clause that pulls in uncoded gold-quartz-vein occurrences while
   rejecting Sb / scheelite / skarn / polymetallic / base-metal text, then re-grade.

## Results

Clean-subset AUC (`auc_gems` on the matched-base arms, `auc_mapped` on the regional
arms), the delta over each arm's own base, and the paired-bootstrap 95% interval.

**Matched base (terrain + aeromag + geology), district grid, `auc_gems`:**

| arm | target | n_pos (mapped) | base | with features | Δ | 95% CI | P(Δ>0) |
|---|---|--:|--:|--:|--:|:--:|--:|
| struct_generic | round-1 broad net | 82 | 0.540 | 0.646 | +0.106 | +0.050 to +0.164 | 0.999 |
| struct_generic | **typed 36a** | 48 | 0.541 | 0.679 | **+0.138** | +0.063 to +0.209 | 0.9995 |
| struct_groves | **typed 36a** | 48 | 0.541 | **0.712** | **+0.170** | +0.108 to +0.234 | 1.000 |

`struct_generic` is the four named features. `struct_groves` adds NE×NW
fault-intersection and second-order-splay proximity to them: the orogenic-gold
control places deposits in the splays and intersections, not on the trunk.

**Crippled regional base (terrain-free, aeromag-free), dispersed labels,
`auc_mapped` delta** (direction and generalization, not magnitude):

| labels | grid | n_pos (mapped) | struct_generic Δ | 95% CI | struct_groves Δ | 95% CI |
|---|---|--:|--:|:--:|--:|:--:|
| statewide 36a | wider | 64 | +0.300 | +0.237 to +0.367 | +0.370 | +0.306 to +0.436 |
| statewide 36a | central | 54 | +0.216 | +0.135 to +0.295 | +0.239 | +0.154 to +0.320 |
| 36a + recovery | wider | 67 | +0.334 | +0.269 to +0.399 | +0.380 | +0.302 to +0.457 |
| 36a + recovery | central | 55 | +0.129 | +0.064 to +0.192 | +0.161 | +0.093 to +0.227 |

Every dispersed delta clears zero with P(Δ>0) at or near 1.0. The deltas are
larger than the matched-base numbers because the regional base is near-random by
construction; read them for sign and significance, not size.

## The verdict

**Survives, and promotes.** Typing the labels does not erase the structure signal,
it sharpens it: the same four named features move from +0.106 on the noisy broad
net to +0.138 on the typed 36a target, and the bootstrap interval clears zero on
both. The screen is now a confirmed marginal signal.

On the usable-layer gate (auc_gems > 0.70): the four named features alone land at
0.679, just short. The geology says why, and supplies the fix. The control is not
bare fault proximity but the splays and intersections of the NE and NW fault sets;
adding those two proximities (`struct_groves`) takes the typed model to 0.712, past
the gate, with a CI of +0.108 to +0.234. So the usable layer is the typed target
plus the intersection/splay features, and the four generic distances are the floor
beneath it.

The direction is not a district artifact. Dispersed to statewide-ARDF 36a labels
and to the disciplined-recovery set, on regional structure maps that reach past
central Nome, `struct_generic` stays positive on every grid with intervals above
zero. The held-out eastern subset (Solomon/Council, beyond the central grid the
model never trained on) carries the same sign once the eastern labels are powered
enough to measure.

## What this does and does not establish

**Establishes.** Under leak-guarded spatial CV, the structure control the round-1
screen pointed at is real on a properly typed lode target: the four named features
clear zero (+0.138, +0.063 to +0.209) over a matched base, and with the documented
intersection/splay control the typed model reaches a usable auc_gems of 0.712. The
features are ground properties that exist independent of the occurrence labels, so
this is not the F3 self-marking confound. The signal generalizes off the district.

**Does not establish.** That the district lode model is solved: 0.712 is a usable
marginal layer, not a sited drill target, and the clean test still lives inside the
mapped structure footprint, so cells with no structural mapping are still scored
blind. That the magnitude transfers: the dispersed deltas (+0.13 to +0.38) ride a
deliberately weakened base and are not comparable to the matched-base +0.138 /
+0.170. That typing made the target larger: typing the district set to 36a makes
it smaller and cleaner (54 positives versus the broad net's 93); "larger and less
restriction-prone" comes from the dispersal to the wider grid, not from typing. And
nothing here changes the placer story, which the geophysics did not move and which
waits on the deferred beach-line LiDAR.

## Provenance and reproduce

No model is re-run in this consolidation. The AUCs and intervals are read from the
committed outputs of the merged rebuild PRs:

```
# regenerate the consolidated JSON from the four rounds' committed outputs
PYTHONPATH=src python -m scripts.nome_placer.typed_label_lode_rebuild_consolidate
# -> data/derived/nome_placer/typed_label_lode_rebuild/typed_label_lode_rebuild.json
```

Sources, each with its own driver and seeded CV:

- round 1 anchor: `scripts/nome_placer/newlayers_geophys_rebaseline.py`,
  `scripts/nome_placer/newlayers_bootstrap.py`
  (`docs/newlayers_geophys_structure_rebaseline.md`).
- round 2 typed-district rebuild + bootstrap:
  `scripts/nome_placer/lode_structure_sharpen_cv.py`,
  `scripts/nome_placer/lode_groves_bootstrap.py`.
- round 3 dispersal: `scripts/nome_placer/lode_peninsula_generalization.py`.
- round 4a dispersal + disciplined typing:
  `scripts/nome_placer/lode_eastern_splay_confirm.py`.

To reproduce a round from scratch rather than read its committed JSON, run its
driver with `PYTHONPATH=src`; each is seeded (random_state=42) and reproduces its
recorded numbers.
