# Does the splay/intersection refinement clear zero in the east?

Round 3 settled that the lode structure control is not central-Nome-specific: the
NE/NW orientation features predict eastern occurrences they were not trained on.
It left one question open. The Groves splay/intersection *refinement* specifically
(distance to NE×NW intersection nodes and to second-order splays, over and above
plain NE/NW fault proximity) could not be separated from a lithology+fault base on
the nine eastern positives then available. The eastern marginal gain was +0.104
with a 95% CI of [-0.073, +0.282], spanning zero.

This round expands the eastern label set and re-grades the same gain.

**Answer: it does not confirm. With the eastern set expanded to eleven typed
positives, the splay/intersection refinement gain in the east is -0.047, 95% CI
[-0.189, +0.096]. The point estimate is now negative and the interval still spans
zero. The plain NE/NW orientation arm continues to transfer (+0.119, P(Δ>0)=0.97).
Read together: the orientation control generalizes east; the splay/intersection
refinement is Nome-district-specific. The central-district served surface stands,
and nothing is promoted peninsula-wide.**

## Expanding the eastern labels

The constraint round 3 named was statistical power, so the first task was to add
eastern positives without loosening the typed-label discipline that fixed the F3
occurrence-feature leak. Four things were checked:

- **ARDF model_code 36a** is already pulled exhaustively. The full statewide ARDF
  holds nine 36a-coded occurrences east of the central district edge (x = -473300)
  inside the wider grid, and round 3 used all nine. A tenth, Otter Creek
  (x = -394085), sits just beyond the grid's eastern edge; it is one positive and
  is reported as an available extension, not folded into the comparable test.
- **Disciplined text recovery** adds the occurrences ARDF left uncoded. Records
  whose deposit-model text reads as a gold-bearing quartz vein, with antimony,
  scheelite, tungsten, skarn, polymetallic (22c), galena, base-metal, fluorite,
  calcite, replacement and placer descriptions all excluded, recover exactly two
  eastern lodes: Lower and Upper Crooked Creek in the Council area, both described
  as "gold-bearing quartz veins in schistose marble." The nearby 22c polymetallic
  veins and the antimony and galena occurrences are correctly rejected. This is the
  textbook low-sulfide Au-quartz signature, not a widened keyword net.
- **MRDS adds nothing.** The only Alaska MRDS extract on disk covers Interior and
  eastern Alaska, with zero Seward Peninsula records. A fresh western-Alaska MRDS
  pull would be a separate data-acquisition task.
- **The named districts are mostly already central.** The Big Hurrah and Casadepaga
  lode sites the brief named sit at x = -487000 to -490000, west of the eastern
  hold-out boundary, so they are already inside the central training set, not new
  eastern positives.

The result is eleven eastern positives (nine 36a plus the two Crooked Creek lodes),
all on RI 2024-7 or SIM 3131 mapped ground. The grid, structure bands and CV are
the round-3 ones unchanged, so the only thing that moves between rounds is the
eastern label count.

One limit carries through the rest of this report: the eastern Seward Peninsula is
genuinely label-poor for low-sulfide Au-quartz lode. Going from nine to eleven is
the most the on-disk public data supports under typed discipline. The test below is
better powered than round 3, not well powered.

## Numbers (F1 leak-guarded CV, contiguous folds, 1 km dead-zone, seed 42)

| grid | arm | auc_mapped | Δ vs base (mapped) | auc_east | Δ_east | 95% CI (east) | P(Δ_east>0) |
|---|---|---|---|---|---|---|---|
| wider | base | 0.494 | n/a | 0.800 | n/a | n/a | n/a |
| wider | struct_generic | 0.828 | +0.334 | 0.919 | +0.119 | [-0.008, +0.269] | 0.97 |
| **wider** | **struct_groves** | **0.874** | **+0.380** | **0.752** | **-0.047** | **[-0.189, +0.096]** | **0.25** |
| central | base | 0.695 | n/a | n/a | n/a | n/a | n/a |
| central | struct_generic | 0.824 | +0.129 | n/a | n/a | n/a | n/a |
| **central** | **struct_groves** | **0.856** | **+0.161** | n/a | n/a | n/a | n/a |

`struct_generic` adds plain NE/NW fault distance plus fold-hinge and graphitic-host
bands to the base; `struct_groves` swaps the NE/NW fault distances for the splay and
NE×NW-intersection bands, keeping fold-hinge and host. The east column is the OOF
AUC on the eleven held-out eastern occurrences. Paired bootstrap, 2000 resamples,
the same `boot_delta` as the round-1 and gate CIs.

## Reading the result

**The orientation control transfers; the refinement does not.** In the east the
generic NE/NW arm gains +0.119 over base with P(Δ>0)=0.97, almost clearing zero on
eleven positives. Swapping in the splay and intersection bands moves the eastern
gain to -0.047 with P(Δ>0)=0.25. The refinement that sharpens the central model
does not add over a lithology+fault+orientation base in the east, and on this
sample it slightly hurts.

**The two added positives are what moved it.** Round 3's nine 36a-coded positives
gave +0.104 in the east. Adding the two Crooked Creek lodes pulled the gain to
-0.047. Those two genuine low-sulfide Au-quartz occurrences do not sit near the
modeled NE×NW intersection nodes or splays, so including them lowers the
refinement's eastern AUC. That is a signal about where eastern lodes actually sit,
not a labeling error: the Crooked Creek lodes are correctly typed.

**The central control is unchanged.** With the disciplined labels the central
struct_groves gain is +0.161, CI [+0.093, +0.227], whole interval above zero. The
central-district result that the served surface rests on does not depend on the
label-set change.

**A caveat on the small eastern base.** The eastern base AUC is already 0.800, so
the room for any structure arm to add in the east is narrow, and eleven positives
is a thin sample for a marginal test. The correct statement is "cannot confirm, and
the point estimate is negative," not "proven absent." But the direction is now
consistent across two rounds and the refinement has had its expansion.

## What this does and does not settle

Settles: the question round 3 left open. The Groves splay/intersection refinement is
Nome-district-specific. It does not clear zero in the east on the expanded typed
label set, so there is no case for promoting a peninsula-wide splay/intersection
lode surface over the central-district one. The central served surface stands.

Does not settle: whether a materially larger eastern label set (a fresh
western-Alaska MRDS pull, or the full ARDF JSON deposit-type text mined for
uncoded Au-quartz) would change the picture. The eastern Seward Peninsula simply
holds few mapped low-sulfide Au-quartz lodes, and that data limit, not the model,
is what bounds this test.

## Reproduce

```
PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_eastern_splay_confirm
```

Numbers in `data/derived/nome_placer/lode_eastern_splay_confirm/{json,csv}`.
Reuses `src/ai_minerals/data/nome_structure_regional.py` and the round-3 wider grid;
sources staged under `data/raw/sim3131_seward/` and `data/raw/dggs_ri2024_7/`.
