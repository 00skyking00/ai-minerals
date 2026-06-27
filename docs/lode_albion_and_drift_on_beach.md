# Two deferred feature builds: the Albion fault, and drift-on-beach

Round 2 deferred two feature builds because each had a real obstacle: the Albion
fault needed a careful georeference of a paper figure, and the drift-on-beach
placer feature needed a surficial-drift overlay. This is both of them, built and
graded. Both come back negative, and the two negatives are different kinds.

## The Albion fault: built, documented, does not help

The Albion fault is the NE-trending structure that hosts the Rock Creek sheeted-
vein deposit (Otto, Piekenbrock & Odden 2009). It is a deposit-scale structure and
is absent from the district GeMS, so the round-2 lode model never carried it. Round
2 declined to pixel-trace it from the paper figure unattended. This builds it from
the paper's explicit numbers instead, which is a documented construction rather than
a guess:

- **Control point:** the ARDF occurrence "Albion" (model_code 36a), EPSG:3338
  (-542902, 1675320). Cross-checked against Otto 2009 Fig. 3, which is drawn in
  UTM Zone 3 NAD83: the anchor converts to UTM3N N=7,166,061, landing on the
  figure's 7166000 N tick. The control point sits on the paper's own grid.
- **Strike:** azimuth 045 deg ("strikes north 45 deg east", p.951). At Nome the
  UTM zone-3 grid convergence is -0.37 deg, so the bearing maps to grid azimuth.
- **Extent:** 20 km NE ("continues to the northeast for about 20 km", p.951) plus
  3 km SW through the deposit (conservative; the SW reach is less constrained).

The trace is built in UTM Zone 3 NAD83 and reprojected to EPSG:3338. Control points
and provenance are written to `data/derived/nome_geophys/albion_fault/`
(`albion_fault_control.json`, `albion_fault_3338.geojson`). It is a straight-line
idealization at the stated strike, flagged as a first documented geometry, not a
surveyed trace.

**Result: it does not help.** Added on top of the round-2 winner (`struct_groves`,
auc_gems 0.712) and graded on the typed 36a labels:

| arm | auc_gems |
|---|---|
| struct_groves | 0.712 |
| struct_groves + dist_to_albion | 0.691 |

Marginal of `dist_to_albion`: Δ = -0.021, 95% CI [-0.074, +0.034], P(Δ>0) = 0.23.
The interval includes zero; the point is a small negative. The reason is structural
redundancy: the Albion fault is parallel and adjacent to the Anvil fault, and the
NE-oriented bands (`dist_ne_fault`, the intersection and splay proximities) already
carry that NE control. One more straight-line distance to a parallel structure adds
a noisy feature the model splits on without gain. The documented geometry is worth
keeping; as a model feature it is redundant.

## Drift-on-beach: the drift map does not cover the placer model

The Nome raised-beach model holds that gold concentrates where a raised strandline
crosses gold-bearing glacial drift. PR #33 flagged the strandline x drift
intersection as the one genuinely-new beach-line lever not already in the served
base. Building it surfaced two data problems, the second decisive.

**First, AOF 125 maps no discrete glacial-drift unit.** AOF 125 (Riehle et al.
1981) is a reconnaissance surficial map. Its units are beach (Qba/Qbv), alluvium
(Qal), intertidal (Qif), terrace (Qt), undifferentiated deposits (Qu/Qud), and
bedrock (pQb). The Qu/Qud "undifferentiated deposits" (colluvium, eolian,
lagoonal/marine, fluvial; Qud a melt-feature physiography) are the closest proxy
for glacially-influenced cover, but there is no mapped "drift" or "till" unit. So
"drift" can only be a proxy here.

**Second, and decisive: AOF 125 does not cover the placer model grid.** AOF 125
maps the coast from Tolstoi Point east to Cape Nome. In EPSG:3338 its western edge
is x=-527,199. The served placer in-box grid runs to x=-532,975, and all 65 placer
positives sit west of x=-533,429. The two do not overlap: the surficial map is
about 6 km east of the nearest placer positive. The drift-on-beach feature is
therefore undefined across the entire placer model.

| quantity | value |
|---|---|
| strandline cells on the in-box DEM | 93,726 |
| AOF 125 cells inside the placer grid | 0 |
| strandline × drift intersection cells | 0 |
| placer positives inside AOF 125 | 0 of 65 |

The marginal AUC the run reports (Δ = -0.005) is a non-result: the feature is
all-NODATA on the grid, so the model sees a constant. This is the coverage limit
the coordinator asked to handle, stronger than anticipated: not "defined only where
strandlines cross drift", but "the drift map is the wrong stretch of coast".

**What it would take.** A surficial or Quaternary-geology map that actually covers
the central-Nome placer in-box (the Snake River mouth and the Second/Third Beach
trend), with a glacial-drift unit. RI 2024-6 (the companion surficial map) is the
obvious candidate; only its PDF is in the library, not a vector. The drift-on-beach
feature should wait for that vector, not be forced onto AOF 125.

## What both negatives mean

Neither feature is a model improvement, and neither failure is a tuning artefact.
The Albion fault is real but redundant with the NE structure already scored. The
drift-on-beach feature is sound in principle but cannot be graded until a drift map
covers the placer area. Both the documented Albion geometry and the drift-on-beach
pipeline are kept for when better data (a surveyed Albion trace, an in-box drift
map) arrives.

## Reproduce

```
PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_albion_cv
PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.placer_drift_on_beach_cv
```

Numbers: `data/derived/nome_placer/lode_albion/lode_albion.json`,
`data/derived/nome_placer/drift_on_beach/placer_drift_on_beach.json`. Albion
geometry + control points: `data/derived/nome_geophys/albion_fault/`.
