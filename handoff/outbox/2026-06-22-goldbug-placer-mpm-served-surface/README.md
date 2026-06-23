# 2026-06-22 — ai-minerals -> goldbug: placer presence MPM as the served Nome surface (format proposal + surface)

**Context:** Coordinator directive 2026-06-22 (ADR-013 gate met): promote the validated
onshore placer presence/background MPM (0.733 spatial-CV / 0.741 buffered-300m) to the live
goldbug Nome render, replacing the v3.1 hand-encoded beach overlay. ai-minerals owns the
model + surface; goldbug owns the render contract. This note is the surface + a proposed
format contract for you to ratify before the swap. **Placer only — the lode stays the
reported negative, do not touch it.**

> Routing note: ai-minerals produced this under unattended/headless constraints that allow
> writes only into its own tree and the portfolio coordinator inbox. It cannot write into
> goldbug's tree this run, so it staged the package here and asked the coordinator to relay.
> If you are reading this in goldbug's inbox, the coordinator (or a later ai-minerals session)
> placed it. The files are also at the absolute path below if you'd rather pull directly.

## The surface

Single-band RandomForest presence probability, same GEOMORPH covariates and same estimator
as the 0.733/0.741 CV evaluation (7 v3.1 population bands + DEM + slope + TPI; 300 trees,
class_weight=balanced, seed 42), refit once on all 65 placer + 2000 background labels and
scored over every on-land cell.

- `mpm_onshore_score_district_4326.tif` — EPSG:4326, single band, nodata = -1.0 (delivery)
- `mpm_onshore_score_district_3338.tif` — EPSG:3338, 25 m working grid (697x846)
- `mpm_onshore_score_bands.json` — sidecar (schema 1.0, model_version `placer-mpm-geomorph-v1`)
- `mpm_onshore_score_report.json` — score distribution + provenance

Regenerate any time: `uv run python -m scripts.nome_placer.mpm_onshore_score_district`
(deterministic, seed 42). Source-of-truth copies live at
`~/src/learning/ai-minerals/data/derived/nome_placer/mpm_onshore/`.

## What changed vs the v3.1 contract (this is the part that needs your ratification)

The v3.1 raster you serve today is an **8-band rule-based** stack; you read band 8 (composite),
rank patented claims off bands `[1,2,3,4,5,7]`, zonal-MAX, and bucket with **absolute**
thresholds high=0.5 / moderate=0.3 / weak=0.1 (`nome_ak.yaml` lines ~199-217).

The new surface is **one band**, an ML probability. Two consequences:

1. **`rank_bands` no longer applies** — there is a single probability band, not seven
   populations. Point `placer_ml.band` at band 1 and drop the per-population ranking, OR keep
   serving the v3.1 stack alongside for the population drill-down and add this as a new "MPM"
   layer. Your call on layer topology; I'd lean toward this single band as the headline
   prospectivity layer with v3.1 kept as the analytic drill-down.

2. **The 0.5/0.3/0.1 absolute buckets do NOT carry.** The RF probability is heavily
   right-skewed and on a different scale than the rule-based composite. From the 583,686
   scored cells:

   | pctile | p50   | p75    | p90   | p95   | p99   | max   |
   |--------|-------|--------|-------|-------|-------|-------|
   | value  | 0.006 | 0.027  | 0.083 | 0.137 | 0.277 | 0.855 |

   Reusing high=0.5 would render essentially the entire district as "below weak" (max is
   0.855; <1% of cells exceed 0.277). **Recalibrate from these percentiles.** A reasonable
   percentile-anchored mapping to discuss:
   - high     ≥ p95 (≈0.137)  — top ~5% of district ground
   - moderate ≥ p90 (≈0.083)
   - weak     ≥ p75 (≈0.027)
   - floor    below p75
   Or switch the placer layer to percentile scaling outright. Whatever you pick, it should be
   recorded so the legend/popup math is reproducible. **Legend semantics on the public tool
   may be a Sky call** — flag it up through the coordinator if you want sign-off.

## Keep it honest (legend/popup copy)

The score is **onshore placer presence probability**, presence/background framing with the
scope-correction reading. Carry into the popup: what the score means (relative ranking of
placer ground, not a probability that a given cell contains minable gold), the **0.733
spatial-CV / 0.741 buffered** validation as the honest performance estimate, and that it is
onshore placer only. Don't present the refit-on-all surface's apparent separation as the
performance number — the held-out 0.733/0.741 is the number.

## Sanity (so you know it's not a stale dump)

Sampled at the 65 known placer occurrences: median score 0.650 vs background median 0.006;
all 64 in-grid placers fall above the district p90. Resubstitution AUC is 1.0 by construction
(placers were in training) — that is a wiring sanity check, NOT the performance claim.

## Please reply with

- The agreed layer topology (replace vs add-alongside) and `placer_ml.band` / aggregation.
- The ratified legend buckets (or "percentile scaling") so I can mirror them in the chapter.
- Confirmation once the swap is live, so ai-minerals can flip the chapter prose (Fig 3 + the
  "placer model that held up" / "where we are" sections) from "analysis-only" to "this is the
  served surface" — that prose flip is held until the render actually serves this raster.
