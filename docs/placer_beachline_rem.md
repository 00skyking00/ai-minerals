# Placer beach-line REM: the model is already a beach-line model

The plan named strandline / sea-stand geometry as the strongest placer lever not yet in
the model, and asked whether beach-line features built from the NOAA/USACE 2021 1 m
bare-earth LiDAR move the placer number. The first thing the data showed is that the lever
is already pulled: the served placer model is built on strandline-stand-elevation scores.
The new LiDAR re-derives a control the base already has, and it cannot be graded on the
labels because the LiDAR strip reaches almost none of them.

## The base already scores the strandlines

The served placer model's geomorph priors come from `features/coastal_scorer.py`:

- **BL**: cells near a true-beach stand elevation.
- **AP**: abrasion-platform / sloughover cells.
- **buried_BL**: cells where the *bedrock* sits at a documented true-beach stand elevation
  (the buried Third Beach, 80 to 90 ft under the modern surface, the Bear Cub case).

Those are strandline-proximity features, scored at the documented stand elevations. A new
distance-to-Second / distance-to-Third band is the same quantity computed a second way.

## Two tests, one answer

**Test 1 (the 1 m LiDAR, as the plan specified): underpowered.** Only **3 of 65** placer
positives fall inside the narrow NCMP coastal strip (the strip covers 4% of the inbox grid).
The strip-restricted marginal AUCs are noise on three positives:

| arm (on the LiDAR strip) | d_auc_strip |
|---|---|
| rem | -0.026 |
| strand (Second/Third/submerged) | -0.100 |
| abrasion | -0.017 |
| ew_ridge | -0.031 |
| beachline (all) | -0.210 |

A -0.21 on three positives is not a result; it is the sample size.

**Test 2 (powered): the strandline proximity recomputed on the full-coverage IfSAR DEM.**
The inbox IfSAR (25 m, NAVD88-compatible to within +1.4 m, applied as an offset) reaches all
65 positives, so its whole-extent AUC is clean (no coverage proxy). Distance-to-Second /
Third / Fourth, marginal over the POP base:

> **strand_ifsar: auc_full 0.679 -> 0.682, d = +0.003 (all 65 positives).**

Zero, with power. The documented raised-strandline control adds nothing over the base
because the BL / buried_BL priors already encode it. Numbers:
`data/derived/nome_placer/placer_beachline_rem/{json,csv}`.

## What the 1 m LiDAR is actually for

The LiDAR is not redundant in general; it is redundant *for the district presence AUC*. Its
value is the berm-scale detail the 5 m IfSAR blurs: locating a specific buried beach ridge
for a drill program (the Bear Cub buried-Third-Beach use case), not separating placer ground
from background across the district. The `nome_placer_rem` module builds that surface (REM,
per-strandline proximity, abrasion platform, the E-W shore-parallel ridge feature) and is the
infrastructure for that targeting use, kept even though it does not move the presence number.

## The one beach-line feature the base does NOT have

The drift-on-beach intersection, where a stillstand beach crosses the glacial drift (the
documented sweet spot), is genuinely new. It is not a stand-elevation score; it needs the
surficial-drift polygons. The Tolstoi Point to Cape Nome surficial map (AOF 125, DOI
10.14509/42) is now staged at `data/raw/nome_surficial_aof125/` (524 drift polygons in the
shapefile; the GeoPackage export contains empty MapUnitPolys, a DGGS export bug, so the
shapefile is the source). Building distance-to-(strandline ∩ drift) is the next placer
feature, deferred to a feature-build round because it faces the same coastal-coverage limit:
the intersection is only defined where strandlines cross mapped drift, which is the coastal
zone the occurrence labels under-sample.

## What this does not prove

- **It does not prove the strandlines are unimportant.** It proves they are already in the
  model. Removing the BL / buried_BL priors would presumably drop the AUC; that ablation was
  not run here.
- **The +0.003 is on 65 positives at 25 m.** A finer full-coverage DEM (a gridded mosaic of
  the topobathy point cloud, which also reaches the submerged stands) could change the
  margin, but the redundancy with the existing priors is the more likely reason it is flat.
- **The submerged strandlines were not graded.** The 1 m strip reaches -13.7 m (the Submarine
  / -36 to -45 ft stand) but not the deeper -55 / -70 / -80 ft stands; the IfSAR is clamped
  at 0 m. Those need the topobathy grid.
