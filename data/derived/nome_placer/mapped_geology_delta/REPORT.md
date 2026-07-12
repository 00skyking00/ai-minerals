# Mapped geology/structure: presence-CV delta over the 0.679 placer base

**Question (ML step 3 Part A).** Does independently-mapped geology/structure improve
the leak-guarded placer presence model beyond the pure-geomorph baseline
(placer_onshore AUC 0.679)?

**Answer.** No, not once the sampling-effort confound is accounted for. The two
mapped fields that raise placer AUC are the two that fail the occurrence-density
independence check; the fields that pass the check add nothing. Presence skill stays
at 0.679. For the lode model the same fields are decisive, which is the expected
contrast and is discussed at the end.

Base reproduced at 0.6786 under the F1 leak-guarded recipe (variogram range 449.9 m,
900 m blocks, 1 km dead zone, contiguous folds, RandomForest(300, balanced), 65
positives + 2000 background), so every delta below is apples-to-apples against the
published 0.679.

## Inventory: what already existed, and what fed the 0.679 base

The 0.679 placer presence model (`mpm_onshore_presence_cv.py`, re-graded by
`f1_leak_guarded_rebaseline.py`) uses ten features: the seven geomorph population
bands (`bl, ap, tb, ss, bc, qm, buried_bl`) plus IfSAR `dem/slope/tpi`. **No mapped
geology or structure field is in it.** The mapped-geology rasters in
`covariates_mpm/` feed the *lode* model, not the placer base, so the "does mapped
geology help placer presence" question was genuinely open.

| field | source | provenance | in the 0.679 placer base? |
|---|---|---|---|
| `geology_unit_code_nome` | OFR 2009-1254 / SIM 3131 (1:500,000) | polygon `LABEL` rasterized | no (lode only) |
| `dist_to_fault_nome` | OFR 2009-1254 arcs, `ARC_CODE {4,30,60}` | EDT to nearest fault arc | no (lode only) |
| `akmag_nome` | OFR 97-520 composite aeromag (1 km) | GXF reprojected to grid | no (lode only) |
| `dist_to_contact_nome` | OFR 2009-1254 arcs, `ARC_CODE 1` | **built here**, EDT to nearest contact | new |
| GeMS `dist_ne/nw_fault, dist_fold_hinge, carbonaceous_host` | DGGS PDF 94-39 Nome district | `nome_structure` module | no (tested on lode only) |

A related prior result: the round-4B bedrock-contact experiment
(`placer_bedrock_contact_cv.py`) already tested a *surficial* Qs/bedrock contact
proxy against this same 0.679 base and found a null (−0.030, −0.016). The present
work tests the *bedrock* contact and fault linework, which is different linework but
the same base.

The two sources sit at opposite scales, and that matters for the confound:

- **OFR 2009-1254** is a 1:500,000 regional compilation. Over the ~17x21 km placer
  core it carries only 9 contact arcs and 4 fault arcs. Drawn as a regional
  compilation, not where anyone dug, so its confound risk is low but its power is
  also low.
- **GeMS PDF 94-39** is the Nome *mining district* map: a dense structure network and
  the graphitic-host lithology. Its footprint is drawn around the workings, so its
  extent tracks the occurrences. That is the confound the review names, and the
  numbers below confirm it.

## Independence check 1: occurrence-density confound

Line-cell density inside vs outside a 1.5 km buffer of the 65 placer occurrences,
on-land. A ratio near 1 is coverage-uniform; a ratio far above 1 means the linework
is drawn denser where the occurrences are (mapping-where-they-mined).

| field | density in / out ratio | reading |
|---|---|---|
| OFR contacts (`ARC_CODE 1`) | **2.05** | mild concentration |
| OFR faults (`{4,30,60}`) | **11.6** | strong concentration; fails |
| GeMS NE faults | 3.0 | concentration |
| GeMS NW faults | 2.4 | concentration |
| GeMS mapped footprint (extent itself) | 1.68 | coverage proxy |

The GeMS footprint covers **96.9% of the placer positives but only 77% of all
cells**, so any GeMS field scored over the whole grid mixes real structure with a
distance-to-mapped-extent proxy. That is why the GeMS arm is read on its
mapped-only subset below.

## Independence check 2: mapped faults vs the coverage-uniform aeromag

The 1 km composite aeromag (`akmag`) does not care where anyone dug, so a mapped
fault with a real magnetic expression should sit on an aeromag gradient. It does not:

- Spearman(`dist_to_fault`, aeromag intensity) = **0.02** (no relation).
- Spearman(`dist_to_fault`, aeromag horizontal gradient) = **−0.21**.
- Aeromag gradient at mapped-fault cells vs random on-land cells = **0.84** (mapped
  faults sit on *lower* gradient than random ground).

The mapped OFR fault traces carry no aeromag corroboration at this resolution over
this AOI. They are independent of the geophysics, but that independence cuts the
wrong way here: there is no deep-structure signal behind them, only 1:500,000
surface linework that happens to pass through the mining ground (check 1).

## Independence check 3: variance / distribution

Every continuous field has real spread over the grid (no distance field collapsed to
a constant): `dist_to_contact` mean 3.7 km, `dist_to_fault` mean 6.8 km, GeMS
fault-distance means 4.5 km, all with cv 0.5 to 0.9. Two categorical fields are worth
a specific note:

- `geology_unit_code`: Qs 54%, water 23%, Dcs 11%, DOx 8%, Ocs 4.5%. Real variance,
  but Qs (surficial cover) is where the placers already sit, so the geomorph terrain
  base already encodes this split.
- GeMS `carbonaceous_host`: **2.1% of cells, 1.5% of positives.** Effectively
  no-variance for a placer signal. The graphitic Nome Group is the lode host, not the
  placer ground (placers are the reworked alluvium downstream), so this field cannot
  carry placer-presence signal. Drop for placer; it belongs to the lode model.

## The presence-CV delta (placer)

Marginal leak-guarded AUC over the 0.679 base, paired bootstrap (2000), 95% interval.

| arm | AUC (full) | marginal | 95% CI | P(d>0) | independence verdict |
|---|---|---|---|---|---|
| `dist_to_contact` | 0.713 | **+0.035** | [0.007, 0.065] | 0.99 | mild confound (2.05x): suspect |
| `dist_to_fault` | 0.749 | **+0.070** | [0.018, 0.127] | 1.00 | fails check 1 (11.6x) + check 2: drop |
| `host_rock_onehot` | 0.669 | −0.010 | [−0.052, 0.030] | 0.32 | clean but no signal: drop |
| `gems_structure` | 0.690 | +0.012 | [−0.026, 0.051] | 0.72 | no signal; −0.001 on mapped-only |
| `all_ofr_geology` | 0.735 | +0.056 | [0.010, 0.103] | 0.99 | driven by `dist_to_fault`: drop |

The only arms whose CI clears zero are the two distance-to-linework fields, and both
fail or are marked suspect by the independence checks. `dist_to_fault` is the clearest
case: it raises AUC by 0.070, and it is exactly the field with an 11.6x density
concentration inside the occurrence buffer and no aeromag corroboration. With only 4
fault arcs in the AOI, all passing through the mining ground, "distance to the nearest
mapped fault" is a distance-to-where-the-occurrences-are feature in a geology costume.
`dist_to_contact` is milder (2.05x) and has a partly-genuine geological basis (the
Qs/bedrock contact is a real placer control), but at a 2x concentration the +0.035
cannot be separated from the coverage component, so it is not banked as validated
mapped-geology skill.

The independence-clean options add nothing. Host-rock one-hot is a null (the terrain
base already knows the alluvium). GeMS structure is a null on the full grid (+0.012,
CI spans zero) and a dead null on its own clean mapped-only footprint (−0.001, CI
[−0.047, 0.042]), so its small full-grid point estimate is the coverage proxy, not
structure.

**Verdict for placer presence: no trustworthy delta. Presence skill stays at 0.679.**
Drop `dist_to_fault` (fails both checks), `host_rock_onehot` (no signal), and
`gems_structure` (no signal). Treat `dist_to_contact` as suspect, not validation.

## The lode contrast

For the lode presence model the same mapped geology is decisive. Ablating the lode
base to terrain-only (`dem/slope/tpi`) and adding the mapped-geology block back:

| lode_inbox (30 positives) | AUC | marginal | 95% CI | P(d>0) |
|---|---|---|---|---|
| terrain only | 0.573 | | | |
| + mapped geology (geol one-hot, dist_to_fault, akmag) | 0.806 | **+0.233** | [0.126, 0.330] | 1.00 |

Terrain alone barely beats chance for lode; mapped geology carries essentially all of
the 0.806. This is the expected orogenic-gold control: lode occurrences sit on the
structure and lithology the maps draw. The caveat is that the same `dist_to_fault`
field carries the 11.6x occurrence-density confound, so part of the +0.233 is the same
coverage proxy the placer arm exposed. The direction (mapped structure is where lode
signal lives) is not in doubt; the clean-separable share is smaller than the point
estimate, and this in-AOI test cannot fully split structure-control from
occurrence-proximity.

## What this does not prove

- It does not say mapped geology is irrelevant to placers everywhere. It says that
  over this AOI, at these map scales, no mapped-geology field adds placer-presence
  ranking beyond the geomorph base once the sampling-effort confound is removed.
- It does not condemn `dist_to_contact` as geologically meaningless. The Qs/bedrock
  contact is a real placer control; the objection is only that its +0.035 is not
  separable from a 2x coverage concentration on 9 arcs, so it cannot be cited as
  validated skill.
- The independence checks are low-power on the OFR source (9 contacts, 4 faults) and
  on the aeromag cross-check (1 km field, 17 km AOI). A high confound ratio on tiny
  linework counts is a warning, not a precise coefficient; the direction is what the
  verdict rests on.
- The Tuck favorability field is out of scope for this round (gated behind the §4
  hindsight audit) and is not in any arm here.

## Reproduce

`PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_mapped_geology_delta`
writes `placer_mapped_geology_delta.json` (all numbers above) and
`covariates_mpm/dist_to_contact_nome_3338.tif` (the one new field).
