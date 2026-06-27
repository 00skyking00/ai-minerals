# Can RI 2024-6 supply the drift-on-beach overlay? No: wrong map.

The round-4 brief asked to rebuild the drift-on-beach placer feature on the
RI 2024-6 surficial vector, on the stated premise that RI 2024-6 "covers central
Nome and maps the Nome River drift as discrete polygons." Round 3 had walled the
same feature on AOF 125, which lies east of the placer grid with no overlap. The
first step was to check the premise before building anything.

**The premise does not hold. RI 2024-6 is the Casadepaga / Big Hurrah-Council Bluff
surficial map, the eastern companion to the RI 2024-7 bedrock map, not a central
Nome map. Its mapped extent sits east of the Cape Nome placer grid with zero
overlap, and its nearest "Drift of Nome River Age" (Qdn) polygon is about 71 km
from the placer grid. RI 2024-6 cannot supply the strandline×drift intersection for
the central placer model. No model was built; this is reported as a wall and
escalated.**

## What was checked

The downloaded RI 2024-6 GeMS shapefile loads cleanly. The check reprojected its
`GM_MapUnitPolys` layer to EPSG:3338 and measured it against the placer model grid
(the IfSAR DEM the placer features are built on).

| layer | extent (EPSG:3338) | overlaps placer grid? |
|---|---|---|
| placer model grid | [-550400, 1655025, -532975, 1676175] | n/a |
| RI 2024-6 surficial | [-500004, 1656148, -450930, 1716190] | no |
| AOF 125 (round 3) | [-527199, 1530424, -316579, 1684643] | no |

RI 2024-6's western edge (x = -500004) is about 33 km east of the placer grid's
eastern edge (x = -532975). The map does carry the discrete drift units the brief
wanted, including three Qdn "Drift of Nome River Age" polygons, but those are the
Nome River drift type-area lobes on the eastern map. The nearest Qdn polygon is
about 71 km from the placer grid; the nearest discrete-drift polygon of any age is
about 33 km away. Building the strandline×drift feature on RI 2024-6 would produce
an all-NODATA layer over the placer grid, the same result round 3 got from AOF 125.

The map name itself is the tell: the downloaded package is
`ri2024_006_casadepaga_surf_gems`. Casadepaga is the eastern quadrangle, the same
ground RI 2024-7 bedrock covers. RI 2024-6 and RI 2024-7 are an eastern surficial /
bedrock pair, ~70 km from the Cape Nome beach.

## Why both maps miss the placer

The Cape Nome placer model sits on the Nome-town beachline (x ≈ -540000). The two
public surficial vectors both cover ground to its east:

- **AOF 125** (Riehle 1981, "Tolstoi Point to Cape Nome") begins at Cape Nome and
  runs east, just missing the placer grid to its west, and maps no discrete glacial
  drift unit anyway (round 3 used a Qu/Qud proxy).
- **RI 2024-6** (Stevens 2024, Casadepaga) sits ~33 km further east still, and does
  map the Nome River drift as discrete Qdn polygons, but in the eastern type-area,
  ~71 km from the placer.

No surficial drift vector on disk covers the Nome-town placer beach itself. That is
the data wall, and it is a geographic gap in the public mapping, not a modeling
choice.

## Decision

Escalated to the coordinator. No drift-on-beach model is built on RI 2024-6,
because it would reproduce the round-3 AOF 125 null by construction (an all-NODATA
feature). The round-3 conclusion stands: the placer beach-line backbone already
carries the placer signal, and there is no central-Nome discrete-drift vector to
add over it. The open item for the coordinator is whether a surficial map covering
the Cape Nome beach exists (a different DGGS sheet, an older USGS surficial map, or
a digitized drift boundary from the genesis-synthesis sources); if one is found,
the round-3 `placer_drift_on_beach_cv.py` machinery runs on it unchanged.

## Reproduce

```
PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.drift_on_beach_ri2024_6_coverage
```

Numbers in `data/derived/nome_placer/drift_on_beach/drift_on_beach_ri2024_6_coverage.json`.
RI 2024-6 staged under `data/raw/dggs_ri2024_6/` (DGGS webpubs, DOI 10.14509/31054).
