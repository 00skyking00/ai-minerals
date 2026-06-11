"""Anchor districts for the northern-Sierra placer Phase 1 validation gate.

Seven historic hydraulic-mining districts on the Tertiary deep-gravel belt.
Phase 1 (`scorers/usgs_alaska_placer.py`) must place every one of them in
the top decile of the within-AOI index, without any of them appearing in
the training labels. Phase 2 (`scripts/northern_sierra_placer/train_predict_250m.py`)
masks the cells containing these centroids out of every training fold,
every PU bag, and every calibration fold.

v3 refinement (B.0): v2 used GNIS town-center coordinates. GNIS records
"Forest Hill" as the town center, which can be 200m-2km from the actual
hydraulic-pit footprint the district is named for. The v2 anchor gate
failures at Forest Hill (decile 9 in v2 calibrated) and the weak hits at
North San Juan / You Bet / Michigan Bluff traced partly to this offset.
v3 snaps each anchor to the centroid of the nearest Orlando 2016 pit
polygon within 2 km of the GNIS lookup, falling back to GNIS if no pit
polygon sits within the buffer.

Per-anchor snap distances and source polygons (computed in
scripts/v3_refine_anchor_coordinates.py):

  anchor                                method                                snap_dist_m
  Malakoff Diggins / North Bloomfield   INSIDE Malakoff Diggings polygon      386
  North San Juan                        -> Sebastopol Diggins (boundary 151m) 726
  Dutch Flat                            INSIDE Nichols Diggings polygon       147
  You Bet                               -> Hunts Hill (boundary 128m)         683
  Iowa Hill                             -> Iowa Hill polygon (boundary 65m)   231
  Forest Hill                           -> Adams Pit (boundary 1800m)        2051 *
  Michigan Bluff                        -> Big Gun Diggings (boundary 277m)   842

  * Forest Hill is the one low-confidence outlier. The historic district
    spans the divide between the North and Middle Forks of the American
    River with multiple sub-cluster pits. Adams Pit is the nearest single
    polygon but undercounts the district. Lindgren PP73 plate 2 has the
    full Forest Hill Divide pit footprint; digitizing it is queued for v4.

The v2 GNIS coordinates are preserved in `_GNIS_COORDINATES` below for
diagnostic comparison + as the fallback if you want to re-run the v2
gate.
"""

from __future__ import annotations


# v2 GNIS town-center coordinates. (lon, lat) in WGS84.
_GNIS_COORDINATES: dict[str, tuple[float, float]] = {
    "Malakoff Diggins / North Bloomfield": (-120.910, 39.371),
    "North San Juan":                      (-121.105, 39.371),
    "Dutch Flat":                          (-120.842, 39.207),
    "You Bet":                             (-120.913, 39.241),
    "Iowa Hill":                           (-120.851, 39.103),
    "Forest Hill":                         (-120.829, 39.020),
    "Michigan Bluff":                      (-120.741, 39.039),
}


# v3-refined: nearest Orlando 2016 pit polygon centroid within 2 km of GNIS.
# Downstream code (phase1_index, train_predict, validation) reads from here.
ANCHOR_DISTRICTS: dict[str, tuple[float, float]] = {
    "Malakoff Diggins / North Bloomfield": (-120.914297, 39.371977),
    "North San Juan":                      (-121.113098, 39.372812),
    "Dutch Flat":                          (-120.843047, 39.208039),
    "You Bet":                             (-120.909771, 39.235386),
    "Iowa Hill":                           (-120.853609, 39.103461),
    "Forest Hill":                         (-120.841915, 39.035472),
    "Michigan Bluff":                      (-120.731263, 39.038777),
}
