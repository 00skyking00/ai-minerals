"""Arizona porphyry-Cu belt — Path 3 portfolio target.

The third regional MPM project, applying everything learned from
Mother Lode v3.1 and Tanacross Path 2:

  - Cox-Singer-style label cleanup on porphyry-Cu (filter to porphyry-
    family from generic Cu via dep_model regex on MRDS).
  - Magnetic-field derivatives (1VD, HGM, AS, Tilt) computed from the
    raw residual TI grid via `data.magnetic_derivatives.write_derivatives`.
  - Isostatic-residual gravity layer alongside Bouguer.
  - Cox-Singer-cleaned labels + leak-free out-of-fold spatial-block CV
    scoring as the validation default (template:
    `scripts/motherlode_path1_oof.py`).

AOI scoping decision:
  Arizona has multiple porphyry-Cu districts. For a single self-
  contained portfolio AOI, we pick the southeastern Arizona porphyry
  belt: a rectangle covering the Globe-Miami, Pima, Ray, Safford,
  and Morenci districts. This is the densest concentration of
  Laramide-age porphyry copper deposits in the United States and
  contains world-class deposits (Morenci, Ray, Sierrita, Pinto Valley,
  Resolution).

  AOI bbox: 110.4°W to 109.0°W, 32.0°N to 33.7°N (~150 km E-W, ~190 km
  N-S, total ~28,500 km²). Covers Pima, Pinal, Gila, Greenlee, and
  parts of Graham counties.

CRS: EPSG:3742 (NAD83 / Arizona Central, ftUS) is the state-plane
choice for central Arizona, but for cross-region comparability with
the Mother Lode and Klamath work we use EPSG:3310 (California
Albers) — which is also a valid equal-area projection at this
latitude band, though Arizona-centered would be marginally more
accurate. To keep the feature-frame schema consistent across all
four regions, EPSG:3310 is the choice. If accuracy matters more than
schema consistency in a follow-up, switch to EPSG:3742 here.

Data sources:
  - MRDS for Cu / Mo occurrences (porphyry-Cu label after Cox-Singer
    filter).
  - SGMC for geology with MAJOR1/2/3 fine-grained lithology.
  - NGDB for stream-sediment chemistry across the porphyry-Cu
    pathfinder suite (Cu, Mo, Pb, Zn, Au, Ag, As, Sb).
  - USGS national magnetic + Bouguer + isostatic gravity grids,
    clipped to the AOI.
  - Sentinel-2 SWIR alteration indices for Arizona's exposed
    porphyry hosts (the desert climate gives much better spectral
    visibility than the Mother Lode foothills' vegetation cover).
  - 30 m DEM from USGS National Map.

Path 3 status: Region config defined; data fetches and feature build
queued.
"""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


_ARIZONA_AOI = AOI(
    name="arizona",
    min_lon=-110.4, min_lat=32.0,
    max_lon=-109.0, max_lat=33.7,
)


ARIZONA = Region(
    slug="arizona",
    aoi=_ARIZONA_AOI,
    working_crs="EPSG:3310",
    data_prefix="arizona",

    occurrences_source="mrds",
    geochem_source="ngdb",
    geology_source="usgs_sgmc",
    geophysics_source="usgs",
    drillhole_source=None,

    deposit_classes={
        # Cox-Singer porphyry-Cu codes. Includes 17 (Porphyry Cu),
        # 21a (Porphyry Cu-Mo), 20c (Porphyry Cu, low-F / distal).
        # 21b (Porphyry Mo low-F) is Mo-dominant; we exclude it from
        # the porphyry-Cu positive label, parallel to how Tanacross
        # Path 2 cleaned the label.
        "porphyry_cu":     ("usgs:17", "usgs:21a", "usgs:20c"),
        "porphyry_mo":     ("usgs:21b",),  # tracked but not a positive
        "skarn_cu":        ("usgs:18b",),  # separate label for diagnostics
    },

    occurrence_commodity_filter=("cu", "copper"),

    pathfinder_elements=(
        # Porphyry-Cu pathfinder suite. Au and Ag are subordinate but
        # present in many porphyry systems (the "porphyry-Cu-Au"
        # subtype, model 20c). As, Sb, and W are alteration-halo
        # indicators useful for distal targeting.
        "Cu", "Mo", "Pb", "Zn", "Au", "Ag", "As", "Sb", "W", "Bi", "Te",
    ),

    raw_paths={
        "occurrences":   DATA_RAW / "mrds" / "mrds_arizona.gpkg",
        "geochem":       DATA_RAW / "ngdb" / "ngdb_sediment_arizona.gpkg",
        "geology":       DATA_RAW / "sgmc" / "sgmc_geology_arizona.gpkg",
        "geology_arcs":  DATA_RAW / "sgmc" / "sgmc_structure_arizona.gpkg",
        "dem":           DATA_RAW / "dem" / "dem_arizona.tif",
        "sentinel2":     DATA_RAW / "sentinel2" / "s2_mean_arizona.tif",
        "magnetic":      DATA_RAW / "gsc_geophysics" / "magnetic_arizona.tif",
        "gravity":       DATA_RAW / "gsc_geophysics" / "gravity_arizona.tif",
        "gravity_isostatic": (
            DATA_RAW / "gsc_geophysics" / "gravity_isostatic_arizona.tif"
        ),
        "magnetic_1vd":              DATA_RAW / "gsc_geophysics" / "magnetic_1vd_arizona.tif",
        "magnetic_hgm":              DATA_RAW / "gsc_geophysics" / "magnetic_hgm_arizona.tif",
        "magnetic_analytic_signal":  DATA_RAW / "gsc_geophysics" / "magnetic_analytic_signal_arizona.tif",
        "magnetic_tilt":             DATA_RAW / "gsc_geophysics" / "magnetic_tilt_arizona.tif",
    },
)
