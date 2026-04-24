"""BC Golden Triangle — v2 region config (stub; populated in phase B)."""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


# Placeholder AOI — to be refined from BC MINFILE property polygons in phase B1.
_BCGT_AOI = AOI(
    name="BCGT",
    min_lon=-131.5, min_lat=56.0,
    max_lon=-129.5, max_lat=57.5,
)


BCGT = Region(
    slug="bcgt",
    aoi=_BCGT_AOI,
    working_crs="EPSG:3005",     # BC Albers Equal Area
    data_prefix="bcgt",

    # Adapters not yet implemented — phase B populates these.
    occurrences_source="bc_minfile",
    geochem_source="bcgs_rgs",
    geology_source="bcgs_digital",
    geophysics_source="gsc",
    drillhole_source="bcgs_geofile",

    deposit_classes={
        "porphyry_family": ("bc:L03", "bc:L04"),
        "epithermal":      ("bc:H04", "bc:H05"),
        "skarn":           ("bc:K01", "bc:K02"),
    },

    occurrence_commodity_filter=("cu", "copper", "au", "gold", "ag", "silver"),

    # Add Hg/Tl/Ba for epithermal pathfinders.
    pathfinder_elements=("Ag", "As", "Au", "Bi", "Cu", "Mo", "Pb", "Sb", "Te", "Zn", "Hg", "Tl", "Ba"),

    raw_paths={
        # populated in phase B as BC fetch modules land
    },
)
