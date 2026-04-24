"""California Mother Lode — v3 region config (stub; populated in phase C)."""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


# Placeholder AOI covering the Sierra Nevada foothills orogenic-gold belt.
_MOTHERLODE_AOI = AOI(
    name="MotherLode",
    min_lon=-121.5, min_lat=37.5,
    max_lon=-119.5, max_lat=40.0,
)


MOTHERLODE = Region(
    slug="motherlode",
    aoi=_MOTHERLODE_AOI,
    working_crs="EPSG:3310",     # California Albers Equal Area (NAD83)
    data_prefix="motherlode",

    # Adapters to be implemented in phase C. USGS SGMC covers CA at national scale.
    occurrences_source="mrds",
    geochem_source="ngdb",         # USGS NGDB instead of AGDB4
    geology_source="usgs_sgmc",    # USGS SGMC works nationwide
    geophysics_source="usgs",
    drillhole_source=None,         # No structured public drill-hole DB for Mother Lode

    deposit_classes={
        # Cox-&-Singer codes for orogenic/lode Au deposits.
        "orogenic_gold":    ("usgs:36a", "usgs:36b"),
        "low_sulfidation":  ("usgs:25c",),
    },

    occurrence_commodity_filter=("au", "gold"),

    # Au/As/Sb/Hg/W pathfinders dominate orogenic-Au systems.
    pathfinder_elements=("Au", "As", "Sb", "Hg", "W", "Ag", "Cu", "Pb", "Zn"),

    raw_paths={
        # populated in phase C
    },
)
