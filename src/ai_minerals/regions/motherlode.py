"""California Mother Lode — orogenic-Au v3 region config.

AOI covers the Sierra Nevada foothills from south of Yosemite (~37.5°N)
up through El Dorado County (~40°N), bounded -121.5° to -119.5° W. The
core Mother Lode Belt itself is a thin N-S strip from Mariposa through
El Dorado within this AOI; we keep the surrounding terrain in the AOI
so the model can learn what does NOT host orogenic Au (Central Valley
sediments to the west, Sierra Nevada batholith to the east).

CRS: EPSG:3310, California Albers Equal Area (NAD83). The same CRS
will be used for the Klamath/Trinity AOI (Phase 5 cross-region transfer
test) so the two feature frames are directly comparable.

Data sources are all USGS / conterminous-US:
- MRDS for occurrence labels (commodity-filtered to Au + gold; MRDS
  does not consistently carry Cox-Singer dep_type, so the deposit-class
  codes below document intent rather than acting as the actual filter).
- USGS SGMC for bedrock geology (uses GENERALIZED_LITH controlled
  vocabulary).
- USGS gravity grids (Bouguer + isostatic) for geophysics.
- NRCan EMAG2 for residual aeromagnetic.
- USGS NGDB stream-sediment for geochemistry pathfinders.
- Copernicus GLO-30 for DEM.
- Sentinel-2 mosaic for alteration indices.

There is no structured public drill-hole database for California Mother
Lode (the BCGS-GeoFile-2025-11 equivalent does not exist), so
drillhole_source is None.
"""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


_MOTHERLODE_AOI = AOI(
    name="MotherLode",
    min_lon=-121.5, min_lat=37.5,
    max_lon=-119.5, max_lat=40.0,
)


MOTHERLODE = Region(
    slug="motherlode",
    aoi=_MOTHERLODE_AOI,
    working_crs="EPSG:3310",
    data_prefix="motherlode",

    occurrences_source="mrds",
    geochem_source="ngdb",
    geology_source="usgs_sgmc",
    geophysics_source="usgs",
    drillhole_source=None,

    deposit_classes={
        "orogenic_gold":   ("usgs:36a", "usgs:36b"),
        "low_sulfidation": ("usgs:25c",),
    },

    occurrence_commodity_filter=("au", "gold"),

    pathfinder_elements=(
        "Au", "As", "Sb", "Hg", "W", "Ag", "Cu", "Pb", "Zn", "Mo", "Bi", "Te",
    ),

    raw_paths={
        "occurrences":   DATA_RAW / "mrds" / "mrds_motherlode.gpkg",
        "geochem":       DATA_RAW / "ngdb" / "ngdb_sediment_motherlode.gpkg",
        "geology":       DATA_RAW / "sgmc" / "sgmc_geology_motherlode.gpkg",
        "geology_arcs":  DATA_RAW / "sgmc" / "sgmc_structure_motherlode.gpkg",
        "dem":           DATA_RAW / "dem" / "dem_motherlode.tif",
        "sentinel2":     DATA_RAW / "sentinel2" / "s2_mean_motherlode.tif",
        "magnetic":      DATA_RAW / "gsc_geophysics" / "magnetic_motherlode.tif",
        "gravity":       DATA_RAW / "gsc_geophysics" / "gravity_motherlode.tif",
        "gravity_isostatic": (
            DATA_RAW / "gsc_geophysics" / "gravity_isostatic_motherlode.tif"
        ),
        # v3.1 magnetic-field derivatives (computed by
        # `data.magnetic_derivatives.write_derivatives`).
        "magnetic_1vd":              DATA_RAW / "gsc_geophysics" / "magnetic_1vd_motherlode.tif",
        "magnetic_hgm":              DATA_RAW / "gsc_geophysics" / "magnetic_hgm_motherlode.tif",
        "magnetic_analytic_signal":  DATA_RAW / "gsc_geophysics" / "magnetic_analytic_signal_motherlode.tif",
        "magnetic_tilt":             DATA_RAW / "gsc_geophysics" / "magnetic_tilt_motherlode.tif",
    },
)
