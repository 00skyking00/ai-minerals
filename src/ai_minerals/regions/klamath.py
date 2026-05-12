"""California-Oregon Klamath / Trinity belt — orogenic-Au cross-region target.

This region exists to support the cross-region transfer test in Phase 5
of the Mother Lode v3 work: train on Sierra Mother Lode, score Klamath,
report whether known Klamath orogenic-Au districts (Yreka, Liberty,
Carrville, Helena) get ranked high.

AOI is a rectangle covering the Klamath Mountains terrane-quilt: roughly
-124°W to -122°W, 40°N to 42°N. This includes Trinity County (Sky's
family heritage region) and adjacent Siskiyou + Humboldt counties on
the California side, plus a sliver of southwestern Oregon. The AOI
crosses the CA-OR state line, so the MRDS shapefile fetcher takes both
states' bundles.

CRS: EPSG:3310 (California Albers Equal Area), same as Mother Lode, so
feature frames from the two regions are directly comparable. Note that
EPSG:3310 is California-centered; cells north of the CA border get a
slightly skewed projection but this is acceptable for the modest area
we're crossing.

Data sources are identical to Mother Lode (USGS MRDS / SGMC / NGDB /
gravity / magnetic / DEM / Sentinel-2). Adapter selection is the same
so the model trained on Mother Lode applies directly without retraining.
"""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


_KLAMATH_AOI = AOI(
    name="Klamath",
    min_lon=-124.0, min_lat=40.0,
    max_lon=-122.0, max_lat=42.0,
)


KLAMATH = Region(
    slug="klamath",
    aoi=_KLAMATH_AOI,
    working_crs="EPSG:3310",
    data_prefix="klamath",

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
        "occurrences":   DATA_RAW / "mrds" / "mrds_klamath.gpkg",
        "geochem":       DATA_RAW / "ngdb" / "ngdb_sediment_klamath.gpkg",
        "geology":       DATA_RAW / "sgmc" / "sgmc_geology_klamath.gpkg",
        "geology_arcs":  DATA_RAW / "sgmc" / "sgmc_structure_klamath.gpkg",
        "dem":           DATA_RAW / "dem" / "dem_klamath.tif",
        "sentinel2":     DATA_RAW / "sentinel2" / "s2_mean_klamath.tif",
        "magnetic":      DATA_RAW / "gsc_geophysics" / "magnetic_klamath.tif",
        "gravity":       DATA_RAW / "gsc_geophysics" / "gravity_klamath.tif",
        "gravity_isostatic": (
            DATA_RAW / "gsc_geophysics" / "gravity_isostatic_klamath.tif"
        ),
        # v3.1 magnetic-field derivatives.
        "magnetic_1vd":              DATA_RAW / "gsc_geophysics" / "magnetic_1vd_klamath.tif",
        "magnetic_hgm":              DATA_RAW / "gsc_geophysics" / "magnetic_hgm_klamath.tif",
        "magnetic_analytic_signal":  DATA_RAW / "gsc_geophysics" / "magnetic_analytic_signal_klamath.tif",
        "magnetic_tilt":             DATA_RAW / "gsc_geophysics" / "magnetic_tilt_klamath.tif",
    },
)
