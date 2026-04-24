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

    # BC deposit-profile codes are lowercased to match adapter convention.
    # L03 = porphyry Cu ± Mo ± Au, L04 = porphyry Cu-Au-Mo
    # H04 = epithermal Au-Ag (high-sulfidation), H05 = (low-sulfidation)
    # K01 = Cu skarn, K02 = Au skarn
    # G06 = Besshi/sedex VMS, G07 = Kuroko/bimodal volcanic VMS (Eskay Creek)
    deposit_classes={
        "porphyry":   ("bc:l03", "bc:l04"),
        "epithermal": ("bc:h04", "bc:h05"),
        "skarn":      ("bc:k01", "bc:k02"),
        "vms":        ("bc:g06", "bc:g07"),
    },

    occurrence_commodity_filter=("cu", "copper", "au", "gold", "ag", "silver"),

    # Add Hg/Tl/Ba for epithermal pathfinders.
    pathfinder_elements=("Ag", "As", "Au", "Bi", "Cu", "Mo", "Pb", "Sb", "Te", "Zn", "Hg", "Tl", "Ba"),

    raw_paths={
        "occurrences":   DATA_RAW / "bcgs_minfile/minfile_bcgt.gpkg",
        "geochem":       DATA_RAW / "bcgs_rgs/rgs_bcgt.parquet",
        "geology":       DATA_RAW / "bcgs_geology/bedrock_bcgt.gpkg",
        "geology_arcs":  DATA_RAW / "bcgs_geology/faults_bcgt.gpkg",
        "dem":           DATA_RAW / "dem/dem_bcgt.tif",
        "sentinel2":     DATA_RAW / "sentinel2/s2_mean_bcgt.tif",
        "magnetic":      DATA_RAW / "gsc_geophysics/magnetic_bcgt.tif",
        "gravity":       DATA_RAW / "gsc_geophysics/gravity_bcgt.tif",
        "drillholes":    DATA_RAW / "bcgs_drillholes/bcgs_drillholes_bcgt.gpkg",
    },
)
