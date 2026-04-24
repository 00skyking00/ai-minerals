"""Eastern Alaska porphyry belt — v1 region config."""

from __future__ import annotations

from ai_minerals.aoi import EASTERN_ALASKA
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


EASTAK = Region(
    slug="eastak",
    aoi=EASTERN_ALASKA,
    working_crs="EPSG:3338",     # NAD83 Alaska Albers Equal Area
    data_prefix="eastak",

    occurrences_source="ardf",
    geochem_source="agdb4",
    geology_source="usgs_sgmc",
    geophysics_source="usgs",
    drillhole_source="kenorland",

    deposit_classes={
        "porphyry":        ("usgs:17", "usgs:20c", "usgs:21a", "usgs:21b"),
        "porphyry_strict": ("usgs:21a",),
    },

    occurrence_commodity_filter=("cu", "copper"),

    pathfinder_elements=("Ag", "As", "Au", "Bi", "Cu", "Mo", "Pb", "Sb", "Te", "Zn"),

    raw_paths={
        "occurrences":    DATA_RAW / "ardf/ardf_eastak.gpkg",
        "occurrences_mrds": DATA_RAW / "mrds/mrds_eastak.geojson",
        "geochem":        DATA_RAW / "agdb4/agdb4_samples_eastak.parquet",
        "geochem_bv_zip": DATA_RAW / "agdb4/AGDB4_text.zip",
        "geology":        DATA_RAW / "geology_ak/geology_eastak.gpkg",
        "geology_arcs":   DATA_RAW / "geology_ak/sim3340/sim3340_gdb/AKgeol_web_gdb/geologic_data/AKStategeol.gdb",
        "dem":            DATA_RAW / "dem/dem_eastak.tif",
        "sentinel2":      DATA_RAW / "sentinel2/s2_mosaic_eastak.tif",
        "magnetic":       DATA_RAW / "geophysics/magnetic_eastak.tif",
        "gravity":        DATA_RAW / "geophysics/gravity_eastak.tif",
        "drillholes":     DATA_RAW / "kenorland/kenorland_tanacross_collars.csv",
    },
)
