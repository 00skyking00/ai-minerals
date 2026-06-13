"""BC Golden Triangle — v2 region config (stub; populated in phase B)."""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


# Golden Triangle AOI: covers Brucejack, Snip, Eskay Creek, KSM/Kerr-Sulphurets-
# Mitchell, Galore Creek, Schaft Creek, Red Chris, Red Mountain. North edge
# extended to 58.0°N to include Red Chris (57.66°N) with buffer.
_BCGT_AOI = AOI(
    name="BCGT",
    min_lon=-131.5, min_lat=56.0,
    max_lon=-129.5, max_lat=58.0,
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


# BCGT mining-district centroids for the B.2 retrospective benchmark.
#
# Identified by DBSCAN (eps=12 km, min_samples=2) on the 47 post-2010
# Cu+ cells in `data/derived/bcgt/bcgs_pre_post_2010_overlay.parquet`,
# then named by lat/lon proximity to public mining-district references.
# See `scripts/bcgt/identify_b2_clusters.py` for the derivation script.
#
# Each entry holds a centroid in 500 m grid index space (row, col) plus
# lat/lon for human reference, the count of post-2010 Cu+ cells and
# holes in the cluster, and a short description of the district.
# `prepare_b2_inputs.py` reads this table to drive the per-subarea
# benchmark setup.
BCGT_B2_CLUSTERS = {
    "KSM": {
        "center_row": 131,
        "center_col": 101,
        "center_lat": 56.5473,
        "center_lon": -130.7547,
        "n_post2010_cuplus_cells": 17,
        "n_post2010_cuplus_holes": 42,
        "description": "Kerr-Sulphurets-Mitchell porphyry-Cu-Au district",
    },
    "Brucejack": {
        "center_row": 104,
        "center_col": 169,
        "center_lat": 56.4459,
        "center_lon": -130.1907,
        "n_post2010_cuplus_cells": 2,
        "n_post2010_cuplus_holes": 2,
        "description": "Pretium Brucejack epithermal Au-Ag",
    },
    "Red_Chris": {
        "center_row": 406,
        "center_col": 203,
        "center_lat": 57.8080,
        "center_lon": -130.0575,
        "n_post2010_cuplus_cells": 13,
        "n_post2010_cuplus_holes": 61,
        "description": "Imperial Red Chris porphyry-Cu-Au",
    },
    "Snip": {
        "center_row": 198,
        "center_col": 75,
        "center_lat": 56.8412,
        "center_lon": -131.0068,
        "n_post2010_cuplus_cells": 2,
        "n_post2010_cuplus_holes": 6,
        "description": "Snip / Iskut River vicinity",
    },
    "BCGT_Central_N": {
        "center_row": 277,
        "center_col": 141,
        "center_lat": 57.2122,
        "center_lon": -130.5071,
        "n_post2010_cuplus_cells": 3,
        "n_post2010_cuplus_holes": 9,
        "description": "Central BCGT exploration cluster (north of More Creek)",
    },
    "BCGT_Central_S": {
        "center_row": 220,
        "center_col": 119,
        "center_lat": 56.9494,
        "center_lon": -130.6551,
        "n_post2010_cuplus_cells": 3,
        "n_post2010_cuplus_holes": 6,
        "description": "Central BCGT exploration cluster (south of Telegraph Creek)",
    },
    "SE_BCGT_Iskut": {
        "center_row": 67,
        "center_col": 199,
        "center_lat": 56.2890,
        "center_lon": -129.9314,
        "n_post2010_cuplus_cells": 5,
        "n_post2010_cuplus_holes": 10,
        "description": "Southeast BCGT exploration (Bronson Slope / Iskut River area)",
    },
}
