"""Calaveras placer-Au region — coarse transfer-test AOI.

Calaveras County, central Sierra Nevada foothills. Same Cox-Singer
deposit class taxonomy as the northern-Sierra build (`placer_tertiary`
USGS 39b + `placer_quaternary` USGS 39a) but tilted toward the modern-
drainage population: the well-known Calaveras placer districts (Mokelumne
Hill, San Andreas, Murphys, Angels Camp, Carson Hill) cluster on
Quaternary stream and gravel-bench placers, with the Tertiary deep-gravel
signal more localized than in the Yuba / American river systems further
north.

Used by `scripts/northern_sierra_placer/calaveras_transfer.py` as a
coarse out-of-region transfer target. Not a full-build region: the data
inventory below mirrors `northern_sierra_placer.py` for path-key
compatibility, but Calaveras-specific raw files (paleochannel-likelihood
raster, NHDPlus HR for HUC 1804, Calaveras-clipped CGS 2010 geology) may
not exist on disk. The transfer driver tolerates missing features with
NaN-fills + warnings and documents which features it actually computed.

CRS: EPSG:3310 (California Albers Equal Area, NAD83), matching the
northern-Sierra placer and motherlode lode rasters for direct comparison
on a shared 250 m grid.
"""

from __future__ import annotations

from ai_minerals.aoi import CALAVERAS_PLACER
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


CALAVERAS_PLACER_REGION = Region(
    slug="calaveras_placer",
    aoi=CALAVERAS_PLACER,
    working_crs="EPSG:3310",
    data_prefix="calaveras_placer",

    occurrences_source="mrds",
    geochem_source="ngdb",
    geology_source="cgs_2010",
    geophysics_source="usgs",
    drillhole_source=None,

    deposit_classes={
        "placer_tertiary":   ("usgs:39b",),
        "placer_quaternary": ("usgs:39a",),
    },

    occurrence_commodity_filter=("au", "gold"),

    pathfinder_elements=("Au", "As", "Sb", "Hg", "Bi", "W"),

    raw_paths={
        # MRDS for Calaveras sits inside the motherlode MRDS pull (Calaveras Co
        # is inside the motherlode AOI). The transfer driver reuses that file
        # and AOI-clips on read.
        "occurrences":   DATA_RAW / "mrds" / "mrds_motherlode.gpkg",
        "geochem":       DATA_RAW / "ngdb" / "ngdb_sediment_motherlode.gpkg",
        "geochem_nure":  DATA_RAW / "nure_iicpms" / "nure_western_us.gpkg",

        # CGS 2010 geology is a statewide FeatureServer pull; a Calaveras-clipped
        # extract may or may not exist. Fall back to the northern-Sierra extract
        # at use time if the Calaveras-specific file is absent.
        "geology":       DATA_RAW / "cgs_2010" / "cgs_geology_calaveras.gpkg",
        "geology_arcs":  DATA_RAW / "cgs_2010" / "cgs_faults_calaveras.gpkg",

        "dem":           DATA_RAW / "dem" / "dem_motherlode.tif",
        "lidar_dem":     DATA_RAW / "3dep_lidar" / "3dep_1m_calaveras.tif",

        "hydraulic_pits": DATA_RAW / "hydraulic_pits" / "hydraulic_mine_pits_ca.gpkg",
        "nhd_flowlines":  DATA_RAW / "nhd_hr" / "nhd_flowlines_calaveras.gpkg",

        # Lode-Au seeds reuse the motherlode MRDS pull. The
        # distance-downstream-from-lode feature filters out placer dep_types
        # at compute time.
        "lode_mrds":     DATA_RAW / "mrds" / "mrds_motherlode.gpkg",

        # Optional Phase 2 paleochannel-likelihood raster; absent unless a
        # Calaveras-specific precompute has been run.
        "paleochannel_likelihood":
                         DATA_RAW / "3dep_lidar" / "paleochannel_likelihood_calaveras.tif",

        # Geophysics — Calaveras sits inside the motherlode lode-raster extent.
        "magnetic":      DATA_RAW / "gsc_geophysics" / "magnetic_motherlode.tif",
        "gravity":       DATA_RAW / "gsc_geophysics" / "gravity_motherlode.tif",
    },
)
