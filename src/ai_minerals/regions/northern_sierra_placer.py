"""Northern Sierra deep-gravel placer-Au region.

The northern Sierra deep-gravel belt is the canonical California placer
province: Tertiary buried paleochannels carved by ancestral rivers,
later capped by lava flows, then re-exposed in the Quaternary as the
Yuba and American river systems incised through them. The hydraulic-
mining era (1850s–1884) exposed the deep gravels at Malakoff Diggins,
Dutch Flat, North San Juan, You Bet, Forest Hill, Iowa Hill, and
Michigan Bluff.

The region covers two distinct placer populations with different
geomorphic signatures:

- **Tertiary deep-gravel** (`placer_tertiary`, USGS Cox-Singer 39b) —
  buried paleochannels on interstream ridges that diverge from the
  modern drainage. Mapped via Hydraulic Mine Pit polygons (DOI
  10.5066/F7J38QMD; 167 polygons, Orlando 2016) plus the Lindgren PP73
  paleochannel plates, and (Phase 2) a paleochannel-likelihood raster
  derived from 3DEP LiDAR.

- **Quaternary modern-channel** (`placer_quaternary`, USGS Cox-Singer 39a) —
  modern drainage placers in the active stream network. Mapped via MRDS
  placer points (filtered for dep_type matching placer/alluv/stream/gravel)
  and NHD-network proximity features.

Each population gets its own classifier; the two calibrated rasters
fuse via per-cell max() into the deliverable that the gldbg sibling
repo samples (see `~/src/learning/gldbg/research/plans/gldbg_plan_06_placer-and-region-expansion.md`
and `research/placer_handoff_start_here.md`).

AOI matches the existing motherlode lode-raster extent (37.49–40.01°N,
121.55–119.48°W) so the placer 250 m grid registers cell-for-cell with
the lode raster. goldbug consumes the finished raster via its narrower
scan window in `config/regions/northern_sierra_ca.yaml`; that window
is not a constraint on this AOI.

CRS: EPSG:3310 (California Albers Equal Area, NAD83), matching motherlode
for direct comparison. The deliverable raster is reprojected to EPSG:4326
via `scripts/motherlode/v2_postprocess_250m.py::write_geotiff`.
"""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


_NORTHERN_SIERRA_PLACER_AOI = AOI(
    name="NorthernSierraPlacer",
    min_lon=-121.55, min_lat=37.49,
    max_lon=-119.48, max_lat=40.01,
)


NORTHERN_SIERRA_PLACER = Region(
    slug="northern_sierra_placer",
    aoi=_NORTHERN_SIERRA_PLACER_AOI,
    working_crs="EPSG:3310",
    data_prefix="northern_sierra_placer",

    occurrences_source="mrds",
    geochem_source="ngdb",
    # CGS 2010 Geologic Map of California via the AGOL CGS-hosted endpoint
    # at services2.arcgis.com/zr3KAIbsRSUyARHG/.../GMC_Geology (live as of
    # 2026-05-31). Finer Quaternary subcodes than SGMC ('Qa', 'Qal', 'Qg'
    # for modern alluvium/gravel) feed the Quaternary-alluvium feature.
    # Faults aren't published by this service; geology_arcs still points
    # at the SGMC structure layer.
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
        "occurrences":   DATA_RAW / "mrds" / "mrds_northern_sierra_placer.gpkg",
        "geochem":       DATA_RAW / "ngdb" / "ngdb_sediment_northern_sierra.gpkg",
        "geochem_nure":  DATA_RAW / "nure_iicpms" / "nure_western_us.gpkg",

        "geology":       DATA_RAW / "cgs_2010" / "cgs_geology_northern_sierra.gpkg",
        # CGS 2010 service has no faults layer; reuse SGMC structure file
        # for distance-to-fault.
        "geology_arcs":  DATA_RAW / "sgmc" / "sgmc_structure_northern_sierra.gpkg",

        "dem":           DATA_RAW / "dem" / "dem_northern_sierra.tif",
        "lidar_dem":     DATA_RAW / "3dep_lidar" / "3dep_1m_northern_sierra.tif",

        "hydraulic_pits": DATA_RAW / "hydraulic_pits" / "hydraulic_mine_pits_ca.gpkg",
        "nhd_flowlines":  DATA_RAW / "nhd_hr" / "nhd_flowlines_northern_sierra.gpkg",

        # Lode-Au seeds for distance-downstream-from-lode. Reuses the motherlode
        # MRDS pull (same AOI extent); filter to dev_stat = Past Producer/Producer
        # and exclude placer dep_types at use time. Strict no-overlap-with-placer-
        # positives assertion lives in features/hydrology.py.
        "lode_mrds":     DATA_RAW / "mrds" / "mrds_motherlode.gpkg",

        # Phase 2 paleochannel-likelihood raster, precomputed by
        # scripts/northern_sierra_placer/precompute_paleochannel.py.
        "paleochannel_likelihood":
                         DATA_RAW / "3dep_lidar" / "paleochannel_likelihood_northern_sierra.tif",

        # Geophysics — same grids as motherlode (AOI is identical).
        "magnetic":      DATA_RAW / "gsc_geophysics" / "magnetic_motherlode.tif",
        "gravity":       DATA_RAW / "gsc_geophysics" / "gravity_motherlode.tif",
    },
)
