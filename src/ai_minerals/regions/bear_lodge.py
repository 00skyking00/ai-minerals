"""Bear Lodge Mountains, Wyoming — carbonatite REE province.

Pivot target for the Mt Weld replication: the USGS Mt Weld data
release (data.usgs.gov:63dd8407d34e9fa19a9ad3bf) is a single-deposit
characterization dataset (drill core + open pit samples, CSV only,
no regional rasters) and does not support prospectivity mapping.
Bear Lodge is a geologically analogous carbonatite-derived REE
province in the United States with full public-data coverage via
the existing USGS adapter stack (SGMC, NGDB, MRDS, USGS gravity /
magnetic, USGS DEM).

Hypothesis the experiment tests:
  Does DEEP-SEAM's deviation-network advantage transfer within its
  REE-deposit niche? On Curnamona (7 positives, heavy ILR-PCA + GLCM
  preprocessing), DevNet beat RF by 2-3x. If the same architecture
  also beats RF on Bear Lodge (a different REE province on a
  different continent and a different basement age), the architecture
  has a defensible class-of-problem niche. If it fails on Bear Lodge
  too, the Curnamona result is dataset-specific in an even tighter
  sense than the cross-region experiment already established.

AOI scoping decision:
  The Bear Lodge carbonatite complex (Tertiary alkaline intrusion
  into Precambrian basement) sits near 44.48 N, 104.45 W in
  Crook County, NE Wyoming. The known REE deposit is the Bull Hill
  diatreme system plus its satellites (Whitetail Ridge, Carbon).
  AOI bbox: 44.0 N to 45.0 N, 105.0 W to 103.8 W (about 110 km
  E-W, 110 km N-S, total ~12,000 km^2). Covers the Bear Lodge
  Mountains, parts of Crook + Weston counties WY, and the western
  edge of the Black Hills into SD.

CRS: EPSG:5070 (NAD83 Conus Albers) is the standard CONUS-wide
equal-area projection. We use it here for geographic accuracy at
this latitude; cross-region comparability with the Mother Lode /
Arizona work (which use EPSG:3310 California Albers) is preserved
by reprojecting at the cross-region experiment step rather than at
the per-region grid build.

Data sources (all USGS open data):
  - MRDS for Cox-Singer 36b / 36c / 38a REE-bearing records.
  - SGMC for geology with MAJOR1/2/3 fine-grained lithology.
  - NGDB for stream-sediment chemistry. REE coverage in NGDB is
    sparser than the porphyry-Cu pathfinder suite; we include the
    standard REE-relevant pathfinder set and accept that some
    columns will be empty for many cells.
  - USGS national magnetic + Bouguer + isostatic gravity grids,
    clipped to the AOI. Bear Lodge's carbonatite produces a
    distinctive magnetic anomaly that should make this a strong
    feature.
  - Sentinel-2 indices for the (small) areas of exposed
    carbonatite outcrop. Most of the deposit is under regolith /
    laterite, so S2 is a marginal contributor.
  - 30 m DEM from USGS National Map.
"""

from __future__ import annotations

from ai_minerals.aoi import AOI
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


_BEAR_LODGE_AOI = AOI(
    name="bear_lodge",
    min_lon=-105.0, min_lat=44.0,
    max_lon=-103.8, max_lat=45.0,
)


BEAR_LODGE = Region(
    slug="bear_lodge",
    aoi=_BEAR_LODGE_AOI,
    working_crs="EPSG:5070",
    data_prefix="bear_lodge",

    occurrences_source="mrds",
    geochem_source="ngdb",
    geology_source="usgs_sgmc",
    geophysics_source="usgs",
    drillhole_source=None,

    deposit_classes={
        # Cox-Singer REE-bearing codes. 36b = Thorium-REE veins, 36c
        # = REE pegmatites, 38a = Phoscorite-carbonatite (Bear Lodge
        # is in this family). We label all three as REE positives
        # for the within-niche transfer test.
        "ree":         ("usgs:36b", "usgs:36c", "usgs:38a"),
    },

    occurrence_commodity_filter=("ree", "rare", "lanthanum", "cerium", "yttrium",
                                  "niobium", "thorium"),

    pathfinder_elements=(
        # REE-bearing-deposit pathfinder suite. Light REEs (La, Ce,
        # Nd) carry the strongest signal at Bear Lodge. Y is the
        # heavy-REE proxy. Th, U, Nb often associated. F (fluorine)
        # is a carbonatite alteration indicator but NGDB coverage
        # is thin.
        "La", "Ce", "Nd", "Y", "Th", "U", "Nb",
    ),

    raw_paths={
        "occurrences":   DATA_RAW / "mrds" / "mrds_bear_lodge.gpkg",
        "geochem":       DATA_RAW / "ngdb" / "ngdb_sediment_bear_lodge.gpkg",
        "geology":       DATA_RAW / "sgmc" / "sgmc_geology_bear_lodge.gpkg",
        "geology_arcs":  DATA_RAW / "sgmc" / "sgmc_structure_bear_lodge.gpkg",
        "dem":           DATA_RAW / "dem" / "dem_bear_lodge.tif",
        "sentinel2":     DATA_RAW / "sentinel2" / "s2_mean_bear_lodge.tif",
        "magnetic":      DATA_RAW / "gsc_geophysics" / "magnetic_bear_lodge.tif",
        "gravity":       DATA_RAW / "gsc_geophysics" / "gravity_bear_lodge.tif",
        "gravity_isostatic": (
            DATA_RAW / "gsc_geophysics" / "gravity_isostatic_bear_lodge.tif"
        ),
        "magnetic_1vd":              DATA_RAW / "gsc_geophysics" / "magnetic_1vd_bear_lodge.tif",
        "magnetic_hgm":              DATA_RAW / "gsc_geophysics" / "magnetic_hgm_bear_lodge.tif",
        "magnetic_analytic_signal":  DATA_RAW / "gsc_geophysics" / "magnetic_analytic_signal_bear_lodge.tif",
        "magnetic_tilt":             DATA_RAW / "gsc_geophysics" / "magnetic_tilt_bear_lodge.tif",
    },
)
