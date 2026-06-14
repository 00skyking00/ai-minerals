"""Nome Placer Fields — Cape Nome four-population placer-Au region.

Cape Nome mining district, Seward Peninsula AK. Coastal / glacial / fluvial
placer system that the chapter 5 placer write-up calls out as
structurally distinct from the Sierra deep-gravel model. Four placer
populations land in this region's deposit class list rather than the one
Sierra has, because Sky's calibration on 2026-06-14 and the bearcub
2026-06-14 reply established that the architecture has to represent all
four.

- ``placer_beach`` (BL) -- relict lag gravels on the six documented
  paleo-shoreline stands at Nome (Submarine -20 / -35 / -70 ft,
  Intermediate +20, Second +35-40, Monroeville +40, Third +70-79, Fourth
  +120-125). Primary feature: elevation-relative-to-stand. Third Beach
  is the high-grade exemplar.
- ``placer_beach_confluence`` (BC) -- where a modern or Wisconsinan
  stream meets a paleo-shoreline. Bear Cub MS 1178 is a BC claim; the
  Anvil Creek bonanza on Third Beach is a BC special case. Primary
  feature: distance-to-beach x distance-to-stream-channel interaction
  term.
- ``placer_quaternary`` (QM) -- modern creek placers reworking ~70 ppb
  background gold from Iron Creek and Nome River drift. Anvil Creek
  main reach above and below the Third Beach intersection produces this
  way. Primary feature: distance-to-stream-channel inside an Iron
  Creek or Nome River drift polygon.
- ``placer_tertiary_buried`` (TB) -- older paleochannels carved into
  the highest glacial-drift surfaces and capped, with multiple gold-rich
  gravel layers separated by clay or volcanic ash "false bedrocks."
  Molasses (MS 1179) and Honey (MS 1181) on the Dexter-Anvil divide
  are the canonical example: three pay levels, two false bedrocks
  down to 275 ft, well above the +120-125 ft Fourth Beach maximum.
  The Bulolo Gold Placer (Papua New Guinea, intermediate-lens false
  bedrock at 15-40 ft / actual bedrock at 200 ft) is the architectural
  analog. Primary feature: REM/LRM-detected buried paleochannels +
  depth-aware feature stack.

USGS Cox-Singer subtypes: BL/BC/QM map onto 39a (placer Au-PGE generic);
TB maps onto 39b (Tertiary buried placer). The distinction between BL,
BC, and QM is made through bearcub's per-hole ``deposit_class`` field in
the label table, not through ARDF / MRDS subcodes (ARDF does not split
that finely at Nome).

bearcub contracts (per 2026-06-14 reply, all four pushbacks accepted):

- Drill-hole label table from ``src/bearcub/adapters/drillholes.py``:
  per-hole EPSG:3338 + WGS84 collar, ``deposit_class`` (one of BL / BC /
  QM / TB / barren), ``pay_zone_average_oz_cuyd``, ``pay_binary``,
  ``bedrock_depth_ft``, ``pay_zone_thickness_ft``, ``claim_ms``,
  ``source``, ``ocr_confidence``, ``drill_line_id``, ``drill_year``,
  ``operator``.
- ``hole_layers`` sidecar table with per-interval lithology
  (``hole_id``, ``depth_top_ft``, ``depth_bottom_ft``, ``lithology``,
  ``assay_oz_cuyd_or_null``). Drives the TB false-bedrock feature.
- Bedrock-topo surface as a GeoTIFF with a per-cell GP-kernel-variance
  sidecar. ai-minerals downweights the bedrock feature wherever the
  variance exceeds a threshold.
- Surveyed claim polygons (Bear Cub MS 1178, Early Bear MS 1180,
  Molasses MS 1179, Honey MS 1181) from
  ``data/derived/nome_block_resource/claims_by_ms.geojson`` on the
  bearcub side. EPSG:4326. Drives spatial blocking and Phase 1
  validation gating for the TB population.

Other contracts:

- Beach-line geometry derivation lives in ``features/coastal.py`` in
  this repo (per pushback #1). Contract artefact (GeoJSON polylines +
  distance + elevation-relative-to-stand) is unchanged; either side can
  rebuild from IfSAR.
- District drill-planner is deferred to Phase 3 (per the master plan
  on 2026-06-14). Single-claim GP recommender stays in bearcub.
- Nome raster ships with its own ``bands.json`` and empirical
  ``coverage_threshold`` computed at the Nome model resolution, never
  inheriting the Sierra defaults (goldbug's silent-config term).

Working CRS: EPSG:3338 NAD83 Alaska Albers Equal Area, matching the
existing eastak region. v1 model grid: 25 m. 10 m grid reserved as a
config switch for the AGC-gated re-run when labels can support the
finer resolution; feature derivations run at native IfSAR 5 m
regardless.
"""

from __future__ import annotations

from ai_minerals.aoi import NOME_PLACER
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


NOME_PLACER_REGION = Region(
    slug="nome_placer",
    aoi=NOME_PLACER,
    working_crs="EPSG:3338",
    data_prefix="nome_placer",

    occurrences_source="ardf",
    geochem_source="agdb4",
    # v1 geology uses the Wilson et al. 2015 SIM 3340 statewide map, same
    # as eastak. The Cape Nome / Nome C-6 sub-quadrangle has no DGGS
    # digital surficial-geology publication; AOF 125 (Tolstoi Point) and
    # USGS OF 72-321/322 (Nome C-2/C-3) cover the eastern Seward
    # Peninsula and are ~40-80 km east of the AOI. The ADGGS adapter is
    # wired through `surficial_seward` for v2 AOI expansion and stays out
    # of the v1 geology slot.
    geology_source="usgs_sgmc",
    geophysics_source="usgs",
    drillhole_source="bearcub_nome",

    deposit_classes={
        "placer_beach":            ("usgs:39a",),
        "placer_beach_confluence": ("usgs:39a",),
        "placer_quaternary":       ("usgs:39a",),
        "placer_tertiary_buried":  ("usgs:39b",),
    },

    occurrence_commodity_filter=("au", "gold"),

    pathfinder_elements=("Au", "As", "Sb", "Hg", "Bi"),

    raw_paths={
        # Occurrences: ARDF placer + lode points clipped to the Cape Nome
        # AOI. The lode-seed list for distance-downstream-from-lode is
        # Rock Creek (Nome Group mesothermal Au, ~0.071 oz/ton, ~473k oz
        # contained, 8 mi N of Nome) plus ARDF lode points at Cape Nome,
        # Penny River, and Snake River. Filtered at use time, not here.
        "occurrences":   DATA_RAW / "ardf" / "ardf_nome_placer.gpkg",

        # Geochem: AGDB4 stream-sediment + soil samples clipped to AOI.
        # Same source as eastak.
        "geochem":       DATA_RAW / "agdb4" / "agdb4_samples_nome_placer.parquet",

        # Geology: Wilson et al. 2015 SIM 3340 statewide compilation
        # (same source eastak uses). Quaternary surficial units are
        # coarsely classified; Iron Creek vs Nome River drift is NOT
        # split at this scale. v1 accepts this as the coverage available
        # for the Cape Nome AOI.
        "geology":       DATA_RAW / "geology_ak" / "geology_nome_placer.gpkg",

        # Structural / fault layer: AKStategeol arcs from the
        # statewide compilation (same source eastak uses for faults).
        "geology_arcs":  DATA_RAW / "geology_ak/sim3340/sim3340_gdb/AKgeol_web_gdb/geologic_data/AKStategeol.gdb",

        # Eastern Seward Peninsula surficial geology (ADGGS AOF 125,
        # Tolstoi Point area). Reserved for v2 AOI expansion east of the
        # current bbox; not used by v1 since AOF 125 does NOT cover the
        # Cape Nome AOI itself. See data/adggs_surficial.py docstring.
        "surficial_seward": DATA_RAW / "adggs" / "adggs_surficial_seward_peninsula.gpkg",

        # Elevation: IfSAR Alaska 5 m DEM, mosaicked to the Cape Nome
        # AOI. Used by features/coastal.py for beach-line contour
        # extraction, by features/paleochannel.py for REM/LRM detrending
        # on the TB population, and by features/hydrology.py for
        # stream-channel and watershed derivation.
        "dem":           DATA_RAW / "ifsar_alaska" / "ifsar_5m_nome_placer.tif",

        # Hydrology: NHDPlus HR Alaska, clipped to AOI. Stream lines feed
        # the QM and BC distance-to-channel features.
        "nhd_flowlines": DATA_RAW / "nhd_hr" / "nhd_flowlines_nome_placer.gpkg",

        # Geophysics (optional): USGS Alaska aeromagnetic + gravity
        # composites clipped to AOI. Same fallback pattern eastak uses.
        "magnetic":      DATA_RAW / "geophysics" / "magnetic_nome_placer.tif",
        "gravity":       DATA_RAW / "geophysics" / "gravity_nome_placer.tif",

        # bearcub-delivered contracts. Paths point at the bearcub
        # derived/ tree directly because bearcub is a sibling repo on
        # disk; ai-minerals reads but does not write these. See module
        # docstring for the schema.
        "drillholes":           DATA_RAW / "bearcub_nome" / "drillholes_nome_placer.parquet",
        "hole_layers":          DATA_RAW / "bearcub_nome" / "hole_layers_nome_placer.parquet",
        "bedrock_topo":         DATA_RAW / "bearcub_nome" / "bedrock_topo_nome_placer.tif",
        "bedrock_topo_variance": DATA_RAW / "bearcub_nome" / "bedrock_topo_variance_nome_placer.tif",
        "claims_by_ms":         DATA_RAW / "bearcub_nome" / "claims_by_ms.geojson",
    },
)
