"""Nome Placer Fields — multi-claim placer-Au region (stub; populated as data ingest progresses).

Cape Nome mining district, Seward Peninsula AK. Family-held property
spanning both lowland coastal placer benches (near Bear Cub MS 1178) and
Anvil-area uplands inland.

**Status**: stub. The Bear Cub pilot subproject (24 Murray drill collars,
1925-1955) demonstrates the dark-data → structured-data ingestion pipeline
on the lowland-Bear-Cub node. As additional data sources come online —
Janin drill logs from the Huntington Library, the 3 sister claims, geology
reports and proprietary writeups — they plug into this Region via canonical
adapters.

**Out of scope for the v1 sprint**: only the Bear Cub pilot notebook
(`notebooks/bear_cub/main.qmd`) consumes this Region's slug. Feature
pipeline / models / multi-claim spatial CV are deferred until additional
nodes' data is ingested.
"""

from __future__ import annotations

from ai_minerals.aoi import NOME_PLACER
from ai_minerals.data._common import DATA_RAW
from ai_minerals.regions import Region


NOME_PLACER_REGION = Region(
    slug="nome_placer",
    aoi=NOME_PLACER,
    working_crs="EPSG:3338",     # NAD83 Alaska Albers — same as EastAK
    data_prefix="nome_placer",

    # Nothing wired in yet — the Bear Cub adapter is the first node.
    occurrences_source="",
    geochem_source="",
    geology_source="",
    geophysics_source="",
    drillhole_source="bear_cub",   # only Bear Cub Murray archive so far

    # ARDF NM253 (Nome Offshore Placer) is the relevant USGS reference for
    # placer-Au; populate when the occurrences adapter is wired in.
    deposit_classes={
        "placer_au": ("usgs:39a",),
    },

    occurrence_commodity_filter=("au", "gold"),

    # Placer-Au pathfinders skew toward Hg/Sb/As (sulfide-association) +
    # PGE/Bi for the deeper-source narrative. Light list for now.
    pathfinder_elements=("Au", "As", "Sb", "Hg"),

    raw_paths={
        "drillholes":  DATA_RAW / "bear_cub/bear_cub_collars.csv",
        # Future:
        #   - janin_drillholes:    Charles Janin field-notebook OCR (Huntington Library)
        #   - dark_data_archive:   property writeups, surveyor docs, court records
        #   - upland_anvil_*:      data for the 2 upland claims
        #   - lowland_adjacent_*:  data for the 2nd lowland claim
    },
)
