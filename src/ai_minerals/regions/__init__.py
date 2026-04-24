"""Region configs.

A `Region` bundles everything that varies between EastAK / BC Golden
Triangle / Mother Lode runs: AOI, working CRS, which adapters to use per
data kind, which deposit codes define positives, where the raw files live.

Feature pipeline, models, and notebooks take a `Region` and stay
jurisdiction-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai_minerals.aoi import AOI


@dataclass(frozen=True)
class Region:
    slug: str                    # filename-safe identifier: "eastak", "bcgt", ...
    aoi: AOI
    working_crs: str             # EPSG code for the feature grid
    data_prefix: str             # filename prefix for derived raw files

    # Adapter selection (source-key strings; resolved via ADAPTERS registry)
    occurrences_source: str
    geochem_source: str
    geology_source: str
    geophysics_source: str
    drillhole_source: str | None

    # Semantic deposit class → jurisdiction-prefixed codes tuple.
    # E.g. {"porphyry_family": ("usgs:17", "usgs:20c", "usgs:21a", "usgs:21b")}
    deposit_classes: dict[str, tuple[str, ...]]

    # Commodity substrings to match for the any-mineral-occurrence exclusion mask.
    # Case-insensitive; any match in `commodity` field qualifies a record.
    occurrence_commodity_filter: tuple[str, ...]

    # Pathfinder element symbols to aggregate (per-region; epithermal needs Hg/Tl/Ba).
    pathfinder_elements: tuple[str, ...]

    # Raw-file paths by dataset key. Assemble.py indexes by these keys.
    raw_paths: dict[str, Path] = field(default_factory=dict)


from ai_minerals.regions.eastak import EASTAK  # noqa: E402

__all__ = ["Region", "EASTAK"]
