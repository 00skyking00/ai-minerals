"""Adapter registry.

Each entry maps (data-kind, source-key) → a callable `load(path, aoi, ...)`
that returns a canonical-schema GeoDataFrame (or numpy array for geophysics).

Region configs pick adapters by source-key; downstream code resolves via the
registry. New jurisdictions just register new entries — no caller-site
changes anywhere.
"""

from __future__ import annotations

from ai_minerals.data.adapters.occurrences import ardf as _ardf_occ
from ai_minerals.data.adapters.occurrences import mrds as _mrds_occ
from ai_minerals.data.adapters.occurrences import bc_minfile as _bc_minfile_occ
from ai_minerals.data.adapters.geochem import agdb4 as _agdb4_gc
from ai_minerals.data.adapters.geochem import bcgs_rgs as _bcgs_rgs_gc
from ai_minerals.data.adapters.geochem import ngdb as _ngdb_gc
from ai_minerals.data.adapters.geology import usgs_sgmc as _sgmc_geo
from ai_minerals.data.adapters.geology import bcgs_digital as _bcgs_digital_geo
from ai_minerals.data.adapters.geophysics import usgs as _usgs_geophys
from ai_minerals.data.adapters.drillholes import kenorland as _kenorland_dh
from ai_minerals.data.adapters.drillholes import bcgs_geofile as _bcgs_geofile_dh


ADAPTERS: dict[str, dict[str, object]] = {
    "occurrences": {
        "ardf": _ardf_occ.load,
        "mrds": _mrds_occ.load,
        "bc_minfile": _bc_minfile_occ.load,
    },
    "geochem": {
        "agdb4": _agdb4_gc.load,
        "bcgs_rgs": _bcgs_rgs_gc.load,
        "ngdb": _ngdb_gc.load,
    },
    "geology": {
        "usgs_sgmc": _sgmc_geo.load,
        "bcgs_digital": _bcgs_digital_geo.load,
    },
    "geophysics": {
        "usgs": _usgs_geophys.mask_nodata,  # applies to already-sampled array
        # v2 placeholder: BCGT geophysics grids are NaN-filled (real NRCan
        # integration is v2.1); mask_nodata is idempotent on NaN input.
        "gsc": _usgs_geophys.mask_nodata,
    },
    "drillholes": {
        "kenorland": _kenorland_dh.load,
        "bcgs_geofile": _bcgs_geofile_dh.load,
    },
}


def get_adapter(kind: str, source: str):
    try:
        return ADAPTERS[kind][source]
    except KeyError:
        available = list(ADAPTERS.get(kind, {}))
        raise KeyError(
            f"No adapter for ({kind}, {source}). Known {kind} sources: {available}"
        )
