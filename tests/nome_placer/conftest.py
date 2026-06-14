"""Shared real-data fixtures for the Nome placer test suite.

These fixtures load the actual IfSAR n65w166 tile, the actual
bearcub-delivered drillholes / hole_layers / bedrock_topo / claims_by_ms
files, and the actual EPSG:3338 25 m grid. Tests that exercise the
coastal / scoring / stream code paths should consume these fixtures
rather than mock geometries; the v0 scorer was validated on real data
and the unit suite preserves that level of fidelity.

If the IfSAR tile or any bearcub artifact is missing, the dependent
fixtures `pytest.skip()` with a clear message so the test suite still
runs on environments without the data cache.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import rioxarray  # noqa: F401  -- registers .rio accessor

from ai_minerals.aoi import NOME_PLACER
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION


@pytest.fixture(scope="session")
def nome_aoi():
    return NOME_PLACER


@pytest.fixture(scope="session")
def nome_region():
    return NOME_PLACER_REGION


@pytest.fixture(scope="session")
def ifsar_dem_path(nome_region) -> Path:
    path = Path(nome_region.raw_paths["dem"])
    if not path.exists():
        pytest.skip(
            f"IfSAR DEM not cached at {path}. Run "
            "`.venv/bin/python -c 'from ai_minerals.data.ifsar_alaska import fetch; fetch()'`"
        )
    return path


@pytest.fixture(scope="session")
def ifsar_dem_aoi_proj(ifsar_dem_path, nome_aoi):
    """IfSAR DEM clipped to the Nome AOI and reprojected to EPSG:3338 (working CRS).

    Cached at session scope because reprojecting the n65w166 tile takes
    several seconds.
    """
    from ai_minerals.data.adapters.elevation.ifsar_alaska import load

    da_full = load(ifsar_dem_path, nome_aoi)
    clipped = da_full.rio.clip_box(*nome_aoi.bbox)
    return clipped.rio.reproject("EPSG:3338")


@pytest.fixture(scope="session")
def nome_grid_25m(nome_aoi):
    """25 m grid on the Nome AOI in EPSG:3338."""
    from ai_minerals.grid import build_grid

    return build_grid(nome_aoi, resolution_m=25, working_crs="EPSG:3338")


@pytest.fixture(scope="session")
def bearcub_drillholes(nome_region) -> pd.DataFrame:
    path = Path(nome_region.raw_paths["drillholes"])
    if not path.exists():
        pytest.skip(
            f"bearcub drillholes parquet not at {path}. Need bearcub to deliver."
        )
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def bearcub_hole_layers(nome_region) -> pd.DataFrame:
    path = Path(nome_region.raw_paths["hole_layers"])
    if not path.exists():
        pytest.skip(f"bearcub hole_layers parquet not at {path}.")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def bearcub_claims(nome_region):
    """Family-claim polygons including MS 1178 Bear Cub / 1179 Molasses /
    1180 Early Bear / 1181 Honey. EPSG:4326."""
    path = Path(nome_region.raw_paths["claims_by_ms"])
    if not path.exists():
        pytest.skip(f"bearcub claims geojson not at {path}.")
    return gpd.read_file(path)


@pytest.fixture(scope="session")
def family_claim_polygons(bearcub_claims):
    """The four family claim polygons keyed by name."""
    by_ms = {str(ms): row for ms, row in bearcub_claims.set_index("MS").iterrows()}
    families = {
        "bear_cub": "1178",
        "molasses": "1179",
        "early_bear": "1180",
        "honey": "1181",
    }
    out = {}
    for name, ms in families.items():
        if ms not in by_ms:
            pytest.skip(f"Family claim MS {ms} ({name}) not in claims_by_ms.geojson")
        out[name] = by_ms[ms]
    return out
