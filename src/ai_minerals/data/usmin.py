"""USGS USMIN historic mine features (per-state shapefile bundles) — fetcher.

USMIN is the USGS database of mine, prospect, and surface-mining-feature
points extracted from historical USGS topographic quadrangles. It is
published per-state at `https://mrdata.usgs.gov/usmin/state/usmin-<ST>.zip`.
Each ZIP contains a `<ST>-point.shp` (point features) and a `<ST>-poly.shp`
(polygon features). Schema is uniform across states; the `FTR_TYPE`
attribute is the human-readable feature class ("Gravel Pit", "Placer
Mine", "Mine Shaft", "Adit", "Prospect Pit", etc.).

For the placer model we keep only feature classes likely to mark
historical alluvial / paleoplacer ground:

  Placer Mine, Gravel Pit, Sand Pit, Sand and Gravel Pit,
  Gravel/Borrow Pit - Undifferentiated, Diggings, Tailings - Undifferentiated,
  Mine Dump

Borrow / cinder / clay / quarry / coal / leach / slag are excluded.

The state ZIP for CA is ~7 MB; we materialize the AOI-clipped placer/gravel
points as a single GeoPackage. Schema preserved from source plus a
`source_state` column and a `gda_id` (the USMIN row's `GDA_ID`).

Resolved 2026-05-31: the per-state ZIP URLs at `mrdata.usgs.gov/usmin/`
are the canonical, stable distribution. The ScienceBase parent for the
USMIN program (`52323e8ae4b0f06321418f27`) hosts metadata + a 2014 CDI
presentation, not the shapefile bundles themselves — do not point this
fetcher at ScienceBase.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "usmin"

# USGS MRDATA per-state USMIN ZIPs. Resolved 2026-05-31; landing page at
# `https://mrdata.usgs.gov/usmin/` lists one ZIP per US state + DC + PR.
USMIN_LANDING = "https://mrdata.usgs.gov/usmin/"
USMIN_STATE_URL = "https://mrdata.usgs.gov/usmin/state/usmin-{state}.zip"

# USMIN FTR_TYPE values we keep. The set was derived from inspecting the
# AZ + CA shapefiles and selecting every feature class that plausibly
# marks alluvial or paleoplacer ground or its tailings.
PLACER_FTR_TYPES: frozenset[str] = frozenset(
    {
        "Placer Mine",
        "Hydraulic Mine",
        "Gravel Pit",
        "Sand Pit",
        "Sand and Gravel Pit",
        "Gravel/Borrow Pit - Undifferentiated",
        "Diggings",
        "Tailings - Undifferentiated",
        "Mine Dump",
    }
)


def _state_for_aoi(aoi: AOI) -> str:
    """Pick a USMIN state code for the AOI bbox centroid.

    Small lookup table — the placer model only operates in CA right now,
    but the lookup keeps the fetcher trivially extensible. Defaults to CA
    when the centroid sits in the conterminous US west of the 100th
    meridian (the only USMIN coverage Sierra-style fetchers care about).
    """
    cx = 0.5 * (aoi.min_lon + aoi.max_lon)
    cy = 0.5 * (aoi.min_lat + aoi.max_lat)
    # CA: 32.5-42, -124.5 to -114
    if -124.5 <= cx <= -114.0 and 32.5 <= cy <= 42.0:
        return "CA"
    # AK: 51-72, -180 to -130
    if -180.0 <= cx <= -130.0 and 51.0 <= cy <= 72.0:
        return "AK"
    # AZ: 31-37, -114.8 to -109
    if -114.8 <= cx <= -109.0 and 31.0 <= cy <= 37.0:
        return "AZ"
    raise ValueError(
        f"No USMIN state mapping for AOI {aoi.name} centroid ({cx:.3f}, {cy:.3f}); "
        f"add a clause to _state_for_aoi."
    )


def _download_and_extract(state: str, out_dir: Path) -> Path:
    """Download `usmin-<state>.zip` and unzip into out_dir. Returns the .shp path."""
    extract_root = out_dir / state
    shp_path = extract_root / f"{state}-point.shp"
    if shp_path.exists():
        return shp_path

    extract_root.mkdir(parents=True, exist_ok=True)
    url = USMIN_STATE_URL.format(state=state)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(extract_root)

    if not shp_path.exists():
        hits = list(extract_root.rglob(f"{state}-point.shp"))
        if not hits:
            raise RuntimeError(
                f"USMIN point shapefile not found after extracting {url} into "
                f"{extract_root}; contents: {[p.name for p in extract_root.rglob('*')]}"
            )
        shp_path = hits[0]
    return shp_path


def fetch(aoi: AOI, *, force: bool = False) -> Path:
    """Download USMIN state bundle, clip to AOI + placer/gravel filter, write GPKG.

    Output: `data/raw/usmin/usmin_<aoi.name.lower()>.gpkg` (WGS84).

    The filter keeps only `FTR_TYPE` values in `PLACER_FTR_TYPES`; for CA
    that retains a few thousand of ~80k state-wide points.
    """
    out_dir = dataset_dir(NAME)
    out_path = out_dir / f"usmin_{aoi.name.lower()}.gpkg"

    if out_path.exists() and not force:
        return out_path

    state = _state_for_aoi(aoi)
    shp_path = _download_and_extract(state, out_dir)
    gdf = gpd.read_file(shp_path)
    n_state = len(gdf)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() not in {"EPSG:4326", "WGS 84"}:
        gdf = gdf.to_crs("EPSG:4326")

    aoi_box = box(*aoi.bbox)
    gdf = gdf[gdf.intersects(aoi_box)].copy()
    n_in_aoi = len(gdf)

    if "FTR_TYPE" not in gdf.columns:
        raise RuntimeError(
            f"USMIN shapefile {shp_path} missing FTR_TYPE column; "
            f"got: {list(gdf.columns)}"
        )
    gdf = gdf[gdf["FTR_TYPE"].isin(PLACER_FTR_TYPES)].copy()
    gdf["source_state"] = state
    if "GDA_ID" in gdf.columns:
        gda = pd.to_numeric(gdf["GDA_ID"], errors="coerce").fillna(-1).astype("int64")
    else:
        gda = pd.Series(gdf.index.astype("int64"), index=gdf.index)
    # `usmin_gda_id` rather than `gda_id` because GPKG's case-insensitive
    # field handling collides with the source column `GDA_ID`.
    gdf["usmin_gda_id"] = gda.values

    gdf.to_file(out_path, driver="GPKG")

    write_source_md(
        NAME,
        title="USGS USMIN historic mine features (per-state shapefile bundles)",
        url=USMIN_LANDING,
        license="US public domain (USGS)",
        notes=(
            f"Downloaded `usmin-{state}.zip` from "
            f"`{USMIN_STATE_URL.format(state=state)}` (resolved 2026-05-31), "
            f"unzipped, read `{state}-point.shp` ({n_state:,} state-wide rows), "
            f"clipped to AOI={aoi.name} bbox={aoi.bbox} ({n_in_aoi:,} rows), "
            f"and filtered to FTR_TYPE in {sorted(PLACER_FTR_TYPES)} "
            f"({len(gdf):,} placer/gravel rows). Reprojected to EPSG:4326 if "
            f"the source CRS differed. Schema preserved from source plus "
            f"`source_state` and `usmin_gda_id` (int64 copy of the source "
            f"GDA_ID; renamed to dodge a GPKG case-insensitive field "
            f"collision with the source `GDA_ID` column)."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER

    path = fetch(NORTHERN_SIERRA_PLACER.aoi)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
