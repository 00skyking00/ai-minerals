"""USGS MRDS — fetcher for the per-state shapefile bundles.

The bbox-query API at `mrdata.usgs.gov/mrds/search-bbox.php` returns nested
JSON, and was throwing 500 errors for the Mother Lode AOI as of 2026-05-05.
The per-state shapefile downloads from `mrdata.usgs.gov/mrds/` are an
alternate path: the user picks a state via the geographic-region selector
and downloads a shapefile bundle.

This fetcher takes one or more such per-state shapefile bundles
(unzipped to data/raw/mrds/mrds-fUS<state>-N/), filters spatially to the
AOI, and normalizes the schema so the existing MRDS adapter
(`data/adapters/occurrences/mrds.py`) can read the result without
modification.

Output is a per-region GeoPackage with columns matching what
`data/adapters/occurrences/mrds.py` expects: id, commodity (joined
string), geometry, plus passthrough metadata fields useful for
downstream filtering.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "mrds"


def _normalize_commodity(row: pd.Series) -> str:
    """Join commod1 + commod2 + commod3 into a comma-separated string."""
    parts = []
    for col in ("commod1", "commod2", "commod3"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip().lower())
    return ",".join(parts)


def fetch(
    aoi: AOI,
    working_crs: str,
    state_dirs: list[Path],
    *,
    force: bool = False,
) -> Path:
    """Combine + clip per-state MRDS shapefiles to AOI, normalize, save GPKG.

    `state_dirs` is a list of paths to unzipped MRDS state shapefile
    folders (each contains mrds-fUSnn-N.shp). For Mother Lode, that's
    just data/raw/mrds/mrds-fUS06-1/. For Klamath, CA + OR.
    """
    out_dir = dataset_dir(NAME)
    out_path = out_dir / f"mrds_{aoi.name.lower()}.gpkg"
    if not force and out_path.exists():
        print(f"MRDS clipped artifact present ({out_path.stat().st_size:,} B); "
              f"skipping clip.")
        return out_path

    aoi_bounds = aoi.bbox  # west, south, east, north
    parts = []
    for sd in state_dirs:
        sd = Path(sd)
        shps = list(sd.glob("*.shp"))
        if not shps:
            raise FileNotFoundError(f"no .shp in {sd}")
        for shp in shps:
            print(f"  reading {shp.name}...")
            g = gpd.read_file(shp, bbox=aoi_bounds)
            print(f"    {len(g)} records in AOI bbox")
            parts.append(g)

    if not parts:
        raise RuntimeError(f"no MRDS records found across {state_dirs}")

    gdf = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs=parts[0].crs
    )
    print(f"  combined: {len(gdf)} records across {len(state_dirs)} state(s)")

    gdf["commodity"] = gdf.apply(_normalize_commodity, axis=1)
    gdf["id"] = gdf["mrds_id"].astype(str) if "mrds_id" in gdf.columns else gdf.index.astype(str)

    gdf = gdf.to_crs(working_crs)
    gdf.to_file(out_path, driver="GPKG")
    print(f"  wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    write_source_md(
        NAME,
        title="USGS MRDS, per-state shapefile bundle (geographic-region download)",
        url="https://mrdata.usgs.gov/mrds/",
        license="US public domain (USGS)",
        notes=(
            f"mrds_{aoi.name.lower()}.gpkg: MRDS records from per-state "
            f"shapefiles, AOI-clipped, reprojected to {working_crs}. "
            f"State source dirs: {[str(p) for p in state_dirs]}. "
            f"commod1+commod2+commod3 joined into 'commodity' field for "
            f"the adapter. Fallback path because the bbox-query API "
            f"returned HTTP 500 for the Mother Lode AOI on 2026-05-05."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.regions.motherlode import MOTHERLODE
    fetch(
        MOTHERLODE.aoi,
        working_crs=MOTHERLODE.working_crs,
        state_dirs=[Path("/home/sky/src/learning/ai-minerals/data/raw/mrds/mrds-fUS06-1")],
    )
