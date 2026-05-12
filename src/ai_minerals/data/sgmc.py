"""USGS State Geologic Map Compilation (SGMC) — fetch helper.

Source: USGS Data Series 1052, Horton et al. 2017,
`https://doi.org/10.5066/F7WH2N65`. The full conterminous-US GeoPackage
is 415 MB and is downloaded manually (the ScienceBase API throttles
heavily under concurrent load). This module assumes the GDB has been
unzipped to:

    data/raw/sgmc/USGS_SGMC_Geodatabase/USGS_StateGeologicMapCompilation_ver1.1.gdb/

It reads the SGMC_Geology and SGMC_Structure layers, filters to the
state(s) the AOI overlaps, clips to the AOI bbox, and reprojects to the
region's working CRS. Output GeoPackages live next to the GDB so the
adapter can read them per-region.

The full GDB ships in ESRI:102039 (USA Contiguous Albers Equal Area
Conic, NAD83). We reproject to the region working CRS at fetch time so
downstream code can stay jurisdiction-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import fiona
import geopandas as gpd
from shapely.geometry import box

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "sgmc"

GDB_RELATIVE = (
    "USGS_SGMC_Geodatabase/USGS_StateGeologicMapCompilation_ver1.1.gdb"
)


def _gdb_path() -> Path:
    return dataset_dir(NAME) / GDB_RELATIVE


def _clip_layer(
    layer: str,
    aoi: AOI,
    states: list[str],
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Read a SGMC GDB layer, state-filter, clip to AOI, reproject."""
    gdb = _gdb_path()
    if not gdb.exists():
        raise FileNotFoundError(
            f"SGMC GDB not present at {gdb}. Manually download "
            f"USGS_SGMC_Geodatabase.zip from ScienceBase item "
            f"5888bf4fe4b05ccb964bab9d and unzip into "
            f"data/raw/sgmc/."
        )

    state_set = set(states)
    aoi_bbox = box(*aoi.bbox)  # in EPSG:4326

    # Read with a state filter via fiona's filter+slicing semantics.
    # SGMC_Geology has ~750k rows nationwide; reading just the AOI states
    # keeps memory down.
    rows = []
    with fiona.open(gdb, layer=layer) as src:
        src_crs = src.crs
        for feat in src:
            if feat["properties"].get("STATE") in state_set:
                rows.append(feat)

    print(f"  {layer}: {len(rows)} rows in {sorted(state_set)} pre-clip")

    gdf = gpd.GeoDataFrame.from_features(rows, crs=src_crs)
    if gdf.empty:
        return gdf

    # Reproject AOI bbox into source CRS, clip, then reproject the
    # result to the working CRS.
    aoi_in_src = (
        gpd.GeoSeries([aoi_bbox], crs="EPSG:4326").to_crs(src_crs).iloc[0]
    )
    gdf_clip = gdf[gdf.intersects(aoi_in_src)].copy()
    gdf_clip["geometry"] = gdf_clip.geometry.intersection(aoi_in_src)
    gdf_clip = gdf_clip[~gdf_clip.geometry.is_empty].copy()
    print(f"  {layer}: {len(gdf_clip)} rows after AOI clip")

    return gdf_clip.to_crs(working_crs)


def fetch(
    aoi: AOI,
    working_crs: str,
    states: list[str],
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Clip SGMC geology + structure layers to the AOI and write to GeoPackages.

    Returns (geology_path, structure_path).

    `states` is the list of postal codes overlapping the AOI (e.g. ["CA"]
    for Mother Lode, ["CA", "OR"] for Klamath if the AOI crosses).
    """
    out_dir = dataset_dir(NAME)
    geo_path = out_dir / f"sgmc_geology_{aoi.name.lower()}.gpkg"
    struct_path = out_dir / f"sgmc_structure_{aoi.name.lower()}.gpkg"

    if not force and geo_path.exists() and struct_path.exists():
        print(
            f"SGMC clipped artifacts already present "
            f"({geo_path.stat().st_size:,} + {struct_path.stat().st_size:,} B); "
            f"skipping clip."
        )
        return geo_path, struct_path

    print(f"Clipping SGMC_Geology for AOI={aoi.name}, states={states}...")
    geo = _clip_layer("SGMC_Geology", aoi, states, working_crs)
    if not geo.empty:
        geo.to_file(geo_path, driver="GPKG")
        print(f"  wrote {geo_path} ({geo_path.stat().st_size:,} bytes)")
    else:
        raise RuntimeError(
            f"SGMC_Geology clip for AOI={aoi.name} states={states} "
            f"returned 0 rows. Check states list and AOI bbox."
        )

    print(f"Clipping SGMC_Structure (faults) for AOI={aoi.name}...")
    struct = _clip_layer("SGMC_Structure", aoi, states, working_crs)
    if not struct.empty:
        struct.to_file(struct_path, driver="GPKG")
        print(f"  wrote {struct_path} ({struct_path.stat().st_size:,} bytes)")
    else:
        # Empty fault layer is OK in some AOIs; write an empty file with the right schema.
        struct_path.unlink(missing_ok=True)
        gpd.GeoDataFrame(
            {"geometry": []}, crs=working_crs
        ).to_file(struct_path, driver="GPKG")
        print(f"  no faults in clip; wrote empty {struct_path}")

    write_source_md(
        NAME,
        title="USGS State Geologic Map Compilation (SGMC) v1.1, 2017",
        url="https://doi.org/10.5066/F7WH2N65",
        license="US public domain (USGS)",
        notes=(
            f"sgmc_geology_{aoi.name.lower()}.gpkg: SGMC_Geology polygons "
            f"clipped to AOI={aoi.name} states={states}, reprojected to "
            f"{working_crs}. sgmc_structure_<region>.gpkg: SGMC_Structure "
            f"faults clipped the same way. Source GDB is in ESRI:102039 "
            f"(USA Contiguous Albers Equal Area Conic). Lithology field is "
            f"GENERALIZED_LITH (controlled vocabulary, ~33 values "
            f"nationally). Adapter consumes these clipped artifacts."
        ),
    )
    return geo_path, struct_path


if __name__ == "__main__":
    from ai_minerals.regions.motherlode import MOTHERLODE
    fetch(MOTHERLODE.aoi, working_crs=MOTHERLODE.working_crs, states=["CA"])
