"""USGS NGDB (National Geochemical Database) — fetch helper.

Source: `https://mrdata.usgs.gov/ngdb/sediment/`. The full national bulk
download is `ngdbsed-csv.zip` (~165 MB) which expands to four CSV
tables plus a sample-locations shapefile:
  - `main.csv`: per-sample metadata (lat_wgs84, long_wgs84, state, ...)
  - `bestvalue.csv`: long-format best-value chemistry (one row per
    sample-element pair).
  - `chemistry.csv`: full multi-method results (we use bestvalue
    instead).
  - `ngdbsed.shp`: sample locations only (just lab_id + geometry).

This module assumes the bulk archive has been extracted to:
    data/raw/ngdb/ngdbsed/

It clips the sample set to the AOI (bbox + state filter) and pivots
bestvalue from long-to-wide so each pathfinder element gets its own
column. Output is a per-region GeoPackage that the geochem adapter
consumes.

NGDB pulls together USGS regional surveys, NURE (National Uranium
Resource Evaluation) reconnaissance from the 1970s, and other
contributors. Quality varies by submitter, but for orogenic-Au
pathfinder discrimination at regional scale it is the best-public
geochemistry source for the conterminous US.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "ngdb"

# Per-element output config: (NGDB species code, output column name,
# multiplier from NGDB unit -> ppm).
#
# NGDB units in bestvalue.csv:
#   - Most metals are reported in `ppm`  -> multiplier = 1
#   - Au is typically in `ppb`           -> multiplier = 1/1000
#   - Hg can be `ppm` or `ppb`           -> handled per-row below
#   - Major elements like Al, Fe, Ca, Mg in `pct` (we don't use these for
#     Au prospectivity)
#
# The pathfinder set follows the orogenic-Au mineral systems literature:
# Au itself, As + Sb (both syngenetic with orogenic Au), Hg + W (often
# spatially associated), plus Cu/Pb/Zn/Ag/Mo as supporting context.
PATHFINDERS_PPM = (
    "Au", "As", "Sb", "Hg", "W", "Ag", "Cu", "Pb", "Zn", "Mo", "Bi", "Te",
)

# Below-detection-limit handling: NGDB encodes BDL as a negative qvalue
# (the absolute value is the detection threshold). We mask to NaN.
BDL_MASK_NEGATIVE = True


def _bbox_filter_main(main_path: Path, aoi: AOI, states: list[str]) -> pd.DataFrame:
    """Read main.csv, keep only AOI-state samples in the AOI bbox."""
    print(f"  Reading main.csv (~75 MB)...")
    df = pd.read_csv(
        main_path,
        usecols=[
            "lab_id",
            "state",
            "lat_wgs84",
            "long_wgs84",
            "primary_class",
            "sample_source",
            "date_collect",
        ],
        dtype={"lab_id": str, "state": str, "primary_class": str, "sample_source": str},
        low_memory=False,
    )
    print(f"  main: {len(df):,} samples nationwide")
    df = df[df["state"].isin(states)].copy()
    print(f"  main: {len(df):,} after state filter ({states})")

    west, south, east, north = aoi.bbox
    df = df[
        (df["long_wgs84"].between(west, east))
        & (df["lat_wgs84"].between(south, north))
    ].copy()
    print(f"  main: {len(df):,} after AOI bbox filter")
    return df


def _bestvalue_for_lab_ids(
    bv_path: Path, lab_ids: set[str]
) -> pd.DataFrame:
    """Stream bestvalue.csv, keep only rows for the given lab_id set."""
    print(f"  Streaming bestvalue.csv (~640 MB) filtered to "
          f"{len(lab_ids):,} lab_ids and pathfinder species...")
    keep_species = set(PATHFINDERS_PPM)
    chunks = []
    chunk_iter = pd.read_csv(
        bv_path,
        usecols=["lab_id", "species", "unit", "qvalue"],
        dtype={"lab_id": str, "species": str, "unit": str, "qvalue": float},
        chunksize=500_000,
        low_memory=False,
    )
    for chunk in chunk_iter:
        f = chunk[
            chunk["lab_id"].isin(lab_ids)
            & chunk["species"].isin(keep_species)
        ]
        if not f.empty:
            chunks.append(f)
    bv = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["lab_id", "species", "unit", "qvalue"]
    )
    print(f"  bestvalue: {len(bv):,} pathfinder rows for AOI samples")
    return bv


def _pivot_to_wide(bv: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format bestvalue to per-sample wide with ppm-normalized columns."""
    # Convert to ppm based on unit.
    bv = bv.copy()
    if BDL_MASK_NEGATIVE:
        bv.loc[bv["qvalue"] < 0, "qvalue"] = np.nan

    unit_lower = bv["unit"].str.lower().fillna("")
    multiplier = pd.Series(1.0, index=bv.index)
    multiplier[unit_lower == "ppb"] = 0.001
    multiplier[unit_lower == "pct"] = 10000.0  # 1% = 10000 ppm
    multiplier[unit_lower.str.startswith("ppm")] = 1.0

    bv["ppm"] = bv["qvalue"] * multiplier

    # Pivot. Use mean across duplicates (same sample, same species, multiple methods).
    wide = (
        bv.pivot_table(
            index="lab_id", columns="species", values="ppm", aggfunc="mean"
        )
        .add_suffix("_ppm")
        .reset_index()
    )
    return wide


def fetch(
    aoi: AOI,
    working_crs: str,
    states: list[str],
    *,
    force: bool = False,
) -> Path:
    """Clip NGDB stream-sediment samples to AOI + states, return path to GPKG.

    Output schema:
    - geometry (in working_crs)
    - lab_id (sample_id alias)
    - sample_source, primary_class, date_collect (passthrough metadata)
    - <el>_ppm columns for each pathfinder
    """
    out_dir = dataset_dir(NAME)
    bulk_dir = out_dir / "ngdbsed"
    main_path = bulk_dir / "main.csv"
    bv_path = bulk_dir / "bestvalue.csv"

    if not main_path.exists() or not bv_path.exists():
        raise FileNotFoundError(
            f"NGDB bulk files missing under {bulk_dir}. Expected main.csv + "
            f"bestvalue.csv. Download ngdbsed-csv.zip from "
            f"https://mrdata.usgs.gov/ngdb/sediment/ngdbsed-csv.zip and "
            f"unzip into data/raw/ngdb/."
        )

    out_path = out_dir / f"ngdb_sediment_{aoi.name.lower()}.gpkg"
    if not force and out_path.exists():
        print(
            f"NGDB clipped artifact present ({out_path.stat().st_size:,} B); "
            f"skipping clip."
        )
        return out_path

    print(f"Clipping NGDB sediment for AOI={aoi.name}, states={states}...")
    main_df = _bbox_filter_main(main_path, aoi, states)
    if main_df.empty:
        raise RuntimeError(
            f"NGDB sediment clip for AOI={aoi.name} states={states} returned 0 samples"
        )

    lab_ids = set(main_df["lab_id"].astype(str))
    bv_df = _bestvalue_for_lab_ids(bv_path, lab_ids)
    wide = _pivot_to_wide(bv_df) if not bv_df.empty else pd.DataFrame({"lab_id": []})

    merged = main_df.merge(wide, on="lab_id", how="left")
    print(f"  merged: {len(merged):,} samples × {len(merged.columns)} columns")

    geom = gpd.points_from_xy(merged["long_wgs84"], merged["lat_wgs84"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(merged.drop(columns=["lat_wgs84", "long_wgs84"]),
                           geometry=geom, crs="EPSG:4326")
    gdf = gdf.to_crs(working_crs)

    gdf.to_file(out_path, driver="GPKG")
    print(f"  wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    write_source_md(
        NAME,
        title="USGS National Geochemical Database (NGDB), stream-sediment samples",
        url="https://mrdata.usgs.gov/ngdb/sediment/",
        license="US public domain (USGS)",
        notes=(
            f"ngdb_sediment_{aoi.name.lower()}.gpkg: NGDB stream-sediment "
            f"samples clipped to AOI={aoi.name} states={states}, reprojected "
            f"to {working_crs}. Pathfinder elements (Au, As, Sb, Hg, W, Ag, "
            f"Cu, Pb, Zn, Mo, Bi, Te) pivoted to <el>_ppm columns. "
            f"Below-detection-limit values (negative qvalue) masked to NaN. "
            f"Au converted from ppb to ppm where reported in ppb."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.regions.motherlode import MOTHERLODE
    fetch(MOTHERLODE.aoi, working_crs=MOTHERLODE.working_crs, states=["CA"])
