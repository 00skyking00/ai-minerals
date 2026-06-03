"""USGS NHDPlus High Resolution — staged HUC4 GDB fetch + AOI clip.

Pulls the per-HUC4 NHDPlus HR Beta GeoDatabases from the public TNM S3
bucket, joins NHDFlowline geometries to the NHDPlusFlowlineVAA Value-
Added Attribute table, AOI-buffer-clips, and writes a single GeoPackage
that the adapter consumes.

The northern-Sierra placer AOI overlaps two California HUC4s:

    1802 — Sacramento River basin
    1804 — San Joaquin River basin

Both are downloaded. Each staged zip is ~300-600 MB and the unzipped
GDB is ~1-3 GB, so we cache the GDB on disk and skip re-download when
present.

The 25-km AOI buffer is applied before the clip so downstream-traversal
features (catchment beyond the AOI edge) keep their full upstream paths
intact. This matters because the placer model walks downstream from
positives, and a fragment that ends right at the AOI edge would lose
its downstream context otherwise.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "nhd_hr"

# Public S3 bucket for TNM staged products — no auth, no rate limit
# beyond standard S3 throttling.
S3_BASE = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/"
    "NHDPlusHR/Beta/GDB"
)

# HUC4s covering the northern-Sierra placer AOI. The bbox
# (-121.55, 37.49, -119.48, 40.01) overlaps:
#   1802 Sacramento (drains Yuba/Feather/American)
#   1804 San Joaquin (drains Mokelumne/Stanislaus/Tuolumne/Merced)
NORTHERN_SIERRA_HUC4S: tuple[str, ...] = ("1802", "1804")

# Buffer the AOI by 25 km before clipping flowlines. NHDPlus uses
# geographic coords (EPSG:4269 NAD83); we buffer in a projected CRS
# (CA Albers) and reproject back for filtering.
AOI_BUFFER_M = 25_000
CA_ALBERS_CRS = "EPSG:3310"

# Chunk size for streaming downloads (16 MB).
CHUNK_BYTES = 1 << 24


def _staged_zip_name(huc4: str) -> str:
    return f"NHDPLUS_H_{huc4}_HU4_GDB.zip"


def _gdb_dir(huc4: str) -> Path:
    # When unzipped, the staged HR products produce a single .gdb folder
    # named NHDPLUS_H_<HUC4>_HU4_GDB.gdb.
    return dataset_dir(NAME) / f"NHDPLUS_H_{huc4}_HU4_GDB.gdb"


def _download_huc4(huc4: str, *, force: bool = False) -> Path:
    """Download + unzip the staged HR GDB for one HUC4. Returns the .gdb path."""
    out_dir = dataset_dir(NAME)
    zip_name = _staged_zip_name(huc4)
    zip_path = out_dir / zip_name
    gdb_path = _gdb_dir(huc4)

    if gdb_path.exists() and not force:
        print(f"NHDPlus HR HUC4 {huc4} GDB cached at {gdb_path}; skipping fetch.")
        return gdb_path

    url = f"{S3_BASE}/{zip_name}"
    if not zip_path.exists() or force:
        print(f"Downloading {url}  (~hundreds of MB)")
        with requests.get(url, stream=True, timeout=1800) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                    if chunk:
                        f.write(chunk)
        print(f"  wrote {zip_path} ({zip_path.stat().st_size:,} bytes)")

    print(f"Unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    if not gdb_path.exists():
        # The staged products sometimes nest the .gdb one level down.
        # Recover by finding any .gdb directory matching this HUC4 that
        # was just produced.
        candidates = list(out_dir.glob(f"**/NHDPLUS_H_{huc4}_HU4_GDB.gdb"))
        if not candidates:
            raise RuntimeError(
                f"Unzipped {zip_path.name} but no GDB found at expected "
                f"{gdb_path}; saw {list(out_dir.iterdir())}"
            )
        return candidates[0]
    return gdb_path


def _pick_layer(gdb: Path, candidates: tuple[str, ...]) -> str:
    """Return the first candidate layer name present in the GDB."""
    available = set(fiona.listlayers(gdb))
    for name in candidates:
        if name in available:
            return name
    raise RuntimeError(
        f"None of {candidates} present in {gdb}. Available: {sorted(available)}"
    )


def _load_huc4_flowlines(huc4: str, aoi_buffered_4269) -> gpd.GeoDataFrame:
    """Read NHDFlowline + NHDPlusFlowlineVAA from one HUC4 GDB and join."""
    gdb = _download_huc4(huc4)

    flow_layer = _pick_layer(gdb, ("NHDFlowline",))
    # Older HR snapshots ship the VAA as NHDPlusFlowlineVAA; some
    # mirrors use the unprefixed variant.
    vaa_layer = _pick_layer(gdb, ("NHDPlusFlowlineVAA", "NHDFlowlineVAA"))

    print(f"  HUC4 {huc4}: reading {flow_layer}...")
    flow = gpd.read_file(gdb, layer=flow_layer)
    print(f"    {len(flow):,} flowlines pre-clip")

    # Drop flowlines whose envelope doesn't intersect the buffered AOI
    # before joining VAA — VAA is large (hundreds of MB in memory) and
    # we don't want to carry rows we'll throw out.
    if flow.crs is None:
        # NHDPlus HR ships in EPSG:4269 NAD83; assume that if unset.
        flow = flow.set_crs("EPSG:4269")
    aoi_in_flow_crs = (
        gpd.GeoSeries([aoi_buffered_4269], crs="EPSG:4326")
        .to_crs(flow.crs)
        .iloc[0]
    )
    flow = flow[flow.intersects(aoi_in_flow_crs)].copy()
    print(f"    {len(flow):,} flowlines after AOI-buffer envelope filter")
    if flow.empty:
        return flow

    print(f"  HUC4 {huc4}: reading {vaa_layer}...")
    vaa = gpd.read_file(gdb, layer=vaa_layer)
    # VAA is a non-spatial attribute table; geopandas reads it as a
    # GeoDataFrame with no geometry column. Drop the geometry if present
    # so the join doesn't get confused.
    if "geometry" in vaa.columns:
        vaa = pd.DataFrame(vaa.drop(columns="geometry"))

    # Pick a join key. HR uses NHDPlusID (float64); older legacy snapshots
    # used COMID. Normalize both sides to a canonical "nhdplusid" column.
    flow_key = next(
        (c for c in ("NHDPlusID", "nhdplusid", "COMID", "comid") if c in flow.columns),
        None,
    )
    vaa_key = next(
        (c for c in ("NHDPlusID", "nhdplusid", "COMID", "comid") if c in vaa.columns),
        None,
    )
    if flow_key is None or vaa_key is None:
        raise RuntimeError(
            f"Cannot find join key in HUC4 {huc4} flow={list(flow.columns)} "
            f"vaa={list(vaa.columns)}"
        )

    # Coerce to integer for a clean join — NHDPlusID is stored as
    # float64 in the GDB to fit ESRI's numeric type system.
    flow["_join_id"] = flow[flow_key].astype("int64")
    vaa["_join_id"] = vaa[vaa_key].astype("int64")

    joined = flow.merge(vaa, on="_join_id", how="left", suffixes=("", "_vaa"))
    print(f"    {len(joined):,} joined rows; "
          f"{joined['ArbolateSum'].notna().sum() if 'ArbolateSum' in joined.columns else '?'} with ArbolateSum")
    return joined


def fetch(aoi: AOI, *, force: bool = False) -> Path:
    """Download NHDPlus HR HUC4s overlapping aoi, join VAA, write a clipped GeoPackage.

    The output GeoPackage carries: comid (int64), arbolate_sum (float),
    stream_order (int), fcode (int), hydroseq (int), geometry (LineString,
    EPSG:4326). Clip uses a 25-km buffer around aoi.bbox so downstream-
    traversal beyond the AOI keeps working.

    `force=True` re-downloads the staged zips even if a cached GDB exists.
    """
    out_dir = dataset_dir(NAME)
    out_path = out_dir / "nhd_flowlines_northern_sierra.gpkg"

    if out_path.exists() and not force:
        print(
            f"NHDPlus HR clipped GeoPackage already at {out_path} "
            f"({out_path.stat().st_size:,} bytes); skipping."
        )
        return out_path

    # Build the buffered AOI in EPSG:4326 (we use the bbox polygon as
    # the buffer source, not a tighter geometry — fetch is bbox-driven).
    aoi_box = box(*aoi.bbox)
    aoi_buf_proj = (
        gpd.GeoSeries([aoi_box], crs="EPSG:4326")
        .to_crs(CA_ALBERS_CRS)
        .buffer(AOI_BUFFER_M)
        .iloc[0]
    )
    aoi_buf_4326 = (
        gpd.GeoSeries([aoi_buf_proj], crs=CA_ALBERS_CRS)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    print(
        f"AOI={aoi.name} bbox={aoi.bbox}; buffered by {AOI_BUFFER_M/1000:.0f} km "
        f"for flowline clip"
    )

    pieces: list[gpd.GeoDataFrame] = []
    for huc4 in NORTHERN_SIERRA_HUC4S:
        piece = _load_huc4_flowlines(huc4, aoi_buf_4326)
        if not piece.empty:
            pieces.append(piece)

    if not pieces:
        raise RuntimeError(
            f"No NHDPlus HR flowlines found in AOI={aoi.name} "
            f"across HUC4s {NORTHERN_SIERRA_HUC4S}"
        )

    # Concatenate across HUC4s in their native CRS (all HR HUC4 GDBs ship
    # in EPSG:4269 NAD83) then reproject to WGS84 for the output file.
    src_crs = pieces[0].crs
    merged = gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True), crs=src_crs
    )

    # Final precise clip against the buffered AOI polygon (the per-HUC
    # filter above used envelope intersects; tighten to the buffered
    # polygon now).
    aoi_in_src = (
        gpd.GeoSeries([aoi_buf_4326], crs="EPSG:4326")
        .to_crs(src_crs)
        .iloc[0]
    )
    merged = merged[merged.intersects(aoi_in_src)].copy()
    print(f"Merged + clipped: {len(merged):,} flowlines")

    # Project to WGS84 for the output GeoPackage.
    merged = merged.to_crs("EPSG:4326")

    # Pick canonical column names. NHDPlus HR ships ArbolateSum (km²
    # historically, km of channel in modern HR), StreamOrde (Strahler),
    # Hydroseq (deterministic walk order), and FCode (NHD feature
    # classification, e.g. 46006 perennial stream, 46003 intermittent).
    def _col(df, *names, default=None):
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df), index=df.index)

    out = gpd.GeoDataFrame(
        {
            "geometry": merged.geometry,
            "comid": merged["_join_id"].astype("int64"),
            # NHDPlus HR's VAA ships column names truncated to 10 chars per
            # the legacy ESRI .dbf-style schema: ArbolateSum -> ArbolateSu,
            # not ArbolateSum. Same with HydroSeq (CamelCase 'S' in 'Seq').
            # Tolerate both spellings + various casings.
            "arbolate_sum": pd.to_numeric(
                _col(merged, "ArbolateSu", "ArbolateSum", "arbolatesu", "arbolatesum"),
                errors="coerce",
            ).astype("float64"),
            "stream_order": pd.to_numeric(
                _col(merged, "StreamOrde", "streamorde"), errors="coerce"
            ).fillna(0).astype("int64"),
            "fcode": pd.to_numeric(
                _col(merged, "FCode", "fcode"), errors="coerce"
            ).fillna(0).astype("int64"),
            "hydroseq": pd.to_numeric(
                _col(merged, "HydroSeq", "Hydroseq", "hydroseq"), errors="coerce"
            ).fillna(0).astype("int64"),
            # NHDPlus HR ships a per-flowline longitudinal slope (dimensionless,
            # rise/run) in the VAA table. Used as a cell-level Quaternary feature
            # in v3 Phase D.2 (nearest-reach snap). Some reaches have NaN slope
            # in the source; preserve NaN rather than zero-fill so downstream
            # consumers can distinguish "unknown" from "flat".
            "slope": pd.to_numeric(
                _col(merged, "Slope", "slope"), errors="coerce"
            ).astype("float64"),
        },
        crs="EPSG:4326",
    )

    out.to_file(out_path, driver="GPKG")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    write_source_md(
        NAME,
        title="USGS NHDPlus High Resolution (Beta) — staged HUC4 GDBs",
        url=f"{S3_BASE}/",
        license="US public domain (USGS)",
        notes=(
            f"NHDPlus HR Beta staged products downloaded by HUC4 from the "
            f"public TNM S3 bucket. HUC4s fetched: "
            f"{', '.join(NORTHERN_SIERRA_HUC4S)} (Sacramento + San Joaquin, "
            f"covering the northern-Sierra placer AOI). NHDFlowline geometry "
            f"joined to NHDPlusFlowlineVAA on NHDPlusID. Clipped to AOI "
            f"bbox={aoi.bbox} buffered by {AOI_BUFFER_M/1000:.0f} km so "
            f"downstream-traversal beyond the AOI edge still works. "
            f"Output columns: geometry (LineString, EPSG:4326), comid "
            f"(int64 NHDPlusID), arbolate_sum (float, cumulative upstream "
            f"channel length per NHDPlus HR), stream_order (int, Strahler "
            f"order), fcode (int, NHD feature classification), hydroseq "
            f"(int, NHDPlus downstream-walk key), slope (float, per-flowline "
            f"longitudinal slope, dimensionless rise/run, NaN preserved). "
            f"Cached GDBs remain on disk under data/raw/{NAME}/ for re-runs."
        ),
    )
    return out_path


if __name__ == "__main__":
    nsierra = AOI(
        name="NorthernSierra",
        min_lon=-121.55,
        min_lat=37.49,
        max_lon=-119.48,
        max_lat=40.01,
    )
    p = fetch(nsierra)
    print(f"Done: {p}")
