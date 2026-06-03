"""Assemble the per-cell placer-Au feature DataFrame for the northern Sierra.

Phase E of the placer pipeline. Builds on `build_feature_frame` (which
already wires up DEM derivatives, geophysics, CGS-2010 lithology,
distance-to-fault, NGDB pathfinder aggregations, and is_<class> label
columns) and extends with the placer-specific columns:

  - flow_acc, spi_band, twi, ksn, tpi               (DEM-derived hydrology)
  - geomorphon_terrace_mask                          (GRASS r.geomorphon)
  - paleochannel_likelihood                          (precomputed raster)
  - distance_downstream_from_lode_m                  (NHD-network walk)
  - hydraulic_pit_proximity_m                        (USGS Orlando 2016)
  - is_quaternary_alluvium                           (CGS 2010 PTYPE)
  - catchment_au_hawkes, _as_hawkes, _sb_hawkes      (Hawkes 1976 dual-decay)

Lawley audit discipline: count-style exploration-density features
(`*_count_5km`, `*_has_data_5km`) are dropped before write.

Output: `data/derived/features_northern_sierra_placer_250m.parquet`

Usage:
    .venv/bin/python scripts/northern_sierra_placer_assemble_250m.py
    .venv/bin/python scripts/northern_sierra_placer_assemble_250m.py --resolution-m 500
    .venv/bin/python scripts/northern_sierra_placer_assemble_250m.py --augment-osm-mining

v3 Phase D.4: `--augment-osm-mining` adds OSM mining-tag points
(historic=mine, man_made=mineshaft|adit, landuse=quarry) to
`is_placer_quaternary` as supplementary positive labels (not features).
The OSM data is OpenStreetMap, licensed ODbL 1.0; any downstream product
that uses this augmentation MUST carry the attribution "(c) OpenStreetMap
contributors" in the model card. See
`scripts/northern_sierra_placer_fetch_osm_mining.py` for the fetcher.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from ai_minerals.data._common import DATA_RAW
from ai_minerals.data.adapters import get_adapter
from ai_minerals.features.assemble import build_feature_frame
from ai_minerals.features.hydrology import (
    distance_downstream_from_lode,
    distance_to_lode_m,
    flow_accumulation,
    geomorphon_terrace_mask,
    knickpoint_ksn,
    plan_curvature,
    profile_curvature,
    stream_power_index_band,
    topographic_wetness_index,
    tpi,
)
from ai_minerals.features.placer_geology import (
    distance_to_lithological_contact_m,
    hawkes_dual_decay_catchment,
    hydraulic_pit_proximity_m,
    hydraulic_pit_proximity_m_buffered,
    is_quaternary_alluvium,
    tertiary_terrace_likelihood,
)
from ai_minerals.features.rasters import sample_raster, slope_and_tri
from ai_minerals.grid import build_grid
from ai_minerals.model_rf import count_feature_columns
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER
DEFAULT_RES_M = 250

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DERIVED = REPO_ROOT / "data" / "derived"


def _placer_dep_type_mask(dep_type: pd.Series) -> pd.Series:
    """True for MRDS records whose dep_type matches the placer regex.

    Mirrors `data/adapters/occurrences/mrds.py::_PLACER_DEP_TYPE_RE` so the
    lode-seed filter here drops the same rows the canonical occurrence
    adapter would have classified as placer.
    """
    pattern = r"placer|alluvial|stream.?placer|paleo.?placer|black.?sand|residual|eluvial"
    dep = dep_type.astype("string").fillna("").str.lower()
    return dep.str.contains(pattern, regex=True, na=False)


def _load_lode_mrds_with_dep_type(path: Path) -> gpd.GeoDataFrame:
    """Read the MRDS GPKG raw so dep_type + dev_stat survive for the leakage guard.

    The canonical MRDS adapter strips dep_type/dev_stat to fit the
    occurrences schema; the distance-downstream-from-lode feature needs
    them for filtering and for its placer-leakage assertion.
    """
    gdf = gpd.read_file(path)
    if "dep_type" not in gdf.columns:
        gdf["dep_type"] = pd.Series([None] * len(gdf), dtype="string")
    if "dev_stat" not in gdf.columns:
        gdf["dev_stat"] = pd.Series([None] * len(gdf), dtype="string")
    return gdf


def _filter_lode_seeds(mrds: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep Past Producer / Producer Au lode rows, drop placer dep_types.

    The downstream `distance_downstream_from_lode` re-checks the placer
    filter and refuses to run if any placer dep_type survives. We pre-drop
    here and also keep the column on the frame so the guard sees it.
    """
    dev = mrds["dev_stat"].astype("string").fillna("")
    keep_dev = dev.isin(["Past Producer", "Producer"])
    is_placer = _placer_dep_type_mask(mrds["dep_type"])
    keep = keep_dev & ~is_placer
    out = mrds.loc[keep].copy()
    print(f"  lode_mrds: {len(mrds)} → {len(out)} after dev_stat + dep_type filter")
    return out


def _row_col_to_flat(df: pd.DataFrame, ncols: int) -> np.ndarray:
    """Map (row, col) pairs in df back to flat indices in row-major grid order."""
    return (df["row"].to_numpy(dtype=np.int64) * ncols
            + df["col"].to_numpy(dtype=np.int64))


def _maybe_attach(
    df: pd.DataFrame,
    name: str,
    values_full_grid: np.ndarray | pd.Series | None,
    flat_idx: np.ndarray,
) -> None:
    """Attach a column to `df` by gathering from a full-grid Series/array.

    `values_full_grid` is either:
      - a 1-D array/Series indexed in row-major (row, col) order over the
        full unclipped grid (length = grid.n_cells), or
      - None (the upstream computation was skipped); the column is filled
        with NaN.
    """
    if values_full_grid is None:
        df[name] = np.full(len(df), np.nan, dtype=np.float32)
        return
    arr = np.asarray(values_full_grid).reshape(-1)
    df[name] = arr[flat_idx]


def _dem_array_at_grid(dem_path: Path, grid) -> tuple[np.ndarray, "from_origin"]:
    """Sample the DEM onto `grid` and return the array + a matching transform."""
    dem = sample_raster(dem_path, grid)
    # grid.xs ascending; grid.ys ascending (south->north). The on-disk raster
    # convention is north-up, so the top-left corner sits at
    # (xs[0] - r/2, ys[-1] + r/2). Build a matching Affine.
    r = grid.resolution_m
    transform = from_origin(grid.xs[0] - r / 2, grid.ys[-1] + r / 2, r, r)
    # `sample_raster` returns the array oriented to ascending ys; flip to
    # north-up so the transform aligns with on-disk raster conventions.
    dem_north_up = dem[::-1, :]
    return dem_north_up, transform


def _ravel_north_up_to_grid_order(arr_north_up: np.ndarray) -> np.ndarray:
    """Flip a north-up array back to ascending-y order, then ravel row-major.

    `build_feature_frame` constructs its row/col columns as
    `np.repeat(arange(n_rows), n_cols)` paired with the ascending grid.ys
    convention used by `build_grid` and `grid.centroid_gdf`. North-up
    rasters (as returned from rasterio/whitebox) need a y-flip before they
    ravel in the same order.
    """
    return arr_north_up[::-1, :].ravel()


def _combine_geochem_samples(
    ngdb: gpd.GeoDataFrame, nure: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Concatenate NGDB + NURE samples, keeping shared element columns.

    Both adapters return canonical-schema frames with `<el>_ppm` columns.
    geopandas.concat reconciles non-matching columns by filling NaN, which
    is the right behaviour: a sample only contributes for elements where
    it has a measurement.
    """
    if ngdb is None and nure is None:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if ngdb is None:
        return nure
    if nure is None:
        return ngdb
    if nure.crs is not None and ngdb.crs is not None and ngdb.crs != nure.crs:
        nure = nure.to_crs(ngdb.crs)
    out = pd.concat([ngdb, nure], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=ngdb.crs)


def _inject_placer_positives(
    df: pd.DataFrame,
    *,
    resolution_m: int,
    reclassify_mrds: bool = True,
    augment_osm_mining: bool = False,
) -> pd.DataFrame:
    """Overwrite is_placer_tertiary / is_placer_quaternary with real positives.

    The canonical MRDS adapter is lode-Au-tuned: it assigns codes 36a/36b for
    Au commodity and DROPS placer dep_types (returns empty deposit_codes).
    The placer Region's deposit_classes use codes 39a/39b which the adapter
    never emits, so build_feature_frame leaves both label columns all-zero.

    This function injects the real labels:
      Tertiary  (is_placer_tertiary):  hydraulic-pit polygon centroids
                                       (Hydraulic Mine Pits of California,
                                        DOI 10.5066/F7J38QMD).
      Quaternary (is_placer_quaternary): MRDS records with dep_type matching
                                          the placer regex from
                                          adapters/occurrences/mrds.py.

    Rows are matched by spatial proximity: each positive geometry is snapped
    to its nearest grid cell (the one whose (x, y) centroid is closest in
    the working CRS).

    v3 reclassification (B.0a, opt-in via `reclassify_mrds=True`, default):
    after identifying MRDS placer records, reclassify each as Tertiary if:
      - it sits within 500 m of any Orlando 2016 pit polygon (clearly a
        deep-gravel district), OR
      - it sits on a CGS 2010 PTYPE in {Tvp, Tv} (Tertiary volcanic cap that
        overlies buried Tertiary auriferous gravels).
    Otherwise: Quaternary. Records on bedrock PTYPE (Pz, J, grMz, Mzv, etc.)
    are correctly Quaternary by default — they are bedrock-paystreak placers
    in incised streams where the gravel veneer is too thin for CGS 2010's
    1:750,000 mapping to resolve (the mining works the bedrock-gravel
    contact, but the deposit itself is a Quaternary surficial gravel that
    just isn't mapped at this resolution). v2 treated all MRDS placers as
    Quaternary; v3 only reassigns the ones with explicit Tertiary signal.
    Preserve v2 behavior with `reclassify_mrds=False`.
    """
    df = df.copy()
    from ai_minerals.data.adapters.occurrences.mrds import _PLACER_DEP_TYPE_RE
    from ai_minerals.grid import build_grid

    grid = build_grid(REGION.aoi, resolution_m=resolution_m, working_crs=REGION.working_crs)
    centroids = grid.centroid_gdf()  # has row, col, x, y, geometry in working CRS

    # Build a lookup from (row, col) -> df row index so we can write labels
    # straight into the AOI-clipped df.
    rc_to_idx: dict[tuple[int, int], int] = {
        (int(r), int(c)): int(i) for i, (r, c) in enumerate(zip(df["row"].values, df["col"].values))
    }

    def _snap_to_cells(points: gpd.GeoDataFrame) -> list[int]:
        """Snap each point to its nearest grid cell; return df row indices (may dedup)."""
        if points.empty:
            return []
        pts = points.to_crs(REGION.working_crs)
        joined = gpd.sjoin_nearest(
            pts[["geometry"]].reset_index(drop=True),
            centroids[["row", "col", "geometry"]],
            how="left",
            distance_col="_d",
        )
        out: list[int] = []
        for _, row in joined.iterrows():
            key = (int(row["row"]), int(row["col"]))
            if key in rc_to_idx:
                out.append(rc_to_idx[key])
        return out

    # ---- Tertiary: hydraulic-pit polygon centroids.
    pit_path = REGION.raw_paths.get("hydraulic_pits")
    if pit_path is not None and pit_path.exists():
        pit_polys = get_adapter("geology", "hydraulic_pits")(pit_path, REGION.aoi)
        pit_centroids = gpd.GeoDataFrame(
            geometry=pit_polys.to_crs(REGION.working_crs).geometry.centroid,
            crs=REGION.working_crs,
        )
        t_idxs = _snap_to_cells(pit_centroids)
        df["is_placer_tertiary"] = 0
        if t_idxs:
            df.loc[t_idxs, "is_placer_tertiary"] = 1
        print(f"[assemble]   is_placer_tertiary: {df['is_placer_tertiary'].sum()} cells "
              f"({len(pit_centroids)} pit centroids, {len(set(t_idxs))} unique cells)")
    else:
        print("[assemble]   is_placer_tertiary: no pit polygons; left at 0")

    # ---- Quaternary: MRDS placer-flagged records.
    occ_path = REGION.raw_paths.get("occurrences")
    if occ_path is not None and occ_path.exists():
        mrds_raw = gpd.read_file(occ_path)
        if "dep_type" in mrds_raw.columns:
            placer_mask = mrds_raw["dep_type"].fillna("").str.contains(
                _PLACER_DEP_TYPE_RE, regex=True, na=False,
            )
            placer = mrds_raw[placer_mask].copy()
            placer = placer[placer.geometry.notna()]

            # v3 reclassification: split MRDS placers into Tertiary vs Quaternary
            # using the rules in the function docstring. v2 behavior (all MRDS
            # placers -> Quaternary) is reproduced by reclassify_mrds=False.
            if reclassify_mrds and pit_path is not None and pit_path.exists():
                placer_proj = placer.to_crs(REGION.working_crs)
                pit_polys_proj = pit_polys.to_crs(REGION.working_crs)
                pit_buffer = pit_polys_proj.geometry.union_all().buffer(500.0)
                near_pit = placer_proj.geometry.within(pit_buffer).to_numpy()

                # Sample CGS PTYPE under each placer record
                geo_path = REGION.raw_paths.get("geology")
                if geo_path is not None and geo_path.exists():
                    geo = gpd.read_file(geo_path).to_crs(REGION.working_crs)
                    if "PTYPE" in geo.columns:
                        on_geo = gpd.sjoin(
                            placer_proj[["geometry"]].reset_index(),
                            geo[["PTYPE", "geometry"]],
                            how="left", predicate="within",
                        )
                        on_geo = on_geo.drop_duplicates("index", keep="first")
                        on_geo = on_geo.set_index("index").reindex(placer_proj.index)
                        on_tertiary_volcanic = on_geo["PTYPE"].isin({"Tvp", "Tv"}).to_numpy()
                    else:
                        on_tertiary_volcanic = np.zeros(len(placer_proj), dtype=bool)
                else:
                    on_tertiary_volcanic = np.zeros(len(placer_proj), dtype=bool)

                is_tertiary_record = near_pit | on_tertiary_volcanic
                t_records = placer_proj[is_tertiary_record]
                q_records = placer_proj[~is_tertiary_record]
                print(f"[assemble]   v3 MRDS reclassification: "
                      f"{int(near_pit.sum())} within 500 m of a pit polygon, "
                      f"{int(on_tertiary_volcanic.sum())} on Tvp/Tv PTYPE, "
                      f"union = {int(is_tertiary_record.sum())} reclassified Tertiary; "
                      f"{int((~is_tertiary_record).sum())} kept as Quaternary")
            else:
                t_records = placer.iloc[:0]
                q_records = placer

            # Quaternary positives (the records NOT reclassified to Tertiary)
            q_idxs = _snap_to_cells(q_records)
            df["is_placer_quaternary"] = 0
            if q_idxs:
                df.loc[q_idxs, "is_placer_quaternary"] = 1
            print(f"[assemble]   is_placer_quaternary: {df['is_placer_quaternary'].sum()} cells "
                  f"({len(q_records)} MRDS records, {len(set(q_idxs))} unique cells)")

            # Augment Tertiary with the reclassified records
            if reclassify_mrds and len(t_records) > 0:
                aug_idxs = _snap_to_cells(t_records)
                if aug_idxs:
                    df.loc[aug_idxs, "is_placer_tertiary"] = 1
                aug_unique = len(set(aug_idxs))
                base_unique = int(df["is_placer_tertiary"].sum()) - aug_unique
                # base_unique may not match exactly because cell-snap can put a
                # reclassified record on a pit cell (double-counted). Defensive math.
                print(f"[assemble]   is_placer_tertiary: {df['is_placer_tertiary'].sum()} cells "
                      f"(after +{aug_unique} reclassified MRDS records snapped to unique cells)")
        else:
            print("[assemble]   is_placer_quaternary: MRDS has no dep_type column; left at 0")
    else:
        print("[assemble]   is_placer_quaternary: no MRDS file; left at 0")

    # ---- v3 D.4: optional OSM mining-tag augmentation for Quaternary.
    # The four Overpass tags (historic=mine, man_made=mineshaft/adit,
    # landuse=quarry) are sparse but high-precision. We snap each feature
    # geometry (points or polygon centroids) to the nearest grid cell and
    # OR it into is_placer_quaternary. License: OpenStreetMap contributors,
    # ODbL 1.0; model card MUST carry the attribution.
    if augment_osm_mining:
        osm_path = DATA_RAW / "osm_mining" / f"osm_mining_{REGION.data_prefix}.gpkg"
        if osm_path.exists():
            osm = gpd.read_file(osm_path)
            osm = osm[osm.geometry.notna()].copy()
            if osm.crs is not None and osm.crs.to_string() != REGION.working_crs:
                osm = osm.to_crs(REGION.working_crs)
            else:
                osm = osm.set_crs(REGION.working_crs) if osm.crs is None else osm
            # Reduce polygons to centroids so _snap_to_cells gets point geometry.
            osm_pts = gpd.GeoDataFrame(
                geometry=osm.geometry.centroid,
                crs=REGION.working_crs,
            )
            before = int(df["is_placer_quaternary"].sum())
            osm_idxs = _snap_to_cells(osm_pts)
            if osm_idxs:
                df.loc[osm_idxs, "is_placer_quaternary"] = 1
            after = int(df["is_placer_quaternary"].sum())
            print(f"[assemble]   OSM mining-tag augmentation: {len(osm_pts)} features, "
                  f"+{after - before} new Quaternary cells "
                  f"({after} total). License: ODbL; attribute (c) OpenStreetMap "
                  f"contributors in the model card.")
        else:
            print(f"[assemble]   OSM mining augmentation requested but file not "
                  f"present ({osm_path}); skipping. Run "
                  f"scripts/northern_sierra_placer_fetch_osm_mining.py first.")

    return df


def _build_placer_columns(
    df: pd.DataFrame,
    *,
    resolution_m: int,
) -> pd.DataFrame:
    """Compute the placer-specific columns and join them onto df by (row, col)."""
    grid = build_grid(REGION.aoi, resolution_m=resolution_m, working_crs=REGION.working_crs)
    n_cells_full = grid.n_cells
    flat_idx = _row_col_to_flat(df, ncols=grid.shape[1])
    # df is AOI-clipped (build_feature_frame masks to AOI); grid is the bbox
    # rectangle. We don't require len(df) == grid.n_cells, but every df cell's
    # flat index MUST lie inside the grid (downstream scatter into grid-sized
    # arrays assumes this). Silent misalignment from a shape mismatch would
    # corrupt the downstream sampling.
    assert flat_idx.max() < grid.n_cells, (
        f"flat_idx out of range: max={int(flat_idx.max())} vs n_cells={grid.n_cells}"
    )
    assert flat_idx.min() >= 0, f"flat_idx negative: min={int(flat_idx.min())}"

    # ---- DEM-derived hydrology (slope is already on df from build_feature_frame).
    print("[placer] DEM + hydrology derivatives")
    dem_path = REGION.raw_paths["dem"]
    if not dem_path.exists():
        warnings.warn(f"DEM missing at {dem_path}; hydrology features all NaN.")
        _maybe_attach(df, "flow_acc", None, flat_idx)
        _maybe_attach(df, "spi_band", None, flat_idx)
        _maybe_attach(df, "twi", None, flat_idx)
        _maybe_attach(df, "ksn", None, flat_idx)
        _maybe_attach(df, "tpi", None, flat_idx)
    else:
        dem_north_up, transform = _dem_array_at_grid(dem_path, grid)
        slope_north_up, _tri = slope_and_tri(dem_north_up, grid.resolution_m)
        try:
            flow_acc_north_up = flow_accumulation(dem_north_up, transform=transform)
        except RuntimeError as exc:
            warnings.warn(
                f"flow_accumulation unavailable ({exc}); SPI/TWI/ksn fall back to NaN."
            )
            flow_acc_north_up = None

        if flow_acc_north_up is None:
            _maybe_attach(df, "flow_acc", None, flat_idx)
            _maybe_attach(df, "spi_band", None, flat_idx)
            _maybe_attach(df, "twi", None, flat_idx)
            _maybe_attach(df, "ksn", None, flat_idx)
        else:
            spi = stream_power_index_band(flow_acc_north_up, slope_north_up)
            twi = topographic_wetness_index(flow_acc_north_up, slope_north_up)
            ksn = knickpoint_ksn(dem_north_up, flow_acc_north_up)
            _maybe_attach(df, "flow_acc",
                          _ravel_north_up_to_grid_order(flow_acc_north_up), flat_idx)
            _maybe_attach(df, "spi_band",
                          _ravel_north_up_to_grid_order(spi), flat_idx)
            _maybe_attach(df, "twi",
                          _ravel_north_up_to_grid_order(twi), flat_idx)
            _maybe_attach(df, "ksn",
                          _ravel_north_up_to_grid_order(ksn), flat_idx)

        tpi_north_up = tpi(dem_north_up, radius_cells=8)
        _maybe_attach(df, "tpi",
                      _ravel_north_up_to_grid_order(tpi_north_up), flat_idx)

        # v3 Phase B.6: second-derivative curvature features. plan_curvature
        # distinguishes ridges (positive) from valleys (negative); profile
        # curvature picks out convex/concave breaks of slope. Both are
        # Tertiary-relevant (flat-bench vs valley-side discrimination).
        # huc12_id is deferred to v4 (need WBD HUC12 polygon download; the
        # NHDPlus HR GeoPackage at REGION.raw_paths["nhd_flowlines"] only
        # ships flowlines, not the WBD HUC12 layer).
        plan_curv_north_up = plan_curvature(dem_north_up, sigma_cells=2.0)
        prof_curv_north_up = profile_curvature(dem_north_up, sigma_cells=2.0)
        _maybe_attach(df, "plan_curvature",
                      _ravel_north_up_to_grid_order(plan_curv_north_up), flat_idx)
        _maybe_attach(df, "profile_curvature",
                      _ravel_north_up_to_grid_order(prof_curv_north_up), flat_idx)

    # ---- Geomorphon terrace mask (GRASS r.geomorphon; slow on first run).
    print("[placer] geomorphon terrace mask")
    if not dem_path.exists():
        _maybe_attach(df, "geomorphon_terrace_mask", None, flat_idx)
    else:
        try:
            mask = geomorphon_terrace_mask(dem_path)
        except RuntimeError as exc:
            warnings.warn(
                f"geomorphon_terrace_mask unavailable ({exc}); column filled with NaN."
            )
            mask = None
        if mask is None:
            _maybe_attach(df, "geomorphon_terrace_mask", None, flat_idx)
        else:
            # geomorphon_terrace_mask returns an array shaped like the on-disk
            # DEM (north-up). The DEM-at-grid path resampled the DEM with the
            # same transform, so the geomorphon array shares orientation iff
            # the DEM was already at grid resolution. To be safe, resample the
            # cached geomorphon raster onto the grid via the standard path.
            mask_grid = sample_raster(
                dem_path.with_suffix(dem_path.suffix + ".geomorphon.tif"),
                grid,
            )
            # Reduce to terrace classes (1, 7, 8) → {0, 1} after resampling.
            mask_binary = np.isin(
                np.rint(mask_grid).astype(np.int32), (1, 7, 8)
            ).astype(np.float32)
            # mask_binary is in ascending-y (grid.ys) order from sample_raster.
            _maybe_attach(df, "geomorphon_terrace_mask",
                          mask_binary.ravel(), flat_idx)

    # ---- Paleochannel-likelihood raster (precomputed by Phase D).
    print("[placer] paleochannel likelihood")
    pc_path = REGION.raw_paths.get("paleochannel_likelihood")
    if pc_path is None or not pc_path.exists():
        warnings.warn(
            f"paleochannel_likelihood raster missing at {pc_path}; "
            "fill with NaN (Phase D not yet run)."
        )
        _maybe_attach(df, "paleochannel_likelihood", None, flat_idx)
    else:
        # Area-average, not the sample_raster default (bilinear). The paleochannel
        # raster is much finer than the 250 m grid; bilinear point-samples ~4 of
        # the ~625 source cells per target cell and aliases. average aggregates
        # the whole footprint, which is what we want for a continuous morphometric
        # signal being coarsened.
        pc_arr = sample_raster(pc_path, grid, resampling=Resampling.average)
        _maybe_attach(df, "paleochannel_likelihood", pc_arr.ravel(), flat_idx)

    # ---- Distance downstream from lode-Au along NHD flowlines.
    print("[placer] distance downstream from lode")
    lode_path = REGION.raw_paths.get("lode_mrds")
    nhd_path = REGION.raw_paths.get("nhd_flowlines")
    if (
        lode_path is None or not lode_path.exists()
        or nhd_path is None or not nhd_path.exists()
    ):
        warnings.warn(
            "lode_mrds and/or nhd_flowlines missing; "
            "distance_downstream_from_lode_m filled with NaN."
        )
        _maybe_attach(df, "distance_downstream_from_lode_m", None, flat_idx)
    else:
        lode_raw = _load_lode_mrds_with_dep_type(lode_path)
        lode_seeds = _filter_lode_seeds(lode_raw)
        nhd = get_adapter("hydrology", "nhdplus_hr")(nhd_path, REGION.aoi)
        dist_km = distance_downstream_from_lode(lode_seeds, nhd, grid)
        # Returned in km; the column name (per spec) is in meters.
        dist_m = (dist_km.to_numpy(dtype=np.float32) * 1000.0)
        if dist_m.shape[0] != n_cells_full:
            raise RuntimeError(
                f"distance_downstream_from_lode returned {dist_m.shape[0]} "
                f"entries; expected {n_cells_full} (grid.n_cells)."
            )
        _maybe_attach(df, "distance_downstream_from_lode_m", dist_m, flat_idx)

        # Omnidirectional companion: Sierra Tertiary deep-gravels are paleo-
        # channels offset laterally / upstream of the modern Mother Lode trend,
        # so the flow-routed feature is NaN at every anchor. Straight-line
        # distance fires everywhere within the cap and carries the lode-
        # proximity signal the downstream variant intends.
        print("[placer] distance to lode (omnidirectional)")
        dist_eu = distance_to_lode_m(lode_seeds, grid)
        _maybe_attach(df, "distance_to_lode_m",
                      dist_eu.to_numpy(dtype=np.float32), flat_idx)

    # ---- NHD VAA: per-cell reach attributes (v3 Phase D.2).
    # Cell-level features derived from the nearest NHD reach. Quaternary-
    # relevant; Tertiary cells far from modern reaches get NaN.
    print("[placer] NHD VAA snap (stream_order, arbolate_sum, slope)")
    if nhd_path is not None and nhd_path.exists():
        # Reproject NHD into the grid's working CRS (meters) so sjoin_nearest's
        # distance_col is in meters as the column name implies, and so the
        # "nearest" reach is picked under an isotropic projected metric rather
        # than a lat-lon-anisotropic degree metric.
        nhd_for_vaa = (
            get_adapter("hydrology", "nhdplus_hr")(nhd_path, REGION.aoi)
            .to_crs(grid.crs)
        )
        centroids = grid.centroid_gdf()
        vaa_cols = [c for c in ["stream_order", "arbolate_sum", "slope", "hydroseq"]
                    if c in nhd_for_vaa.columns]
        joined = gpd.sjoin_nearest(
            centroids[["row", "col", "geometry"]],
            nhd_for_vaa[vaa_cols + ["geometry"]],
            how="left",
            distance_col="distance_to_nearest_reach_m",
        )
        joined = joined.loc[~joined.index.duplicated(keep="first")].sort_index()
        if "stream_order" in joined.columns:
            so = pd.to_numeric(joined["stream_order"], errors="coerce").to_numpy()
            _maybe_attach(df, "nearest_reach_stream_order", so, flat_idx)
        else:
            _maybe_attach(df, "nearest_reach_stream_order", None, flat_idx)
        if "arbolate_sum" in joined.columns:
            ar = pd.to_numeric(joined["arbolate_sum"], errors="coerce").to_numpy()
            _maybe_attach(df, "nearest_reach_arbolate_sum_km", ar, flat_idx)
        else:
            _maybe_attach(df, "nearest_reach_arbolate_sum_km", None, flat_idx)
        if "slope" in joined.columns:
            sl = pd.to_numeric(joined["slope"], errors="coerce").to_numpy()
            _maybe_attach(df, "nearest_reach_slope", sl, flat_idx)
        else:
            _maybe_attach(df, "nearest_reach_slope", None, flat_idx)
        dn = pd.to_numeric(
            joined["distance_to_nearest_reach_m"], errors="coerce"
        ).to_numpy()
        _maybe_attach(df, "distance_to_nearest_reach_m", dn, flat_idx)
        print(f"  cells with finite stream_order: "
              f"{int(np.isfinite(df['nearest_reach_stream_order']).sum()):,}")
    else:
        warnings.warn(f"NHD flowlines missing at {nhd_path}; NHD VAA features NaN.")
        _maybe_attach(df, "nearest_reach_stream_order", None, flat_idx)
        _maybe_attach(df, "nearest_reach_arbolate_sum_km", None, flat_idx)
        _maybe_attach(df, "nearest_reach_slope", None, flat_idx)
        _maybe_attach(df, "distance_to_nearest_reach_m", None, flat_idx)

    # ---- Hydraulic-pit proximity.
    print("[placer] hydraulic-pit proximity")
    pit_path = REGION.raw_paths.get("hydraulic_pits")
    if pit_path is None or not pit_path.exists():
        warnings.warn(
            f"hydraulic_pits missing at {pit_path}; "
            "hydraulic_pit_proximity_m filled with NaN."
        )
        _maybe_attach(df, "hydraulic_pit_proximity_m", None, flat_idx)
        _maybe_attach(df, "hydraulic_pit_proximity_m_buffered", None, flat_idx)
    else:
        pit_polys = get_adapter("geology", "hydraulic_pits")(pit_path, REGION.aoi)
        prox = hydraulic_pit_proximity_m(pit_polys, grid)
        _maybe_attach(df, "hydraulic_pit_proximity_m",
                      prox.to_numpy(dtype=np.float32), flat_idx)
        # v3 Phase B.2: buffered variant for Tertiary training so the model
        # cannot learn the trivial "distance == 0 → pit centroid → positive"
        # shortcut. NaN within 1 km of any pit polygon.
        prox_buffered = hydraulic_pit_proximity_m_buffered(pit_polys, grid)
        _maybe_attach(df, "hydraulic_pit_proximity_m_buffered",
                      prox_buffered.to_numpy(dtype=np.float32), flat_idx)

    # ---- Quaternary-alluvium boolean.
    print("[placer] Quaternary-alluvium mask")
    geo_path = REGION.raw_paths.get("geology")
    if geo_path is None or not geo_path.exists():
        warnings.warn(
            f"geology missing at {geo_path}; is_quaternary_alluvium filled with NaN."
        )
        _maybe_attach(df, "is_quaternary_alluvium", None, flat_idx)
        _maybe_attach(df, "distance_to_lithological_contact_m", None, flat_idx)
    else:
        geo_poly = get_adapter("geology", REGION.geology_source)(geo_path, REGION.aoi)
        qal = is_quaternary_alluvium(geo_poly, grid)
        # Bool feature; preserve dtype rather than float-cast.
        df["is_quaternary_alluvium"] = qal.to_numpy(dtype=bool)[flat_idx]

        # v3 Phase B.6: distance to nearest CGS 2010 lithology contact.
        # Contact zones favor lode -> placer transport; cells far from any
        # polygon boundary sit inside large homogeneous units. Reuses the
        # same geo_poly load as is_quaternary_alluvium.
        print("[placer] distance to lithological contact")
        contact_dist = distance_to_lithological_contact_m(geo_poly, grid)
        _maybe_attach(df, "distance_to_lithological_contact_m",
                      contact_dist.to_numpy(dtype=np.float32), flat_idx)

    # ---- v3 Phase D.3 (PENDING WIRING — v3.5 follow-up).
    # CGS Jennings 2010 fault-activity classes are available as a separate
    # dataset + adapter, but are not yet wired into this assemble step.
    # The new modules:
    #   - Fetcher:  src/ai_minerals/data/cgs_jennings.py
    #               (fetch_cgs_jennings(aoi) -> Path to per-AOI GPKG)
    #   - Adapter:  src/ai_minerals/data/adapters/geology/cgs_jennings.py
    #               (registered as get_adapter("geology", "cgs_jennings"))
    #   - Feature:  src/ai_minerals/features/hydrology.py
    #               distance_to_fault_by_class(faults, grid)
    #
    # To add the split distance-to-fault features in v3.5:
    #   1. Fetch the dataset:
    #        from ai_minerals.data.cgs_jennings import fetch_cgs_jennings
    #        fetch_cgs_jennings(REGION.aoi)
    #      Writes to data/raw/cgs_jennings/cgs_jennings_<region>.gpkg.
    #   2. Add the GPKG path to REGION.raw_paths under e.g. "fault_classes".
    #   3. Load + split via the registered adapter:
    #        faults = get_adapter("geology", "cgs_jennings")(
    #            REGION.raw_paths["fault_classes"], REGION.aoi
    #        )
    #   4. Rasterize per-class distance:
    #        from ai_minerals.features.hydrology import distance_to_fault_by_class
    #        per_class = distance_to_fault_by_class(faults, grid)
    #      Returns {"pre_quaternary": Series, "quaternary_active": Series}.
    #   5. Attach as "distance_to_prequaternary_fault_m" (Tertiary stack) and
    #      "distance_to_quaternary_fault_m" (Quaternary stack), then add the
    #      column names to TERTIARY_FEATURE_COLUMNS / QUATERNARY_FEATURE_COLUMNS
    #      in config.py.
    # Plan-level rationale: ~/.claude/plans/hazy-humming-lynx.md, section
    # "D.3 Replace SGMC fault arcs with CGS Jennings 2010".

    # ---- Geophysics: magnetic + gravity (v3 Phase D.1).
    # Not direct placer signals; they refine the lode-source-favorability
    # context for the distance_to_lode_m feature. The base features land in
    # the per-population feature stacks via TERTIARY_FEATURE_COLUMNS and
    # QUATERNARY_FEATURE_COLUMNS in config.py.
    #
    # Wiring path: `build_feature_frame` in features/assemble.py already
    # samples REGION.raw_paths["magnetic"] and REGION.raw_paths["gravity"]
    # onto the grid and attaches them as columns on df before we get here
    # (motherlode v2/v3.1 reuse the same rasters). No explicit sample_raster
    # needed in this script; we only verify the columns survived so a
    # regression upstream surfaces loudly instead of silently dropping the
    # geophysics features from the placer stack.
    print("[placer] geophysics: magnetic + gravity (verifying upstream wiring)")
    for _col in ("magnetic", "gravity"):
        if _col not in df.columns:
            warnings.warn(
                f"{_col} column missing from build_feature_frame output; "
                f"placer feature stack will be missing the {_col} feature."
            )
            _maybe_attach(df, _col, None, flat_idx)

    # ---- Tertiary terrace likelihood (depends on tpi, slope, is_quaternary_alluvium).
    # The standard paleochannel composite scores modern-channel-proximity; Sierra
    # Tertiary deep-gravels sit on benches above the modern drainage. This
    # composite (high TPI ∧ low slope ∧ not-Qal) is the bench-shaped twin.
    print("[placer] Tertiary terrace likelihood")
    if "tpi" in df.columns and "slope" in df.columns:
        ttl = tertiary_terrace_likelihood(df)
        df["tertiary_terrace_likelihood"] = ttl.to_numpy(dtype=np.float32)
    else:
        warnings.warn("tpi and/or slope missing; tertiary_terrace_likelihood NaN.")
        df["tertiary_terrace_likelihood"] = np.nan

    # ---- v3 Phase D.5: motherlode_prob as A/B diagnostic feature.
    # The motherlode v2 model's calibrated probability raster overlaps the
    # placer AOI substantially. Sample it onto the placer grid as a single
    # feature; v3 trains both with and without it to detect whether the
    # transitive feature helps generalize or just adds noise. The proper
    # out-of-fold hierarchical model (motherlode predictions per placer
    # spatial-block fold) is v4 work.
    print("[placer] motherlode probability (Phase D.5 A/B diagnostic feature)")
    motherlode_raster = (REPO_ROOT / "data" / "derived" / "motherlode"
                         / "prospectivity_motherlode_v2_250m_calibrated_4326.tif")
    if motherlode_raster.exists():
        ml_arr = sample_raster(motherlode_raster, grid, resampling=Resampling.bilinear)
        _maybe_attach(df, "motherlode_prob", ml_arr.ravel(), flat_idx)
        print(f"  motherlode_prob: cells with finite values: "
              f"{int(np.isfinite(df['motherlode_prob']).sum()):,}")
    else:
        warnings.warn(f"motherlode calibrated raster missing at {motherlode_raster}; "
                      f"motherlode_prob NaN.")
        _maybe_attach(df, "motherlode_prob", None, flat_idx)

    # ---- Hawkes dual-decay catchment aggregates (Au, As, Sb).
    print("[placer] Hawkes dual-decay catchment (Au, As, Sb)")
    geochem_path = REGION.raw_paths.get("geochem")
    nure_path = REGION.raw_paths.get("geochem_nure")
    nhd_loaded = None
    samples: gpd.GeoDataFrame | None = None
    if nhd_path is not None and nhd_path.exists():
        nhd_loaded = get_adapter("hydrology", "nhdplus_hr")(nhd_path, REGION.aoi)
    ngdb = None
    nure = None
    if geochem_path is not None and geochem_path.exists():
        ngdb = get_adapter("geochem", "ngdb")(
            geochem_path, REGION.aoi, elements=REGION.pathfinder_elements,
        )
    else:
        warnings.warn(f"NGDB geochem missing at {geochem_path}; skipping for Hawkes.")
    if nure_path is not None and nure_path.exists():
        nure = get_adapter("geochem", "nure_iicpms")(
            nure_path, REGION.aoi, elements=REGION.pathfinder_elements,
        )
    else:
        warnings.warn(f"NURE geochem missing at {nure_path}; skipping for Hawkes.")
    samples = _combine_geochem_samples(ngdb, nure)

    element_columns = {
        "Au_ppm": "catchment_au_hawkes",
        "As_ppm": "catchment_as_hawkes",
        "Sb_ppm": "catchment_sb_hawkes",
    }
    if nhd_loaded is None or samples is None or len(samples) == 0:
        warnings.warn(
            "Hawkes catchment skipped (missing NHD or geochem samples); "
            "columns filled with NaN."
        )
        for col_name in element_columns.values():
            _maybe_attach(df, col_name, None, flat_idx)
    else:
        for element, col_name in element_columns.items():
            if element not in samples.columns:
                warnings.warn(
                    f"Hawkes: element {element!r} absent from samples; "
                    f"{col_name} filled with NaN."
                )
                _maybe_attach(df, col_name, None, flat_idx)
                continue
            series = hawkes_dual_decay_catchment(
                samples, nhd_loaded, grid, element=element,
            )
            _maybe_attach(df, col_name,
                          series.to_numpy(dtype=np.float32), flat_idx)

    return df


def assemble(*, resolution_m: int, augment_osm_mining: bool = False) -> pd.DataFrame:
    """Build the per-cell placer feature frame and return it."""
    t0 = time.time()
    print(f"[assemble] base frame ({REGION.slug}, resolution={resolution_m} m)")
    df = build_feature_frame(REGION, resolution_m=resolution_m)

    assert "is_placer_tertiary" in df.columns, (
        "build_feature_frame did not produce is_placer_tertiary. "
        "Check REGION.deposit_classes."
    )
    assert "is_placer_quaternary" in df.columns, (
        "build_feature_frame did not produce is_placer_quaternary. "
        "Check REGION.deposit_classes."
    )

    # Lawley audit discipline: drop *_count_5km and *_has_data_5km columns.
    drop_cols = count_feature_columns(list(df.columns))
    if drop_cols:
        print(f"[assemble] dropping {len(drop_cols)} count features: "
              f"{drop_cols[:6]}{'...' if len(drop_cols) > 6 else ''}")
        df = df.drop(columns=drop_cols)

    # The canonical MRDS adapter assigns lode-Au codes (36a/36b) for the Au
    # commodity and DROPS placer-flagged records. The placer Region's
    # deposit_classes use codes 39a/39b which the MRDS adapter never emits,
    # so build_feature_frame leaves is_placer_tertiary / is_placer_quaternary
    # all-zero. We inject the real positives here:
    #   Tertiary: hydraulic-pit polygon centroids (Hydraulic Mine Pits CA)
    #   Quaternary: MRDS records with dep_type matching the placer regex
    df = _inject_placer_positives(
        df,
        resolution_m=resolution_m,
        augment_osm_mining=augment_osm_mining,
    )

    df = _build_placer_columns(df, resolution_m=resolution_m)

    elapsed_min = (time.time() - t0) / 60.0
    print(f"[assemble] done: {len(df):,} cells × {len(df.columns)} columns "
          f"in {elapsed_min:.1f} min")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution-m", type=int, default=DEFAULT_RES_M,
        help=f"Grid resolution in meters (default: {DEFAULT_RES_M}).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output parquet path (default: data/derived/features_<slug>_<res>m.parquet).",
    )
    parser.add_argument(
        "--augment-osm-mining", action="store_true",
        help=(
            "Augment is_placer_quaternary with OSM mining-tag features "
            "(historic=mine, man_made=mineshaft|adit, landuse=quarry) from "
            "data/raw/osm_mining/osm_mining_<region>.gpkg. Fetched by "
            "scripts/northern_sierra_placer_fetch_osm_mining.py. License: "
            "ODbL; model card must attribute (c) OpenStreetMap contributors."
        ),
    )
    args = parser.parse_args(argv)

    out_path = args.out or (
        DATA_DERIVED / f"features_{REGION.data_prefix}_{args.resolution_m}m.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = assemble(
        resolution_m=args.resolution_m,
        augment_osm_mining=args.augment_osm_mining,
    )
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(df):,} rows × {len(df.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
