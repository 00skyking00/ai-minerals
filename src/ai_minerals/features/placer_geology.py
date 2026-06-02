"""Placer-specific geology / catchment features for the northern Sierra model.

Three per-grid-cell features feed both the Phase 1 weighted index and the
Phase 2 supervised models:

- `hydraulic_pit_proximity_m`: distance (m) to the nearest USGS Hydraulic
  Mine Pit polygon boundary. Cells inside a pit get 0.
- `is_quaternary_alluvium`: boolean from a centroid-in-polygon spatial join
  against the CGS 2010 geology surface, using the adapter-provided
  `is_quaternary_alluvium` column (falling back to a PTYPE regex).
- `hawkes_dual_decay_catchment`: dilution-corrected upstream-catchment
  aggregation (Hawkes 1976) of a stream-sediment element onto each cell,
  weighting samples by a two-decay-length kernel along NHDPlus HR reaches.

The Hawkes function honours a `fold_mask` to keep spatial-block CV honest:
when a fold's training samples are passed in, the feature recomputes from
only those samples, so a test-fold sample never contributes to a feature
seen by the model at fit time.
"""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from ai_minerals.grid import Grid


# Fallback regex for CGS PTYPE codes that denote Quaternary alluvium / gravel /
# fan / lake / wash. Matches the inclusive pattern used by the cgs_2010 adapter
# (`_Q_ALLUV_RE`) — kept here so the feature still works against an older
# adapter pull that doesn't carry the precomputed `is_quaternary_alluvium`
# column.
_Q_ALLUV_FALLBACK_RE = re.compile(
    r"^Q(a|al|g|f|l|w|oa|s|ls|p|e|c|pf|d|m|t|ya|oal|oa[lo]?)?$",
    re.IGNORECASE,
)

# Sentinel large distance for sjoin_nearest's max_distance cap. We want it
# large enough that no grid cell inside the AOI gets clipped, but finite so
# the operation does not degenerate to all-vs-all.
_NEAREST_PIT_CAP_M = 500_000.0


def tertiary_terrace_likelihood(
    df: pd.DataFrame,
    *,
    tpi_col: str = "tpi",
    slope_col: str = "slope",
    qal_col: str = "is_quaternary_alluvium",
    tpi_clip_pct: tuple[float, float] = (50.0, 99.0),
    slope_cap_deg: float = 15.0,
) -> pd.Series:
    """Composite feature for Sierra Tertiary deep-gravel terrace likelihood.

    The standard `paleochannel_likelihood` raster (REM + LRM + GMI) targets
    UNINCISED paleochannels in modern lowlands and lights up canyon bottoms.
    Sierra Tertiary auriferous gravels are the OPPOSITE: paleo-channels that
    the Quaternary system has incised through, leaving the pay strata on
    hillside benches and terraces ABOVE the modern drainage. The composite
    here scores those benches by combining three signals:

      tpi_high:   normalized `tpi` clipped at `tpi_clip_pct` — captures cells
                  that sit above the local mean (benches, ridges, terrace
                  remnants). Below the lower clip → 0.
      slope_low:  `max(0, 1 - slope / slope_cap_deg)` — planated terrace
                  surfaces are low-slope after Tertiary base-leveling. Cells
                  steeper than `slope_cap_deg` zero out.
      not_qal:    `~is_quaternary_alluvium` — Tertiary deep-gravels are
                  not in young alluvium (the gravels are pre-incision).

    The composite is the geometric mean of the three; geometric mean (rather
    than weighted sum) means any single component near zero zeros out the
    composite, which matches the geology: a Tertiary terrace candidate needs
    all three signatures (elevated AND low-slope AND not-young-alluvium).

    Returns a [0, 1] Series aligned to df.index. Cells with missing inputs
    produce NaN.
    """
    if tpi_col not in df.columns:
        raise KeyError(f"tertiary_terrace_likelihood needs '{tpi_col}' column")
    if slope_col not in df.columns:
        raise KeyError(f"tertiary_terrace_likelihood needs '{slope_col}' column")

    tpi_vals = df[tpi_col].to_numpy(dtype=np.float32)
    slope_vals = df[slope_col].to_numpy(dtype=np.float32)
    if qal_col in df.columns:
        qal_vals = df[qal_col].fillna(False).to_numpy(dtype=bool)
    else:
        qal_vals = np.zeros(len(df), dtype=bool)

    finite_tpi = np.isfinite(tpi_vals)
    if not finite_tpi.any():
        return pd.Series(np.full(len(df), np.nan, dtype=np.float32),
                         index=df.index, name="tertiary_terrace_likelihood")
    lo_pct, hi_pct = tpi_clip_pct
    lo = np.nanpercentile(tpi_vals, lo_pct)
    hi = np.nanpercentile(tpi_vals, hi_pct)
    tpi_high = np.zeros_like(tpi_vals)
    if hi > lo:
        clipped = np.clip(tpi_vals, lo, hi)
        tpi_high = ((clipped - lo) / (hi - lo)).astype(np.float32)
    tpi_high[~finite_tpi] = np.nan

    slope_low = np.where(
        np.isfinite(slope_vals),
        np.clip(1.0 - slope_vals / slope_cap_deg, 0.0, 1.0),
        np.nan,
    ).astype(np.float32)

    not_qal = (~qal_vals).astype(np.float32)

    eps = 1e-6
    composite = np.cbrt(
        (tpi_high + eps) * (slope_low + eps) * (not_qal + eps)
    ).astype(np.float32)
    composite[~np.isfinite(tpi_high) | ~np.isfinite(slope_low)] = np.nan
    return pd.Series(composite, index=df.index, name="tertiary_terrace_likelihood")


def hydraulic_pit_proximity_m(
    pit_polys: gpd.GeoDataFrame, grid: Grid
) -> pd.Series:
    """Per-grid-cell distance (m) to the nearest hydraulic-pit polygon boundary.

    Returns 0 inside any pit. The result is indexed identically to
    `grid.centroid_gdf()` (range index 0..n_cells-1, in row-major order).
    """
    centroids = grid.centroid_gdf()
    pits = pit_polys.to_crs(grid.crs)

    # Keep only the geometry column on the right side so the join is cheap
    # and the returned frame stays small.
    pits_geom = pits[["geometry"]].copy()

    joined = gpd.sjoin_nearest(
        centroids,
        pits_geom,
        how="left",
        max_distance=_NEAREST_PIT_CAP_M,
        distance_col="_pit_dist_m",
    )
    # A centroid can tie between two polygons at exactly equal distance;
    # sjoin_nearest emits one row per tie. Drop dupes keeping the first.
    joined = joined.loc[~joined.index.duplicated(keep="first")]

    # Distance is NaN only if no pit was within the cap, which inside the AOI
    # should not happen; we still defend against it by filling with the cap.
    dist = joined["_pit_dist_m"].fillna(_NEAREST_PIT_CAP_M).to_numpy(dtype=np.float32)
    return pd.Series(dist, index=centroids.index, name="hydraulic_pit_proximity_m")


def is_quaternary_alluvium(
    geo_poly: gpd.GeoDataFrame, grid: Grid
) -> pd.Series:
    """Per-grid-cell boolean for Quaternary alluvial cover.

    Spatial-joins grid centroids against `geo_poly` (CGS 2010 schema). Uses
    the adapter-provided `is_quaternary_alluvium` column when present;
    otherwise regex-falls-back on `ptype` / `PTYPE`.
    """
    centroids = grid.centroid_gdf()
    poly = geo_poly.to_crs(grid.crs)

    # Decide which column we'll read off the polygon side.
    if "is_quaternary_alluvium" in poly.columns:
        flag_col = poly["is_quaternary_alluvium"].astype(bool)
        right = gpd.GeoDataFrame(
            {"_qal_flag": flag_col, "geometry": poly.geometry},
            crs=poly.crs,
        )
    elif "ptype" in poly.columns or "PTYPE" in poly.columns:
        ptype_series = (poly["ptype"] if "ptype" in poly.columns else poly["PTYPE"]).astype("string")
        regex_flag = ptype_series.fillna("").map(
            lambda s: bool(_Q_ALLUV_FALLBACK_RE.match(s.strip()))
        )
        right = gpd.GeoDataFrame(
            {"_qal_flag": regex_flag.astype(bool), "geometry": poly.geometry},
            crs=poly.crs,
        )
    elif "lith_group" in poly.columns and "age_era" in poly.columns:
        # USGS SGMC fallback: "Unconsolidated, undifferentiated" → lith_group
        # "surficial"; pair with age_era == "cenozoic" to approximate
        # Quaternary-alluvium cover. Coarser than CGS 2010's Qa/Qal/Qg subcodes
        # but the only signal SGMC carries.
        sgmc_flag = (
            (poly["lith_group"].astype("string") == "surficial")
            & (poly["age_era"].astype("string") == "cenozoic")
        )
        right = gpd.GeoDataFrame(
            {"_qal_flag": sgmc_flag.astype(bool), "geometry": poly.geometry},
            crs=poly.crs,
        )
    else:
        raise ValueError(
            "geo_poly has neither `is_quaternary_alluvium`, `ptype`/`PTYPE`, "
            "nor SGMC `lith_group`/`age_era`; cannot determine Quaternary-"
            "alluvium membership."
        )

    joined = gpd.sjoin(centroids, right, how="left", predicate="within")
    # A centroid can land on a polygon boundary and match two polygons. Pick
    # the first non-null flag per centroid; True wins ties (a cell in *any*
    # Q-alluvial polygon counts).
    flag_per_centroid = (
        joined.groupby(level=0)["_qal_flag"]
        .apply(lambda s: bool(s.dropna().any()))
        .reindex(centroids.index, fill_value=False)
        .astype(bool)
    )
    return pd.Series(
        flag_per_centroid.to_numpy(), index=centroids.index, name="is_quaternary_alluvium"
    )


def _build_hydroseq_index(
    nhd_network: gpd.GeoDataFrame,
) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    """Build reach lookup tables once for the upstream walk.

    Returns:
      hydroseq_by_comid : COMID -> hydroseq (float; NHD stores as float64)
      arbolate_by_comid : COMID -> arbolate_sum (km of cumulative upstream channel)
      reach_position    : COMID -> integer position in the input frame, so we can
                          recover the underlying geometry from a positional lookup
    """
    hydroseq_by_comid: dict[int, float] = {}
    arbolate_by_comid: dict[int, float] = {}
    reach_position: dict[int, int] = {}

    comid_arr = nhd_network["comid"].astype("int64").to_numpy()
    hydroseq_arr = nhd_network["hydroseq"].astype("float64").to_numpy()
    arbolate_arr = nhd_network["arbolate_sum"].astype("float64").to_numpy()

    for i, comid in enumerate(comid_arr):
        c = int(comid)
        hydroseq_by_comid[c] = float(hydroseq_arr[i])
        arbolate_by_comid[c] = float(arbolate_arr[i])
        reach_position[c] = i

    return hydroseq_by_comid, arbolate_by_comid, reach_position


def hawkes_dual_decay_catchment(
    samples: gpd.GeoDataFrame,
    nhd_network: gpd.GeoDataFrame,
    grid: Grid,
    element: str,
    *,
    near_decay_km: float = 2.0,
    far_decay_km: float = 15.0,
    alpha: float = 0.3,
    fold_mask: np.ndarray | None = None,
    cell_mask: np.ndarray | None = None,
) -> pd.Series:
    """Hawkes-style dual-decay catchment aggregation of an element column.

    For each cell, find every sample whose snapped NHD reach lies upstream of
    the cell's snapped reach, compute the along-channel distance via
    arbolate-sum differencing, and weight the sample's `<element>` value by
        w(d) = exp(-d / near_decay_km) + alpha * exp(-d / far_decay_km)
    The per-cell value is the weight-normalized mean. Cells with no upstream
    sample get NaN.

    `fold_mask` (bool array of length len(samples), True = training fold) is
    applied **before** the snap step, so re-running the feature per spatial-
    block fold never lets a test-fold sample contribute to its own cell.

    `cell_mask` (bool array of length grid.n_cells, True = compute this cell)
    restricts the hot loop to a subset of cells. The per-fold Hawkes refold
    in train_predict_250m.py uses this to compute only the test-fold cells,
    avoiding ~90% wasted compute over the full grid. Cells outside the mask
    return NaN. Default None = compute every cell.

    Upstream test (NHDPlus HR convention): a sample reach is upstream of a
    cell reach iff its `hydroseq` is strictly greater. Arbolate-sum is
    monotonically decreasing downstream, so the channel distance from sample
    to cell is `sample.arbolate_sum - cell.arbolate_sum` (km), clipped at 0.
    """
    if element not in samples.columns:
        raise ValueError(
            f"hawkes_dual_decay_catchment: column {element!r} not in samples; "
            f"available columns: {list(samples.columns)}"
        )

    # 0. Apply fold mask BEFORE the snap. Re-projecting and snapping are the
    #    expensive steps; masking first keeps fold re-runs cheap and also
    #    eliminates any chance of a test-fold sample influencing a feature.
    s = samples
    if fold_mask is not None:
        mask = np.asarray(fold_mask, dtype=bool)
        if mask.shape[0] != len(s):
            raise ValueError(
                f"fold_mask length {mask.shape[0]} != samples length {len(s)}"
            )
        s = s.loc[mask].copy()

    # Drop samples with NaN element values up front; they cannot contribute.
    val_col = s[element].astype("float64")
    s = s.loc[val_col.notna()].copy()
    if len(s) == 0:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            index=grid.centroid_gdf().index,
            name=f"hawkes_catchment_{element}",
        )

    centroids = grid.centroid_gdf()
    s = s.to_crs(grid.crs)
    nhd = nhd_network.to_crs(grid.crs)

    # 1. Snap each sample to nearest NHD reach (COMID).
    nhd_keys = nhd[["comid", "geometry"]].copy()
    sample_snap = gpd.sjoin_nearest(
        s[[element, "geometry"]],
        nhd_keys,
        how="left",
        distance_col="_snap_dist_m",
    )
    sample_snap = sample_snap.loc[~sample_snap.index.duplicated(keep="first")]
    sample_snap = sample_snap.loc[sample_snap["comid"].notna()].copy()
    if len(sample_snap) == 0:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            index=centroids.index,
            name=f"hawkes_catchment_{element}",
        )
    sample_snap["comid"] = sample_snap["comid"].astype("int64")

    # 2. Snap each cell to nearest NHD reach.
    cell_snap = gpd.sjoin_nearest(
        centroids[["geometry"]],
        nhd_keys,
        how="left",
        distance_col="_snap_dist_m",
    )
    cell_snap = cell_snap.loc[~cell_snap.index.duplicated(keep="first")]
    # A cell with no reach in the AOI gets NaN COMID — its feature stays NaN.
    cell_has_reach = cell_snap["comid"].notna()
    cell_comid = cell_snap["comid"].where(cell_has_reach, -1).astype("int64").to_numpy()

    # 3. Reach lookups (build once).
    hydroseq_by_comid, arbolate_by_comid, _ = _build_hydroseq_index(nhd)

    # Group samples by their snapped COMID for the upstream walk. Pre-extract
    # numpy arrays of (hydroseq, arbolate_sum, value) per sample-COMID.
    sample_comid_arr = sample_snap["comid"].to_numpy(dtype=np.int64)
    sample_value_arr = sample_snap[element].to_numpy(dtype=np.float64)
    sample_hydroseq = np.array(
        [hydroseq_by_comid.get(int(c), np.nan) for c in sample_comid_arr],
        dtype=np.float64,
    )
    sample_arbolate = np.array(
        [arbolate_by_comid.get(int(c), np.nan) for c in sample_comid_arr],
        dtype=np.float64,
    )

    # Drop samples whose snapped COMID isn't in the network table (shouldn't
    # happen given step 1 used the same `nhd`, but be defensive).
    sample_valid = np.isfinite(sample_hydroseq) & np.isfinite(sample_arbolate)
    sample_hydroseq = sample_hydroseq[sample_valid]
    sample_arbolate = sample_arbolate[sample_valid]
    sample_value_arr = sample_value_arr[sample_valid]

    if sample_hydroseq.size == 0:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            index=centroids.index,
            name=f"hawkes_catchment_{element}",
        )

    # 4. For each cell, accumulate weighted sum and weight total over upstream
    #    samples. Vectorize over samples per cell.
    #
    #    Memory: n_cells * n_samples can blow up. We loop over cells but
    #    vectorize the per-cell sample sweep with numpy boolean masks. For
    #    grids ~10^5 cells and ~10^4 samples this is acceptable; if it
    #    becomes a bottleneck we can group cells by COMID to amortize.
    #
    #    cell_mask: when set, iterate only over masked-true cells. The v2 per-
    #    fold Hawkes refold computed all 800k cells then discarded ~90% via
    #    test-fold indexing; passing a cell_mask cuts the iteration to the
    #    test fold (typically ~50-100k cells) and gives a 5-10x speedup on
    #    per-fold runs without changing the per-cell math.
    out_sum = np.zeros(grid.n_cells, dtype=np.float64)
    out_wsum = np.zeros(grid.n_cells, dtype=np.float64)
    out_count = np.zeros(grid.n_cells, dtype=np.int64)

    # Precompute decay constants as floats (avoid div-by-zero if a caller
    # passes 0; treat as "no contribution from that band").
    near = float(near_decay_km)
    far = float(far_decay_km)

    if cell_mask is None:
        cell_iter = range(grid.n_cells)
    else:
        cm = np.asarray(cell_mask, dtype=bool)
        if cm.shape[0] != grid.n_cells:
            raise ValueError(
                f"cell_mask length {cm.shape[0]} != grid.n_cells {grid.n_cells}"
            )
        cell_iter = np.flatnonzero(cm).tolist()

    for ci in cell_iter:
        if not cell_has_reach.iat[ci]:
            continue
        c_comid = int(cell_comid[ci])
        c_hydro = hydroseq_by_comid.get(c_comid)
        c_arbolate = arbolate_by_comid.get(c_comid)
        if c_hydro is None or c_arbolate is None:
            continue
        # Upstream-of-cell: sample hydroseq strictly greater (NHDPlus HR).
        # Inclusive equality (same reach as the cell) also contributes —
        # the channel distance is then 0 and weight is 1+alpha, which is
        # the maximal weight, appropriate for "sample is on this reach".
        upstream_mask = sample_hydroseq >= c_hydro
        if not upstream_mask.any():
            continue
        d_km = sample_arbolate[upstream_mask] - c_arbolate
        # arbolate_sum is monotonically non-decreasing upstream → downstream
        # in NHDPlus HR's accumulation convention. If a sample's arbolate_sum
        # comes out less than the cell's despite the hydroseq test (rare
        # network-coding edge case), clip distance at 0 rather than drop.
        d_km = np.clip(d_km, 0.0, None)
        w_near = np.exp(-d_km / near) if near > 0 else np.zeros_like(d_km)
        w_far = alpha * np.exp(-d_km / far) if far > 0 else np.zeros_like(d_km)
        w = w_near + w_far
        vals = sample_value_arr[upstream_mask]
        out_sum[ci] = float(np.sum(w * vals))
        out_wsum[ci] = float(np.sum(w))
        out_count[ci] = int(w.size)

    result = np.full(grid.n_cells, np.nan, dtype=np.float32)
    has_w = out_wsum > 0
    result[has_w] = (out_sum[has_w] / out_wsum[has_w]).astype(np.float32)

    return pd.Series(result, index=centroids.index, name=f"hawkes_catchment_{element}")


# --- CLI smoke test ----------------------------------------------------------

def _smoke() -> None:
    """Tiny synthetic run: 10x10 grid, 5 samples, 3 pit polys, 1 NHD reach."""
    from ai_minerals.aoi import AOI

    # Build a 10x10 grid at 100 m in EPSG:3310 over a tiny patch of the
    # northern Sierra (37.49,-121.55) corner.
    aoi = AOI(name="smoke", min_lon=-121.55, min_lat=37.49,
              max_lon=-121.549, max_lat=37.491)
    # Bypass build_grid (which depends on AOI extents matching resolution);
    # construct a Grid directly in EPSG:3310 meters.
    xs = np.arange(-50.0, 950.0, 100.0)  # 10 cell centers
    ys = np.arange(-50.0, 950.0, 100.0)
    grid = Grid(xs=xs, ys=ys, resolution_m=100, crs="EPSG:3310")

    # 3 fake pit polygons (squares).
    pits = gpd.GeoDataFrame(
        {
            "pit_id": [1, 2, 3],
            "pit_name": ["A", "B", "C"],
            "data_source": ["smoke", "smoke", "smoke"],
            "area_acres": [1.0, 2.0, 3.0],
            "source": ["HYDRAULIC_PITS"] * 3,
            "geometry": [
                Polygon([(100, 100), (200, 100), (200, 200), (100, 200)]),
                Polygon([(500, 500), (600, 500), (600, 600), (500, 600)]),
                Polygon([(800, 800), (900, 800), (900, 900), (800, 900)]),
            ],
        },
        crs="EPSG:3310",
    )
    # Reproject pits to WGS84 to mimic the adapter's contract.
    pits_wgs = pits.to_crs("EPSG:4326")

    dist = hydraulic_pit_proximity_m(pits_wgs, grid)
    print(f"hydraulic_pit_proximity_m: shape={dist.shape}, min={dist.min():.1f}, max={dist.max():.1f}")

    # Geology polygon covering half the grid as Q-alluvium.
    geo = gpd.GeoDataFrame(
        {
            "lith_class": [1, 2],
            "lith_group": ["surficial", "intrusive"],
            "is_quaternary_alluvium": [True, False],
            "ptype": ["Qa", "Kgr"],
            "source": ["CGS_2010", "CGS_2010"],
            "geometry": [
                Polygon([(0, 0), (1000, 0), (1000, 500), (0, 500)]),
                Polygon([(0, 500), (1000, 500), (1000, 1000), (0, 1000)]),
            ],
        },
        crs="EPSG:3310",
    )
    qal = is_quaternary_alluvium(geo, grid)
    print(f"is_quaternary_alluvium: shape={qal.shape}, n_true={int(qal.sum())}")

    # 1 NHD reach as a diagonal line crossing the grid, single COMID. Add a
    # second reach to exercise the upstream-walk hydroseq comparison.
    nhd = gpd.GeoDataFrame(
        {
            "comid": [101, 102],
            "arbolate_sum": [5.0, 10.0],  # 102 is upstream of 101 (more cumulative)
            "stream_order": [2, 1],
            "hydroseq": [1000.0, 2000.0],  # higher hydroseq = upstream
            "fcode": [46006, 46006],
            "source": ["NHDPlus_HR", "NHDPlus_HR"],
            "geometry": [
                LineString([(0, 100), (1000, 100)]),     # downstream reach
                LineString([(0, 800), (1000, 800)]),     # upstream reach
            ],
        },
        crs="EPSG:3310",
    )

    # 5 fake stream-sediment samples. Place some near each reach so they
    # snap deterministically.
    samples = gpd.GeoDataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
            "Au_ppm": [0.01, 0.05, 0.2, 0.03, 0.1],
            "source": ["smoke"] * 5,
            "geometry": [
                Point(100, 110),   # snaps to reach 101 (downstream)
                Point(500, 110),   # snaps to reach 101
                Point(900, 800),   # snaps to reach 102 (upstream)
                Point(500, 800),   # snaps to reach 102
                Point(100, 800),   # snaps to reach 102
            ],
        },
        crs="EPSG:3310",
    )

    hawkes = hawkes_dual_decay_catchment(samples, nhd, grid, element="Au_ppm")
    print(
        f"hawkes_dual_decay_catchment: shape={hawkes.shape}, "
        f"n_finite={int(hawkes.notna().sum())}, "
        f"min={np.nanmin(hawkes.values):.4f}, max={np.nanmax(hawkes.values):.4f}"
    )

    # Smoke-test the fold mask: drop sample s3 (the highest grade on the
    # upstream reach), expect the upstream-reach cells' values to drop.
    mask = np.array([True, True, False, True, True])
    hawkes_folded = hawkes_dual_decay_catchment(
        samples, nhd, grid, element="Au_ppm", fold_mask=mask
    )
    print(
        f"hawkes with fold_mask (drop s3): n_finite={int(hawkes_folded.notna().sum())}, "
        f"max={np.nanmax(hawkes_folded.values):.4f}"
    )


if __name__ == "__main__":
    _smoke()
