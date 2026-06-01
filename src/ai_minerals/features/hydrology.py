"""Hydrology / paleochannel terrain features for the placer-Au model.

Per-cell terrain-derivative features computed from a DEM aligned to a
`Grid`: D8 flow accumulation, Stream Power Index (trapezoidal-bandpass
membership), Topographic Wetness Index, channel-steepness (ksn),
Topographic Position Index, a geomorphon-derived terrace mask, and
network-distance-downstream-from-lode along NHDPlus flowlines.

The heavy hydrology routing (`flow_accumulation`) uses Whitebox via the
MIT-licensed `whitebox-workflows` Python frontend. The geomorphon
terrace mask shells out to GRASS `r.geomorphon`. Both are optional —
install via the `paleochannel` extra:

    uv sync --extra paleochannel       # whitebox-workflows
    apt install grass                  # geomorphon (Linux)
    brew install grass                 # geomorphon (macOS)

The remaining functions (`stream_power_index_band`,
`topographic_wetness_index`, `knickpoint_ksn`, `tpi`,
`distance_downstream_from_lode`) are pure numpy / geopandas and need no
extra system tools.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine, from_origin
from scipy.ndimage import uniform_filter

if TYPE_CHECKING:
    from ai_minerals.grid import Grid


# Placer regex mirrors src/ai_minerals/data/adapters/occurrences/mrds.py
# so the leakage guard here flags the same records the orogenic-Au
# pipeline already filters out. Kept as a local copy rather than an
# import so the guard is self-contained.
_PLACER_DEP_TYPE_PATTERN = (
    r"placer|alluvial|stream.?placer|paleo.?placer|black.?sand|residual|eluvial"
)


# ---------------------------------------------------------------------------
# Flow accumulation (Whitebox)
# ---------------------------------------------------------------------------

def flow_accumulation(
    dem_array: np.ndarray,
    *,
    transform: Affine,
    nodata: float | None = None,
) -> np.ndarray:
    """D8 flow accumulation via Whitebox. Returns upstream cell counts.

    Writes the DEM to a tempfile (Whitebox needs an on-disk raster),
    runs WhiteboxTools `D8FlowAccumulation` with `out_type='cells'`, and
    reads the result back into a numpy array shaped like `dem_array`.

    `transform` is a rasterio.transform.Affine matching `dem_array`'s
    pixel layout (typically the grid's `from_origin(...)` transform).
    `nodata` is written into the temp GeoTIFF so Whitebox treats those
    cells as no-data; defaults to NaN-compatible -32768.
    """
    try:
        from whitebox_workflows import WbEnvironment
    except ImportError as exc:
        raise RuntimeError(
            "whitebox-workflows is not installed. Install with "
            "`uv sync --extra paleochannel` (adds whitebox-workflows>=1.3)."
        ) from exc

    if nodata is None:
        nodata = -32768.0

    arr = np.where(np.isfinite(dem_array), dem_array, nodata).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="wbw_flowacc_") as tmp:
        tmp_dir = Path(tmp)
        dem_path = tmp_dir / "dem.tif"
        out_path = tmp_dir / "flowacc.tif"

        profile = {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 1,
            "dtype": "float32",
            "transform": transform,
            "nodata": nodata,
            "compress": "lzw",
        }
        with rasterio.open(dem_path, "w", **profile) as dst:
            dst.write(arr, 1)

        wbe = WbEnvironment()
        wbe.working_directory = str(tmp_dir)
        dem_ras = wbe.read_raster(str(dem_path))
        # whitebox-workflows >= 2.0 disabled the flat WbEnvironment tool
        # methods; tools now live on category objects (here:
        # `wbe.hydrology`). Single-step D8 still conditions hydrologically
        # internally per the WBT default.
        flowacc = wbe.hydrology.d8_flow_accum(
            input=dem_ras, out_type="cells", log_transform=False
        )
        wbe.write_raster(flowacc, str(out_path))

        with rasterio.open(out_path) as src:
            out = src.read(1).astype(np.float32)
            src_nodata = src.nodata

    if src_nodata is not None:
        out = np.where(out == src_nodata, np.nan, out)
    return out


# ---------------------------------------------------------------------------
# Stream Power Index (bandpass membership)
# ---------------------------------------------------------------------------

def stream_power_index_band(
    flow_acc: np.ndarray,
    slope: np.ndarray,
    *,
    band: tuple[float, float] = (3.0, 6.0),
) -> np.ndarray:
    """SPI = ln(flow_acc * tan(slope_rad)); trapezoidal-bandpass in [0, 1].

    Detachment / placer-transport competence lies in the mid SPI range,
    not the extremes (Roy, Upton & Craw 2018): below the band there's
    not enough stream power to transport gold; above it, the channel is
    flushing through without depositing. Membership ramps 0→1 over
    [band[0]-0.5, band[0]], stays at 1 in [band[0], band[1]], ramps 1→0
    over [band[1], band[1]+0.5], and is 0 elsewhere.

    Cells with slope==0 or flow_acc<=0 cannot compute ln(SPI); they get
    NaN (the bandpass treats NaN as out-of-band and the result is NaN,
    not 0, so downstream code can distinguish "no data" from "out of
    band"). If you'd rather get 0, mask the result before consumption.
    """
    lo, hi = band
    if not lo < hi:
        raise ValueError(f"band must be (low, high) with low < high; got {band}")

    slope_rad = np.deg2rad(slope)
    tan_s = np.tan(slope_rad)

    spi = np.full_like(flow_acc, np.nan, dtype=np.float32)
    valid = np.isfinite(flow_acc) & np.isfinite(slope) & (flow_acc > 0) & (tan_s > 0)
    spi[valid] = np.log(flow_acc[valid] * tan_s[valid])

    out = np.full_like(spi, np.nan, dtype=np.float32)
    v = np.isfinite(spi)
    s = spi[v]
    m = np.zeros_like(s, dtype=np.float32)
    # ramp up over [lo - 0.5, lo]
    up = (s >= lo - 0.5) & (s < lo)
    m[up] = (s[up] - (lo - 0.5)) / 0.5
    # plateau in [lo, hi]
    plateau = (s >= lo) & (s <= hi)
    m[plateau] = 1.0
    # ramp down over (hi, hi + 0.5]
    down = (s > hi) & (s <= hi + 0.5)
    m[down] = ((hi + 0.5) - s[down]) / 0.5
    # outside band stays 0
    out[v] = m
    return out


# ---------------------------------------------------------------------------
# Topographic Wetness Index (Beven & Kirkby 1979)
# ---------------------------------------------------------------------------

def topographic_wetness_index(
    flow_acc: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """TWI = ln(SCA / tan(slope_rad)); flow_acc is in upstream cell count.

    Convention: caller passes `flow_acc` as cell count (the
    `flow_accumulation` default). On a regular grid the contour width
    equals the cell side, so specific catchment area in cell-count units
    is just `flow_acc` (i.e. SCA in cell-side units). We compute TWI in
    those scale-free units; the resulting raster is monotone in the
    physical TWI, so it ranks identically and is what a tree model
    consumes.

    Slope is in degrees; converted to radians internally. Cells with
    slope==0 or flow_acc<=0 return NaN.
    """
    slope_rad = np.deg2rad(slope)
    tan_s = np.tan(slope_rad)

    out = np.full_like(flow_acc, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(flow_acc)
        & np.isfinite(slope)
        & (flow_acc > 0)
        & (tan_s > 0)
    )
    out[valid] = np.log(flow_acc[valid] / tan_s[valid])
    return out


# ---------------------------------------------------------------------------
# Channel-steepness ksn (Wobus et al. 2006)
# ---------------------------------------------------------------------------

def knickpoint_ksn(
    dem_array: np.ndarray,
    flow_acc: np.ndarray,
    *,
    theta: float = 0.45,
) -> np.ndarray:
    """Steepness index ksn = slope * A^theta, theta = 0.45 (Wobus 2006).

    `dem_array` is needed only for its shape — slope here is computed
    locally rather than re-using `slope_and_tri` because ksn is
    insensitive to whether slope is degrees or the gradient magnitude
    (the per-cell value is what matters for ranking, not the unit). We
    use the gradient magnitude (m/m) for numerical simplicity.

    `flow_acc` is in upstream cell counts; we treat that as a stand-in
    for drainage area A (constant cell area drops out under
    normalization). Output is per-cell ksn min-max-normalized to [0, 1]
    so it composes cleanly with the other [0, 1] hydrology features.

    No river-network masking: returned for every cell. Downstream code
    can mask via a flow_acc threshold (e.g. only cells with flow_acc >
    100 are "channel-like").
    """
    dz_dy, dz_dx = np.gradient(dem_array.astype(np.float32))
    slope_mag = np.hypot(dz_dx, dz_dy)

    A = np.where(np.isfinite(flow_acc) & (flow_acc > 0), flow_acc, np.nan)
    ksn = slope_mag * np.power(A, theta, where=np.isfinite(A), out=np.full_like(A, np.nan))

    finite = np.isfinite(ksn)
    if not finite.any():
        return np.full_like(ksn, np.nan, dtype=np.float32)
    lo = np.nanmin(ksn[finite])
    hi = np.nanmax(ksn[finite])
    if hi <= lo:
        return np.where(finite, 0.0, np.nan).astype(np.float32)
    out = np.full_like(ksn, np.nan, dtype=np.float32)
    out[finite] = (ksn[finite] - lo) / (hi - lo)
    return out


# ---------------------------------------------------------------------------
# Topographic Position Index
# ---------------------------------------------------------------------------

def tpi(dem_array: np.ndarray, *, radius_cells: int = 8) -> np.ndarray:
    """TPI = elevation - mean(elevation in the (2r+1)^2 window).

    Positive on ridges and interfluves, negative in valleys and on
    Tertiary-gravel benches that have been re-incised, near-zero on
    uniform slopes. Uses a square box mean via `uniform_filter` (cheap,
    O(N) regardless of radius). The strict TPI definition uses an
    annulus rather than a filled box; for the placer-feature use case
    the box approximation is fine and an order of magnitude faster.
    """
    size = 2 * int(radius_cells) + 1
    arr = dem_array.astype(np.float32)
    # Fill non-finite cells with the global mean for the box average, then
    # restore the NaN mask on output so downstream code doesn't see
    # spurious values where the DEM was no-data.
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        fill = float(np.nanmean(arr)) if (~nan_mask).any() else 0.0
        filled = np.where(nan_mask, fill, arr)
    else:
        filled = arr
    local_mean = uniform_filter(filled, size=size, mode="reflect")
    out = arr - local_mean
    if nan_mask.any():
        out = np.where(nan_mask, np.nan, out)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Geomorphon terrace mask (GRASS r.geomorphon)
# ---------------------------------------------------------------------------

# GRASS r.geomorphon class codes (Jasiewicz & Stepinski 2013).
# 1=flat, 2=peak, 3=ridge, 4=shoulder, 5=spur, 6=slope, 7=hollow,
# 8=footslope, 9=valley, 10=pit. Terrace surfaces in the deep-gravel
# context map to flat (1), footslope (8), and hollow (7).
_TERRACE_GEOMORPHON_CLASSES = (1, 7, 8)


def geomorphon_terrace_mask(
    dem_path: Path,
    *,
    search_radius_m: float = 80.0,
    search_radius_cells: int | None = None,
) -> np.ndarray:
    """Geomorphon-based binary terrace mask (1 = terrace-like, 0 otherwise).

    Calls GRASS `r.geomorphon` in a temporary mapset, then reduces the
    10-class output to a binary {flat, hollow, footslope} mask suitable
    for use as a paleochannel-bench prior.

    `search_radius_m` is the geomorphon lookup radius (L) in ground meters,
    converted to cells from the DEM's resolution so the same ground extent is
    searched whatever the internal resolution. ~80 m is what Jasiewicz &
    Stepinski recommend for fluvial geomorphology. Pass `search_radius_cells`
    to override with an explicit cell count.

    Caches the raw 10-class geomorphon raster at
    `<dem_path>.geomorphon.tif` so repeated calls skip the GRASS
    shell-out.

    Raises RuntimeError if `grass` is not on PATH.
    """
    dem_path = Path(dem_path)
    cache_path = dem_path.with_suffix(dem_path.suffix + ".geomorphon.tif")

    if not cache_path.exists():
        grass_bin = _which("grass")
        if grass_bin is None:
            raise RuntimeError(
                "GRASS GIS is required for geomorphon_terrace_mask but `grass` "
                "is not on PATH. Install with `apt install grass` (Linux) or "
                "`brew install grass` (macOS)."
            )
        if search_radius_cells is None:
            with rasterio.open(dem_path) as src:
                res_x_deg = abs(src.transform.a)
                if src.crs is not None and src.crs.is_geographic:
                    import math
                    lat = src.transform.f + src.transform.e * (src.height / 2.0)
                    res_m = res_x_deg * 111_320.0 * math.cos(math.radians(lat))
                else:
                    res_m = res_x_deg
            search_radius_cells = max(3, int(round(search_radius_m / res_m)))
        _run_grass_geomorphon(
            grass_bin=grass_bin,
            dem_path=dem_path,
            out_path=cache_path,
            search_cells=int(search_radius_cells),
        )

    with rasterio.open(cache_path) as src:
        cls = src.read(1)
        nodata = src.nodata

    mask = np.isin(cls, _TERRACE_GEOMORPHON_CLASSES).astype(np.float32)
    if nodata is not None:
        mask = np.where(cls == nodata, np.nan, mask)
    return mask


def _which(cmd: str) -> str | None:
    import shutil
    return shutil.which(cmd)


def _run_grass_geomorphon(
    *,
    grass_bin: str,
    dem_path: Path,
    out_path: Path,
    search_cells: int,
) -> None:
    """Run GRASS r.geomorphon in a throwaway location, write GeoTIFF result."""
    with tempfile.TemporaryDirectory(prefix="grass_geomorphon_") as tmp:
        gisdb = Path(tmp) / "grassdata"
        location = "loc"
        mapset = "PERMANENT"
        # GRASS will create the location from a georeferenced DEM via -c.
        # Then we run a one-shot exec inside that mapset.
        script = (
            f"r.in.gdal input={dem_path} output=dem --overwrite -o\n"
            f"g.region raster=dem\n"
            f"r.geomorphon elevation=dem forms=geom search={search_cells} --overwrite\n"
            f"r.out.gdal input=geom output={out_path} format=GTiff "
            f"createopt='COMPRESS=LZW' --overwrite\n"
        )
        script_path = Path(tmp) / "run.sh"
        script_path.write_text(script)

        # `grass -c <dem> <gisdb>/<location>/<mapset> --exec sh script.sh`
        # creates the location from the DEM's projection.
        cmd = [
            grass_bin,
            "-c",
            str(dem_path),
            "-e",
            str(gisdb / location),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        cmd = [
            grass_bin,
            str(gisdb / location / mapset),
            "--exec",
            "sh",
            str(script_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Distance to nearest lode-Au point (omnidirectional)
# ---------------------------------------------------------------------------

def distance_to_lode_m(
    lode_points: gpd.GeoDataFrame,
    grid: "Grid",
    *,
    max_m: float = 25_000.0,
) -> pd.Series:
    """Per-cell Euclidean distance (m) to the nearest lode-Au MRDS point.

    The companion to `distance_downstream_from_lode` for placer geology where
    flow-routing doesn't capture the lode-to-placer relationship. Sierra Tertiary
    deep-gravels are paleo-channels offset laterally or upstream of the modern
    Mother Lode trend, so the flow-routed feature is NaN at all anchor districts
    (Malakoff, Dutch Flat, You Bet, etc.). This feature fires at every cell
    within `max_m` of any lode point regardless of direction.

    LEAKAGE GUARD: same `_assert_no_placer` check as the downstream variant.

    Returns m, capped at `max_m` (cells beyond → NaN, consistent with the
    downstream variant's semantics).
    """
    _assert_no_placer(lode_points)
    if len(lode_points) == 0:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            name="distance_to_lode_m",
        )

    centroids = grid.centroid_gdf()
    lode = lode_points.to_crs(grid.crs)[["geometry"]].copy()
    joined = gpd.sjoin_nearest(
        centroids,
        lode,
        how="left",
        max_distance=max_m,
        distance_col="_lode_dist_m",
    )
    joined = joined.loc[~joined.index.duplicated(keep="first")].sort_index()
    dist = joined["_lode_dist_m"].to_numpy(dtype=np.float32)
    return pd.Series(dist, index=centroids.index, name="distance_to_lode_m")


# ---------------------------------------------------------------------------
# Distance downstream from lode-Au along NHD flowlines
# ---------------------------------------------------------------------------

def distance_downstream_from_lode(
    lode_points: gpd.GeoDataFrame,
    nhd_network: gpd.GeoDataFrame,
    grid: "Grid",
    *,
    max_km: float = 50.0,
) -> pd.Series:
    """Per-cell km downstream from the nearest MRDS lode-Au point along NHD.

    Algorithm:
      1. Snap each lode point to its nearest NHD reach.
      2. For each lode-seeded reach, walk downstream via the NHDPlus
         `hydroseq` ordering (smaller hydroseq = further downstream by
         NHDPlus convention) and tag every reach downstream of any lode
         seed with the *minimum* (seed_arbolate_sum) among the seeds
         that reach it. This is the "nearest upstream lode" assignment.
      3. For each grid centroid, find its nearest NHD reach. Distance =
         max(0, reach_arbolate_sum - seed_arbolate_sum). Cells whose
         nearest reach has no upstream lode seed get NaN. Cells beyond
         `max_km` get NaN.

    Returns a `pd.Series` aligned to `grid.centroid_gdf()`'s row order
    (one entry per cell, in row-major order), values in km.

    LEAKAGE GUARD: asserts `lode_points` has no placer-flagged records.
    The caller must pre-filter via the MRDS adapter's dev_stat and
    dep_type rules; this function refuses to silently accept placer
    seeds that would leak the label.
    """
    _assert_no_placer(lode_points)

    required = {"comid", "arbolate_sum"}
    missing = required - set(nhd_network.columns)
    if missing:
        raise ValueError(
            f"nhd_network is missing required columns {missing}; "
            f"got {list(nhd_network.columns)}"
        )
    if "hydroseq" not in nhd_network.columns:
        raise ValueError(
            "nhd_network is missing 'hydroseq'; cannot walk downstream. "
            "Re-fetch via data/nhdplus_hr.py (which writes Hydroseq into "
            "the canonical GeoPackage)."
        )

    if len(lode_points) == 0:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            name="dist_downstream_from_lode_km",
        )

    # Project everything into the grid CRS for metric distances + nearest joins.
    net = nhd_network.to_crs(grid.crs).copy()
    lode = lode_points.to_crs(grid.crs)[["geometry"]].copy()

    net["arbolate_sum"] = pd.to_numeric(net["arbolate_sum"], errors="coerce")
    net["hydroseq"] = pd.to_numeric(net["hydroseq"], errors="coerce")
    net = net.dropna(subset=["arbolate_sum", "hydroseq"]).reset_index(drop=True)
    if net.empty:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            name="dist_downstream_from_lode_km",
        )

    # 1. Snap each lode point to its nearest reach; capture that reach's
    #    arbolate_sum + hydroseq as the seed.
    snapped = gpd.sjoin_nearest(
        lode,
        net[["geometry", "comid", "arbolate_sum", "hydroseq"]],
        how="left",
    )
    snapped = snapped.dropna(subset=["arbolate_sum", "hydroseq"])
    if snapped.empty:
        return pd.Series(
            np.full(grid.n_cells, np.nan, dtype=np.float32),
            name="dist_downstream_from_lode_km",
        )

    seed_hydroseqs = snapped["hydroseq"].to_numpy()
    seed_arbolates = snapped["arbolate_sum"].to_numpy()

    # 2. Per reach: find the smallest seed_arbolate_sum among lode seeds
    #    that are upstream of this reach (i.e. seed_hydroseq >= reach_hydroseq,
    #    NHDPlus convention: hydroseq strictly decreases downstream).
    #    The min over seeds picks the *nearest* lode upstream (smallest
    #    distance gap downstream → smallest arbolate_sum offset).
    net_hydroseq = net["hydroseq"].to_numpy()
    net_arbolate = net["arbolate_sum"].to_numpy()
    seed_assigned = _nearest_upstream_seed(
        net_hydroseq=net_hydroseq,
        seed_hydroseqs=seed_hydroseqs,
        seed_arbolates=seed_arbolates,
    )
    net["seed_arbolate_sum"] = seed_assigned

    # 3. For every grid centroid, nearest NHD reach → distance from seed.
    centroids = grid.centroid_gdf()  # already in grid.crs
    joined = gpd.sjoin_nearest(
        centroids[["row", "col", "geometry"]],
        net[["geometry", "arbolate_sum", "seed_arbolate_sum"]],
        how="left",
    )
    # sjoin_nearest can return multiple matches when distances tie; keep first.
    joined = joined.loc[~joined.index.duplicated(keep="first")].sort_index()

    dist_km = joined["arbolate_sum"] - joined["seed_arbolate_sum"]
    dist_km = dist_km.where(dist_km >= 0, np.nan)
    dist_km = dist_km.where(dist_km <= max_km, np.nan)

    out = pd.Series(
        dist_km.to_numpy(dtype=np.float32),
        name="dist_downstream_from_lode_km",
    )
    return out


def _assert_no_placer(lode_points: gpd.GeoDataFrame) -> None:
    """Refuse to run if any lode_points record carries a placer dep_type.

    The placer-Au pipeline derives its positive labels from these same
    MRDS rows when filtered with the placer regex; using them as
    "upstream lode" seeds would leak the label into the feature. The
    MRDS adapter (`data/adapters/occurrences/mrds.py`) writes a
    `dep_type` column on the raw extract; the caller should drop placer
    rows there (or pass an already-filtered subset).
    """
    if "dep_type" not in lode_points.columns:
        raise AssertionError(
            "distance_downstream_from_lode: lode_points has no `dep_type` "
            "column. The placer-leakage guard can only verify filtering when "
            "this column is present. Pull lode_points from the MRDS adapter "
            "(which writes dep_type) and pre-filter to dev_stat in "
            "(Past Producer, Producer) and drop placer dep_types before "
            "calling."
        )
    dep = lode_points["dep_type"].astype("string").str.lower().fillna("")
    placer_hits = dep.str.contains(_PLACER_DEP_TYPE_PATTERN, regex=True, na=False)
    if placer_hits.any():
        n = int(placer_hits.sum())
        raise AssertionError(
            f"distance_downstream_from_lode: {n} of {len(lode_points)} lode_points "
            f"records match the placer dep_type regex "
            f"({_PLACER_DEP_TYPE_PATTERN!r}). These would leak the label. "
            "Pre-filter to dev_stat in (Past Producer, Producer) and drop "
            "placer dep_types before calling."
        )


def _nearest_upstream_seed(
    *,
    net_hydroseq: np.ndarray,
    seed_hydroseqs: np.ndarray,
    seed_arbolates: np.ndarray,
) -> np.ndarray:
    """For each reach, return the arbolate_sum of the closest upstream seed.

    NHDPlus convention: along a single downstream walk, hydroseq strictly
    decreases. A seed at hydroseq H_s is upstream of a reach at H_r iff
    H_s >= H_r. Among all such seeds, "closest" = largest seed_arbolate
    (smallest downstream gap), but that's only meaningful if both seed
    and reach sit on the same downstream path. We approximate with a
    pure-hydroseq search; full network-walk is left for a future revision
    and is acceptable here because the calling code caps results at
    `max_km` and discards cells whose nearest reach yields a negative
    or out-of-range distance (the natural filter for off-network seeds).

    Returns an array shaped like `net_hydroseq`; entries with no
    upstream seed are NaN.
    """
    order = np.argsort(seed_hydroseqs)  # ascending
    sorted_h = seed_hydroseqs[order]
    sorted_a = seed_arbolates[order]

    out = np.full_like(net_hydroseq, np.nan, dtype=np.float64)
    # For each reach hydroseq H_r, find seeds with H_s >= H_r → the
    # rightmost block of sorted_h. arbolate_sum grows downstream, so
    # distance from a seed to a reach is (reach.arbolate - seed.arbolate).
    # The "closest upstream" seed (smallest non-negative distance) is the
    # one with the LARGEST seed.arbolate_sum among the upstream candidates.
    # searchsorted gives the leftmost insert point for H_r in sorted_h;
    # everything at index >= idx is an upstream-by-hydroseq seed.
    idx = np.searchsorted(sorted_h, net_hydroseq, side="left")
    for i, start in enumerate(idx):
        if start >= len(sorted_a):
            continue
        # Largest seed_arbolate_sum among upstream candidates = closest upstream.
        out[i] = float(np.max(sorted_a[start:]))
    return out


# ---------------------------------------------------------------------------
# CLI smoke test (no Whitebox / NHD needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dem = (
        100.0
        + 30.0 * rng.standard_normal((100, 100)).cumsum(axis=0) / 50.0
        + 10.0 * rng.standard_normal((100, 100))
    ).astype(np.float32)
    # Fake "flow accumulation" as a positive-only random surface so the
    # SPI / TWI / ksn calls have something nontrivial to chew on.
    flow_acc = (1.0 + rng.exponential(scale=50.0, size=dem.shape)).astype(np.float32)
    # Fake slope in degrees in a realistic range.
    slope = np.clip(
        np.abs(np.gradient(dem)[0]) * 5.0,
        0.0, 60.0,
    ).astype(np.float32)

    spi_band = stream_power_index_band(flow_acc, slope)
    twi = topographic_wetness_index(flow_acc, slope)
    ksn = knickpoint_ksn(dem, flow_acc)
    tpi_arr = tpi(dem, radius_cells=8)

    print(f"dem.shape={dem.shape}, dtype={dem.dtype}")
    print(f"spi_band.shape={spi_band.shape}, nan_frac={np.isnan(spi_band).mean():.3f}")
    print(f"twi.shape={twi.shape}, nan_frac={np.isnan(twi).mean():.3f}")
    print(f"ksn.shape={ksn.shape}, range=({np.nanmin(ksn):.3f}, {np.nanmax(ksn):.3f})")
    print(f"tpi.shape={tpi_arr.shape}, range=({np.nanmin(tpi_arr):.3f}, {np.nanmax(tpi_arr):.3f})")
    print("OK")
