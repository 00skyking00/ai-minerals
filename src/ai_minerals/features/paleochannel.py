"""Paleochannel-likelihood raster builder for the Tertiary deep-gravel branch.

Composites four detection signals into a single per-cell paleochannel-
likelihood raster, following `research/Paleo_River_channel_detection.md`:

  1. Relative Elevation Model (REM) via the `riverrem` package — detrends
     the regional gradient to expose terraces, benches, and abandoned
     channels that absolute DEMs bury. Strongest single signal where
     1 m LiDAR is flown.

  2. Local Relief Model (LRM) — `dem - focal_mean(dem, lrm_kernel_m)`.
     Cheap, generic; surfaces disconnected paleochannels via local
     micro-topography. Robust where 1 m LiDAR isn't available.

  3. GeoMorphic Index components:
       - Geomorphon valleys via GRASS `r.geomorphon` (ridge class masked).
         Reused from `features/hydrology.py::geomorphon_terrace_mask`.
       - D-infinity flow accumulation via Whitebox `DInfFlowAccumulation`.
       - Black Top-Hat morphology: `closing(dem) - dem` (scipy.ndimage).
       - Multiscale Elevation Percentile: low-percentile band over a
         medium window (scipy.ndimage.percentile_filter).
     Each component normalized to [0, 1] and averaged into the GMI score.

  4. Composite: weighted average of REM, LRM, GMI; normalized to [0, 1];
     low-relief flat areas (slope < 1°) get a small boost.

The Attention-UNet segmentation step in the paleochannel doc is Phase 3,
not built here.

Where 3DEP 1 m isn't flown, the caller passes a 10 m DEM and the REM/LRM
signals degrade gracefully (lower resolution = coarser terraces); the
GMI still works at 10 m. Coverage gaps are logged into the sidecar JSON
written by `scripts/northern_sierra_placer_precompute_paleochannel.py`.

This module is **import-light** — heavy packages (riverrem, whitebox,
scikit-image) are imported inside the function body so the module
imports cleanly even when the `paleochannel` optional dep group isn't
installed. Install with `uv sync --extra paleochannel`. GRASS must be
on the PATH (system package).
"""

from __future__ import annotations

import resource
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

_PAGE_BYTES = resource.getpagesize()


def _log_rss(label: str) -> None:
    """Print current + peak resident-set size. The whole-DEM arrays here are
    ~5.6 GB each at 10 m over the northern Sierra footprint, so this is how we
    see which stage crosses the OOM wall. Cheap: two reads, no allocation."""
    with open("/proc/self/statm") as fh:
        cur_gb = int(fh.read().split()[1]) * _PAGE_BYTES / 1e9
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9
    print(f"    [mem] {label:<26} cur={cur_gb:5.2f} GB  peak={peak_gb:5.2f} GB", flush=True)


def _read_dem(dem_path: Path) -> tuple[np.ndarray, Affine, CRS]:
    with rasterio.open(dem_path) as src:
        arr = src.read(1, masked=False)
        # Cast to float32 only if needed (the 3DEP tiles are already Float32),
        # and replace nodata in place. The old `np.where(arr==nd, nan, arr)`
        # allocated a *second* full-resolution copy (~5.6 GB at 10 m over this
        # footprint) on top of `arr`; the in-place write leaves only the
        # transient boolean mask. See _log_rss output for the difference.
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        nodata = src.nodatavals[0] if src.nodatavals else None
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.transform, src.crs


def _write_geotiff(arr: np.ndarray, transform: Affine, crs: CRS, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
        compress="deflate",
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Robust min-max to [0, 1] using p1/p99 clipping. NaN preserved."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        out = np.zeros_like(arr, dtype=np.float32)
        out[~np.isfinite(arr)] = np.nan
        return out
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _compute_rem(
    dem: np.ndarray, dem_path: Path, transform: Affine, crs: CRS, *,
    rem_source: str = "flow", nhd_path: Path | None = None,
) -> np.ndarray:
    """Relative Elevation Model: elevation relative to the local channel network.

    rem_source:
      "nhd": rasterize NHDPlus HR flowlines (from K.1 fetch) and detrend against
        the nearest mapped channel. Authoritative USGS hydrography, no network
        dependency at runtime, far better headwater tributary coverage than OSM
        for CONUS. Falls back to flow-REM if the GeoPackage is missing or empty
        in the DEM bounds. The right choice for any US AOI where NHD has been
        fetched.
      "flow": derive the channel network from the DEM's own D-infinity flow
        accumulation and detrend against the nearest channel's elevation. No
        network, no external data, deterministic — the right choice for any AOI
        where NHD coverage doesn't exist (Alaska bush, Canada, ROW).
      "osm": query OpenStreetMap river centerlines via riverrem. Depends on a
        live Overpass query at runtime, which rate-limits aggressively; on any
        failure falls back to flow-REM. Retained for ROW work and parity testing,
        not recommended for US AOIs where "nhd" is available.

    All three branches return the same polarity: channel-level cells sit near the
    top of the range, uplands near the bottom, so high REM = paleochannel-proximal.
    """
    if rem_source == "nhd":
        try:
            return _compute_rem_from_nhd(dem, transform, crs, nhd_path=nhd_path)
        except Exception as exc:
            print(f"  WARNING: NHD REM unavailable ({type(exc).__name__}: {exc}); "
                  f"using flow-derived REM instead.")
        return _compute_rem_from_flow(dem, dem_path)
    if rem_source == "osm":
        try:
            return _compute_rem_osm(dem_path)
        except Exception as exc:
            print(f"  WARNING: OSM REM unavailable ({type(exc).__name__}: {exc}); "
                  f"using flow-derived REM instead.")
    return _compute_rem_from_flow(dem, dem_path)


def _compute_rem_osm(dem_path: Path) -> np.ndarray:
    """REM from OSM river centerlines via riverrem. Raises on any failure so the
    caller can fall back. Returns height-above-river negated (channel high)."""
    # riverrem 0.0.1 uses bare `import gdal`/`osr`/`ogr`; shim the legacy names
    # so its import succeeds on modern GDAL.
    import sys as _sys
    from osgeo import gdal as _osgeo_gdal, ogr as _osgeo_ogr, osr as _osgeo_osr
    _sys.modules.setdefault("gdal", _osgeo_gdal)
    _sys.modules.setdefault("ogr", _osgeo_ogr)
    _sys.modules.setdefault("osr", _osgeo_osr)
    # riverrem 0.0.1 calls `osmnx.geometries_from_bbox(*bbox, tags=...)` with
    # bbox = (north, south, east, west) — the pre-2.0 signature. osmnx 2.x
    # replaced it with `features_from_bbox((west, south, east, north), tags)`. A
    # bare alias collides on `tags`; this adapter maps the old call onto the new.
    import osmnx as _osmnx
    if not hasattr(_osmnx, "geometries_from_bbox") and hasattr(_osmnx, "features_from_bbox"):
        def _geometries_from_bbox(*bbox, tags=None, **_ignored):
            north, south, east, west = bbox
            return _osmnx.features_from_bbox((west, south, east, north), tags)
        _osmnx.geometries_from_bbox = _geometries_from_bbox
    from riverrem.REMMaker import REMMaker

    out_dir = dem_path.parent / "_rem_cache"
    out_dir.mkdir(exist_ok=True, parents=True)
    maker = REMMaker(dem=str(dem_path), out_dir=str(out_dir))
    rem_path = Path(maker.make_rem())
    arr, _, _ = _read_dem(rem_path)
    return -arr  # height-above-river -> channel-relative (channel high after normalize)


def _compute_rem_from_nhd(
    dem: np.ndarray, transform: Affine, crs: CRS, *,
    nhd_path: Path | None = None, min_stream_order: int = 2,
) -> np.ndarray:
    """REM detrended against NHDPlus HR flowlines.

    Rasterizes the NHD flowline GeoPackage onto the DEM grid (any cell touched
    by a stream-order >= 2 flowline becomes a channel cell), then runs the same
    Euclidean feature transform + elevation lookup the flow-derived REM uses.
    Output polarity matches: channel-level high, uplands low.

    Why stream_order >= 2: order-1 cells are NHD's ephemeral first-order
    headwaters, which in this dataset include irrigation ditches and
    intermittent draws that aren't relevant to placer channel geometry. Dropping
    them is the NHD analog of the flow-REM's "top 5% accumulation" channel
    threshold and produces a comparable density of channel cells.

    Raises on missing GPKG or empty intersection; caller falls back to flow-REM.
    """
    from scipy import ndimage
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds

    if nhd_path is None or not Path(nhd_path).exists():
        raise FileNotFoundError(
            f"NHD flowlines GeoPackage not found at {nhd_path}; run K.1 fetch"
        )

    gdf = gpd.read_file(nhd_path)
    if "stream_order" in gdf.columns:
        gdf = gdf[gdf["stream_order"].fillna(0) >= min_stream_order]
    if len(gdf) == 0:
        raise RuntimeError(f"No NHD flowlines with stream_order >= {min_stream_order}")

    if gdf.crs is None or str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)

    minx, miny, maxx, maxy = array_bounds(dem.shape[0], dem.shape[1], transform)
    gdf = gdf.cx[minx:maxx, miny:maxy]
    if len(gdf) == 0:
        raise RuntimeError("No NHD flowlines intersect the DEM bounds")

    print(f"    NHD-REM: {len(gdf):,} flowlines (stream_order >= {min_stream_order})")

    channel = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=dem.shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True,
    ).astype(bool)
    del gdf

    finite = np.isfinite(dem)
    channel &= finite
    if not channel.any():
        raise RuntimeError("NHD rasterization produced no channel cells")

    idx = ndimage.distance_transform_edt(
        ~channel, return_distances=False, return_indices=True,
    )
    del channel
    nearest_channel_elev = dem[tuple(idx)]
    del idx
    rem = (nearest_channel_elev - dem).astype(np.float32)
    rem[~finite] = np.nan
    return rem


def _compute_rem_from_flow(
    dem: np.ndarray, dem_path: Path, *, channel_percentile: float = 95.0,
) -> np.ndarray:
    """Network-free REM via Height-Above-Nearest-Drainage.

    Channels are the high-flow-accumulation cells of the DEM's own D-infinity
    routing (the top (100 - channel_percentile)%). Each cell's REM is the
    elevation of its nearest channel cell minus its own elevation, so channel-
    level cells sit near the maximum after normalization and uplands sit low —
    the same polarity as the riverrem path. Needs no OSM/Overpass and is
    deterministic for a given DEM. The nearest-channel lookup is a Euclidean
    feature transform (not flow-routed), which is a fair approximation at the
    30 m scale this runs at."""
    from scipy import ndimage

    flow = _dinf_flow_accumulation(dem_path)  # log1p(D-inf accumulation), full grid
    finite = np.isfinite(dem) & np.isfinite(flow)
    thr = np.percentile(flow[finite], channel_percentile)
    channel = finite & (flow >= thr)
    del flow
    if not channel.any():
        raise RuntimeError("flow-derived REM found no channel cells")
    # For every cell, the (row, col) of its nearest channel cell: EDT on the
    # complement, returning indices into the channel set.
    idx = ndimage.distance_transform_edt(~channel, return_distances=False, return_indices=True)
    del channel
    nearest_channel_elev = dem[tuple(idx)]
    del idx
    rem = (nearest_channel_elev - dem).astype(np.float32)
    rem[~np.isfinite(dem)] = np.nan
    return rem


def _pixel_size_m(transform: Affine, crs: CRS, n_rows: int) -> tuple[float, float]:
    """(north-south, east-west) ground resolution in meters per cell.

    For a projected CRS already in meters, the transform scales are meters and
    are returned directly. For a geographic CRS (degrees), convert with the
    standard ~111.32 km per degree of latitude, scaling the east-west term by
    cos(center latitude). 3DEP tiles arrive in EPSG:4269 (degrees), so dividing
    a meter-valued kernel width by the raw degree scale (~9.3e-5) would inflate
    the kernel ~111,000x and collapse the LRM into `dem - global_mean`.
    """
    res_y, res_x = abs(transform.e), abs(transform.a)
    if crs is not None and crs.is_geographic:
        import math
        center_lat = transform.f + transform.e * (n_rows / 2.0)
        m_per_deg = 111_320.0
        return res_y * m_per_deg, res_x * m_per_deg * math.cos(math.radians(center_lat))
    return res_y, res_x


def _odd_cells(kernel_m: float, res_m: float) -> int:
    """Kernel width in cells (>= 3, odd) covering `kernel_m` meters of ground."""
    k = max(3, int(round(kernel_m / res_m)))
    return k + 1 if k % 2 == 0 else k


def _focal_mean(arr: np.ndarray, kernel: int | tuple[int, int]) -> np.ndarray:
    """Mean over a (possibly anisotropic) kernel; NaN-aware (NaN cells excluded).

    `kernel` is a cell count or an (rows, cols) pair; the latter keeps the
    ground footprint square on a geographic grid where cells aren't. Buffers are
    freed as soon as they're consumed: at 10 m over this footprint each full
    array is ~5.6 GB, so the old five-live-arrays version peaked near 33 GB."""
    from scipy import ndimage

    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0).astype(np.float32)
    summed = ndimage.uniform_filter(filled, size=kernel, mode="nearest")
    del filled
    counts = ndimage.uniform_filter(finite.astype(np.float32), size=kernel, mode="nearest")
    del finite
    with np.errstate(invalid="ignore", divide="ignore"):
        summed /= counts  # in-place: counts==0 yields nan/inf, masked next
    summed[counts == 0] = np.nan
    return summed


def _compute_lrm(dem: np.ndarray, transform: Affine, crs: CRS, kernel_m: float) -> np.ndarray:
    """Local Relief Model: dem - focal_mean(dem). Positive on humps, negative in hollows."""
    res_y_m, res_x_m = _pixel_size_m(transform, crs, dem.shape[0])
    kernel = (_odd_cells(kernel_m, res_y_m), _odd_cells(kernel_m, res_x_m))
    return (dem - _focal_mean(dem, kernel)).astype(np.float32)


def _black_top_hat(
    dem: np.ndarray, *, kernel_m: float = 150.0, res_m: tuple[float, float] = (10.0, 10.0),
) -> np.ndarray:
    """Closing(dem) - dem. Positive in linear depressions (e.g., abandoned channels).

    `kernel_m` is the structuring-element footprint in ground meters, converted
    to cells per axis from `res_m` so the feature covers the same ground extent
    whatever the internal resolution (the DEM grid is non-square in meters)."""
    from scipy import ndimage

    kernel = (_odd_cells(kernel_m, res_m[0]), _odd_cells(kernel_m, res_m[1]))
    finite = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    closed = ndimage.grey_closing(finite, size=kernel)
    out = (closed - dem).astype(np.float32)
    out[~np.isfinite(dem)] = np.nan
    return out


def _multiscale_elevation_percentile(
    dem: np.ndarray, *, percentile: float = 10.0, kernel_m: float = 250.0,
    res_m: tuple[float, float] = (10.0, 10.0),
) -> np.ndarray:
    """MEP: persistent-low-points detector. dem - percentile_filter(dem, p)."""
    from scipy import ndimage

    kernel = (_odd_cells(kernel_m, res_m[0]), _odd_cells(kernel_m, res_m[1]))
    finite = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    low = ndimage.percentile_filter(finite, percentile=percentile, size=kernel)
    out = (dem - low).astype(np.float32)
    out[~np.isfinite(dem)] = np.nan
    return out


def _dinf_flow_accumulation(dem_path: Path) -> np.ndarray:
    """Whitebox DInfFlowAccumulation. Returns a 2-D float32 array aligned to dem_path.

    Caches the raster at `<dem>.dinf_facc.tif`; with rem_source="flow" both the
    REM channel and the GMI need it, so the cache avoids running the (expensive)
    whitebox routing twice on the same DEM."""
    out_path = dem_path.with_suffix(".dinf_facc.tif")
    if out_path.exists():
        arr, _, _ = _read_dem(out_path)
        return np.log1p(arr).astype(np.float32)

    try:
        from whitebox_workflows import WbEnvironment
    except ImportError as exc:
        raise RuntimeError(
            "_dinf_flow_accumulation requires whitebox-workflows; "
            "install with `uv sync --extra paleochannel`."
        ) from exc

    wbe = WbEnvironment()
    wbe.working_directory = str(dem_path.parent)
    dem = wbe.read_raster(str(dem_path))
    # whitebox-workflows >= 2.0 disabled the flat WbEnvironment tool
    # methods; tools now live on category objects (here: `wbe.hydrology`).
    # The 2.x rename also dropped the underscore in d_inf_* → dinf_*.
    filled = wbe.hydrology.fill_depressions(dem=dem)
    flow_dir = wbe.hydrology.dinf_pointer(dem=filled)
    flow_acc = wbe.hydrology.dinf_flow_accum(input=flow_dir, input_is_pointer=True)
    # whitebox-workflows >= 2.0 moved compression flags into an `options`
    # dict; bare `compress=True` kwarg is rejected.
    wbe.write_raster(flow_acc, str(out_path), options={"compress": True})
    arr, _, _ = _read_dem(out_path)
    return np.log1p(arr).astype(np.float32)  # log to compress dynamic range


def _gmi(
    dem: np.ndarray, dem_path: Path, *, geomorphon_terrace: np.ndarray | None,
    res_m: tuple[float, float] = (10.0, 10.0),
) -> np.ndarray:
    """GeoMorphic Index: averaged signal across geomorphon valleys, BTH, MEP, D-inf flow."""
    # Running NaN-aware mean. The old version appended every component to a
    # list and then `np.stack`ed it into one (N x full-resolution) array — a
    # ~22 GB single allocation at 10 m here, on top of the components it copied
    # from. Accumulating sum + per-cell count and freeing each component as it
    # lands reproduces np.nanmean(stack) exactly while holding ~2 arrays.
    acc: np.ndarray | None = None
    cnt: np.ndarray | None = None

    def _accumulate(component: np.ndarray) -> None:
        nonlocal acc, cnt
        if acc is None:
            acc = np.zeros(component.shape, dtype=np.float32)
            cnt = np.zeros(component.shape, dtype=np.uint8)
        finite = np.isfinite(component)
        acc[finite] += component[finite]
        cnt[finite] += 1

    if geomorphon_terrace is not None:
        _accumulate(_normalize_01(geomorphon_terrace.astype(np.float32)))
    _accumulate(_normalize_01(_black_top_hat(dem, res_m=res_m)))
    _log_rss("gmi: +BTH")
    _accumulate(_normalize_01(_multiscale_elevation_percentile(dem, res_m=res_m)))
    _log_rss("gmi: +MEP")
    try:
        _accumulate(_normalize_01(_dinf_flow_accumulation(dem_path)))
        _log_rss("gmi: +Dinf")
    except RuntimeError:
        # whitebox not installed; degrade gracefully.
        pass

    out = np.full(acc.shape, np.nan, dtype=np.float32)
    valid = cnt > 0
    with np.errstate(invalid="ignore"):
        out[valid] = acc[valid] / cnt[valid]
    _log_rss("gmi: mean done")
    return out


def _decimate_dem(src_path: Path, target_res_m: float) -> Path:
    """Write an area-averaged copy of the DEM at ~target_res_m and return its path.

    GDAL resamples during the windowed read, so peak memory tracks the *output*
    size, not the 1.4-Gcell input. Returns src_path unchanged if it's already at
    or coarser than the target. The morphometric operators here are 80-250 m
    low-pass filters feeding a 250 m model grid, so ~30 m internal resolution is
    indistinguishable downstream from native 10 m while using ~9x less memory."""
    with rasterio.open(src_path) as src:
        res_y_m, res_x_m = _pixel_size_m(src.transform, src.crs, src.height)
        fy = max(1, int(round(target_res_m / res_y_m)))
        fx = max(1, int(round(target_res_m / res_x_m)))
        if fy <= 1 and fx <= 1:
            return src_path
        h, w = src.height // fy, src.width // fx
        arr = src.read(1, out_shape=(h, w), resampling=Resampling.average)
        t = src.transform
        new_t = Affine(t.a * src.width / w, t.b, t.c, t.d, t.e * src.height / h, t.f)
        out_path = src_path.with_suffix(f".ds{int(round(target_res_m))}m.tif")
        nodata = src.nodatavals[0] if src.nodatavals else None
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=src.crs, transform=new_t, nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(arr.astype(np.float32), 1)
    return out_path


def build_paleochannel_likelihood(
    lidar_dem_path: Path,
    out_path: Path,
    *,
    rem_radius_m: float = 200.0,
    lrm_kernel_m: float = 100.0,
    weights: tuple[float, float, float] = (0.45, 0.20, 0.35),
    downsample_to_m: float | None = None,
    rem_source: str = "flow",
    nhd_path: Path | None = None,
) -> Path:
    """Composite REM + LRM + GMI into a single per-cell paleochannel-likelihood raster.

    lidar_dem_path: input DEM (3DEP 1 m where flown; 10 m fallback OK).
    out_path: written GeoTIFF in the (possibly downsampled) DEM's CRS.
    weights: (rem_w, lrm_w, gmi_w); must sum to ~1.0.
    downsample_to_m: if set and finer than the DEM, process at this ground
        resolution instead. The kernels are meter-parameterized, so the feature
        is resolution-independent; native 10 m over a large footprint needs a
        ~17 GB floor that won't fit a 16 GB box, and the 250 m model grid can't
        use sub-30 m detail anyway.
    rem_source: "nhd" (NHDPlus HR flowlines; best US data, no network),
        "flow" (DEM-derived; no external data, works anywhere), or "osm"
        (riverrem/Overpass; rate-limited, falls back to flow). See _compute_rem.
    nhd_path: GeoPackage of NHD flowlines used when rem_source="nhd". Region's
        raw_paths["nhd_flowlines"] is the canonical source; default None raises
        via the dispatcher's FileNotFoundError fallback to flow-REM.

    nhd_network is intentionally NOT a parameter: the GMI's flow-accumulation
    component computes its own flow routing from the DEM directly via
    Whitebox D-infinity, which is more faithful to local paleochannel geometry
    than snapping to a synthetic NHD line.

    Returns out_path.
    """
    w_rem, w_lrm, w_gmi = weights

    if downsample_to_m is not None:
        coarse = _decimate_dem(lidar_dem_path, downsample_to_m)
        if coarse != lidar_dem_path:
            print(f"  downsampled DEM -> {coarse.name} (~{downsample_to_m:.0f} m internal resolution)")
            lidar_dem_path = coarse

    dem, transform, crs = _read_dem(lidar_dem_path)
    res_m = _pixel_size_m(transform, crs, dem.shape[0])
    _log_rss(f"after _read_dem {dem.shape}")

    # Fold each channel into the weighted composite as soon as it's computed,
    # then drop it. The old version normalized rem/lrm/gmi into three more full
    # arrays and summed them with all six live at once (~40 GB at 10 m); this
    # holds the running composite plus at most one channel.
    print(f"  REM (source={rem_source}; nhd/osm fall back to flow if unavailable)")
    rem = _compute_rem(
        dem, lidar_dem_path, transform, crs,
        rem_source=rem_source, nhd_path=nhd_path,
    )
    composite = (w_rem * _normalize_01(rem)).astype(np.float32)
    del rem
    _log_rss("after REM")

    print(f"  LRM kernel={lrm_kernel_m}m")
    lrm = _compute_lrm(dem, transform, crs, lrm_kernel_m)
    composite += w_lrm * _normalize_01(lrm)
    del lrm
    _log_rss("after LRM")

    # Geomorphon terrace mask is computed by hydrology.py; pass through
    # if available, otherwise the GMI just averages BTH + MEP + D-inf.
    try:
        from ai_minerals.features.hydrology import geomorphon_terrace_mask
        terrace = geomorphon_terrace_mask(lidar_dem_path)
    except Exception:
        terrace = None

    print("  GMI (geomorphon, BTH, MEP, D-inf flow)")
    gmi = _gmi(dem, lidar_dem_path, geomorphon_terrace=terrace, res_m=res_m)
    del dem
    composite += w_gmi * _normalize_01(gmi)
    del gmi
    _log_rss("after GMI")

    composite /= sum(weights)
    composite = _normalize_01(composite)
    _log_rss("after composite")

    _write_geotiff(composite, transform, crs, out_path)
    return out_path
