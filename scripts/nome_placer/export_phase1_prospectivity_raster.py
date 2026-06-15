"""Export the Phase 1 Nome placer prospectivity raster as a multi-band GeoTIFF.

This is the concrete Phase 1 model output: per-cell scores for each
of the six populations (BL, AP, TB, BC, QM, buried_BL) plus the
per-cell composite max. Multi-band GeoTIFF on the 25 m EPSG:3338
working grid; reprojected copy in EPSG:4326 for goldbug consumption.

Inputs are the v1.5 Phase 1 stack:
- IfSAR Alaska DEM (clipped + reprojected to EPSG:3338)
- DEM-derived streams (flow-accumulation threshold 100)
- Bearcub bedrock_topo + variance (Phase 1.5 buried-BL detection)

Bands (in order):
  1: bl              - true beach (Present + Second + Third + Fourth)
  2: ap              - abrasion-platform / sloughover (Submarine + ...)
  3: tb              - Tertiary buried high-bench (Tuck +300 ft / +600 ft)
  4: bc              - beach-stream confluence (BL/AP x stream proximity)
  5: qm              - off-beach modern creek (in-drift x stream proximity)
  6: buried_bl       - buried true beach (bedrock at stand elevation)
  7: composite       - per-cell max() across all six populations

Goldbug consumption: this raster + the bands.json sidecar replace
the placeholder rough-affine TIFFs. Goldbug samples the composite
band as the "Nome placer prospectivity" overlay; per-population
bands are available for analytic drill-down.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

from ai_minerals.aoi import NOME_PLACER
from ai_minerals.data.adapters.elevation.ifsar_alaska import load
from ai_minerals.features.coastal_scorer import score_all_populations
from ai_minerals.features.hydrology import (
    distance_to_stream_m, flow_accumulation, streams_from_flow_accumulation,
)
from ai_minerals.features.lithology import (
    elevation_of_bedrock_m, load_bedrock_surface,
)
from ai_minerals.grid import build_grid
from ai_minerals.regions.nome_placer import NOME_PLACER_REGION


BANDS_IN_ORDER: list[tuple[str, str, str]] = [
    ("bl", "True beach (BL)", "Tuck 1942: Present + Second + Third + Fourth beach stands"),
    ("ap", "Abrasion platform / sloughover (AP)", "Tuck 1942: Submarine + Intermediate + Monroeville"),
    ("tb", "Tertiary buried high-bench (TB)", "Tuck 1942: head-of-Dexter-Creek high-bench gravel at +300 ft / +600 ft"),
    ("bc", "Beach-stream confluence (BC)", "max(BL, buried_BL, AP) x Gaussian(distance_to_stream)"),
    ("qm", "Off-beach modern creek (QM)", "in-drift elevation band x Gaussian(distance_to_stream)"),
    ("buried_bl", "Buried true beach (Phase 1.5)", "bedrock at stand elevation; lifts Bear Cub MS 1178 BC"),
    ("composite", "Composite score", "per-cell max() across six populations"),
]


def main() -> None:
    out_dir = Path("data/derived/nome_placer/prospectivity_v1p5")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading IfSAR DEM + reprojecting to EPSG:3338 ...")
    da_full = load(NOME_PLACER_REGION.raw_paths["dem"], NOME_PLACER)
    dem = da_full.rio.clip_box(*NOME_PLACER.bbox).rio.reproject("EPSG:3338")

    print("Computing flow-accumulation streams ...")
    flow_acc = flow_accumulation(dem.values, transform=dem.rio.transform())
    streams = streams_from_flow_accumulation(
        flow_acc, transform=dem.rio.transform(), crs=str(dem.rio.crs),
    )

    print("Building grid + distance-to-stream ...")
    grid = build_grid(NOME_PLACER, resolution_m=25, working_crs="EPSG:3338")
    d_stream = distance_to_stream_m(streams, grid)

    print("Loading bearcub bedrock-topo for buried-BL ...")
    bedrock = load_bedrock_surface(
        Path(NOME_PLACER_REGION.raw_paths["bedrock_topo"]),
        Path(NOME_PLACER_REGION.raw_paths["bedrock_topo_variance"]),
        grid,
    )
    bedrock_elev = elevation_of_bedrock_m(dem, grid, bedrock)

    print("Scoring all populations ...")
    scores = score_all_populations(
        dem, grid,
        distance_to_stream_m=d_stream,
        bedrock_elevation_m=bedrock_elev,
    )

    # Reshape per-cell Series to (H, W)
    H, W = grid.shape
    band_arrays = []
    for band_key, _, _ in BANDS_IN_ORDER:
        series = getattr(scores, band_key)
        arr = np.asarray(series, dtype=np.float32).reshape((H, W))
        # Replace NaN with 0 for goldbug consumption (composite is naturally
        # 0 in nodata cells from the inter-band max).
        arr = np.where(np.isnan(arr), 0.0, arr)
        band_arrays.append(arr)
    stack = np.stack(band_arrays, axis=0)

    # Write EPSG:3338 GeoTIFF (working CRS)
    transform_3338 = from_origin(
        west=grid.xs[0] - 12.5,  # cell centre to NW corner offset
        north=grid.ys[-1] + 12.5,
        xsize=25.0,
        ysize=25.0,
    )
    # The grid's centroid_gdf has rows ordered south-to-north, but raster
    # convention is north-to-south. Flip the row axis to match.
    stack_north_up = stack[:, ::-1, :]

    out_3338 = out_dir / "nome_placer_prospectivity_v1p5_3338.tif"
    with rasterio.open(
        out_3338, "w",
        driver="GTiff", height=H, width=W,
        count=stack.shape[0], dtype="float32",
        crs=CRS.from_epsg(3338), transform=transform_3338,
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        for i, arr in enumerate(stack_north_up, start=1):
            dst.write(arr, i)
            dst.set_band_description(i, BANDS_IN_ORDER[i - 1][1])
    print(f"  EPSG:3338 raster -> {out_3338} ({out_3338.stat().st_size:,} B)")

    # Reproject to EPSG:4326 for goldbug
    out_4326 = out_dir / "nome_placer_prospectivity_v1p5_4326.tif"
    with rasterio.open(out_3338) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            crs="EPSG:4326",
            transform=dst_transform,
            width=dst_w,
            height=dst_h,
        )
        with rasterio.open(out_4326, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear,
                )
                dst.set_band_description(i, BANDS_IN_ORDER[i - 1][1])
    print(f"  EPSG:4326 raster -> {out_4326} ({out_4326.stat().st_size:,} B)")

    # bands.json sidecar
    bands_meta = {
        "schema_version": "1.0",
        "raster_3338": out_3338.name,
        "raster_4326": out_4326.name,
        "resolution_m": 25,
        "working_crs": "EPSG:3338",
        "delivery_crs": "EPSG:4326",
        "bands": [
            {"index": i + 1, "key": k, "name": n, "description": d, "value_range": [0.0, 1.0]}
            for i, (k, n, d) in enumerate(BANDS_IN_ORDER)
        ],
        "phase": "1.5 v1 (buried-BL feeding BC; KDTree-fast streams)",
        "validation_gate": {
            "anvil_creek_qm_p90": 1.000,
            "third_beach_bl_p90": 1.000,
            "submarine_beach_bl_p90": 1.000,
            "molasses_ms_1179_tb_p90": 0.402,
            "honey_ms_1181_tb_p90": 0.402,
            "bear_cub_ms_1178_bc_p90": 0.967,
            "passed_count": 6,
            "total_landmarks": 6,
        },
        "coverage_threshold": {
            "description": (
                "Goldbug-side contract: cells where composite >= 0.5 are "
                "the v1 high-prospectivity ground (top quartile). Cells "
                "below 0.3 are the v1 low-prospectivity floor. Tune per "
                "goldbug's display preference."
            ),
            "high_prospectivity_min": 0.50,
            "low_prospectivity_max": 0.30,
        },
    }
    meta_path = out_dir / "bands.json"
    meta_path.write_text(json.dumps(bands_meta, indent=2))
    print(f"  bands.json sidecar -> {meta_path}")

    # Deliver to goldbug
    goldbug_inbox = Path(
        "/home/sky/src/learning/gldbg/handoff/inbox/"
        "2026-06-14-from-ai-minerals-nome-prospectivity-v1p5"
    )
    goldbug_inbox.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in (out_3338, out_4326, meta_path):
        shutil.copy(f, goldbug_inbox / f.name)
    print(f"\nDelivered to goldbug: {goldbug_inbox}")


if __name__ == "__main__":
    main()
