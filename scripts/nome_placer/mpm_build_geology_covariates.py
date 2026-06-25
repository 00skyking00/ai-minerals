"""Build geology MPM covariates on the Nome target grid.

Source: USGS OFR 2009-1254 (digital data for SIM 3131, Bedrock Geologic Map of
the Seward Peninsula). Nome quad polygons (nm_ddp) + arcs (nm_dda), NAD27
geographic. Produces:
  (a) geology_unit_code_nome_3338.tif  - integer LABEL codes + legend JSON
  (b) dist_to_fault_nome_3338.tif      - metres to nearest mapped fault
Faults = arc ARC_CODE in {4, 30, 60} (normal / uncertain / concealed).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

RAW = Path("data/raw/nome_mpm/geol/nmgeol_dd")
OUT = Path("data/derived/nome_placer/covariates_mpm")
TEMPLATE = Path(
    "data/derived/nome_placer/prospectivity_v1p5/"
    "nome_placer_prospectivity_v1p5_v3p1_3338.tif"
)
TARGET_CRS = "EPSG:3338"
FAULT_CODES = {4, 30, 60}  # normal (certain), uncertain, concealed


def main() -> None:
    with rasterio.open(TEMPLATE) as t:
        transform = t.transform
        width, height = t.width, t.height
        bounds = t.bounds
        crs = t.crs

    # ---- (a) geology unit-code raster -------------------------------------
    polys = gpd.read_file(RAW / "nm_ddp.shp").to_crs(TARGET_CRS)
    aoi_polys = polys.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    labels_in_aoi = sorted(aoi_polys["LABEL"].dropna().unique().tolist())
    print(f"polygons: {len(polys)} total, {len(aoi_polys)} intersect AOI; "
          f"{len(labels_in_aoi)} units in AOI: {labels_in_aoi}")

    # Stable code map: 0 reserved for nodata/unmapped. Code by sorted label.
    code_map = {lab: i + 1 for i, lab in enumerate(labels_in_aoi)}
    NODATA_U = 0
    shapes = [
        (geom, code_map[lab])
        for geom, lab in zip(polys.geometry, polys["LABEL"])
        if lab in code_map and geom is not None and not geom.is_empty
    ]
    unit = rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=NODATA_U, dtype="int32", all_touched=False,
    )
    unmapped = int((unit == NODATA_U).sum())
    print(f"unit raster: {unmapped} unmapped cells "
          f"({unmapped/(height*width)*100:.2f}%)")

    unit_path = OUT / "geology_unit_code_nome_3338.tif"
    with rasterio.open(
        unit_path, "w", driver="GTiff", dtype="int32", count=1,
        width=width, height=height, crs=crs, transform=transform,
        nodata=NODATA_U, compress="deflate", tiled=True,
    ) as o:
        o.write(unit.astype("int32"), 1)
        o.update_tags(source="USGS OFR 2009-1254 nm_ddp (SIM 3131 Seward Pen.)")

    legend = {str(code): lab for lab, code in code_map.items()}
    legend["0"] = "<unmapped/nodata>"
    legend_path = OUT / "geology_unit_code_legend.json"
    legend_path.write_text(json.dumps(legend, indent=2))
    print(f"wrote {unit_path} + {legend_path.name}")

    # ---- (b) distance-to-fault raster -------------------------------------
    arcs = gpd.read_file(RAW / "nm_dda.shp").to_crs(TARGET_CRS)
    faults = arcs[arcs["ARC_CODE"].isin(FAULT_CODES)].copy()
    n_aoi_faults = len(
        faults.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    )
    print(f"faults: {len(faults)} arcs in quad (codes {sorted(FAULT_CODES)}), "
          f"{n_aoi_faults} intersect AOI")

    if len(faults) == 0:
        print("NO fault linework available -> skipping dist_to_fault raster")
        return

    fault_shapes = [
        (g, 1) for g in faults.geometry if g is not None and not g.is_empty
    ]
    px = abs(transform.a)  # 25 m
    # Rasterize on a grid padded outward so a cell's nearest fault can lie
    # outside the AOI (faults in the quad but beyond the bbox still anchor
    # edge distances). Pad by PAD_M, EDT on the padded grid, then crop back.
    PAD_M = 8000.0
    pad = int(round(PAD_M / px))
    from rasterio.transform import Affine
    big_transform = Affine(
        transform.a, transform.b, transform.c - pad * px,
        transform.d, transform.e, transform.f + pad * px,
    )
    big_h, big_w = height + 2 * pad, width + 2 * pad
    fault_mask = rasterize(
        fault_shapes, out_shape=(big_h, big_w), transform=big_transform,
        fill=0, dtype="uint8", all_touched=True,
    )
    n_fault_cells_aoi = int(fault_mask[pad:pad + height, pad:pad + width].sum())
    print(f"fault cells (padded grid {big_h}x{big_w}, pad={pad}); "
          f"in-AOI fault cells={n_fault_cells_aoi}")

    # distance to nearest fault cell; faults are 0, others measured.
    dist_big = distance_transform_edt(fault_mask == 0, sampling=px)
    dist = dist_big[pad:pad + height, pad:pad + width].astype("float32")
    print(f"dist_to_fault (m): min/mean/max = "
          f"{dist.min():.1f}/{dist.mean():.1f}/{dist.max():.1f}")

    dist_path = OUT / "dist_to_fault_nome_3338.tif"
    with rasterio.open(
        dist_path, "w", driver="GTiff", dtype="float32", count=1,
        width=width, height=height, crs=crs, transform=transform,
        nodata=np.float32(-1.0), compress="deflate", predictor=2, tiled=True,
    ) as o:
        o.write(dist, 1)
        o.update_tags(
            source="USGS OFR 2009-1254 nm_dda ARC_CODE in {4,30,60}",
            method="scipy.ndimage.distance_transform_edt, 25 m sampling",
        )
    print(f"wrote {dist_path}")


if __name__ == "__main__":
    main()
