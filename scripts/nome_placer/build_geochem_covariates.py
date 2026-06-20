#!/usr/bin/env python
"""Interpolate AGDB4 stream-sediment pathfinder geochemistry onto the Nome MPM grid.

Builds log-space IDW surfaces for Au/As/Sb/Bi/Hg/W from stream-sediment samples
(the drainage-catchment-integrating medium). Placer/HMC and beach samples are
deliberately excluded: they sit on the known pay creeks and would be near-circular
with the ARDF placer labels. Below-detection values (stored as negatives in the
AGDB4 best-value tables) are substituted with |DL|/2 before the log transform.

Output: a 6-band float32 GeoTIFF aligned to the v3.1 prospectivity grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

CSV = "data/raw/nome_mpm/agdb4_nome.csv"
GRID = "data/derived/nome_placer/prospectivity_v1p5/nome_placer_prospectivity_v1p5_v3p1_3338.tif"
OUT = "data/derived/nome_placer/covariates_mpm/geochem_ss_idw_log_3338.tif"
ELEMENTS = ["Au_ppm", "As_ppm", "Sb_ppm", "Bi_ppm", "Hg_ppm", "W_ppm"]
STREAM_SED = {"stream", "stream sediment", "stream/river", "sediment"}
K = 12          # nearest neighbours for IDW
POWER = 2.0     # IDW distance power
NODATA = -9999.0


def clean_element(s: pd.Series) -> pd.Series:
    """AGDB4 best-value: negative = below-detection (negated DL). Use |DL|/2; log10."""
    v = pd.to_numeric(s, errors="coerce")
    v = v.mask(v < 0, v.abs() / 2.0)   # below-detection -> half the detection limit
    v = v.where(v > 0)                 # drop zeros / non-positive
    return np.log10(v)


def idw(sample_xy: np.ndarray, sample_val: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    tree = cKDTree(sample_xy)
    dist, idx = tree.query(grid_xy, k=min(K, len(sample_xy)))
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.power(np.maximum(dist, 1e-6), POWER)
    # exact hit: collapse weight onto the coincident sample
    exact = dist < 1e-6
    w = np.where(exact.any(axis=1, keepdims=True), np.where(exact, 1.0, 0.0), w)
    return (w * sample_val[idx]).sum(axis=1) / w.sum(axis=1)


def main() -> None:
    df = pd.read_csv(CSV, low_memory=False)
    ss = df[df["SAMPLE_SOURCE"].astype(str).str.strip().str.lower().isin(STREAM_SED)].copy()
    print(f"stream-sed samples: {len(ss)} of {len(df)}")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    sx, sy = tr.transform(ss["LONGITUDE"].values, ss["LATITUDE"].values)

    with rasterio.open(GRID) as g:
        H, W = g.height, g.width
        transform, crs = g.transform, g.crs
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    gx, gy = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    grid_xy = np.column_stack([gx, gy])
    sample_xy_all = np.column_stack([sx, sy])

    bands = np.full((len(ELEMENTS), H, W), NODATA, dtype="float32")
    descriptions = []
    for i, el in enumerate(ELEMENTS):
        logv = clean_element(ss[el])
        m = logv.notna().values
        n = int(m.sum())
        surf = idw(sample_xy_all[m], logv.values[m], grid_xy).reshape(H, W)
        bands[i] = surf.astype("float32")
        descriptions.append(f"log10_{el}_ss_idw")
        print(f"{el:8s} n={n:5d}  log10 surface min/med/max = "
              f"{surf.min():.3f}/{np.median(surf):.3f}/{surf.max():.3f}")

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with rasterio.open(GRID) as g:
        profile = g.profile
    profile.update(count=len(ELEMENTS), dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(bands)
        for i, d in enumerate(descriptions, 1):
            dst.set_band_description(i, d)
    print(f"wrote {OUT}  ({len(ELEMENTS)} bands, {H}x{W}, {crs})")

    # sanity: interpolated Au at ARDF placer points should beat the AOI background
    import geopandas as gpd
    ardf = gpd.read_file("data/raw/nome_mpm/ardf_nome.geojson").to_crs("EPSG:3338")
    au = bands[0]
    # classify placer via model_code/site_type if available
    txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str)).str.lower()
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    placer = ardf[mc.str.startswith("39") | txt.str.contains("placer")]
    vals = []
    at = rasterio.transform.AffineTransformer(transform)
    for geom in placer.geometry:
        r, c = at.rowcol(geom.x, geom.y)
        if 0 <= r < H and 0 <= c < W:
            vals.append(au[r, c])
    vals = np.array(vals)
    print(f"SANITY Au log10: AOI grid median={np.median(au):.3f}  "
          f"placer-pts median={np.median(vals):.3f} (n={len(vals)})  "
          f"-> placer {'ABOVE' if np.median(vals) > np.median(au) else 'NOT above'} background")


if __name__ == "__main__":
    main()
