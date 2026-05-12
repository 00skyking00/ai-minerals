"""Path 5 Stage A: DEEP-SEAM-style feature decomposition for Tanacross.

Two transformations applied to the v3.2 feature frame:

1. **ILR-PCA on geochem** — replicates DEEP-SEAM's `ilrrpca` step.
   Take the NGDB pathfinder elements (Cu, Mo, Pb, Zn, Au, Ag, As, Sb,
   W, Bi, Te), apply Aitchison's isometric-log-ratio transformation to
   convert compositional data into Euclidean coordinates, then PCA to
   reduce to 7 principal components (DEEP-SEAM's pc7 choice).
   Geometric-mean replacement for zeros / NaN. Implementation does not
   depend on `composition_stats`; ILR is computed from the closed-form
   orthonormal basis directly.

2. **GLCM textural features on geophysics** — replicates DEEP-SEAM's
   `_dissimilarity / _correlation / _mean / _std` features. For each
   geophysical raster (magnetic, magnetic_1vd, magnetic_hgm,
   magnetic_analytic_signal, magnetic_tilt, gravity, gravity_isostatic_hp),
   bin to 16 gray levels, compute Gray-Level Co-occurrence Matrix in a
   5x5 sliding window, extract four standard texture metrics per
   pixel: dissimilarity, correlation, mean, std.

Output:
  data/derived/features_eastak_500m_v3p3.parquet

The new feature frame replaces the 11 raw NGDB columns with 7 ILR-PCA
components, and adds 4 texture metrics x 7 geophysical layers = 28 new
texture columns. Net: keeps non-geochem non-geophysics features the
same, replaces geochem with PCA, adds GLCM textures.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops

DATA_RAW = Path("/home/sky/src/learning/ai-minerals/data/raw")
DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")

IN_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p2.parquet"
OUT_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p3.parquet"

PATHFINDER_ELEMENTS = [
    "Cu", "Mo", "Pb", "Zn", "Au", "Ag", "As", "Sb", "W", "Bi", "Te",
]
GEOCHEM_AGGS = ["mean", "max"]
GEOPHYS_RASTERS = [
    ("magnetic", DATA_RAW / "gsc_geophysics" / "magnetic_eastak.tif"),
    ("magnetic_1vd", DATA_RAW / "gsc_geophysics" / "magnetic_1vd_eastak.tif"),
    ("magnetic_hgm", DATA_RAW / "gsc_geophysics" / "magnetic_hgm_eastak.tif"),
    ("magnetic_analytic_signal", DATA_RAW / "gsc_geophysics" / "magnetic_analytic_signal_eastak.tif"),
    ("magnetic_tilt", DATA_RAW / "gsc_geophysics" / "magnetic_tilt_eastak.tif"),
    ("gravity", DATA_RAW / "geophysics" / "gravity_eastak.tif"),
]
TEXTURE_PROPS = ["dissimilarity", "correlation", "contrast", "homogeneity"]
GLCM_LEVELS = 16
GLCM_WINDOW = 5  # 5x5 window
N_PCA_COMPONENTS = 7


def ilr_transform(X: np.ndarray) -> np.ndarray:
    """Aitchison's isometric-log-ratio. X assumed strictly positive,
    rows are compositions. Returns (n, D-1) array."""
    n, D = X.shape
    assert (X > 0).all(), "ILR requires strictly positive compositions"
    log_X = np.log(X)
    out = np.empty((n, D - 1), dtype=np.float64)
    for i in range(D - 1):
        # ILR coordinate i = sqrt((i+1) * (D-i-1) / D) * log(geo_mean(x_1..x_{i+1}) / geo_mean(x_{i+2}..x_D))
        # Using the orthonormal Egozcue basis.
        left_logmean = log_X[:, : i + 1].mean(axis=1)
        right_logmean = log_X[:, i + 1 :].mean(axis=1)
        scale = np.sqrt((i + 1) * (D - i - 1) / D)
        out[:, i] = scale * (left_logmean - right_logmean)
    return out


def compute_glcm_per_pixel(arr: np.ndarray, levels: int = 16, win: int = 5) -> dict[str, np.ndarray]:
    """Sliding-window GLCM texture metrics. Returns dict of texture_metric -> (H, W) array.

    Quantizes input to `levels` bins after winsorizing (5th-95th percentile),
    then computes GLCM per window with offset (1, 0) (horizontal neighbor).
    """
    finite = np.isfinite(arr)
    if not finite.any():
        out = {p: np.full(arr.shape, np.nan, dtype=np.float32) for p in TEXTURE_PROPS}
        return out
    lo, hi = np.percentile(arr[finite], [5, 95])
    span = max(hi - lo, 1e-9)
    quantized = np.clip((arr - lo) / span, 0, 1)
    quantized = (quantized * (levels - 1)).astype(np.uint8)
    # Replace non-finite with bin 0 (texture in NaN regions is meaningless; will be masked out later)
    quantized[~finite] = 0

    H, W = arr.shape
    half = win // 2
    out = {p: np.full((H, W), np.nan, dtype=np.float32) for p in TEXTURE_PROPS}

    # Iterate; for 270k cells with 5x5 window this is the slow part.
    # Vectorize as much as possible by batching graycomatrix calls.
    for r in range(half, H - half):
        for c in range(half, W - half):
            patch = quantized[r - half : r + half + 1, c - half : c + half + 1]
            glcm = graycomatrix(
                patch, distances=[1], angles=[0], levels=levels,
                symmetric=True, normed=True,
            )
            for p in TEXTURE_PROPS:
                out[p][r, c] = float(graycoprops(glcm, p)[0, 0])
    return out


def main() -> None:
    print("=== Path 5 Stage A: Tanacross v3.3 decomposed features ===")
    df = pd.read_parquet(IN_PARQUET)
    print(f"input: {df.shape}")

    # 1. ILR-PCA on geochem.
    print("\n[1/2] ILR-PCA on geochem")
    geochem_cols = []
    for el in PATHFINDER_ELEMENTS:
        for agg in GEOCHEM_AGGS:
            col = f"{el.lower()}_{agg}_5km"
            if col in df.columns:
                geochem_cols.append(col)
    print(f"  {len(geochem_cols)} geochem columns: {geochem_cols[:6]}{'...' if len(geochem_cols) > 6 else ''}")

    if not geochem_cols:
        print("  no geochem columns found; skipping ILR-PCA")
        pca_components = None
    else:
        gc = df[geochem_cols].to_numpy()
        # Replace NaN with column median, then ensure strictly positive.
        col_median = np.nanmedian(gc, axis=0)
        gc = np.where(np.isnan(gc), col_median, gc)
        # Detection-limit replacement: any value <= 0 becomes 0.5x the smallest positive in that column.
        for j in range(gc.shape[1]):
            col = gc[:, j]
            pos_min = col[col > 0].min() if (col > 0).any() else 1e-6
            col[col <= 0] = pos_min * 0.5
            gc[:, j] = col
        ilr = ilr_transform(gc)
        print(f"  ILR shape: {ilr.shape}, range [{ilr.min():.3f}, {ilr.max():.3f}]")

        pca = PCA(n_components=N_PCA_COMPONENTS)
        pca_components = pca.fit_transform(ilr)
        print(f"  PCA components: {pca_components.shape}, "
              f"explained variance ratio: {pca.explained_variance_ratio_.round(3).tolist()}")
        print(f"  cumulative: {pca.explained_variance_ratio_.cumsum().round(3).tolist()}")

    # 2. GLCM textural features on geophysics rasters.
    print("\n[2/2] GLCM textural features on geophysics rasters")
    n_rows = int(df["row"].max()) + 2
    n_cols = int(df["col"].max()) + 2

    texture_features = {}
    t0 = time.time()
    for layer_name, raster_path in GEOPHYS_RASTERS:
        if not raster_path.exists():
            print(f"  {layer_name}: raster missing, skipping")
            continue
        # Reconstruct a 2D grid of this layer's values from the feature frame.
        # The feature frame's column has the value at (row, col).
        col_in_df = layer_name
        if col_in_df not in df.columns:
            print(f"  {layer_name}: column not in feature frame, skipping")
            continue
        grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        grid[df["row"].to_numpy(), df["col"].to_numpy()] = df[col_in_df].to_numpy()
        print(f"  {layer_name}: {grid.shape}, computing GLCM...", flush=True)
        layer_t0 = time.time()
        textures = compute_glcm_per_pixel(grid, levels=GLCM_LEVELS, win=GLCM_WINDOW)
        for prop, tex_grid in textures.items():
            col_name = f"glcm_{layer_name}_{prop}"
            texture_features[col_name] = tex_grid[df["row"].to_numpy(), df["col"].to_numpy()]
        print(f"  {layer_name}: done in {time.time()-layer_t0:.0f}s")

    elapsed = time.time() - t0
    print(f"\nGLCM total: {elapsed/60:.1f} min")
    print(f"new texture columns: {len(texture_features)}")

    # 3. Build v3.3 feature frame: drop raw geochem, add PCA components, add textures.
    print("\nbuilding v3.3 feature frame")
    out = df.copy()
    if pca_components is not None:
        for i in range(N_PCA_COMPONENTS):
            out[f"geochem_pca{i+1}"] = pca_components[:, i]
        # Drop raw geochem columns (replaced by PCA components).
        out = out.drop(columns=geochem_cols)
        print(f"  dropped {len(geochem_cols)} raw geochem columns, added {N_PCA_COMPONENTS} PCA columns")
    for col_name, vals in texture_features.items():
        out[col_name] = vals
    print(f"  added {len(texture_features)} GLCM texture columns")
    print(f"  output shape: {out.shape}")

    out.to_parquet(OUT_PARQUET)
    print(f"\nsaved {OUT_PARQUET}")


if __name__ == "__main__":
    main()
