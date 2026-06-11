"""Path 3 Stage C: AZ DEEP-SEAM-style feature decomposition.

Mirrors `eastak_path5_decomposed_features.py` for Arizona:
  - ILR-PCA on the NGDB pathfinder elements (Cu, Mo, Pb, Zn, Au, Ag,
    As, Sb, W, Bi, Te) to 7 principal components.
  - GLCM textural features (mean, std, dissimilarity, correlation) on
    each geophysics raster (magnetic, 4 derivatives, Bouguer + isostatic
    gravity), 5x5 sliding window.

Output: data/derived/features_arizona_500m_decomposed.parquet
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

IN_PARQUET = DATA_DERIVED / "features_arizona_500m.parquet"
OUT_PARQUET = DATA_DERIVED / "features_arizona_500m_decomposed.parquet"

PATHFINDER_ELEMENTS = [
    "Cu", "Mo", "Pb", "Zn", "Au", "Ag", "As", "Sb", "W", "Bi", "Te",
]
GEOCHEM_AGGS = ["mean", "max"]
GEOPHYS_LAYERS = [
    "magnetic", "magnetic_1vd", "magnetic_hgm",
    "magnetic_analytic_signal", "magnetic_tilt",
    "gravity", "gravity_isostatic",
]
TEXTURE_PROPS = ["dissimilarity", "correlation", "contrast", "homogeneity"]
GLCM_LEVELS = 16
GLCM_WINDOW = 5
N_PCA_COMPONENTS = 7


def ilr_transform(X: np.ndarray) -> np.ndarray:
    n, D = X.shape
    assert (X > 0).all(), "ILR requires strictly positive compositions"
    log_X = np.log(X)
    out = np.empty((n, D - 1), dtype=np.float64)
    for i in range(D - 1):
        left = log_X[:, : i + 1].mean(axis=1)
        right = log_X[:, i + 1 :].mean(axis=1)
        scale = np.sqrt((i + 1) * (D - i - 1) / D)
        out[:, i] = scale * (left - right)
    return out


def compute_glcm_per_pixel(arr: np.ndarray, levels: int = 16, win: int = 5) -> dict[str, np.ndarray]:
    finite = np.isfinite(arr)
    if not finite.any():
        return {p: np.full(arr.shape, np.nan, dtype=np.float32) for p in TEXTURE_PROPS}
    lo, hi = np.percentile(arr[finite], [5, 95])
    span = max(hi - lo, 1e-9)
    quantized = np.clip((arr - lo) / span, 0, 1)
    quantized = (quantized * (levels - 1)).astype(np.uint8)
    quantized[~finite] = 0
    H, W = arr.shape
    half = win // 2
    out = {p: np.full((H, W), np.nan, dtype=np.float32) for p in TEXTURE_PROPS}
    for r in range(half, H - half):
        for c in range(half, W - half):
            patch = quantized[r - half : r + half + 1, c - half : c + half + 1]
            glcm = graycomatrix(patch, distances=[1], angles=[0], levels=levels,
                                symmetric=True, normed=True)
            for p in TEXTURE_PROPS:
                out[p][r, c] = float(graycoprops(glcm, p)[0, 0])
    return out


def main() -> None:
    print("=== Path 3 Stage C: Arizona DEEP-SEAM-style decomposition ===")
    df = pd.read_parquet(IN_PARQUET)
    print(f"input: {df.shape}")

    # 1. ILR-PCA on geochem.
    print("\n[1/2] ILR-PCA on geochem")
    geochem_cols = [
        f"{el.lower()}_{agg}_5km" for el in PATHFINDER_ELEMENTS for agg in GEOCHEM_AGGS
        if f"{el.lower()}_{agg}_5km" in df.columns
    ]
    print(f"  {len(geochem_cols)} geochem columns")
    pca_components = None
    if geochem_cols:
        gc = df[geochem_cols].to_numpy()
        col_median = np.nanmedian(gc, axis=0)
        gc = np.where(np.isnan(gc), col_median, gc)
        for j in range(gc.shape[1]):
            col = gc[:, j]
            pos_min = col[col > 0].min() if (col > 0).any() else 1e-6
            col[col <= 0] = pos_min * 0.5
            gc[:, j] = col
        ilr = ilr_transform(gc)
        pca = PCA(n_components=min(N_PCA_COMPONENTS, ilr.shape[1]))
        pca_components = pca.fit_transform(ilr)
        print(f"  ILR shape: {ilr.shape}, PCA components: {pca_components.shape}")
        print(f"  explained variance ratio: {pca.explained_variance_ratio_.round(3).tolist()}")
        print(f"  cumulative: {pca.explained_variance_ratio_.cumsum().round(3).tolist()}")

    # 2. GLCM textural features.
    print("\n[2/2] GLCM textural features")
    n_rows = int(df["row"].max()) + 2
    n_cols = int(df["col"].max()) + 2
    texture_features = {}
    t0 = time.time()
    for layer_name in GEOPHYS_LAYERS:
        if layer_name not in df.columns:
            print(f"  {layer_name}: column missing in feature frame, skipping")
            continue
        grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        grid[df["row"].to_numpy(), df["col"].to_numpy()] = df[layer_name].to_numpy()
        layer_t0 = time.time()
        textures = compute_glcm_per_pixel(grid, levels=GLCM_LEVELS, win=GLCM_WINDOW)
        for prop, tex_grid in textures.items():
            col_name = f"glcm_{layer_name}_{prop}"
            texture_features[col_name] = tex_grid[df["row"].to_numpy(), df["col"].to_numpy()]
        print(f"  {layer_name}: done in {time.time()-layer_t0:.0f}s")
    elapsed = time.time() - t0
    print(f"\nGLCM total: {elapsed/60:.1f} min")

    # Build decomposed feature frame.
    out = df.copy()
    if pca_components is not None:
        for i in range(pca_components.shape[1]):
            out[f"geochem_pca{i+1}"] = pca_components[:, i]
        out = out.drop(columns=geochem_cols)
        print(f"  dropped {len(geochem_cols)} raw geochem columns, added {pca_components.shape[1]} PCA columns")
    for col_name, vals in texture_features.items():
        out[col_name] = vals
    print(f"  added {len(texture_features)} GLCM texture columns")
    print(f"  output shape: {out.shape}")

    out.to_parquet(OUT_PARQUET)
    print(f"\nsaved {OUT_PARQUET}")


if __name__ == "__main__":
    main()
