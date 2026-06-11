"""Path 2 Stage 2A: add isostatic-residual-style gravity + fine-grained Alaska lithology to Eastak feature frame.

Two changes vs `features_eastak_500m_v3p1.parquet`:

1. **gravity_isostatic_hp**: high-pass-filtered Bouguer gravity computed by
   subtracting a Gaussian-smoothed (sigma = 10 cells = 5 km) regional
   field from the existing `gravity` column. This is a pragmatic
   substitute for a real USGS Alaska isostatic-residual grid. The
   result captures crustal-density variations on shorter wavelengths.

2. **major1/2/3_class equivalents**: factorized integer codes for
   `STATE_UNITNAME`, `NSACLASS`, `NSASUB` from Alaska geology. Parallel
   to Mother Lode's MAJOR1/2/3. One-hot encoded downstream by
   `add_lithology_onehot`.

Output: features_eastak_500m_v3p2.parquet
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point

DATA_RAW = Path("/home/sky/src/learning/ai-minerals/data/raw")
DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")

IN_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p1.parquet"
OUT_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p2.parquet"
GEOLOGY_PATH = DATA_RAW / "geology_ak" / "geology_eastak.gpkg"
WORKING_CRS = "EPSG:6393"


def main() -> None:
    print("=== Path 2 Stage 2A: Eastak feature-frame rebuild ===")
    df = pd.read_parquet(IN_PARQUET)
    print(f"input: {df.shape}")

    # 1. High-pass gravity feature.
    print("\n[1/2] computing high-pass gravity (substitute for isostatic residual)")
    n_rows = int(df["row"].max()) + 2
    n_cols = int(df["col"].max()) + 2
    grav_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grav_grid[df["row"].to_numpy(), df["col"].to_numpy()] = df["gravity"].to_numpy()

    # Fill NaN with mean for filter; mask later.
    valid_mask = ~np.isnan(grav_grid)
    fill_val = float(np.nanmean(grav_grid))
    grav_filled = np.where(valid_mask, grav_grid, fill_val)
    sigma_cells = 10.0  # 10 cells * 500m = 5km regional wavelength
    grav_smooth = gaussian_filter(grav_filled, sigma=sigma_cells)
    grav_hp = grav_filled - grav_smooth
    grav_hp = np.where(valid_mask, grav_hp, np.nan)

    df["gravity_isostatic_hp"] = grav_hp[df["row"].to_numpy(), df["col"].to_numpy()]
    finite = np.isfinite(df["gravity_isostatic_hp"])
    print(f"  gravity_isostatic_hp: median={df['gravity_isostatic_hp'][finite].median():.3f}, "
          f"range=[{df['gravity_isostatic_hp'][finite].min():.3f}, "
          f"{df['gravity_isostatic_hp'][finite].max():.3f}], "
          f"NaN frac={(~finite).mean()*100:.1f}%")

    # 2. Fine-grained Alaska lithology codes parallel to MAJOR1/2/3.
    print("\n[2/2] assigning Alaska fine-grained lithology codes")
    geo = gpd.read_file(GEOLOGY_PATH).to_crs(WORKING_CRS)
    print(f"  geology polygons: {len(geo):,}")

    grid_pts = gpd.GeoDataFrame(
        df[["row", "col", "x", "y"]].copy(),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=WORKING_CRS,
    )

    joined = gpd.sjoin(grid_pts, geo[["STATE_UNITNAME", "NSACLASS", "NSASUB", "geometry"]],
                       how="left", predicate="within")
    joined = joined.drop_duplicates(subset=["row", "col"])

    for src_col, target_col in [
        ("STATE_UNITNAME", "major1_class"),
        ("NSACLASS", "major2_class"),
        ("NSASUB", "major3_class"),
    ]:
        s = joined[src_col].astype("string").fillna("").str.strip().str.lower()
        codes_, _ = pd.factorize(s.where(s != "", other="(none)"))
        df[target_col] = pd.Series(
            codes_.astype("int64"), index=joined.index
        ).reindex(df.index, fill_value=-1)
        n_unique = (codes_ >= 0).sum() and len(np.unique(codes_[codes_ >= 0]))
        print(f"  {target_col} (from {src_col}): {n_unique} unique codes, "
              f"{(df[target_col] >= 0).sum():,} cells assigned")

    print(f"\nsaving to {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET)
    print(f"  shape: {df.shape}")
    print("done.")


if __name__ == "__main__":
    main()
