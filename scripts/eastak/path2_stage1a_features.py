"""Path 2 Stage 1A: rebuild Eastak feature frame with v3.1 magnetic derivatives + Cox-Singer-style porphyry label cleanup.

Two changes vs the existing `features_eastak_500m.parquet`:

1. **Magnetic derivatives**: compute HGM, analytic signal, and tilt from
   the existing `magnetic_eastak.tif` via `data/magnetic_derivatives.py`,
   sample onto the Eastak grid, add as new columns. The 1VD column is
   already present and stays as-is.

2. **Cox-Singer-style porphyry-Cu label**: build a new column
   `is_porphyry_clean` from ARDF records where dep_model contains
   "porphyry", comm_main contains Cu or copper, and dep_model does
   NOT contain "polymetallic" (composite-deposit confounders). Keeps the
   pure porphyry-Cu / porphyry-Cu-Mo records and excludes ambiguous
   polymetallic + porphyry composites.

Output: `data/derived/features_eastak_500m_v3p1.parquet` (does not
overwrite the existing v1-era frame).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from ai_minerals.data.magnetic_derivatives import write_derivatives

DATA_RAW = Path("/home/sky/src/learning/ai-minerals/data/raw")
DATA_DERIVED = Path("/home/sky/src/learning/ai-minerals/data/derived")

MAG_RASTER = DATA_RAW / "gsc_geophysics" / "magnetic_eastak.tif"
DERIV_DIR = DATA_RAW / "gsc_geophysics"
ARDF_PATH = DATA_RAW / "ardf" / "ardf_eastak.gpkg"
IN_PARQUET = DATA_DERIVED / "features_eastak_500m.parquet"
OUT_PARQUET = DATA_DERIVED / "features_eastak_500m_v3p1.parquet"


def sample_raster_at_xy(raster_path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a raster at given (x, y) coordinates in the raster's CRS."""
    with rasterio.open(raster_path) as src:
        coords = list(zip(x.tolist(), y.tolist()))
        return np.array([v[0] for v in src.sample(coords)], dtype=np.float32)


def main() -> None:
    print("=== Path 2 Stage 1A: Eastak feature-frame rebuild ===")
    df = pd.read_parquet(IN_PARQUET)
    print(f"input feature frame: {df.shape}")
    print(f"  is_porphyry positives: {(df['is_porphyry'] == 1).sum()}")
    print(f"  is_porphyry_strict:    {(df['is_porphyry_strict'] == 1).sum()}")

    # 1. Generate magnetic derivatives (HGM, AS, Tilt). 1VD already exists; the
    #    function will rewrite it but the value is the same.
    print("\n[1/3] generating magnetic derivatives from magnetic_eastak.tif")
    deriv_paths = write_derivatives(MAG_RASTER, DERIV_DIR)
    for k, p in deriv_paths.items():
        print(f"  {k}: {p.name}")

    # 2. Sample the new derivatives onto the existing grid (use feature-frame x/y).
    print("\n[2/3] sampling derivatives onto Eastak grid")
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    for key in ("magnetic_hgm", "magnetic_analytic_signal", "magnetic_tilt"):
        # Source raster names use the suffix without "magnetic_" prefix repeated
        # write_derivatives outputs "magnetic_<key>_<region>.tif" or similar;
        # use deriv_paths directly.
        raster_path = deriv_paths.get(key)
        if raster_path is None or not raster_path.exists():
            raise RuntimeError(f"derivative raster missing: {key}")
        vals = sample_raster_at_xy(raster_path, x, y)
        df[key] = vals
        print(f"  {key}: median={np.nanmedian(vals):.3f}, "
              f"range=[{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}], "
              f"NaN frac={np.isnan(vals).mean()*100:.1f}%")

    # 3. Cox-Singer-style porphyry-Cu label cleanup.
    print("\n[3/3] Cox-Singer-style porphyry-Cu label cleanup")
    ardf = gpd.read_file(ARDF_PATH)
    print(f"  ARDF records: {len(ardf):,}")

    dm = ardf["dep_model"].fillna("").str.lower()
    cm = ardf["comm_main"].fillna("").str.lower() + " " + ardf["comm_other"].fillna("").str.lower()

    has_porphyry = dm.str.contains("porphyry", regex=True)
    has_cu = cm.str.contains(r"\bcu\b|copper", regex=True)
    is_polymetallic_composite = dm.str.contains("polymetallic", regex=True)

    porph_mask = has_porphyry & has_cu & ~is_polymetallic_composite
    clean_records = ardf[porph_mask].copy()
    print(f"  porphyry-Cu records (cleaned): {len(clean_records):,}")
    print(f"  excluded composites (polymetallic+porphyry): {(has_porphyry & has_cu & is_polymetallic_composite).sum()}")

    # Map cleaned records to grid cells. Reproject to working CRS first.
    if clean_records.crs is None:
        clean_records = clean_records.set_crs("EPSG:4326")
    # Working CRS for Eastak: EPSG:6393 (Alaska Albers)
    target_crs = "EPSG:6393"
    clean_records = clean_records.to_crs(target_crs)

    # Use KDTree nearest-cell lookup against the feature-frame (x, y) grid.
    # More robust than computing row/col from x/y because it sidesteps any
    # off-by-one or half-cell offset between the raster and the parquet.
    from scipy.spatial import cKDTree
    grid_xy = df[["x", "y"]].to_numpy()
    tree = cKDTree(grid_xy)

    rec_xy = []
    valid_recs = []
    for _, rec in clean_records.iterrows():
        if rec.geometry is None or rec.geometry.is_empty:
            continue
        rec_xy.append((rec.geometry.x, rec.geometry.y))
        valid_recs.append(rec)

    if rec_xy:
        rec_xy = np.array(rec_xy)
        distances, idx = tree.query(rec_xy)
        # Only accept matches within one cell-width (500 m) of the record.
        ok = distances <= 500.0
        df["is_porphyry_clean"] = np.zeros(len(df), dtype=np.uint8)
        df.loc[df.index[idx[ok]], "is_porphyry_clean"] = 1
        n_assigned = int(ok.sum())
        n_outside = int((~ok).sum())
        max_dist = float(distances.max())
    else:
        df["is_porphyry_clean"] = np.zeros(len(df), dtype=np.uint8)
        n_assigned = 0
        n_outside = 0
        max_dist = 0.0

    print(f"  records assigned to grid cells: {n_assigned}")
    print(f"  records outside grid (>500 m to nearest cell): {n_outside}")
    print(f"  max nearest-cell distance: {max_dist:.0f} m")
    n_pos = int((df["is_porphyry_clean"] == 1).sum())
    print(f"  is_porphyry_clean positive cells: {n_pos}")

    # Diagnostic: how does this compare to the existing is_porphyry?
    overlap = ((df["is_porphyry"] == 1) & (df["is_porphyry_clean"] == 1)).sum()
    only_existing = ((df["is_porphyry"] == 1) & (df["is_porphyry_clean"] == 0)).sum()
    only_clean = ((df["is_porphyry"] == 0) & (df["is_porphyry_clean"] == 1)).sum()
    print(f"\n  label set comparison:")
    print(f"    is_porphyry only (will be excluded by cleanup): {only_existing}")
    print(f"    is_porphyry_clean only (newly added):           {only_clean}")
    print(f"    both labels agree:                              {overlap}")

    # Save.
    print(f"\nsaving to {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET)
    print(f"  shape: {df.shape}")
    print("done.")


if __name__ == "__main__":
    main()
