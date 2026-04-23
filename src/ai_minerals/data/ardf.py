"""USGS Alaska Resource Data File (ARDF).

There's no per-quadrangle download endpoint, so we grab the whole dataset
(~2.5 MB) and filter locally. Tanacross quadrangle code is 'TC'.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "ardf"
ZIP_URL = "https://mrdata.usgs.gov/ardf/ardf.zip"
LANDING_URL = "https://mrdata.usgs.gov/ardf/"


def fetch(*, force: bool = False) -> Path:
    """Download and extract the full ARDF shapefile bundle."""
    out_dir = dataset_dir(NAME)
    zip_path = out_dir / "ardf.zip"

    if zip_path.exists() and not force:
        print(f"ARDF zip already present at {zip_path} ({zip_path.stat().st_size:,} B); skipping.")
    else:
        print(f"Downloading ARDF from {ZIP_URL}")
        with requests.get(ZIP_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"Wrote {zip_path} ({zip_path.stat().st_size:,} bytes)")

    extract_dir = out_dir / "ardf"
    if not extract_dir.exists() or force:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    write_source_md(
        NAME,
        title="USGS Alaska Resource Data File (ARDF) — full shapefile",
        url=LANDING_URL,
        license="US public domain (USGS)",
        notes=(
            "Full ARDF shapefile bundle. Filter to a specific 1:250k quadrangle "
            "(e.g. 'TC' for Tanacross) using the 'quad' attribute in the table."
        ),
    )
    return zip_path


def load_quadrangle(
    quad: str = "TC",
    *,
    aoi: "AOI | None" = None,  # type: ignore[name-defined]  # noqa: F821
) -> "geopandas.GeoDataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Return the ARDF records for a given 1:250k quadrangle code.

    If `aoi` is given, also filter records to those whose point geometry lies
    within the AOI polygon. ARDF's `quad_250` field is an editorial label, not
    a strict geographic test: ~half of TC-labeled records actually have
    coordinates in adjacent quadrangles (Big Delta, Mt Hayes, etc.). Combining
    both filters gives clean positives with valid feature coverage.
    """
    import geopandas as gpd

    extract_dir = dataset_dir(NAME) / "ardf"
    shp_paths = list(extract_dir.rglob("*.shp"))
    if not shp_paths:
        raise FileNotFoundError(f"No .shp in {extract_dir}")
    primary = next(
        (p for p in shp_paths if "ardf" in p.stem.lower()), shp_paths[0]
    )
    print(f"Loading {primary}")
    gdf = gpd.read_file(primary)

    quad_col = "quad_250"
    if quad_col not in gdf.columns:
        raise RuntimeError(f"Expected '{quad_col}' in ARDF schema: {list(gdf.columns)}")

    # Filter by quadrangle label (two-letter code or full name).
    normalized = gdf[quad_col].astype(str).str.strip().str.upper()
    target_code = quad.upper()
    target_name = {"TC": "TANACROSS", "NM": "NOME"}.get(target_code, target_code)
    by_label = gdf[normalized.isin({target_code, target_name})].copy()
    print(f"ARDF: {len(by_label):,} records labeled {quad!r} (of {len(gdf):,} total).")

    if aoi is None:
        return by_label

    aoi_series = gpd.GeoSeries([aoi.polygon], crs=aoi.crs)
    if by_label.crs != aoi_series.crs:
        aoi_series = aoi_series.to_crs(by_label.crs)
    within = by_label[by_label.within(aoi_series.iloc[0])].copy()
    dropped = len(by_label) - len(within)
    print(
        f"ARDF: after AOI filter, {len(within):,} records within {aoi.name} "
        f"(dropped {dropped} labeled {quad!r} but with coords outside the bbox)."
    )
    return within


if __name__ == "__main__":
    from ai_minerals.aoi import TANACROSS

    fetch()
    sub = load_quadrangle("TC", aoi=TANACROSS)
    out_path = dataset_dir(NAME) / "ardf_tc.gpkg"
    sub.to_file(out_path, driver="GPKG")
    print(f"Wrote {out_path}")
