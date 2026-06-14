"""ADGGS Seward Peninsula surficial geology.

Coverage caveat for v1: the Cape Nome / Nome C-6 sub-quadrangle (where
Bear Cub MS 1178 and the Molasses / Honey claims sit) has no DGGS
digital surficial-geology publication. Nearest DGGS digital surficials
are AOF 125 (Tolstoi Point area, ~80 km east on the eastern Seward
Peninsula) and USGS OF 72-321 / OF 72-322 (Nome C-2 / C-3 quads,
~40 km east of the AOI). v1 fetches AOF 125 only; the region module
falls back to the Wilson et al. 2015 statewide map (sim3340, the same
source eastak uses) for the actual Cape Nome AOI coverage.

Data quality note on AOF 125: the published GeoPackage has an empty
``MapUnitPolys`` table; the main surficial map units appear to be
distributed only in the shapefile or AK GeMS variant of the same
publication. The fetcher falls back to ``GeologicPolys`` which on this
pub contains 425 ice-wedge polygon features only. The adapter pattern
still ships clean and works for any future publication that populates
MapUnitPolys. Since v1 region uses sim3340 for geology, this gap is
documented rather than blocking.

When the AOI extends east in a future iteration, or when a Nome C-6
surficial publication appears, this adapter is the integration point.
The pattern (GeMS GeoPackage -> canonical polygons via
``data/adapters/geology/adggs_surficial.py``) handles any DGGS GeMS pub
with minimal additions to the PUBLICATIONS dict.

Output: a canonical GeoPackage at
``data/raw/adggs/adggs_surficial_seward_peninsula.gpkg`` for the
region's ``raw_paths["surficial_seward"]`` entry. Per-publication
extracted folders remain under ``data/raw/adggs/<pub>/extracted/`` for
debug.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from ai_minerals.data._common import dataset_dir, write_source_md


NAME = "adggs"
MERGED_BASENAME = "adggs_surficial_seward_peninsula"

PUBLICATIONS = {
    "aof125": {
        "title": "AOF 125 -- Surficial geology of the Tolstoi Point - Cape Nome area",
        "landing": "https://dggs.alaska.gov/pubs/id/42",
        "geopackage_zip":
            "https://dggs.alaska.gov/webpubs/data/aof125_tolstoi_pt_cape_nome_surf_gems_geopackage.zip",
        "shapefile_zip":
            "https://dggs.alaska.gov/webpubs/data/aof125_tolstoi_pt_cape_nome_surf_gems_shapfile.zip",
    },
}


def _download(url: str, out_path: Path, timeout: int = 600) -> None:
    """Stream a ZIP to disk."""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def _extract(zip_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP, then recursively unzip any nested ``.zip`` it contained.

    DGGS publication bundles wrap the actual GeoPackage in an inner ZIP at
    ``data/<basename>_geopackage.zip`` alongside README + metadata folders;
    we unwrap both layers so the loader finds the .gpkg directly.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    # One level of nested-zip unwrap. DGGS bundles only ever nest once.
    for nested in list(extract_dir.rglob("*.zip")):
        target = nested.with_suffix("")
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(target)


def _find_gpkg(extracted_dir: Path) -> Path:
    """Locate the .gpkg file inside an extracted ADGGS publication folder."""
    candidates = list(extracted_dir.rglob("*.gpkg"))
    if not candidates:
        raise FileNotFoundError(
            f"No .gpkg found under {extracted_dir}; ADGGS pub may use a non-GeMS layout."
        )
    # Prefer the largest .gpkg (the data package; smaller ones are usually
    # symbology auxiliaries).
    return max(candidates, key=lambda p: p.stat().st_size)


_PREFERRED_LAYERS = ("MapUnitPolys", "GeologicPolys")


def _read_mapunit_polys(gpkg_path: Path) -> gpd.GeoDataFrame:
    """Read the surficial-polygons layer from a GeMS GeoPackage.

    GeMS publications can vary on which table holds the actual map-unit
    polygons. AOF 125 stores them in ``GeologicPolys`` and leaves
    ``MapUnitPolys`` empty (zero rows). Other pubs use MapUnitPolys as
    the primary. Try the preferred layers in order; prefer the first
    that returns rows. Fall back to the first non-empty polygon layer.
    """
    layers = gpd.list_layers(gpkg_path)["name"].tolist()
    for preferred in _PREFERRED_LAYERS:
        match = next((n for n in layers if n.lower() == preferred.lower()), None)
        if match is None:
            continue
        gdf = gpd.read_file(gpkg_path, layer=match)
        if len(gdf) > 0:
            return gdf
    # Last-ditch: first non-empty polygon layer.
    for n in layers:
        try:
            gdf = gpd.read_file(gpkg_path, layer=n)
        except Exception:
            continue
        if len(gdf) == 0:
            continue
        if gdf.geom_type.iloc[0] in ("Polygon", "MultiPolygon"):
            return gdf
    raise ValueError(
        f"No non-empty polygon layer found in {gpkg_path}; layers={layers}"
    )


def fetch(*, force: bool = False) -> Path:
    """Download AOF 125 + PDF 94-39, extract, merge into a single canonical GeoPackage.

    Returns the path of the merged GeoPackage. Per-pub extracted folders
    remain available under ``data/raw/adggs/<pub>/extracted/`` for debug.
    """
    out_dir = dataset_dir(NAME)
    merged_path = out_dir / f"{MERGED_BASENAME}.gpkg"

    if merged_path.exists() and not force:
        print(f"ADGGS merged surficial already present at {merged_path}; skipping.")
        return merged_path

    pub_paths: dict[str, Path] = {}
    for pub_key, pub in PUBLICATIONS.items():
        pub_dir = out_dir / pub_key
        pub_dir.mkdir(parents=True, exist_ok=True)
        zip_path = pub_dir / f"{pub_key}_geopackage.zip"
        extracted = pub_dir / "extracted"

        if extracted.exists() and any(extracted.iterdir()) and not force:
            print(f"ADGGS {pub_key}: already extracted at {extracted}.")
        else:
            url = pub["geopackage_zip"]
            print(f"ADGGS {pub_key}: downloading {url}")
            _download(url, zip_path)
            _extract(zip_path, extracted)
            print(f"  extracted to {extracted}")
        pub_paths[pub_key] = _find_gpkg(extracted)

    # Read MapUnitPolys from each and concatenate.
    parts: list[gpd.GeoDataFrame] = []
    for pub_key, gpkg in pub_paths.items():
        gdf = _read_mapunit_polys(gpkg)
        gdf["pub_key"] = pub_key
        gdf["pub_title"] = PUBLICATIONS[pub_key]["title"]
        parts.append(gdf)

    # Reproject everything to the first part's CRS before concatenating.
    base_crs = parts[0].crs
    parts_aligned = [p.to_crs(base_crs) if p.crs != base_crs else p for p in parts]
    merged = gpd.GeoDataFrame(
        pd.concat(parts_aligned, ignore_index=True, sort=False),
        crs=base_crs,
    )
    merged.to_file(merged_path, driver="GPKG")
    print(f"ADGGS: wrote merged {merged_path} ({len(merged):,} polygons; CRS={base_crs}).")

    pub_lines = "\n".join(
        f"- **{k}**: {v['title']} (landing: <{v['landing']}>)"
        for k, v in PUBLICATIONS.items()
    )
    write_source_md(
        NAME,
        title="ADGGS Seward Peninsula surficial geology",
        url=next(iter(PUBLICATIONS.values()))["landing"],
        license="Public domain (DGGS Alaska publications)",
        notes=(
            f"{pub_lines}\n\n"
            f"Merged canonical: `{MERGED_BASENAME}.gpkg` "
            f"({len(merged):,} polygons). GeMS schema; loader at "
            f"`data/adapters/geology/adggs_surficial.py` reads the "
            f"MapUnit or Symbol column and emits canonical polygons.\n\n"
            "Coverage caveat: AOF 125 covers the eastern Seward Peninsula "
            "(Tolstoi Point area), not the Cape Nome AOI itself. v1 region "
            "uses sim3340 statewide for the geology slot; ADGGS goes to "
            "`raw_paths[\"surficial_seward\"]` for v2 expansion."
        ),
    )
    return merged_path


if __name__ == "__main__":
    fetch()
