"""Alaska IfSAR-derived elevation via TNM 1/3 arc-second.

USGS National Map distributes Alaska elevation under several products. The
2010-onwards Interferometric Synthetic Aperture Radar (IfSAR) collections
feed the 1 and 1/3 arc-second seamless layers where coverage is flown; the
rest of Alaska falls back to the 2 arc-second (~60 m) seamless layer.

At Nome (64.5 deg N), 1/3 arc-second is ~10 m in latitude and ~4.5 m in
longitude. The Seward Peninsula has IfSAR coverage; the 1/3 arc-second
tile `n65w166` covering the Cape Nome AOI (lat 64.45-64.62, lon -165.50
to -165.20) is the IfSAR-derived 10 m product.

This fetcher reuses the TNM machinery already in `data/threedep.py`. Tile
URLs follow the pattern
`https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n{lat:02d}w{lon:03d}/USGS_13_n{lat:02d}w{lon:03d}.tif`.
Same path the CONUS 10 m fetcher uses; only the AOI selects Alaska tiles.

The 5 m IfSAR DTM is published at the per-project, per-cell level under
`OPR/Projects/AK_IFSAR_*` and `OPR/Projects/AK_TyphoonMerbok_*` (the 2022
Norton Sound storm recollect, which covers Nome). That higher resolution
is a future enhancement; this fetcher targets the 1/3 arc-second seamless
mosaic which is enough for the v1 25 m model grid.

Output: a single mosaicked GeoTIFF at
`data/raw/ifsar_alaska/ifsar_5m_nome_placer.tif` (named for the eventual
5 m upgrade target; v1 content is the 10 m IfSAR-derived seamless layer)
plus a sidecar `*_meta.json` recording the actual resolution and the
tile manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_minerals.aoi import AOI, NOME_PLACER
from ai_minerals.data._common import dataset_dir, write_source_md
from ai_minerals.data.threedep import (
    _fetch_tnm_tile,
    _mosaic,
    _resolution_meters,
    _tnm_tile_names,
)


NAME = "ifsar_alaska"
OUT_BASENAME = "ifsar_5m_nome_placer"

TNM_BASE = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
)


def fetch(aoi: AOI = NOME_PLACER, *, force: bool = False) -> Path:
    """Download and mosaic the Alaska IfSAR-derived 1/3 arc-second tiles for `aoi`.

    Returns the path of the mosaicked GeoTIFF. Writes a sidecar
    `*_meta.json` with the actual resolution, tile list, and any
    coverage gaps.
    """
    out_dir = dataset_dir(NAME)
    tile_dir = out_dir / "_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUT_BASENAME}.tif"

    if out_path.exists() and not force:
        print(f"IfSAR Alaska mosaic already present at {out_path}; skipping.")
        return out_path

    tiles = _tnm_tile_names(aoi)
    print(f"IfSAR Alaska: {len(tiles)} TNM tile(s) cover the AOI: {tiles}")

    tile_paths: list[Path] = []
    gaps: list[str] = []
    for tile in tiles:
        tile_path = tile_dir / f"USGS_13_{tile}.tif"
        if _fetch_tnm_tile(tile, tile_path):
            tile_paths.append(tile_path)
        else:
            gaps.append(tile)

    if not tile_paths:
        raise RuntimeError(
            f"IfSAR Alaska: no tiles fetched for AOI {aoi.name}; gaps={gaps}. "
            f"Confirm TNM 13/TIFF/current coverage at {TNM_BASE}."
        )

    _mosaic(tile_paths, out_path)

    res_m = _resolution_meters(out_path)
    meta = {
        "name": NAME,
        "out_basename": OUT_BASENAME,
        "aoi": aoi.name,
        "aoi_bbox": list(aoi.bbox),
        "source_tier": "TNM 1/3 arc-second seamless (Alaska IfSAR-derived)",
        "source_url_pattern": f"{TNM_BASE}/{{tile}}/USGS_13_{{tile}}.tif",
        "tiles_fetched": [p.name for p in tile_paths],
        "tiles_gap": gaps,
        "actual_resolution_m": res_m,
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    write_source_md(NAME, lines=[
        f"# {NAME} ({OUT_BASENAME})",
        "",
        f"USGS National Map 1/3 arc-second seamless Alaska elevation, IfSAR-derived where flown.",
        f"AOI: {aoi.name} = {tuple(aoi.bbox)}",
        f"Resolution: ~{res_m} m",
        f"Tiles fetched: {[p.name for p in tile_paths]}",
        f"Gaps: {gaps}",
        "",
        f"5 m IfSAR DTM (per-project at OPR/Projects/AK_IFSAR_*) is a future upgrade path; v1 uses the 1/3 arc-second mosaic.",
    ])
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} B); resolution ~{res_m} m.")
    return out_path


if __name__ == "__main__":
    fetch()
