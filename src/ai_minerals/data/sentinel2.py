"""Sentinel-2 L2A composite via Microsoft Planetary Computer STAC.

Computing a true median-over-time composite for a 1°×2° AOI across many
scenes at 10-60 m resolution is memory-heavy — stackstac has to materialize
every scene's pixel stack per tile before it can reduce. For a regional
portfolio demo that's overkill; a small "best-N scenes, mean" composite is
good enough and far lighter.

Strategy:
  1. Search for all low-cloud scenes in the snow-free window.
  2. Pick the N lowest-cloud scenes.
  3. Take the mean (skipping NaN) — cheaper than median under dask.
  4. Stream to COG with dask.

Resolution default is 60 m — coarse enough to fit in memory, fine enough for
regional alteration indices.
"""

from __future__ import annotations

from pathlib import Path

import planetary_computer
import pystac_client
import stackstac

from ai_minerals.aoi import AOI, WORKING_CRS
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "sentinel2"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_DATETIME = "2024-07-10/2024-08-25"
DEFAULT_MAX_CLOUD = 10
DEFAULT_SCENE_LIMIT = 6

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def fetch(
    aoi: AOI,
    *,
    datetime: str = DEFAULT_DATETIME,
    max_cloud: int = DEFAULT_MAX_CLOUD,
    resolution: int = 60,
    scene_limit: int = DEFAULT_SCENE_LIMIT,
) -> Path:
    """Fetch a Sentinel-2 L2A mean composite over the AOI.

    Uses the `scene_limit` lowest-cloud scenes in the window, rather than all
    matching scenes, and takes the mean over time — cheaper under dask than
    a full-stack median.
    """
    import rioxarray  # noqa: F401

    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = client.search(
        collections=[COLLECTION],
        bbox=aoi.bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    print(
        f"Sentinel-2: {len(items)} candidate items for AOI={aoi.name}, "
        f"datetime={datetime}, cloud<{max_cloud}%."
    )
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 items. Try relaxing max_cloud (current={max_cloud}) "
            f"or widening datetime (current={datetime})."
        )

    # Rank by cloud cover; keep the cleanest N scenes.
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 999))
    kept = items[:scene_limit]
    clouds = [it.properties.get("eo:cloud_cover", "?") for it in kept]
    print(f"  Keeping the {len(kept)} cleanest scenes; cloud%: {clouds}")

    stack = stackstac.stack(
        kept,
        assets=BANDS,
        epsg=int(WORKING_CRS.split(":")[1]),
        bounds_latlon=aoi.bbox,
        resolution=resolution,
        chunksize=512,  # smaller tiles keep dask memory bounded
    )
    print(f"  Raw stack shape: {stack.shape}")

    composite = stack.mean("time", skipna=True)
    composite.name = "sentinel2_l2a_mean"
    print(f"  Composite shape: {composite.shape}")

    out_path = dataset_dir(NAME) / f"s2_mean_{aoi.name.lower()}.tif"
    composite.rio.write_crs(WORKING_CRS, inplace=True)
    composite.rio.to_raster(
        out_path,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    write_source_md(
        NAME,
        title="Sentinel-2 L2A mean composite (Planetary Computer)",
        url=f"{STAC_URL}/collections/{COLLECTION}",
        license=(
            "Copernicus Sentinel data — free, full, and open. "
            "https://sentinels.copernicus.eu/web/sentinel/terms-conditions"
        ),
        notes=(
            f"AOI={aoi.name}, datetime={datetime}, max_cloud={max_cloud}%, "
            f"scene_limit={scene_limit}, bands={BANDS}, resolution={resolution} m, "
            f"reprojected to {WORKING_CRS}. Composite is mean-over-time across "
            "the {scene_limit} lowest-cloud scenes. Streamed via dask."
        ),
    )
    return out_path


if __name__ == "__main__":
    from ai_minerals.aoi import TANACROSS

    path = fetch(TANACROSS)
    print(f"Wrote {path}")
