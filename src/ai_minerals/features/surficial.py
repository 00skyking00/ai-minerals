"""Per-cell masks from the DGGS PDF 94-39 surficial geology.

bearcub published PDF 94-39 ("Digital geologic map of the Nome mining
district") on 2026-06-15 as
``data/raw/bearcub_nome/nome_surface_geology.gpkg`` with a coarse
``surface_class`` column. This module rasterizes those polygons onto
the working grid so per-cell features can key off surface_class.

Three masks the Nome placer model wants:

- ``drift_mask`` (``surface_class == "glacial_drift_outwash"``): the
  QM population's in-drift mask. Bear Cub MS 1178 and most of the
  Bourbon / Specimen / Anvil drainages sit on Qd2 Nome River drift
  here. Replaces the v1 elevation-range proxy in ``score_qm``.
- ``schist_mask`` (``surface_class == "bedrock_schist"``): seeds the
  areal lode-source-distance feature. Per the Tuck note about
  gold-tungsten disseminated over a wide area, the seed is the
  whole schist polygon set, not Rock Creek alone.
- ``tailings_mask`` (``surface_class == "placer_tailings"``): the Qh
  "was-mined" polygons. Phase 2 supervised model uses these as a
  weighted auxiliary positive label set (lower weight than a drill
  hole with a measured grade, but real signal that the ground was
  workable).

All masks return as a ``pd.Series`` aligned with ``grid.centroid_gdf()``,
``float32`` in {0.0, 1.0}, so downstream code can multiply through.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree

from ai_minerals.aoi import AOI
from ai_minerals.data.adapters.geology.adggs_surficial import load as load_adggs
from ai_minerals.grid import Grid


DRIFT_CLASS = "glacial_drift_outwash"
SCHIST_CLASS = "bedrock_schist"
TAILINGS_CLASS = "placer_tailings"


def load_surface_geology(path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Load PDF 94-39 surficial polygons via the ADGGS loader.

    Thin alias so callers don't need to import the loader by name. The
    returned GDF carries the ``surface_class`` column when the source
    file has it.
    """
    return load_adggs(path, aoi)


def surface_class_mask(
    polys: gpd.GeoDataFrame,
    grid: Grid,
    surface_class: str,
) -> pd.Series:
    """Per-grid-cell membership in polygons of a given ``surface_class``.

    Returns a float Series of {0.0, 1.0} indexed identically to
    ``grid.centroid_gdf()``. Uses an STRtree intersection check at
    cell centroids; reprojects polygons to the grid CRS first.
    """
    if "surface_class" not in polys.columns:
        raise ValueError(
            "surface geology GDF has no 'surface_class' column; "
            "load via adggs_surficial.load with PDF 94-39 input."
        )

    centroids = grid.centroid_gdf()
    selected = polys[polys["surface_class"] == surface_class]
    if len(selected) == 0:
        return pd.Series(
            np.zeros(len(centroids), dtype=np.float32),
            index=centroids.index,
            name=f"surface_class_{surface_class}",
        )

    if selected.crs != centroids.crs:
        selected = selected.to_crs(centroids.crs)

    tree = STRtree(selected.geometry.to_numpy())
    points = centroids.geometry.to_numpy()
    # STRtree.query returns (input_indices, tree_indices) for any
    # bbox intersection; we follow up with a true contains check.
    in_input, in_tree = tree.query(points, predicate="intersects")
    hit = np.zeros(len(points), dtype=bool)
    hit[np.unique(in_input)] = True

    return pd.Series(
        hit.astype(np.float32),
        index=centroids.index,
        name=f"surface_class_{surface_class}",
    )


def drift_mask(polys: gpd.GeoDataFrame, grid: Grid) -> pd.Series:
    """Per-cell glacial-drift / outwash mask (Qd1-Qd4, Qo2, Qo4)."""
    return surface_class_mask(polys, grid, DRIFT_CLASS).rename("drift_mask")


def schist_mask(polys: gpd.GeoDataFrame, grid: Grid) -> pd.Series:
    """Per-cell Nome Group schist + bedrock mask (lode-source seed)."""
    return surface_class_mask(polys, grid, SCHIST_CLASS).rename("schist_mask")


def tailings_mask(polys: gpd.GeoDataFrame, grid: Grid) -> pd.Series:
    """Per-cell placer-mine tailings mask (Qh; was-mined auxiliary label)."""
    return surface_class_mask(polys, grid, TAILINGS_CLASS).rename("tailings_mask")
