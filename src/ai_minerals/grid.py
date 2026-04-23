"""Regular analysis grid over an AOI in the working CRS.

Used as the pixel-scale basis for feature engineering and modeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ai_minerals.aoi import AOI, WORKING_CRS


@dataclass(frozen=True)
class Grid:
    """A regular grid aligned to round numbers in EPSG:3338.

    Attributes
    ----------
    xs, ys : 1-D arrays of pixel-center coordinates (EPSG:3338 meters), increasing.
    resolution_m : int, pixel edge length in meters.
    crs : str, the coordinate reference system (always WORKING_CRS for v1).
    """

    xs: np.ndarray
    ys: np.ndarray
    resolution_m: int
    crs: str = WORKING_CRS

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.ys), len(self.xs))

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the grid extent (pixel edges)."""
        r = self.resolution_m / 2
        return (self.xs[0] - r, self.ys[0] - r, self.xs[-1] + r, self.ys[-1] + r)

    def centroid_gdf(self) -> gpd.GeoDataFrame:
        """Return a flat GeoDataFrame of pixel centroids with row/col indices."""
        xv, yv = np.meshgrid(self.xs, self.ys)
        rows, cols = np.meshgrid(
            np.arange(len(self.ys)), np.arange(len(self.xs)), indexing="ij"
        )
        df = pd.DataFrame(
            {
                "row": rows.ravel(),
                "col": cols.ravel(),
                "x": xv.ravel(),
                "y": yv.ravel(),
            }
        )
        geom = [Point(x, y) for x, y in zip(df["x"], df["y"])]
        return gpd.GeoDataFrame(df, geometry=geom, crs=self.crs)


def build_grid(aoi: AOI, resolution_m: int = 500) -> Grid:
    """Build a grid over the AOI, aligned to round multiples of `resolution_m`.

    Grid cells are positioned so their centers fall on `resolution_m * n + r/2`
    coordinates — this makes merging with other `resolution_m`-aligned rasters
    straightforward and predictable across re-runs.
    """
    aoi_series = gpd.GeoSeries([aoi.polygon], crs=aoi.crs).to_crs(WORKING_CRS)
    minx, miny, maxx, maxy = aoi_series.total_bounds
    r = resolution_m
    x0 = int(np.floor(minx / r)) * r + r / 2
    y0 = int(np.floor(miny / r)) * r + r / 2
    x1 = int(np.ceil(maxx / r)) * r - r / 2
    y1 = int(np.ceil(maxy / r)) * r - r / 2
    xs = np.arange(x0, x1 + r / 2, r)
    ys = np.arange(y0, y1 + r / 2, r)
    return Grid(xs=xs, ys=ys, resolution_m=r, crs=WORKING_CRS)
