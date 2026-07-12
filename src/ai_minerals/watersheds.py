"""DEM catchment (watershed) delineation for the drill-gold LODO harness.

Why this module exists
----------------------
The drill-gold capture harness holds out *independent drainages*, and the
adversarial review (ml_step3_g1_validation_plan_2026-07-12 §4) is explicit that a
drainage must be a **DEM catchment polygon**, not a distance-threshold point
cluster. A point cluster (the retired ``assign_drainages`` at 1.5 km single
linkage) would let one creek's seven drill lines fake up to seven "drainages" and
shrink the confidence interval fraudulently. A catchment is defined by the
terrain: two blocks are in the same watershed iff surface water from both leaves
the DEM at the same outlet.

Method: D8 flow routing on a depression-filled, flat-resolved DEM, delineating the
whole catchment that drains to each grid-edge / interior-pit outlet
(``pyflwdir.FlwdirRaster.basins``). This is knob-free: no channel-initiation
threshold or pour-point choice enters, so the partition is a property of the
terrain alone. Depression filling and flat resolution are the hard parts of D8
(a naive steepest-descent routing fragments low-relief valley floors into dozens
of spurious micro-basins); ``pyflwdir`` (Deltares, Eilander et al. 2021, the
engine behind ``hydromt``) does them correctly, so Little Creek's drilled valley
comes out as one 42 km2 catchment rather than the ~16 micro-slivers a hand-rolled
router produced.

Limitation (documented, not hidden): basins are truncated at the DEM edge, so a
catchment that continues past the grid is only its in-grid portion. For the Nome
placer core each creek exits south to the coast inside the grid, so this does not
split a drilled creek.

CRS contract: coordinates are projected metres (EPSG:3338 for Nome). The DEM must
be in the same CRS; ``watershed_at`` maps (x, y) metres to a basin id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features as rio_features
from shapely.geometry import shape

NODATA_ID = -1


@dataclass
class Watersheds:
    """A DEM catchment labelling: a basin-id raster plus the geo-referencing to
    query it and polygonise it."""

    labels: np.ndarray          # (H, W) int32; basin id per cell, NODATA_ID where off-basin
    transform: rasterio.Affine
    crs: object
    valid: np.ndarray           # (H, W) bool on-land mask
    uparea_km2: np.ndarray      # (H, W) upstream contributing area per cell (km2)
    n_basins: int
    extras: dict = field(default_factory=dict)

    def watershed_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Basin id at each (x, y) metre coordinate; NODATA_ID if off-grid/off-basin."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        cols, rows = ~self.transform * (x, y)
        rows = np.floor(rows).astype(int)
        cols = np.floor(cols).astype(int)
        H, W = self.labels.shape
        out = np.full(len(x), NODATA_ID, int)
        inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        out[inside] = self.labels[rows[inside], cols[inside]]
        return out

    def basin_area_km2(self, basin_id: int) -> float:
        """Catchment area (km2) of a basin = its cell count times cell area."""
        px = abs(self.transform.a) * abs(self.transform.e) / 1e6
        return float((self.labels == basin_id).sum() * px)

    def basin_sizes(self) -> dict[int, int]:
        """Cell count per basin id (off-basin cells excluded)."""
        m = self.labels != NODATA_ID
        ids, counts = np.unique(self.labels[m], return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}

    def polygons(self, only_ids: set[int] | None = None):
        """(shapely geometry, basin_id) features; optionally a subset of basin ids."""
        lab = self.labels.astype(np.int32)
        mask = self.labels != NODATA_ID
        if only_ids is not None:
            mask = mask & np.isin(lab, list(only_ids))
        return [(shape(g), int(v))
                for g, v in rio_features.shapes(lab, mask=mask, transform=self.transform)]


def delineate(dem_path: Path) -> Watersheds:
    """Delineate DEM catchments from a single-band DEM raster (projected metres)."""
    import pyflwdir

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)
        nod = ds.nodata
        transform, crs = ds.transform, ds.crs
    valid = np.ones(dem.shape, bool) if nod is None else (dem != nod)
    elv = np.where(valid, dem, np.nan)

    flw = pyflwdir.from_dem(data=elv, nodata=np.nan, transform=transform, latlon=False)
    basins = np.asarray(flw.basins(), np.int64)      # 0 = nodata/outside
    labels = np.where(basins > 0, basins, NODATA_ID).astype(np.int32)
    uparea = np.asarray(flw.upstream_area(unit="km2"), np.float32)
    uparea = np.where(np.isfinite(uparea), uparea, 0.0)
    n_basins = int((np.unique(labels) != NODATA_ID).sum())
    return Watersheds(
        labels=labels, transform=transform, crs=crs, valid=valid, uparea_km2=uparea,
        n_basins=n_basins,
        extras={"dem": str(dem_path), "n_cells_valid": int(valid.sum()),
                "engine": f"pyflwdir {pyflwdir.__version__}",
                "method": "D8 whole-catchment basins to edge/pit outlets (knob-free)"},
    )
