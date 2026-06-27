"""Nome lode-control structural & lithologic covariates -> EPSG:3338 grids.

Source: Alaska DGGS PDF 94-39, the Nome mining district GeMS geodatabase
(``nome_mining_district_gems.gdb``), the most detailed public structural map of
the district. These covariates target the *documented* orogenic-gold control at
Nome (USGS Bull. 1786; Piekenbrock & Odden 2009, Econ. Geol. v.104): quartz
veins on high-angle faults in two regional sets (NE- and NW-trending), hosted in
the carbonaceous (graphitic) part of the Nome Group, near fold hinges.

These are ground-property features (structure, lithology) that exist regardless
of whether gold was ever found, so a leak-guarded AUC gain is transferable
signal rather than the occurrence-coverage confound that sank the F3 KG fusion.

Bands produced (each aligned to a caller-supplied model grid):

  dist_ne_fault       metres to the nearest NE-trending fault/lineament segment
                      (strike azimuth in [22.5, 67.5)); the primary Rock Creek /
                      Albion-fault control direction.
  dist_nw_fault       metres to the nearest NW-trending segment (azimuth in
                      [112.5, 157.5)); the conjugate set. Kept SEPARATE from NE
                      because the orientation, not mere fault proximity, is the
                      control.
  dist_fold_hinge     metres to the nearest mapped fold axis (anticline /
                      overturned anticline), the Groves-2018 fold-hinge trap.
  carbonaceous_host   1.0 where the cell falls in a graphitic Nome Group schist
                      unit (pCPzsg / pCPzspm / pCPzsgb), else 0.0. The graphitic
                      metasediment is the lithologic host; being conductive it
                      should also read as an EM low (independent cross-check
                      against the DIGHEM/SkyTEM resistivity, done in the report).

A companion ``gems_extent`` mask (1.0 inside the GeMS bedrock footprint) is
written so the re-baseline can restrict AUC to mapped cells: the GeMS covers
central Nome where occurrences cluster, so whole-extent AUC mixes structural
signal with a mapping-coverage proxy exactly as the DIGHEM footprint does.

Faults and lineaments are split at the SEGMENT level (each consecutive vertex
pair classified by its own azimuth), so a fault that bends contributes its NE
and NW reaches to the correct set.
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString

# Let PROJ pull the Alaska NAD27->NAD83 NADCON grid so the GeMS (NAD27 / UTM 3N)
# linework registers to the same datum as the model grids; without it the warp
# falls back to a ~160 m ballpark shift that would smear distance-to-fault by
# several in-box cells. dighem.ensure_nadcon_grid() places it offline-first.
os.environ.setdefault("PROJ_NETWORK", "ON")

GEMS = Path(
    "data/raw/adggs/pdf94_39/extracted/"
    "nome_mining_district_gems_db_pkg/nome_mining_district_gems.gdb"
)
TARGET_CRS = "EPSG:3338"
NODATA = -1.0
PAD_M = 8000.0  # distance-transform halo so a cell's nearest structure may sit
#                 just outside the model grid (matches mpm_build_geology_covariates)

# Quadrant azimuth bins (trend folded to [0,180)). 45-deg wide, centred on the
# two diagonal sets; N-S and E-W segments fall in neither and anchor no covariate.
NE_RANGE = (22.5, 67.5)
NW_RANGE = (112.5, 157.5)

# Graphitic Nome Group schist units (explicit "graphitic" in the GeMS
# DescriptionOfMapUnits): the carbonaceous lode host.
GRAPHITIC_UNITS = {"pCPzsg", "pCPzspm", "pCPzsgb"}

STRUCT_BANDS = ["dist_ne_fault", "dist_nw_fault", "dist_fold_hinge", "carbonaceous_host"]
DIST_BANDS = ["dist_ne_fault", "dist_nw_fault", "dist_fold_hinge"]


def _segments(gdf: gpd.GeoDataFrame) -> tuple[list[LineString], np.ndarray]:
    """Explode line features into 2-point segments and their trend azimuths [0,180)."""
    segs: list[LineString] = []
    az: list[float] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            xy = np.asarray(part.coords)
            for a, b in zip(xy[:-1], xy[1:]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                if dx == 0 and dy == 0:
                    continue
                segs.append(LineString([a, b]))
                az.append(np.degrees(np.arctan2(dx, dy)) % 180.0)
    return segs, np.asarray(az)


def _in_range(az: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (az >= lo) & (az < hi)


def load_oriented_structures() -> dict[str, list[LineString]]:
    """Return NE/NW fault+lineament segments and fold-axis lines in EPSG:3338."""
    caf = gpd.read_file(GEMS, layer="ContactsAndFaults").to_crs(TARGET_CRS)
    faults = caf[caf["Type"].astype(str).str.startswith("fault")]
    lines = gpd.read_file(GEMS, layer="GeologicLines").to_crs(TARGET_CRS)  # lineaments
    folds = gpd.read_file(GEMS, layer="StructureLines").to_crs(TARGET_CRS)  # fold axes

    structural = gpd.GeoDataFrame(
        geometry=list(faults.geometry) + list(lines.geometry), crs=TARGET_CRS
    )
    segs, az = _segments(structural)
    ne = [s for s, keep in zip(segs, _in_range(az, *NE_RANGE)) if keep]
    nw = [s for s, keep in zip(segs, _in_range(az, *NW_RANGE)) if keep]
    fold_segs, _ = _segments(folds)
    return {"dist_ne_fault": ne, "dist_nw_fault": nw, "dist_fold_hinge": fold_segs}


def _dist_to(geoms: list[LineString], template_path: Path) -> tuple[np.ndarray, dict]:
    """Distance (m) from each template cell to the nearest geometry, padded halo."""
    with rasterio.open(template_path) as t:
        transform, width, height = t.transform, t.width, t.height
        profile = t.profile.copy()
    px = abs(transform.a)
    pad = int(round(PAD_M / px))
    big = Affine(transform.a, transform.b, transform.c - pad * px,
                 transform.d, transform.e, transform.f + pad * px)
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    mask = rasterize(shapes, out_shape=(height + 2 * pad, width + 2 * pad),
                     transform=big, fill=0, dtype="uint8", all_touched=True)
    dist_big = distance_transform_edt(mask == 0, sampling=px)
    dist = dist_big[pad:pad + height, pad:pad + width].astype("float32")
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    return dist, profile


def _polys_mask(units: set[str] | None, template_path: Path) -> tuple[np.ndarray, dict]:
    """Rasterize MapUnitPolys (a unit subset, or all when units is None) to 1.0/0.0."""
    with rasterio.open(template_path) as t:
        transform, width, height = t.transform, t.width, t.height
        profile = t.profile.copy()
    mup = gpd.read_file(GEMS, layer="MapUnitPolys").to_crs(TARGET_CRS)
    if units is not None:
        mup = mup[mup["MapUnit"].isin(units)]
    shapes = [(g, 1) for g in mup.geometry if g is not None and not g.is_empty]
    arr = rasterize(shapes, out_shape=(height, width), transform=transform,
                    fill=0, dtype="uint8", all_touched=False).astype("float32")
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    return arr, profile


def _write(arr: np.ndarray, profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


def build_structure_bands(template_path: Path, out_dir: Path) -> dict[str, Path]:
    """Build the structural/lithologic covariate GeoTIFFs aligned to a model grid.

    Returns {band_name: path} for STRUCT_BANDS plus the ``gems_extent`` mask.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    structures = load_oriented_structures()
    for band, geoms in structures.items():
        dist, profile = _dist_to(geoms, template_path)
        paths[band] = _write(dist, profile, out_dir / f"struct_{band}.tif")

    host, hprof = _polys_mask(GRAPHITIC_UNITS, template_path)
    paths["carbonaceous_host"] = _write(host, hprof, out_dir / "struct_carbonaceous_host.tif")

    extent, eprof = _polys_mask(None, template_path)
    paths["gems_extent"] = _write(extent, eprof, out_dir / "struct_gems_extent.tif")
    return paths
