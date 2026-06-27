"""Peninsula-wide lode-control structure/lithology covariates -> EPSG:3338 grids.

The central-Nome module (``nome_structure``) reads only the Nome mining district
GeMS (DGGS PDF 94-39), which stops at the central district. To test whether the
Groves-2018 structural control on the lode AUC (auc_gems 0.712 on typed 36a
labels) is district-specific or peninsula-wide, the same features have to be
rebuilt on bedrock maps that reach *beyond* central Nome. Two public sources do:

  SIM 3131   USGS Scientific Investigations Map 3131 (Till et al. 2011), bedrock
             geologic map of the Seward Peninsula, 1:500,000, GIS database at
             USGS OF 2009-1254. Per-quadrangle arc/polygon coverages; we read the
             Nome (``nm``) and Solomon (``so``) quadrangles, the southern belt
             that carries the 36a lode occurrences. NAD27 / UTM 3N (EPSG:26703).
             Faults are ARC_CODE 4 (normal), 30 (sense uncertain) and 60
             (concealed); graphitic host units are the LABELs whose
             DescriptionOfMapUnits text says "graphitic" (DOx, Dcs at this scale).

  RI 2024-7  Alaska DGGS Report of Investigation 2024-7 (Werdon et al. 2024),
             bedrock geologic map of the Big Hurrah-Council-Bluff (Casadepaga)
             area, southern Seward Peninsula, DOI 10.14509/31308. AK GeMS
             schema, the same as the central district GeMS and AOF 125, so its
             ContactsAndFaults / MapUnitPolys / StructureLines layers load with
             the same logic. This is the *detailed* eastern map (1835 fault
             features, 223 fold axes) that the regional SIM 3131 lacks.

Bands written (same names as ``nome_structure`` so the lode driver's arm
definitions are reused unchanged):

  dist_ne_fault, dist_nw_fault   distance to nearest NE-/NW-trending fault segment
  dist_fault_intersection        distance to nearest NE x NW intersection node
  dist_splay                     distance to nearest second-order splay. Regionally
                                 there are no named trunks (SIM 3131 faults carry
                                 no name), so a TRUNK is redefined as a fault whose
                                 whole-line length is in the top quintile, and a
                                 splay is a non-trunk fault segment within
                                 ``SPLAY_NEAR_TRUNK_M`` of the trunk union. This is
                                 a length proxy for the central module's
                                 named-trunk definition; stated as a caveat.
  dist_fold_hinge                distance to nearest mapped fold axis (RI 2024-7
                                 StructureLines; SIM 3131 maps no fold axes, so
                                 fold coverage is eastern-heavy)
  carbonaceous_host              1.0 in a graphitic unit, else 0.0
  gems_extent                    1.0 inside the mapped bedrock footprint (SIM 3131
                                 covers the whole wider grid, so unlike the central
                                 GeMS this is near-complete and carries little
                                 coverage-proxy signal)

A companion ``geology_unit_code`` raster + legend (SIM 3131 LABEL rasterised to
an integer code) and a ``dist_to_fault`` band are written by
``build_regional_base_bands`` for the matched terrain-free base model.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

# nome_structure sets PROJ_NETWORK=ON (NADCON NAD27->NAD83) and carries the
# shared raster helpers + azimuth bins; reuse them so the two modules stay aligned.
from ai_minerals.data import nome_structure as ns
from ai_minerals.data.nome_structure import (
    NE_RANGE, NW_RANGE, NODATA, SPLAY_NEAR_TRUNK_M, TARGET_CRS,
    _dist_to, _in_range, _segments, _write,
)

SIM3131_DIR = Path("data/raw/sim3131_seward")
RI2024_OPEN = Path("data/raw/dggs_ri2024_7/extracted/pkg/"
                   "casadepaga_bedrock_gems_db_wo_stations-open")
SIM_QUADS = ("nm", "so")  # Nome + Solomon quadrangles (the southern lode belt)
FAULT_ARC_CODES = {4, 30, 60}  # SIM 3131: normal / sense-uncertain / concealed faults
TRUNK_LENGTH_QUANTILE = 0.80   # regional splay: a "trunk" is a top-quintile-length fault

STRUCT_BANDS = ["dist_ne_fault", "dist_nw_fault", "dist_fold_hinge", "carbonaceous_host"]
GROVES_BANDS = ["dist_fault_intersection", "dist_splay"]
SHARP_BANDS = GROVES_BANDS + ["dist_ne_fault", "dist_nw_fault",
                              "dist_fold_hinge", "carbonaceous_host"]


# --------------------------------------------------------------------------- #
# Source loaders (reprojected to EPSG:3338)
# --------------------------------------------------------------------------- #
def _sim_arcs() -> gpd.GeoDataFrame:
    parts = [gpd.read_file(SIM3131_DIR / f"{q}geol" / f"{q}geola.shp").to_crs(TARGET_CRS)
             for q in SIM_QUADS]
    return pd.concat(parts, ignore_index=True)


def _sim_polys() -> gpd.GeoDataFrame:
    parts = [gpd.read_file(SIM3131_DIR / f"{q}geol" / f"{q}geolp.shp").to_crs(TARGET_CRS)
             for q in SIM_QUADS]
    return pd.concat(parts, ignore_index=True)


def _ri_layer(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(RI2024_OPEN / f"GM_{name}.shp").to_crs(TARGET_CRS)


def sim_graphitic_labels() -> set[str]:
    """SIM 3131 LABELs whose DescriptionOfMapUnits text mentions 'graphitic'."""
    meta = Path("data/raw/sim3131_seward/SP_meta.txt")
    labels = set(_sim_polys()["LABEL"].astype(str))
    if not meta.exists():
        return {"DOx", "Dcs"}  # the nm+so graphitic units, from the SIM 3131 metadata
    text = meta.read_text(errors="ignore")
    import re
    graf = set()
    for lab in labels:
        for m in re.finditer(r":\s*" + re.escape(lab) + r"\s*-\s*([^\n]+)", text):
            if "graphit" in m.group(1).lower():
                graf.add(lab)
    return graf or {"DOx", "Dcs"}


def ri_graphitic_units() -> set[str]:
    """RI 2024-7 MapUnits whose DescriptionOfMapUnits row mentions 'graphitic'."""
    csv = RI2024_OPEN / "DescriptionOfMapUnits.csv"
    if not csv.exists():
        return {"DOg", "DOi", "DOm", "DOms", "DOq", "DOqs", "DOs", "DOsq", "DOx"}
    d = pd.read_csv(csv)
    out = set()
    for _, r in d.iterrows():
        blob = " ".join(str(r[c]) for c in d.columns).lower()
        if "graphit" in blob and str(r["MapUnit"]) not in ("nan", ""):
            out.add(str(r["MapUnit"]))
    return out


def load_regional_faults() -> gpd.GeoDataFrame:
    """All fault lines from SIM 3131 (codes 4/30/60) + RI 2024-7 (Type ~ fault)."""
    arcs = _sim_arcs()
    sim_faults = arcs[arcs["ARC_CODE"].isin(FAULT_ARC_CODES)]
    caf = _ri_layer("ContactsAndFaults")
    ri_faults = caf[caf["Type"].astype(str).str.lower().str.startswith("fault")]
    return gpd.GeoDataFrame(
        geometry=list(sim_faults.geometry) + list(ri_faults.geometry), crs=TARGET_CRS
    )


def load_regional_folds() -> list[LineString]:
    """Fold axes (RI 2024-7 StructureLines; SIM 3131 maps no fold axes)."""
    sl = _ri_layer("StructureLines")
    folds = sl[sl["Category"].astype(str).str.lower().eq("fold")] if "Category" in sl else sl
    segs, _ = _segments(folds)
    return segs


def load_regional_oriented() -> dict[str, list[LineString]]:
    """NE / NW fault segments + fold-axis segments, regional sources combined."""
    segs, az = _segments(load_regional_faults())
    ne = [s for s, keep in zip(segs, _in_range(az, *NE_RANGE)) if keep]
    nw = [s for s, keep in zip(segs, _in_range(az, *NW_RANGE)) if keep]
    return {"dist_ne_fault": ne, "dist_nw_fault": nw, "dist_fold_hinge": load_regional_folds()}


def load_regional_intersections() -> list:
    """NE x NW fault intersection nodes (the Groves structure-intersection target)."""
    o = load_regional_oriented()
    if not o["dist_ne_fault"] or not o["dist_nw_fault"]:
        return []
    ne = unary_union(MultiLineString(o["dist_ne_fault"]))
    nw = unary_union(MultiLineString(o["dist_nw_fault"]))
    inter = ne.intersection(nw)
    if inter.is_empty:
        return []
    geoms = list(inter.geoms) if inter.geom_type.startswith("Multi") else [inter]
    return [g for g in geoms if g.geom_type == "Point"]


def load_regional_splays() -> list[LineString]:
    """Second-order splays via a length-trunk proxy (no named faults regionally).

    Trunk = a whole fault line whose length is in the top ``TRUNK_LENGTH_QUANTILE``
    quintile; splay = a segment of a non-trunk fault within ``SPLAY_NEAR_TRUNK_M``
    of the trunk union (but not coincident). A length proxy for the central
    module's named-trunk splay; stated as a caveat in the report.
    """
    faults = load_regional_faults()
    lengths = faults.geometry.length.to_numpy()
    if len(lengths) == 0:
        return []
    thresh = float(np.quantile(lengths, TRUNK_LENGTH_QUANTILE))
    is_trunk = lengths >= thresh
    trunk = unary_union(list(faults.geometry[is_trunk]))
    other_segs, _ = _segments(faults[~is_trunk])
    return [seg for seg in other_segs if 1.0 < seg.distance(trunk) <= SPLAY_NEAR_TRUNK_M]


def regional_graphitic_polys() -> gpd.GeoDataFrame:
    """Graphitic-unit polygons from SIM 3131 (LABEL) + RI 2024-7 (MapUnit)."""
    sp = _sim_polys()
    sp_g = sp[sp["LABEL"].astype(str).isin(sim_graphitic_labels())]
    mup = _ri_layer("MapUnitPolys")
    ucol = "MapUnit" if "MapUnit" in mup else [c for c in mup.columns if c.lower() == "mapunit"][0]
    ri_g = mup[mup[ucol].astype(str).isin(ri_graphitic_units())]
    return gpd.GeoDataFrame(geometry=list(sp_g.geometry) + list(ri_g.geometry), crs=TARGET_CRS)


def regional_extent_polys() -> gpd.GeoDataFrame:
    """All mapped bedrock polygons (SIM 3131 + RI 2024-7) for the extent mask."""
    sp = _sim_polys()
    sp = sp[~sp["LABEL"].astype(str).isin(["water", "nan", ""])]
    mup = _ri_layer("MapUnitPolys")
    return gpd.GeoDataFrame(geometry=list(sp.geometry) + list(mup.geometry), crs=TARGET_CRS)


# --------------------------------------------------------------------------- #
# Raster builders
# --------------------------------------------------------------------------- #
def _polys_to_mask(polys: gpd.GeoDataFrame, template_path: Path) -> tuple[np.ndarray, dict]:
    from rasterio.features import rasterize
    with rasterio.open(template_path) as t:
        transform, width, height = t.transform, t.width, t.height
        profile = t.profile.copy()
    shapes = [(g, 1) for g in polys.geometry if g is not None and not g.is_empty]
    arr = rasterize(shapes, out_shape=(height, width), transform=transform,
                    fill=0, dtype="uint8", all_touched=False).astype("float32")
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    return arr, profile


def build_regional_structure_bands(template_path: Path, out_dir: Path) -> dict[str, Path]:
    """Write the regional Groves structure + host + extent bands to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    oriented = load_regional_oriented()
    for band, segs in oriented.items():
        dist, prof = _dist_to(segs, template_path)
        paths[band] = _write(dist, prof, out_dir / f"struct_{band}.tif")
    dist, prof = _dist_to(load_regional_intersections(), template_path)
    paths["dist_fault_intersection"] = _write(dist, prof, out_dir / "struct_dist_fault_intersection.tif")
    dist, prof = _dist_to(load_regional_splays(), template_path)
    paths["dist_splay"] = _write(dist, prof, out_dir / "struct_dist_splay.tif")
    host, hprof = _polys_to_mask(regional_graphitic_polys(), template_path)
    paths["carbonaceous_host"] = _write(host, hprof, out_dir / "struct_carbonaceous_host.tif")
    extent, eprof = _polys_to_mask(regional_extent_polys(), template_path)
    paths["gems_extent"] = _write(extent, eprof, out_dir / "struct_gems_extent.tif")
    return paths


def build_regional_base_bands(template_path: Path, out_dir: Path) -> dict[str, Path]:
    """Write the matched terrain-free base inputs: SIM 3131 geology code + legend,
    and a distance-to-any-regional-fault band."""
    from rasterio.features import rasterize
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    sp = _sim_polys()
    sp = sp[~sp["LABEL"].astype(str).isin(["nan", ""])]
    labels = sorted(sp["LABEL"].astype(str).unique())
    code = {lab: i + 1 for i, lab in enumerate(labels)}  # 0 = nodata/unmapped
    with rasterio.open(template_path) as t:
        transform, width, height = t.transform, t.width, t.height
        profile = t.profile.copy()
    shapes = [(g, code[str(lab)]) for g, lab in zip(sp.geometry, sp["LABEL"])
              if g is not None and not g.is_empty]
    arr = rasterize(shapes, out_shape=(height, width), transform=transform,
                    fill=0, dtype="int32", all_touched=False)
    gprof = profile.copy()
    gprof.update(dtype="int32", count=1, nodata=0, compress="deflate")
    gpath = out_dir / "geology_unit_code.tif"
    with rasterio.open(gpath, "w", **gprof) as dst:
        dst.write(arr.astype("int32"), 1)
    paths["geology_unit_code"] = gpath
    (out_dir / "geology_legend.json").write_text(json.dumps({str(v): k for k, v in code.items()}, indent=2))

    dist, prof = _dist_to(list(load_regional_faults().geometry), template_path)
    paths["dist_to_fault"] = _write(dist, prof, out_dir / "dist_to_fault.tif")
    return paths
