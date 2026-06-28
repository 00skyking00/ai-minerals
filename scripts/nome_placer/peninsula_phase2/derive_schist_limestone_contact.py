"""Phase-2 prep: derive the schist-limestone contact from the on-disk geology.

The peninsula-scale geology (Solomon/Bendeleben/Teller sheets) is NOT on disk;
only the Nome quadrangle geology (Bundtzen et al. 1994, nmgeol_dd) is. This
script answers the Phase-2 feasibility question on the data we have: are the
schist and the limestone/marble units cleanly separable, and is a
schist-carbonate contact line derivable?

Method: classify the geology polygons (nm_ddp) by their map-unit LABEL into
schist / carbonate-marble / other, dissolve each class, and take the shared
boundary (schist_union.boundary intersect carbonate_union.boundary) as the
contact. Write the contact line layer + a distance raster aligned to the Nome
DEM grid, and compare it to the prebuilt dist_to_contact.tif used in round 5.

Classification (by LABEL, ages from the unit table; flagged where uncertain):
  schist:    PzZh, Ocs, Dcs, Zn, Zo  (pelitic + calc-schist, Nome Group)
  carbonate: Oim, Pzmm (marble), DOx (Devonian-Ordovician carbonate; the one
             whose assignment most needs the legend confirmed)
  other:     Qs, Kg, Kgu, Kdi, water (surficial, intrusive, water)

Run: uv run python -m scripts.nome_placer.peninsula_phase2.derive_schist_limestone_contact
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from shapely import line_merge
from shapely.geometry import MultiLineString

GEOL = Path("data/raw/nome_mpm/geol/nmgeol_dd/nm_ddp.shp")
DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
EXISTING = Path("data/derived/nome_placer/bedrock_contact/dist_to_contact.tif")
OUT = Path("data/derived/nome_placer/peninsula_phase2")

SCHIST = {"PzZh", "Ocs", "Dcs", "Zn", "Zo"}
CARBONATE = {"DOx", "Oim", "Pzmm"}


def densify_to_points(geom, step_m: float = 50.0) -> np.ndarray:
    """Sample points every step_m along all line components of geom."""
    pts = []
    lines = []
    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        lines = [g for g in geom.geoms if g.geom_type == "LineString"]
    for ln in lines:
        n = max(2, int(ln.length // step_m) + 1)
        for i in range(n + 1):
            p = ln.interpolate(min(i * step_m, ln.length))
            pts.append((p.x, p.y))
    return np.array(pts) if pts else np.empty((0, 2))


def contact_length_km(schist_polys, carb_polys) -> tuple[object, float]:
    su = schist_polys.union_all()
    cu = carb_polys.union_all()
    contact = su.boundary.intersection(cu.boundary)
    length = 0.0
    if not contact.is_empty:
        length = contact.length / 1000.0
    return contact, length


def main() -> None:
    p = gpd.read_file(GEOL).to_crs("EPSG:3338")
    p["rock"] = p["LABEL"].map(
        lambda l: "schist" if l in SCHIST else ("carbonate" if l in CARBONATE else "other"))
    counts = p.groupby(["rock", "LABEL"]).size().to_dict()
    n_schist = int((p.rock == "schist").sum())
    n_carb = int((p.rock == "carbonate").sum())

    schist = p[p.rock == "schist"]
    carb = p[p.rock == "carbonate"]
    contact, length_km = contact_length_km(schist, carb)
    # sensitivity: drop the uncertain DOx unit from the carbonate set
    carb_nodox = p[p.LABEL.isin(CARBONATE - {"DOx"})]
    _, length_km_nodox = contact_length_km(schist, carb_nodox) if len(carb_nodox) else (None, 0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    if not contact.is_empty:
        merged = line_merge(contact) if contact.geom_type != "LineString" else contact
        gpd.GeoDataFrame({"kind": ["schist_carbonate_contact"]},
                         geometry=[merged], crs="EPSG:3338").to_file(
            OUT / "schist_carbonate_contact_nome_3338.geojson", driver="GeoJSON")

    # Distance raster on the Nome DEM grid.
    with rasterio.open(DEM) as ds:
        H, W = ds.height, ds.width
        T = ds.transform
        profile = ds.profile.copy()
        dem = ds.read(1); nod = ds.nodata
    pts = densify_to_points(contact, 50.0)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = rasterio.transform.xy(T, rows.ravel(), cols.ravel())
    cell_xy = np.column_stack([np.asarray(cx), np.asarray(cy)])
    if len(pts):
        tree = cKDTree(pts)
        d, _ = tree.query(cell_xy, k=1)
        dist = d.reshape(H, W).astype(np.float32)
    else:
        dist = np.full((H, W), -1.0, dtype=np.float32)
    dist = np.where(dem == nod, -1.0, dist)
    profile.update(dtype="float32", count=1, nodata=-1.0, compress="lzw")
    with rasterio.open(OUT / "dist_to_schist_carbonate_contact_3338.tif", "w", **profile) as dst:
        dst.write(dist, 1)

    # Compare to the prebuilt contact raster round 5 used.
    cmp = {}
    if EXISTING.exists():
        with rasterio.open(EXISTING) as ds:
            ex = ds.read(1).astype(float); exn = ds.nodata
        ex = np.where(ex == exn, np.nan, ex)
        mine = np.where(dist == -1.0, np.nan, dist)
        ok = np.isfinite(ex) & np.isfinite(mine)
        if ok.sum() > 100:
            from scipy.stats import spearmanr
            rho = float(spearmanr(ex[ok], mine[ok]).correlation)
            cmp = {"spearman_vs_existing": round(rho, 3),
                   "median_existing_m": round(float(np.nanmedian(ex)), 0),
                   "median_mine_m": round(float(np.nanmedian(mine)), 0),
                   "interpretation": ("high rho => the prebuilt raster is this same "
                                      "schist-carbonate contact; low rho => it is a "
                                      "different contact set (explains the round-5 null)")}

    report = {
        "geology_source": "Nome quadrangle only (Bundtzen et al. 1994, nmgeol_dd); "
                          "peninsula sheets NOT on disk",
        "n_polygons": int(len(p)),
        "n_schist_polygons": n_schist, "n_carbonate_polygons": n_carb,
        "polygons_by_class_label": {f"{r}/{l}": int(c) for (r, l), c in counts.items()},
        "schist_carbonate_contact_km": round(length_km, 1),
        "contact_km_without_DOx": round(length_km_nodox, 1),
        "separable_at_nome_quad": bool(n_schist > 0 and n_carb > 0 and length_km > 0),
        "compare_to_prebuilt_dist_to_contact": cmp,
        "DOx_caveat": ("DOx (21 polygons, the largest non-schist unit) is classed as "
                       "carbonate by map-code + Devonian-Ordovician age; confirm against "
                       "the SIM/Bundtzen unit legend before peninsula scale-up"),
        "peninsula_gap": ("only the Nome quad geology is on disk. Phase-2 at peninsula "
                          "scale needs the Solomon/Bendeleben/Teller bedrock geology "
                          "(USGS SIM 3131 if peninsula-wide, else AK DGGS / Till 2011 "
                          "Seward Peninsula compilation) with comparable schist/marble units"),
    }
    (OUT / "schist_carbonate_contact_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
