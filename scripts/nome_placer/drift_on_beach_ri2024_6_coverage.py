"""Round 4B: does RI 2024-6 supply a drift overlay for the central Nome placer?

The round-4 brief asked to rebuild the drift-on-beach placer feature on RI 2024-6
(DGGS, DOI 10.14509/31054), on the premise that RI 2024-6 "covers central Nome and
maps the Nome River drift as discrete polygons." Round 3 had data-walled the feature
on AOF 125 because AOF 125 lies east of the placer grid with zero overlap.

This checks the premise before any model is built. It loads the downloaded RI 2024-6
surficial GeMS vector, isolates the discrete drift units (including Qdn, "Drift of
Nome River Age"), and measures their footprint against the placer model grid.

Finding: RI 2024-6 is the Casadepaga / Big Hurrah-Council Bluff surficial map, the
eastern companion to RI 2024-7 bedrock. Its mapped extent sits ~33 km east of the
Cape Nome placer grid with zero overlap. Its three Qdn polygons are the Nome River
drift type-area lobes on the eastern map, not on the placer beachline. So RI 2024-6
cannot supply the strandline×drift intersection for the central placer model, the
same wall as AOF 125, further east. No model is built; this is reported as a wall.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.drift_on_beach_ri2024_6_coverage
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union

RI2024_6 = Path("data/raw/dggs_ri2024_6/extracted/shp/"
                "casadepaga_surficial_gems_db-open/GM_MapUnitPolys.shp")
AOF125 = Path("data/raw/nome_surficial_aof125/shp/"
              "tolstoi_point_cape_nome_surficial-open/GM_MapUnitPolys.shp")
PLACER_DEM = Path("data/raw/nome_mpm/ifsar_dem_3338.tif")
NOME_RIVER_DRIFT = "Qdn"            # "Drift of Nome River Age" (RI 2024-6 DMU)
DRIFT_PREFIXES = ("Qd",)           # all RI 2024-6 discrete drift units (Qdn/Qds/Qdu/...)
OUT_DIR = Path("data/derived/nome_placer/drift_on_beach")


def _unit_col(gdf: gpd.GeoDataFrame) -> str:
    return "MapUnit" if "MapUnit" in gdf else [c for c in gdf.columns if c.lower() == "mapunit"][0]


def _placer_grid_box() -> tuple[box, list[int], float]:
    with rasterio.open(PLACER_DEM) as ds:
        b = ds.bounds
        res = abs(ds.transform.a)
    return box(b.left, b.bottom, b.right, b.top), [round(b.left), round(b.bottom),
                                                   round(b.right), round(b.top)], res


def _coverage(vec_path: Path, name: str, grid: box) -> dict:
    g = gpd.read_file(vec_path).to_crs("EPSG:3338")
    uc = _unit_col(g)
    tb = [round(v) for v in g.total_bounds]
    drift = g[g[uc].astype(str).str.startswith(DRIFT_PREFIXES)]
    qdn = g[g[uc].astype(str) == NOME_RIVER_DRIFT]
    drift_union = unary_union(list(drift.geometry)) if len(drift) else None
    qdn_union = unary_union(list(qdn.geometry)) if len(qdn) else None
    x_overlap = bool(max(tb[0], grid.bounds[0]) < min(tb[2], grid.bounds[2]))
    y_overlap = bool(max(tb[1], grid.bounds[1]) < min(tb[3], grid.bounds[3]))
    return {
        "vector": str(vec_path),
        "extent_3338": tb,
        "n_polys": int(len(g)),
        "n_discrete_drift_polys": int(len(drift)),
        "drift_units_present": sorted(set(drift[uc].astype(str))),
        "n_nome_river_drift_Qdn_polys": int(len(qdn)),
        "x_overlaps_placer_grid": x_overlap,
        "y_overlaps_placer_grid": y_overlap,
        "overlaps_placer_grid": bool(x_overlap and y_overlap),
        "min_dist_grid_to_any_drift_m": (round(float(grid.distance(drift_union)))
                                         if drift_union is not None else None),
        "min_dist_grid_to_Qdn_m": (round(float(grid.distance(qdn_union)))
                                   if qdn_union is not None else None),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, grid_bounds, res = _placer_grid_box()

    ri = _coverage(RI2024_6, "RI2024-6", grid)
    aof = _coverage(AOF125, "AOF125", grid) if AOF125.exists() else {"note": "AOF125 not on disk"}

    out = {
        "question": "Does RI 2024-6 supply a discrete-drift overlay covering the central "
                    "Nome placer grid, enabling the strandline x drift intersection feature?",
        "premise_checked": "brief stated RI 2024-6 'covers central Nome and maps the Nome "
                           "River drift as discrete polygons'",
        "placer_model_grid": {"dem": str(PLACER_DEM), "bounds_3338": grid_bounds, "res_m": res},
        "ri2024_6": ri,
        "aof125_for_context": aof,
        "finding": (
            "RI 2024-6 is the Casadepaga / Big Hurrah-Council Bluff surficial map (DGGS RI "
            "2024-6, Stevens 2024), the eastern companion to RI 2024-7 bedrock. Its mapped "
            f"extent {ri['extent_3338']} sits east of the placer grid {grid_bounds} with "
            f"{'NO' if not ri['overlaps_placer_grid'] else 'some'} overlap "
            f"(nearest discrete-drift polygon ~{ri['min_dist_grid_to_any_drift_m']} m, nearest "
            f"Qdn Nome-River-drift polygon ~{ri['min_dist_grid_to_Qdn_m']} m from the grid). "
            f"It has {ri['n_nome_river_drift_Qdn_polys']} Qdn polygons, but they are the "
            "type-area lobes on the eastern map, not on the Cape Nome beachline. RI 2024-6 "
            "therefore cannot supply the strandline x drift intersection for the central "
            "placer model. This is the AOF 125 wall (round 3) repeated further east."
        ),
        "decision": "ESCALATE to coordinator. No drift-on-beach model built on RI 2024-6: "
                    "it would be all-NODATA over the placer grid, identical to the round-3 "
                    "AOF 125 null. The round-3 result (the beach-line backbone already "
                    "carries the placer signal; no central-Nome discrete-drift vector exists) "
                    "stands until a surficial drift map covering the Cape Nome beach is found.",
    }
    (OUT_DIR / "drift_on_beach_ri2024_6_coverage.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"ri2024_6_extent_3338": ri["extent_3338"],
                      "placer_grid_3338": grid_bounds,
                      "overlaps_placer_grid": ri["overlaps_placer_grid"],
                      "min_dist_grid_to_Qdn_m": ri["min_dist_grid_to_Qdn_m"],
                      "n_Qdn_polys": ri["n_nome_river_drift_Qdn_polys"],
                      "aof125_overlaps": aof.get("overlaps_placer_grid")}, indent=2))
    print(f"wrote {OUT_DIR / 'drift_on_beach_ri2024_6_coverage.json'}")


if __name__ == "__main__":
    main()
