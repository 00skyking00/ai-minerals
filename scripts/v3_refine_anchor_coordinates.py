"""v3 Phase B.0: refine anchor district coordinates against Orlando 2016 pit polygons.

The v2 anchor lookups used GNIS town-center coordinates, which can be 200m-2km
off the actual hydraulic-pit footprint each district names. This script:

  1. Loads the Orlando 2016 pit polygon set from data/raw/hydraulic_pits/
  2. For each anchor district, finds the nearest pit polygon to the GNIS coord
  3. If within 2km (the "this is reasonably the right district" buffer): use the
     polygon's centroid as the refined anchor coordinate.
  4. Otherwise: keep the GNIS coord and flag as low-confidence.
  5. Re-samples the v2 Tertiary calibrated raster at GNIS vs refined coords
     and reports the decile-rank change.

Outputs:
  - prints a refined ANCHOR_DISTRICTS dict ready to paste into
    src/ai_minerals/regions/_northern_sierra_anchors.py
  - prints a v2-vs-v3 anchor scoring diagnostic so we can see whether the
    refinement moves the v2 gate failures into the top decile

Usage:
    .venv/bin/python scripts/v3_refine_anchor_coordinates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER as REGION
from ai_minerals.regions._northern_sierra_anchors import _GNIS_COORDINATES

SNAP_BUFFER_M = 2_000.0


def refine() -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    """Return (refined_dict, provenance_df)."""
    pits = gpd.read_file(REGION.raw_paths["hydraulic_pits"]).to_crs(REGION.working_crs)
    to_wcrs = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    to_wgs = Transformer.from_crs(REGION.working_crs, "EPSG:4326", always_xy=True)

    refined: dict[str, tuple[float, float]] = {}
    rows: list[dict] = []

    for name, (lon, lat) in _GNIS_COORDINATES.items():
        x, y = to_wcrs.transform(lon, lat)
        pt = Point(x, y)
        pits["_d_boundary"] = pits.geometry.distance(pt)
        pits["_d_centroid"] = pits.geometry.centroid.distance(pt)
        inside = pits[pits.geometry.contains(pt)]
        if len(inside) > 0:
            poly = inside.iloc[0]
            cx, cy = poly.geometry.centroid.x, poly.geometry.centroid.y
            clon, clat = to_wgs.transform(cx, cy)
            method = f"INSIDE {poly['Pit_Name']}"
            snap_m = float(poly["_d_centroid"])
            confidence = "high"
        else:
            nearest = pits.nsmallest(1, "_d_boundary").iloc[0]
            if nearest["_d_boundary"] <= SNAP_BUFFER_M:
                cx, cy = nearest.geometry.centroid.x, nearest.geometry.centroid.y
                clon, clat = to_wgs.transform(cx, cy)
                method = f"-> {nearest['Pit_Name']} (boundary={nearest['_d_boundary']:.0f}m)"
                snap_m = float(nearest["_d_centroid"])
                confidence = "medium"
            else:
                clon, clat = lon, lat
                method = "GNIS fallback (no pit polygon within 2km)"
                snap_m = 0.0
                confidence = "low"

        refined[name] = (round(clon, 6), round(clat, 6))
        rows.append({
            "anchor": name,
            "method": method,
            "snap_dist_m": snap_m,
            "gnis_lon": lon, "gnis_lat": lat,
            "refined_lon": round(clon, 6), "refined_lat": round(clat, 6),
            "confidence": confidence,
        })

    return refined, pd.DataFrame(rows)


def sample_v2_calibrated_at(anchors: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Sample the v2 Tertiary calibrated raster at each anchor coord; return a
    table with p_cal and decile rank per anchor."""
    parquet = (Path(__file__).resolve().parents[1]
               / "data" / "derived" / "northern_sierra_placer"
               / "pop_calibrated_placer_tertiary_250m.parquet")
    if not parquet.exists():
        return pd.DataFrame()
    cal = pd.read_parquet(parquet)
    all_scores = cal["p_cal"].to_numpy()
    rows = []
    for name, (lon, lat) in anchors.items():
        pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(REGION.working_crs).iloc[0]
        dx = cal.x - pt.x; dy = cal.y - pt.y
        i = (dx * dx + dy * dy).idxmin()
        p = float(cal.loc[i, "p_cal"])
        pct_above = float((all_scores >= p).mean())
        decile = min(9, int(pct_above * 10))
        rows.append({"anchor": name, "p_cal": p, "decile": decile})
    return pd.DataFrame(rows)


def main() -> int:
    refined, prov = refine()
    print("=" * 110)
    print("PHASE B.0: anchor coordinate refinement")
    print("=" * 110)
    print()
    print("Per-anchor refinement summary:")
    print(prov.to_string(index=False))
    print()
    print("v3 ANCHOR_DISTRICTS literal:")
    print("{")
    for name, (clon, clat) in refined.items():
        print(f'    "{name}": ({clon}, {clat}),')
    print("}")
    print()

    # v2 vs v3 calibrated-raster scoring at the anchor coordinates.
    v2_at_gnis = sample_v2_calibrated_at(_GNIS_COORDINATES)
    v2_at_refined = sample_v2_calibrated_at(refined)
    if not v2_at_gnis.empty and not v2_at_refined.empty:
        merged = v2_at_gnis.merge(
            v2_at_refined, on="anchor", suffixes=("_v2", "_v3"),
        )
        merged["decile_change"] = merged["decile_v3"] - merged["decile_v2"]
        print("v2 calibrated raster sampled at GNIS (v2) vs refined (v3) coords:")
        print(merged[["anchor", "p_cal_v2", "decile_v2",
                      "p_cal_v3", "decile_v3", "decile_change"]].to_string(index=False))
        n_better = int((merged["decile_change"] < 0).sum())
        n_same = int((merged["decile_change"] == 0).sum())
        n_worse = int((merged["decile_change"] > 0).sum())
        print()
        print(f"Refinement effect on v2 calibrated raster:")
        print(f"  {n_better} anchors moved UP in decile rank (better)")
        print(f"  {n_same} anchors unchanged")
        print(f"  {n_worse} anchors moved DOWN in decile rank (worse)")
        print(f"  net: {-merged['decile_change'].sum()} decile improvement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
