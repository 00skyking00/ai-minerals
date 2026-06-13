"""Identify BCGT mining-district clusters from the post-2010 Cu+ point cloud.

DBSCAN over the per-cell post-2010 Cu+ counts in the BCGT working
area, then map clusters to named mining districts using
hand-validated lat/lon proximity to public references (Red Chris,
KSM, Brucejack, etc.). Writes the resulting centroid table to
`src/ai_minerals/regions/bcgt.py` as a constant the B.2 benchmark
infrastructure reads from.

The point cloud has only 47 post-2010 Cu+ cells across the full
BCGT working area, so cluster sizes are small (2-17 cells). Six
clusters at `eps=12 km, min_samples=2` cover the named districts
that have any post-2010 sampling.

Run once per overlay refresh; the output is checked in to
`src/ai_minerals/regions/bcgt.py` as `BCGT_B2_CLUSTERS`.

Output: data/derived/bcgt/b2_clusters_centroids.json
        data/derived/bcgt/fig_b2_cluster_map.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from sklearn.cluster import DBSCAN

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "data/derived/bcgt/bcgs_pre_post_2010_overlay.parquet"
OUT_JSON = REPO / "data/derived/bcgt/b2_clusters_centroids.json"
OUT_MAP = REPO / "data/derived/bcgt/fig_b2_cluster_map.png"

DBSCAN_EPS_M = 12000.0
DBSCAN_MIN_SAMPLES = 2

# Hand-validated district names mapped by approximate lat/lon
# proximity to public mining-district references. Cluster centroid
# lat/lon ranges are wide bands; each cluster from DBSCAN is matched
# to whichever named district lies inside its band.
NAMED_DISTRICT_CENTROIDS = {
    # (lat, lon) reference points from public sources
    "KSM": (56.55, -130.75),         # Kerr-Sulphurets-Mitchell porphyry
    "Brucejack": (56.47, -130.20),   # Pretium Brucejack gold-silver
    "Red_Chris": (57.69, -129.83),   # Imperial Red Chris porphyry
    "Schaft_Creek": (57.35, -131.05),
    "Galore_Creek": (57.07, -131.40),
    "Snip": (56.65, -131.30),
    "Eskay_Creek": (56.62, -130.43),
    "Red_Mountain": (55.83, -129.51),
}

# Centroid-to-district matching tolerance (degrees lat/lon).
# Match cluster to closest district within this tolerance; otherwise
# keep as descriptive name based on coordinates.
DISTRICT_MATCH_TOL_DEG = 0.30


def haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Quick degree-space distance for centroid matching."""
    return float(np.hypot(lat1 - lat2, (lon1 - lon2) * np.cos(np.radians(lat1))))


def match_cluster_to_district(lat: float, lon: float) -> str:
    """Return the closest named district within tolerance, or a
    descriptive coordinate-based name."""
    best_name = None
    best_dist = float("inf")
    for name, (ref_lat, ref_lon) in NAMED_DISTRICT_CENTROIDS.items():
        d = haversine_deg(lat, lon, ref_lat, ref_lon)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist <= DISTRICT_MATCH_TOL_DEG:
        return best_name
    # Fall back to a descriptive name pinned to half-degree bins
    return f"UnnamedCluster_lat{lat:.1f}_lon{lon:.1f}".replace(".", "p").replace("-", "neg")


def main() -> int:
    df = pd.read_parquet(OVERLAY)
    post = df[df["post_2010_cu_positive_n_holes"] > 0].copy()
    print(f"{len(post)} post-2010 Cu+ cells in BCGT working area")

    db = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES)
    db.fit(post[["x", "y"]].values)
    post["cluster"] = db.labels_

    transformer = pyproj.Transformer.from_crs(
        "EPSG:3005", "EPSG:4326", always_xy=True,
    )

    clusters = {}
    used_district_names = set()
    for cid in sorted(post["cluster"].unique()):
        if cid == -1:
            continue
        sub = post[post["cluster"] == cid]
        x_c = float(sub["x"].mean())
        y_c = float(sub["y"].mean())
        lon, lat = transformer.transform(x_c, y_c)
        row_c = int(sub["row"].mean().round())
        col_c = int(sub["col"].mean().round())
        n_cells = len(sub)
        n_holes = int(sub["post_2010_cu_positive_n_holes"].sum())
        max_cu = float(sub["post_2010_max_cu_ppm"].max())

        district_name = match_cluster_to_district(lat, lon)
        # Deduplicate (two clusters could match the same district name)
        original = district_name
        suffix = 2
        while district_name in used_district_names:
            district_name = f"{original}_{suffix}"
            suffix += 1
        used_district_names.add(district_name)

        clusters[district_name] = {
            "cluster_id": int(cid),
            "center_row": row_c,
            "center_col": col_c,
            "center_x_m": x_c,
            "center_y_m": y_c,
            "center_lat": float(lat),
            "center_lon": float(lon),
            "n_post2010_cuplus_cells": n_cells,
            "n_post2010_cuplus_holes": n_holes,
            "max_cu_ppm": max_cu,
        }
        print(f"  {district_name:>30s}: "
              f"n_cells={n_cells:2d}, n_holes={n_holes:3d}, "
              f"row~{row_c:3d}, col~{col_c:3d}, "
              f"lat={lat:.3f}, lon={lon:.3f}, max_cu={max_cu:5.0f}ppm")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "dbscan_eps_m": DBSCAN_EPS_M,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
            "n_cells_total": len(post),
            "n_clusters": len(clusters),
            "clusters": clusters,
        }, f, indent=2)
    print(f"wrote {OUT_JSON}")

    # Map: plot all post-2010 Cu+ cells colored by cluster, with district labels
    fig, ax = plt.subplots(figsize=(7, 9))
    palette = plt.get_cmap("tab10")
    for i, (name, info) in enumerate(clusters.items()):
        sub = post[post["cluster"] == info["cluster_id"]]
        ax.scatter(sub["x"] / 1000.0, sub["y"] / 1000.0,
                   c=[palette(i % 10)], s=80, edgecolor="black",
                   linewidth=0.5, label=f"{name} (n={len(sub)})")
        ax.annotate(name.replace("_", " "),
                    (info["center_x_m"] / 1000.0, info["center_y_m"] / 1000.0),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    noise = post[post["cluster"] == -1]
    if len(noise) > 0:
        ax.scatter(noise["x"] / 1000.0, noise["y"] / 1000.0,
                   c="lightgray", s=40, marker="x", label=f"noise (n={len(noise)})")
    ax.set_xlabel("EPSG:3005 x (km)")
    ax.set_ylabel("EPSG:3005 y (km)")
    ax.set_title(
        f"BCGT post-2010 Cu+ clusters (DBSCAN eps={DBSCAN_EPS_M/1000:.0f} km, "
        f"min_samples={DBSCAN_MIN_SAMPLES})"
    )
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_MAP, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_MAP}")

    print()
    print("Copy this into src/ai_minerals/regions/bcgt.py as BCGT_B2_CLUSTERS:")
    print()
    print("BCGT_B2_CLUSTERS = {")
    for name, info in clusters.items():
        print(f"    {name!r}: {{")
        print(f"        'center_row': {info['center_row']},")
        print(f"        'center_col': {info['center_col']},")
        print(f"        'center_lat': {info['center_lat']:.4f},")
        print(f"        'center_lon': {info['center_lon']:.4f},")
        print(f"        'n_post2010_cuplus_cells': {info['n_post2010_cuplus_cells']},")
        print(f"        'n_post2010_cuplus_holes': {info['n_post2010_cuplus_holes']},")
        print(f"    }},")
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
