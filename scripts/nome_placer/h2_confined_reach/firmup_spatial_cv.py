"""H2 firm-up: does the lode-distance coarseness gradient hold up spatially?

The H2 headline -- placer gold coarseness fines downstream from the schist-hosted
36a lode (Spearman rho -0.53, p 0.016, n=20) -- is a straight-line association
that treats all 20 placers as independent. Placers in one drainage share a source
area and one downstream-fining gold population, so they are pseudo-replicates: the
naive p over-states significance. This re-tests the gradient with the drainage,
not the placer, as the unit of independence.

Method
  1. Reconstruct the n=20 lode-control set exactly as test_lode_control.py (typed
     alluvial-stream placers with a coarseness rank, nearest 36a schist-hosted lode
     within 15 km, straight-line distance).
  2. Assign each placer to a drainage by tracing it downstream on the H2 D8 network
     (recv_row/recv_col) to its outlet; two placers whose downstream paths merge
     (share any cell) are in the same drainage. B = number of distinct drainages =
     the effective n. A proximity-cluster count (single-linkage at 3 km) is reported
     as a sanity check on that definition.
  3. Cluster bootstrap by drainage (resample whole drainages with replacement) ->
     a rho distribution whose spread reflects drainage-level, not placer-level,
     uncertainty. 95% CI + P(rho<0).
  4. Restricted permutation: shuffle coarseness WITHIN drainages (kills any
     within-drainage gradient, keeps the between-drainage structure) -> one-sided p.
     Free permutation (shuffle all 20) is reported as the naive null for contrast.
  5. Between-drainage Spearman (drainage-median coarseness vs drainage-median
     distance, n=B) and the per-drainage within-drainage Spearman where computable.
  6. Plain verdict: does the negative rho survive once drainages are the unit?

Deterministic (fixed seeds). Run from the repo root:
  .venv/bin/python scripts/nome_placer/h2_confined_reach/firmup_spatial_cv.py
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

OUT = Path("data/derived/nome_placer/h2_confined_reach")
STAGED = Path("data/derived/nome_placer/peninsula_phase2/"
              "peninsula_ardf_placer_lode_3338.geojson")
GEMS = Path("data/raw/dggs_ri2024_7/extracted/pkg/"
            "casadepaga_bedrock_gems_db_wo_stations-open/GM_MapUnitPolys.shp")
MAX_LODE_M = 15_000.0
SUBBASIN_MERGE_M = 5_000.0  # downstream horizon for the tributary-scale sub-basin unit
PROX_CLUSTER_M = 3_000.0   # single-linkage threshold for the sanity-check count
N_BOOT = 5000
N_PERM = 5000


def lode_control_set() -> gpd.GeoDataFrame:
    """The exact n=20 set from test_lode_control.py: typed alluvial placers with a
    coarseness rank, distance to the nearest 36a schist-hosted lode within 15 km."""
    polys = gpd.read_file(GEMS).to_crs(3338)
    mb = polys.total_bounds
    staged = gpd.read_file(STAGED).to_crs(3338)
    lode = staged[staged.deposit_class == "lode"].copy()
    halo = 5000.0
    lode = lode.cx[mb[0] - halo:mb[2] + halo, mb[1] - halo:mb[3] + halo]
    lode = lode[~lode.model_code.astype(str).str.contains("39")]
    ltree = cKDTree(np.column_stack([lode.geometry.x, lode.geometry.y]))

    typed = gpd.read_file(OUT / "placers_typed.geojson").to_crs(3338)
    al = typed[(typed.geol_type == "alluvial-stream")
               & typed.coarseness_rank.notna()].copy()
    al["cls"] = al.coarseness_rank.astype(int)
    d, _ = ltree.query(np.column_stack([al.geometry.x, al.geometry.y]), k=1)
    al["lode_m"] = np.where(d <= MAX_LODE_M, d, np.nan)
    al = al[al.lode_m.notna()].reset_index(drop=True)
    al["n_lodes_used"] = int(len(lode))
    return al


def trace_downstream(r: int, c: int, rr, rc, H: int, W: int, cellsize: float,
                     max_dist_m: float | None = None,
                     max_steps: int = 500_000) -> set[int]:
    """Flat indices (r*W+c) of the downstream path from (r,c).

    Traced to the outlet when max_dist_m is None; capped at max_dist_m of
    along-channel travel otherwise. The cap keeps separate tributaries apart
    (the filled DEM routes whole river systems to a shared coastal sink, so an
    uncapped trace merges everything into a couple of mega-basins)."""
    path: set[int] = set()
    steps = 0
    dist = 0.0
    diag = cellsize * np.sqrt(2.0)
    while True:
        fi = r * W + c
        if fi in path:                 # cycle guard (filled DEM should be acyclic)
            break
        path.add(fi)
        if max_dist_m is not None and dist >= max_dist_m:
            break
        nr, nc = int(rr[r, c]), int(rc[r, c])
        if nr < 0 or nc < 0 or nr >= H or nc >= W:
            break                      # outlet / off-grid
        dist += diag if (nr != r and nc != c) else cellsize
        r, c = nr, nc
        steps += 1
        if steps > max_steps:
            break
    return path


def assign_components(al, transform, H, W, rr, rc, cellsize,
                      max_dist_m: float | None) -> np.ndarray:
    """Component id per placer: components of the 'downstream paths merge' graph,
    with the trace capped at max_dist_m (None = full trace to outlet)."""
    paths = []
    for geom in al.geometry:
        row, col = rowcol(transform, geom.x, geom.y)
        row = int(np.clip(row, 0, H - 1))
        col = int(np.clip(col, 0, W - 1))
        paths.append(trace_downstream(row, col, rr, rc, H, W, cellsize, max_dist_m))
    n = len(paths)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if paths[i] & paths[j]:
                adj[i, j] = adj[j, i] = 1
    _, labels = connected_components(adj, directed=False)
    return labels


def proximity_clusters(al: gpd.GeoDataFrame, thresh_m: float) -> int:
    """Single-linkage cluster count at thresh_m (sanity check on the drainage count)."""
    xy = np.column_stack([al.geometry.x, al.geometry.y])
    n = len(xy)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(xy[i] - xy[j])) <= thresh_m:
                adj[i, j] = adj[j, i] = 1
    n_comp, _ = connected_components(adj, directed=False)
    return int(n_comp)


def safe_spearman(cls, dist):
    if len(np.unique(cls)) < 2 or len(np.unique(dist)) < 2:
        return np.nan
    return float(spearmanr(cls, dist)[0])


def one_sided_p(perm: np.ndarray, obs: float) -> float:
    """How often a permuted rho is as negative as the observed (negative) rho."""
    perm = perm[~np.isnan(perm)]
    return float(np.mean(perm <= obs))


def perm_within(cls, dist, groups, rng, n) -> np.ndarray:
    """Shuffle coarseness within each component, recompute the pooled Spearman."""
    out = []
    for _ in range(n):
        perm = cls.copy()
        for ix in groups.values():
            perm[ix] = rng.permutation(cls[ix])
        out.append(safe_spearman(perm, dist))
    return np.array(out)


def main() -> None:
    al = lode_control_set()
    cls = al.cls.to_numpy()
    dist = al.lode_m.to_numpy()
    obs_rho, obs_p = spearmanr(cls, dist)
    obs_rho = float(obs_rho)

    with rasterio.open(OUT / "recv_row.tif") as src:
        transform, H, W = src.transform, src.height, src.width
        cellsize = abs(transform.a)
        rr = src.read(1)
    rc = rasterio.open(OUT / "recv_col.tif").read(1)

    # Two drainage scales: whole river system (full trace) and tributary sub-basin
    # (trace capped at SUBBASIN_MERGE_M). The sub-basin is the independence unit.
    major = assign_components(al, transform, H, W, rr, rc, cellsize, None)
    sub = assign_components(al, transform, H, W, rr, rc, cellsize, SUBBASIN_MERGE_M)
    al["major"], al["sub"] = major, sub
    g_major = {int(b): np.where(major == b)[0] for b in np.unique(major)}
    g_sub = {int(b): np.where(sub == b)[0] for b in np.unique(sub)}
    B_sub = len(g_sub)
    sub_sizes = sorted((len(ix) for ix in g_sub.values()), reverse=True)

    # within each MAJOR drainage: does the gradient reproduce internally?
    within_major = []
    for b, ix in sorted(g_major.items()):
        if len(ix) >= 2 and len(np.unique(cls[ix])) >= 2:
            within_major.append({"drainage": int(b), "n": int(len(ix)),
                                 "rho": round(safe_spearman(cls[ix], dist[ix]), 3)})

    # cluster bootstrap by SUB-BASIN (resample whole sub-basins with replacement)
    rng = np.random.default_rng(0)
    sub_ids = np.array(list(g_sub))
    boot = []
    for _ in range(N_BOOT):
        chosen = rng.choice(sub_ids, size=B_sub, replace=True)
        idx = np.concatenate([g_sub[int(b)] for b in chosen])
        r = safe_spearman(cls[idx], dist[idx])
        if not np.isnan(r):
            boot.append(r)
    boot = np.array(boot)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    # permutations: within major drainage, within sub-basin, and free (naive null)
    rng = np.random.default_rng(1)
    p_major = one_sided_p(perm_within(cls, dist, g_major, rng, N_PERM), obs_rho)
    p_sub = one_sided_p(perm_within(cls, dist, g_sub, rng, N_PERM), obs_rho)
    free = np.array([safe_spearman(rng.permutation(cls), dist) for _ in range(N_PERM)])
    p_free = one_sided_p(free, obs_rho)
    movable_major = int(sum(len(ix) for ix in g_major.values() if len(ix) >= 2))
    movable_sub = int(sum(len(ix) for ix in g_sub.values() if len(ix) >= 2))

    bg = al.groupby("sub").agg(cls_med=("cls", "median"), dist_med=("lode_m", "median"))
    bgc, bgd = bg.cls_med.to_numpy(), bg.dist_med.to_numpy()
    if len(np.unique(bgc)) >= 2 and len(np.unique(bgd)) >= 2:
        _rb, _pb = spearmanr(bgc, bgd)
        rho_between, p_between = float(_rb), float(_pb)
    else:
        rho_between = p_between = float("nan")

    # verdict (three-way, scale-aware)
    both_neg = bool(within_major) and all(d["rho"] < 0 for d in within_major)
    direction_consistent = (both_neg and rho_between < 0 and float(np.median(boot)) < 0)
    strict_sig = (ci[1] < 0.0) and (p_sub < 0.05)
    rhos_str = ", ".join(f"{d['rho']:+.2f}" for d in within_major)
    boot_med = float(np.median(boot))
    if direction_consistent and strict_sig:
        verdict = ("The gradient survives the spatial structure: the direction holds "
                   "at every scale and the strict within-sub-basin test stays "
                   f"significant (permutation p {p_sub:.3f}, bootstrap 95% CI "
                   f"{ci[0]:+.2f} to {ci[1]:+.2f}).")
    elif direction_consistent:
        verdict = (
            "Direction holds, significance is marginal. The coarseness-vs-"
            "distance gradient is negative at every scale: naive rho "
            f"{obs_rho:+.2f}, between sub-basins rho {rho_between:+.2f} "
            f"(n={B_sub}, p {p_between:.2f}), and it reproduces independently inside "
            f"both major river systems (rho {rhos_str}); {np.mean(boot < 0):.0%} of "
            "sub-basin bootstrap resamples are negative. But once the ~"
            f"{B_sub} sub-basins are the unit of independence the n=20 straight-line "
            f"p ({obs_p:.3f}) was inflated by pseudo-replication: the cluster "
            f"bootstrap 95% CI grazes zero ({ci[0]:+.2f} to {ci[1]:+.2f}) and the "
            f"within-sub-basin permutation is not significant (p {p_sub:.2f}). The "
            "local-source signal is real at the drainage scale (sub-basins nearer the "
            "schist-hosted lode carry coarser gold), but the strict within-drainage "
            "downstream-fining gradient is underpowered, mostly because the confined-"
            "upland clip removes the distal/fine end and class 1 is nearly empty. "
            "Corroborated in direction, not conclusively significant after the "
            "spatial correction.")
    else:
        verdict = ("Once sub-basins are the unit, the gradient does not hold: the n=20 "
                   "straight-line significance was an artifact of pseudo-replication.")

    res = {
        "set": {"n_placers": int(len(al)), "n_lodes_used": int(al.n_lodes_used.iloc[0]),
                "class_counts": {int(k): int(v)
                                 for k, v in al.cls.value_counts().sort_index().items()}},
        "naive": {"spearman_rho": round(obs_rho, 3), "spearman_p": round(float(obs_p), 4),
                  "note": "treats all 20 placers as independent (the headline number)"},
        "effective_independence": {
            "n_major_river_systems_full_trace": len(g_major),
            "n_subbasins_5km_merge": B_sub,
            "n_localities_3km_proximity": proximity_clusters(al, PROX_CLUSTER_M),
            "placers_per_subbasin_desc": sub_sizes,
            "largest_subbasin_n": int(sub_sizes[0]),
            "note": "sub-basin = downstream D8 paths merge within "
                    f"{int(SUBBASIN_MERGE_M)} m; major = full trace to outlet "
                    "(coarse: the filled DEM routes river systems to a shared coastal "
                    "sink); 3 km = single-linkage proximity sanity check",
        },
        "within_major_drainage": within_major,
        "cluster_bootstrap_by_subbasin": {
            "n_subbasins": B_sub, "n_resamples": int(len(boot)),
            "median_rho": round(float(np.median(boot)), 3),
            "ci95": [round(ci[0], 3), round(ci[1], 3)],
            "frac_rho_negative": round(float(np.mean(boot < 0)), 3),
            "excludes_zero": bool(ci[1] < 0.0),
        },
        "permutation_one_sided": {
            "p_within_major_drainage": round(p_major, 4), "movable_major": movable_major,
            "p_within_subbasin": round(p_sub, 4), "movable_subbasin": movable_sub,
            "p_free_naive": round(p_free, 4),
            "note": "within-X permutation shuffles coarseness inside each X and "
                    "recomputes the pooled rho; significant => the gradient is a "
                    "within-X (downstream-fining) effect, not a between-X artifact",
        },
        "between_subbasin": {
            "spearman_rho_subbasin_medians": (round(rho_between, 3)
                                              if not np.isnan(rho_between) else None),
            "spearman_p": (round(p_between, 4) if not np.isnan(p_between) else None),
            "n_subbasins": B_sub,
        },
        "verdict": verdict,
        "standing_limits": [
            "Confined-upland clip removes the wide valleys and coastal plain where "
            "the distal/fine gold ends up, so class 1 (fine) is nearly empty (n=1).",
            "Coarseness is an ordinal mined from century-old narratives that favour "
            "nuggets over fines.",
            "n=20 before any spatial correction; the effective number of independent "
            "drainages is smaller still.",
        ],
    }
    (OUT / "lode_gradient_spatial_firmup.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
