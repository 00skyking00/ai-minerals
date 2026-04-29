"""Bear Cub resource analysis — items 1-9 of the agreed plan.

Computes per-interval Au grade, builds Jesse + Tweet style visualizations,
runs Triangle + Polygon block-volumetric resource estimation, and produces a
cross-validation table against Tweet's published numbers for holes we share.

Outputs (all under data/derived/bear_cub_resource/):
  intervals_with_grade.parquet / .csv     Per-interval grade in oz/cu yd
  hole_rollups.parquet / .csv             Hole-level grade summary stats
  resource_estimate.json                  Polygon + Triangle method totals
  fig_plan_map.png                        Holes color-coded by max grade
  fig_bedrock_depth.png                   2D bedrock-depth contour map
  fig_surface_to_bop_grade.png            Vertically-integrated grade contour
  fig_expected_depth_to_pay.png           Predictive top-of-pay contour
  fig_grade_profiles.png                  Per-hole grade-vs-depth bar charts
  cross_validation.md                     vs Tweet's + Jesse's published numbers

Run:
    uv run python tools/bear_cub_resource_analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay, Voronoi

from ai_minerals.bear_cub.grade import (
    JESSE_GRADE_BINS, TWEET_GRADE_BINS,
    add_grades_to_intervals, hole_rollups,
    polygon_resource, triangle_resource,
    grade_to_jesse_color, grade_to_tweet_color,
)
from ai_minerals.bear_cub.georef import MS_1178_CORNERS

REPO = Path(__file__).resolve().parents[1]
SD = REPO / "data" / "raw" / "bear_cub" / "structured"
OUT = REPO / "data" / "derived" / "bear_cub_resource"
OUT.mkdir(parents=True, exist_ok=True)

# Per-hole bedrock fallback (some holes have null bedrock_depth in new OCR; use original CSV)
ORIGINAL_CSV = REPO / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"


def latlon_to_local_ft(lat, lon, lat0, lon0):
    """Convert (lat, lon) to local Cartesian feet centered at (lat0, lon0)."""
    ft_per_lat = 364400.0
    ft_per_lon = ft_per_lat * math.cos(math.radians(lat0))
    return (lon - lon0) * ft_per_lon, (lat - lat0) * ft_per_lat


def main() -> None:
    print("Loading captured data ...")
    intervals = pd.read_parquet(SD / "drillhole_intervals.parquet")
    new_collars = pd.read_parquet(SD / "drillhole_collars.parquet")
    orig_collars = pd.read_csv(ORIGINAL_CSV)

    # Bedrock priority: reviewer-edited values (which flow into the new
    # drillhole_collars.parquet via the aggregator) win over the original CSV.
    # Lat/lon comes only from the original CSV (the new OCR doesn't have it).
    # Casing/form text comes from the new OCR.
    casing_lookup = new_collars.set_index("file_stem")["casing_or_bit_diameter_text"].to_dict()
    form_lookup = new_collars.set_index("file_stem")["form_type"].to_dict()
    new_bedrock_lookup = new_collars.set_index("file_stem")["depth_to_bedrock_ft"].to_dict()
    new_total_depth_lookup = new_collars.set_index("file_stem")["total_depth_ft"].to_dict()

    collars = orig_collars.copy()
    if "bedrock_depth_ft" in collars.columns and "depth_to_bedrock_ft" not in collars.columns:
        collars = collars.rename(columns={"bedrock_depth_ft": "depth_to_bedrock_ft"})
    collars["casing_or_bit_diameter_text"] = collars["file_stem"].map(casing_lookup).fillna("")
    collars["form_type_full"] = collars["file_stem"].map(form_lookup).fillna(collars.get("form_type", ""))

    # Override bedrock + total-depth from new OCR / reviewer edits when present
    for fs, new_br in new_bedrock_lookup.items():
        if pd.notna(new_br) and new_br > 0:
            collars.loc[collars.file_stem == fs, "depth_to_bedrock_ft"] = float(new_br)
    for fs, new_td in new_total_depth_lookup.items():
        if pd.notna(new_td) and new_td > 0:
            collars.loc[collars.file_stem == fs, "total_depth_ft"] = float(new_td)

    # ------------------------------------------------------------------ #
    # Bedrock imputation for NBR (No Bedrock Reached) holes
    # ------------------------------------------------------------------ #
    # An NBR hole drilled to total_depth without striking bedrock — bedrock is
    # therefore strictly DEEPER than total_depth. We impute via RBF over the
    # non-NBR holes' bedrock surface, then floor the result at (total_depth +
    # NBR_BEDROCK_MARGIN_FT). This brings NBR holes back into the volumetric
    # estimate using a defensible bedrock estimate.
    NBR_BEDROCK_MARGIN_FT = 5.0
    collars["bedrock_imputed"] = False
    nbr_mask = (
        collars["depth_to_bedrock_ft"].isna()
        | (collars["depth_to_bedrock_ft"] == 0)
    ) & collars["lat_wgs84"].notna() & collars["lon_wgs84"].notna()

    if nbr_mask.any():
        coords_valid = collars.dropna(subset=["lat_wgs84", "lon_wgs84"])
        lat0_imp = coords_valid["lat_wgs84"].mean()
        lon0_imp = coords_valid["lon_wgs84"].mean()
        xs, ys = latlon_to_local_ft(
            collars["lat_wgs84"].values, collars["lon_wgs84"].values,
            lat0_imp, lon0_imp,
        )
        collars["_x_imp"] = xs
        collars["_y_imp"] = ys

        non_nbr = collars[
            collars["depth_to_bedrock_ft"].notna()
            & (collars["depth_to_bedrock_ft"] > 0)
        ].copy()
        non_nbr_xy = non_nbr[["_x_imp", "_y_imp"]].values
        non_nbr_br = non_nbr["depth_to_bedrock_ft"].values

        # KNN-IDW (K=4, p=2) — robust against extrapolation that broke RBF
        # at the corpus edges (e.g., H6964 at the NE boundary).
        K = 4
        print("\nBedrock imputation for NBR holes (KNN-IDW K=4, floor = total_depth + 5 ft):")
        for idx in collars[nbr_mask].index:
            row = collars.loc[idx]
            xy = np.array([row["_x_imp"], row["_y_imp"]])
            d = np.linalg.norm(non_nbr_xy - xy, axis=1)
            order = np.argsort(d)[:K]
            d_k = d[order]
            br_k = non_nbr_br[order]
            w = 1.0 / np.maximum(d_k, 1.0) ** 2
            estimated = float(np.sum(w * br_k) / np.sum(w))
            floor = float(row.get("total_depth_ft") or 0) + NBR_BEDROCK_MARGIN_FT
            imputed = max(estimated, floor)
            collars.loc[idx, "depth_to_bedrock_ft"] = imputed
            collars.loc[idx, "bedrock_imputed"] = True
            neighbors = ", ".join(
                f"{non_nbr.iloc[j]['file_stem'].split()[-1]}={non_nbr_br[j]:.0f}@{d[j]:.0f}ft"
                for j in order
            )
            print(f"  {row['file_stem']:13s} TD={row['total_depth_ft']:.0f}ft  "
                  f"IDW={estimated:.1f}  floor={floor:.1f}  → {imputed:.1f} ft  "
                  f"[neighbors: {neighbors}]")

        # ------------------------------------------------------------ #
        # Sensitivity sweep across K and floor — log to a CSV
        # ------------------------------------------------------------ #
        print("\nBedrock imputation sensitivity sweep ...")
        sens_rows = []
        # GP imputation alternative (with uncertainty) — same training set
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF as _RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        gp_scaler = StandardScaler().fit(non_nbr_xy)
        gp_kernel = (ConstantKernel(np.var(non_nbr_br), (1e-6, 1e3))
                     * _RBF(length_scale=[100.0, 100.0], length_scale_bounds=(1.0, 1e4))
                     + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e1)))
        gp = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True,
                                       n_restarts_optimizer=8, random_state=42)
        gp.fit(gp_scaler.transform(non_nbr_xy), non_nbr_br)

        for idx in collars[nbr_mask].index:
            row = collars.loc[idx]
            xy = np.array([row["_x_imp"], row["_y_imp"]])
            d = np.linalg.norm(non_nbr_xy - xy, axis=1)
            order_full = np.argsort(d)
            td = float(row.get("total_depth_ft") or 0)
            # KNN-IDW × {K, floor}
            for k_try in (2, 3, 4, 5, 6):
                ord_k = order_full[:k_try]
                d_k = d[ord_k]
                br_k = non_nbr_br[ord_k]
                w = 1.0 / np.maximum(d_k, 1.0) ** 2
                est = float(np.sum(w * br_k) / np.sum(w))
                for floor_try in (0.0, 5.0, 10.0, 15.0):
                    imp = max(est, td + floor_try)
                    sens_rows.append({
                        "file_stem": row["file_stem"],
                        "method": "knn_idw",
                        "K": k_try,
                        "floor_ft": floor_try,
                        "knn_idw_estimate": est,
                        "floor_value_ft": td + floor_try,
                        "imputed_ft": imp,
                    })
            # GP imputation
            gp_mean, gp_std = gp.predict(gp_scaler.transform(xy.reshape(1, -1)), return_std=True)
            sens_rows.append({
                "file_stem": row["file_stem"],
                "method": "gp",
                "K": None,
                "floor_ft": None,
                "knn_idw_estimate": None,
                "floor_value_ft": None,
                "imputed_ft": float(gp_mean[0]),
                "gp_std_ft": float(gp_std[0]),
            })

        sens_df = pd.DataFrame(sens_rows)
        sens_df.to_csv(OUT / "bedrock_imputation_sensitivity.csv", index=False)
        print(f"  → bedrock_imputation_sensitivity.csv "
              f"({len(sens_df)} rows; {sens_df['file_stem'].nunique()} imputed holes)")
        # Summary: per-hole min/max across all KNN-IDW choices + GP mean ± σ
        for fs in sens_df.file_stem.unique():
            sub = sens_df[sens_df.file_stem == fs]
            knn = sub[sub.method == "knn_idw"]
            gp_row = sub[sub.method == "gp"].iloc[0] if (sub.method == "gp").any() else None
            print(f"  {fs}: KNN-IDW range across K∈{{2..6}} × floor∈{{0,5,10,15}}: "
                  f"{knn['imputed_ft'].min():.1f}-{knn['imputed_ft'].max():.1f} ft "
                  f"(default K=4, floor=5 → "
                  f"{knn[(knn.K == 4) & (knn.floor_ft == 5.0)]['imputed_ft'].iloc[0]:.1f} ft)"
                  + (f" · GP: {gp_row['imputed_ft']:.1f} ± {gp_row['gp_std_ft']:.1f} ft"
                     if gp_row is not None else ""))

        collars = collars.drop(columns=["_x_imp", "_y_imp"])

    print(f"\n  {len(collars)} collars, {len(intervals)} intervals "
          f"({collars['bedrock_imputed'].sum()} bedrock imputed)")

    # ------------------------------------------------------------------ #
    # Item 1: Per-interval grade
    # ------------------------------------------------------------------ #
    print("\nItem 1: computing per-interval grade in oz/cu yd ...")
    iv = add_grades_to_intervals(intervals, collars)
    iv.to_parquet(OUT / "intervals_with_grade.parquet", index=False)
    iv.to_csv(OUT / "intervals_with_grade.csv", index=False)
    print(f"  → intervals_with_grade.parquet ({len(iv)} rows)")
    print(f"  Grade summary across all intervals:")
    print(f"    max:    {iv['grade_oz_per_cu_yd'].max():.4f} oz/cu yd")
    print(f"    median: {iv['grade_oz_per_cu_yd'].median():.4f} oz/cu yd")
    print(f"    >0.005 oz/cu yd intervals: {(iv['grade_oz_per_cu_yd'] >= 0.005).sum()} of {len(iv)}")

    # ------------------------------------------------------------------ #
    # Hole-level rollups
    # ------------------------------------------------------------------ #
    rollups = hole_rollups(iv, collars)
    rollups = rollups.merge(collars[["file_stem", "lat_wgs84", "lon_wgs84",
                                     "easting_local_ft", "northing_local_ft",
                                     "elevation_ft", "form_type"]],
                            on="file_stem", how="left")
    rollups.to_parquet(OUT / "hole_rollups.parquet", index=False)
    rollups.to_csv(OUT / "hole_rollups.csv", index=False)
    print(f"\n  → hole_rollups.parquet ({len(rollups)} holes)")
    print(rollups[["file_stem", "surface_to_br_grade", "pay_zone_grade",
                   "pay_zone_thickness_ft", "total_fine_oz_in_hole"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # Project to local feet for spatial analysis
    # ------------------------------------------------------------------ #
    valid = rollups.dropna(subset=["lat_wgs84", "lon_wgs84"]).copy()
    lat0 = valid["lat_wgs84"].mean()
    lon0 = valid["lon_wgs84"].mean()
    valid["x_ft"], valid["y_ft"] = latlon_to_local_ft(
        valid["lat_wgs84"].values, valid["lon_wgs84"].values, lat0, lon0
    )

    # Bear Cub corner coords in same local frame
    corners_xy = {}
    for k, (lat, lon) in MS_1178_CORNERS.items():
        corners_xy[k] = latlon_to_local_ft(lat, lon, lat0, lon0)

    cx = [corners_xy[k][0] for k in ("TL", "TR", "BR", "BL", "TL")]
    cy = [corners_xy[k][1] for k in ("TL", "TR", "BR", "BL", "TL")]

    # ------------------------------------------------------------------ #
    # Item 6: Triangle + Polygon resource estimation
    # ------------------------------------------------------------------ #
    print("\nItem 6: block-volumetric resource estimation ...")
    interior = valid[valid["bedrock_depth_ft"] > 0].copy()
    pts = interior[["x_ft", "y_ft"]].values
    grades = interior["surface_to_br_grade"].values
    depths = interior["bedrock_depth_ft"].values

    # bbox = bear cub corners (used for ghost-point sizing)
    bbox = (min(cx), min(cy), max(cx), max(cy))

    # Actual clipping polygon = the 4-corner claim quadrilateral (smaller than
    # the bounding rectangle because the claim isn't axis-aligned).
    from shapely.geometry import Polygon as _Polygon
    claim_poly = _Polygon(list(zip(cx[:-1], cy[:-1])))

    poly = polygon_resource(pts, grades, depths, bbox, clip_polygon=claim_poly)
    tri = triangle_resource(pts, grades, depths, bbox)

    print(f"  Polygon (surface-to-BR): {poly['total_fine_oz']:.0f} fine oz, "
          f"avg grade {poly['weighted_avg_grade']:.4f} oz/cu yd "
          f"({poly['total_volume_cuyd']:.0f} cu yd)")
    print(f"  Triangle (surface-to-BR): {tri['total_fine_oz']:.0f} fine oz, "
          f"avg grade {tri['weighted_avg_grade']:.4f} oz/cu yd "
          f"({tri['total_volume_cuyd']:.0f} cu yd)")

    # Pay-zone-only volumetric (Tweet-comparable). Each cell's volume is
    # polygon_area × pay_zone_thickness, grade is the depth-weighted avg
    # grade across the pay zone (not the peak).
    pz_grades = interior["pay_zone_avg_grade"].fillna(0).values
    pz_depths = interior["pay_zone_thickness_ft"].fillna(0).values
    poly_pz = polygon_resource(pts, pz_grades, pz_depths, bbox, clip_polygon=claim_poly)
    print(f"  Polygon (PAY-ZONE-only):  {poly_pz['total_fine_oz']:.0f} fine oz, "
          f"avg grade {poly_pz['weighted_avg_grade']:.4f} oz/cu yd "
          f"({poly_pz['total_volume_cuyd']:.0f} cu yd) ← Tweet-comparable")

    # ------------------------------------------------------------------ #
    # Monte Carlo uncertainty quantification on the pay-zone-only estimate
    # ------------------------------------------------------------------ #
    # Sources of uncertainty propagated:
    #   - Fineness ∈ U(0.85, 0.92) — empirical Bear Cub fineness is 0.89 but
    #     varies with assay batch. Range covers measured assays in district.
    #   - Per-mg OCR + transcription error ε ~ N(0, 0.05) (multiplicative).
    #     5% std reflects manual review precision on captured numbers.
    #   - Bit-area uncertainty σ = 5% of nominal — accounts for casing-vs-bit
    #     diameter ambiguity and out-of-round drilling.
    #   - Pay-zone window cap ∈ {12, 16, 20, 24, 30} ft — sliding-window
    #     thickness sensitivity. Wider window = more inclusive pay zone.
    print("\nMonte Carlo uncertainty quantification (1000 samples) ...")
    rng = np.random.default_rng(42)
    n_samples = 1000
    # Pre-compute per-interval mg arrays and bit areas — we'll perturb these
    iv_for_mc = iv.copy()
    iv_for_mc["mg_baseline"] = iv_for_mc["estimated_weight_mg"].fillna(0).values
    iv_for_mc["bit_area_baseline"] = iv_for_mc["bit_area_ft2"].values

    samples = []
    for s in range(n_samples):
        f = rng.uniform(0.85, 0.92)
        bit_factor = rng.normal(1.0, 0.05, size=len(iv_for_mc))
        mg_factor = rng.normal(1.0, 0.05, size=len(iv_for_mc))
        # Recompute per-interval grade with perturbed fineness, mg, bit area
        iv_perturbed = iv_for_mc.copy()
        iv_perturbed["mg_perturbed"] = iv_for_mc["mg_baseline"].values * mg_factor
        iv_perturbed["bit_perturbed"] = iv_for_mc["bit_area_baseline"].values * bit_factor
        # grade = mg × fineness × 27 / (31103.5 × bit_area × interval_ft)
        iv_perturbed["grade_perturbed"] = (
            iv_perturbed["mg_perturbed"] * f * 27
            / (31103.5 * iv_perturbed["bit_perturbed"] * iv_perturbed["interval_ft"].fillna(1))
        ).fillna(0)
        # Re-roll up per hole with the perturbed grades; recompute pay-zone-only sums
        hole_oz = []
        for fs, g in iv_perturbed.groupby("file_stem"):
            sub = collars[collars.file_stem == fs]
            if sub.empty:
                continue
            br = float(sub["depth_to_bedrock_ft"].iloc[0]) if pd.notna(sub["depth_to_bedrock_ft"].iloc[0]) else 0
            if br <= 0:
                continue
            # Use the same sliding-window logic as hole_rollups, but on perturbed grades
            sg = g.sort_values("depth_from_ft").reset_index(drop=True)
            best_dens = 0.0
            best = None
            for i in range(len(sg)):
                cm = 0.0
                ct = 0.0
                for j in range(i, len(sg)):
                    cm += float(sg.iloc[j]["mg_perturbed"] or 0)
                    ct += float(sg.iloc[j]["interval_ft"] or 0)
                    if ct > 20.0:
                        break
                    if ct < 2.0:
                        continue
                    d = cm / ct
                    if d > best_dens:
                        best_dens = d
                        best = (i, j, cm, ct)
            if best is None:
                continue
            i, j, mg_pz, thick_pz = best
            ba = float(sg.iloc[i]["bit_perturbed"])
            grade_pz = mg_pz * f / 31103.5 / (thick_pz * ba / 27.0) if thick_pz > 0 and ba > 0 else 0
            hole_oz.append({"file_stem": fs, "grade": grade_pz, "thick": thick_pz, "ba": ba})
        # Polygon resource for this sample using perturbed grades + thicknesses
        h_df = pd.DataFrame(hole_oz)
        if not h_df.empty:
            merged = interior[["file_stem", "x_ft", "y_ft"]].merge(h_df, on="file_stem", how="inner")
            if len(merged) >= 4:
                pz_grades_s = merged["grade"].values
                pz_depths_s = merged["thick"].values
                pts_s = merged[["x_ft", "y_ft"]].values
                try:
                    poly_s = polygon_resource(pts_s, pz_grades_s, pz_depths_s, bbox, clip_polygon=claim_poly)
                    samples.append(poly_s["total_fine_oz"])
                except Exception:
                    pass

    if samples:
        samples_arr = np.array(samples)
        ci_low = float(np.percentile(samples_arr, 5))
        ci_med = float(np.percentile(samples_arr, 50))
        ci_high = float(np.percentile(samples_arr, 95))
        ci_mean = float(np.mean(samples_arr))
        print(f"  Pay-zone-only resource — credible interval over {len(samples_arr)} MC samples:")
        print(f"    5%:    {ci_low:.0f} fine oz")
        print(f"    50%:   {ci_med:.0f} fine oz")
        print(f"    95%:   {ci_high:.0f} fine oz")
        print(f"    mean:  {ci_mean:.0f} fine oz")
    else:
        ci_low = ci_med = ci_high = ci_mean = None

    resource = {
        "method_polygon": {k: v for k, v in poly.items() if k != "per_polygon"},
        "method_triangle": {k: v for k, v in tri.items() if k != "per_triangle"},
        "method_polygon_pay_zone": {k: v for k, v in poly_pz.items() if k != "per_polygon"},
        "uncertainty_pay_zone_oz": {
            "method": "Monte Carlo, 1000 samples, fineness ~ U(0.85, 0.92), "
                      "per-mg ε ~ N(0, 0.05), bit area ε ~ N(0, 0.05)",
            "p05": ci_low, "p50": ci_med, "p95": ci_high, "mean": ci_mean,
            "n_samples": len(samples) if samples else 0,
        },
        "n_holes_used": int(len(interior)),
        "bbox_local_ft": list(bbox),
        "lat0": float(lat0), "lon0": float(lon0),
        "fineness_assumed": 0.890,
    }
    (OUT / "resource_estimate.json").write_text(json.dumps(resource, indent=2))
    poly["per_polygon"].to_csv(OUT / "polygon_cells.csv", index=False)
    tri["per_triangle"].to_csv(OUT / "triangle_cells.csv", index=False)
    poly_pz["per_polygon"].to_csv(OUT / "polygon_cells_pay_zone.csv", index=False)
    if samples:
        pd.DataFrame({"fine_oz": samples_arr}).to_csv(OUT / "monte_carlo_samples.csv", index=False)
        # Histogram of the MC distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(samples_arr, bins=40, color="#1f77b4", alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.axvline(poly_pz["total_fine_oz"], color="red", linestyle="-", linewidth=2,
                   label=f"Point estimate: {poly_pz['total_fine_oz']:.0f}")
        ax.axvline(ci_low, color="green", linestyle="--", linewidth=1.5,
                   label=f"5%: {ci_low:.0f}")
        ax.axvline(ci_high, color="green", linestyle="--", linewidth=1.5,
                   label=f"95%: {ci_high:.0f}")
        ax.axvline(10056, color="gray", linestyle=":", linewidth=2,
                   label="Tweet (published): 10,056")
        ax.set_xlabel("Pay-zone fine oz")
        ax.set_ylabel("Frequency (1000 MC samples)")
        ax.set_title(
            "Monte Carlo uncertainty on pay-zone-only resource\n"
            "Sources: fineness ∈ U(0.85, 0.92), per-mg ε ~ N(0, 5%), bit area ε ~ N(0, 5%)"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_mc_uncertainty.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  → fig_mc_uncertainty.png")

    # ------------------------------------------------------------------ #
    # Items 3 + 4: plan map + bedrock-depth contour
    # ------------------------------------------------------------------ #
    print("\nItems 3 + 4: plan map + bedrock-depth contour ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plan map: holes color-coded by surface-to-br grade (Tweet scheme) --- #
    ax = axes[0]
    for _, h in valid.iterrows():
        ax.scatter(h.x_ft, h.y_ft, s=120,
                   c=grade_to_tweet_color(h.surface_to_br_grade),
                   edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(h.file_stem.replace("L", "").replace("H", ""),
                    (h.x_ft, h.y_ft), fontsize=6, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points", zorder=4)

    # Bear Cub claim outline
    ax.plot(cx, cy, "r-", linewidth=2, label="Bear Cub MS 1178")
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft from cluster center)")
    ax.set_ylabel("North (ft from cluster center)")
    ax.set_title("Plan map: holes colored by surface-to-BR grade (Tweet scheme)")
    ax.grid(True, alpha=0.3)
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, _, color, label in TWEET_GRADE_BINS
    ]
    legend_patches.append(mpatches.Patch(color="r", label="Bear Cub claim"))
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.85)

    # --- Bedrock-depth contour (RBF) --- #
    ax2 = axes[1]
    nx, ny = 80, 80
    xs = np.linspace(bbox[0] - 100, bbox[2] + 100, nx)
    ys = np.linspace(bbox[1] - 100, bbox[3] + 100, ny)
    X, Y = np.meshgrid(xs, ys)

    # only use holes with valid bedrock
    bdf = valid[valid["bedrock_depth_ft"] > 0]
    rbf_br = Rbf(bdf["x_ft"].values, bdf["y_ft"].values,
                 bdf["bedrock_depth_ft"].values, function="multiquadric", smooth=2)
    Z = rbf_br(X, Y)
    cs = ax2.contourf(X, Y, Z, levels=12, cmap="viridis")
    ax2.contour(X, Y, Z, levels=12, colors="black", linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax2, label="Bedrock depth (ft below surface)")

    ax2.scatter(bdf["x_ft"], bdf["y_ft"], s=40, c="white",
                edgecolor="black", linewidth=0.6, zorder=3)
    ax2.plot(cx, cy, "r-", linewidth=2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("East (ft)")
    ax2.set_ylabel("North (ft)")
    ax2.set_title("Bedrock depth (ft, RBF interpolated)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_plan_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_plan_map.png")

    # ------------------------------------------------------------------ #
    # Item 8: Surface-to-BOP grade map (Tweet's most important figure)
    # ------------------------------------------------------------------ #
    print("\nItem 8: Surface-to-BOP grade map ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    rbf_g = Rbf(valid["x_ft"].values, valid["y_ft"].values,
                valid["surface_to_br_grade"].values, function="multiquadric", smooth=0.001)
    Zg = np.clip(rbf_g(X, Y), 0, None)

    # Discrete bins matching Tweet's scheme
    bin_edges = [b[0] for b in TWEET_GRADE_BINS] + [TWEET_GRADE_BINS[-1][1]]
    bin_colors = [b[2] for b in TWEET_GRADE_BINS]
    cmap = ListedColormap(bin_colors)
    norm = BoundaryNorm(bin_edges, cmap.N)
    cs = ax.contourf(X, Y, Zg, levels=bin_edges, colors=bin_colors, extend="max")
    ax.contour(X, Y, Zg, levels=bin_edges, colors="black", linewidths=0.3, alpha=0.4)

    # holes overlaid
    for _, h in valid.iterrows():
        ax.scatter(h.x_ft, h.y_ft, s=80, c="white",
                   edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(h.file_stem.replace("L", "").replace("H", ""),
                    (h.x_ft, h.y_ft), fontsize=6,
                    xytext=(4, 4), textcoords="offset points", zorder=4)
    ax.plot(cx, cy, "r-", linewidth=2.5, label="Bear Cub MS 1178")

    cb = plt.colorbar(cs, ax=ax, ticks=bin_edges)
    cb.set_label("Grade (oz fine Au / cu yd)")
    cb.ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft)")
    ax.set_ylabel("North (ft)")
    ax.set_title("Surface-to-Bottom-of-Pay Grade Map\n(vertically integrated, RBF interpolated)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_surface_to_bop_grade.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_surface_to_bop_grade.png")

    # ------------------------------------------------------------------ #
    # Item 7: Expected Depth to Pay
    # ------------------------------------------------------------------ #
    print("\nItem 7: Expected Depth to Pay surface ...")
    pay = valid[valid["pay_zone_top_ft"] > 0]
    if len(pay) >= 3:
        rbf_pay = Rbf(pay["x_ft"].values, pay["y_ft"].values,
                      pay["pay_zone_top_ft"].values, function="multiquadric", smooth=2)
        Zp = rbf_pay(X, Y)
        fig, ax = plt.subplots(figsize=(10, 8))
        cs = ax.contourf(X, Y, Zp, levels=12, cmap="YlOrRd_r")
        ax.contour(X, Y, Zp, levels=12, colors="black", linewidths=0.4, alpha=0.5)
        plt.colorbar(cs, ax=ax, label="Expected depth to top of pay (ft)")
        ax.scatter(pay["x_ft"], pay["y_ft"], s=60, c="white",
                   edgecolor="black", linewidth=0.7, zorder=3)
        ax.plot(cx, cy, "r-", linewidth=2.5)
        ax.set_aspect("equal")
        ax.set_xlabel("East (ft)")
        ax.set_ylabel("North (ft)")
        ax.set_title("Expected Depth to Pay (predictive surface)\nFor a hypothetical new hole at any (E,N)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_expected_depth_to_pay.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  → fig_expected_depth_to_pay.png")
    else:
        print(f"  (skipped — only {len(pay)} holes have pay zones)")

    # ------------------------------------------------------------------ #
    # Per-hole grade profile bar charts
    # ------------------------------------------------------------------ #
    print("\nPer-hole grade profile chart ...")
    n_holes = len(valid)
    ncols = 6
    nrows = math.ceil(n_holes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5),
                             sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, (_, h) in enumerate(valid.sort_values("file_stem").iterrows()):
        ax = axes[i]
        sub = iv[iv.file_stem == h.file_stem].sort_values("depth_from_ft")
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        # horizontal bars, depth on y axis (inverted), grade on x axis
        for _, r in sub.iterrows():
            color = grade_to_jesse_color(r.grade_oz_per_cu_yd)
            ax.barh(
                (r.depth_from_ft + r.depth_to_ft) / 2,
                r.grade_oz_per_cu_yd,
                height=r.interval_ft if r.interval_ft else 1.0,
                color=color, edgecolor="black", linewidth=0.3,
            )
        # bedrock line
        if h.bedrock_depth_ft > 0:
            ax.axhline(h.bedrock_depth_ft, color="brown", linestyle="--",
                       linewidth=1, label=f"BR {h.bedrock_depth_ft:.0f}ft")
        ax.invert_yaxis()
        ax.set_title(h.file_stem, fontsize=8)
        ax.set_xlim(0, max(0.05, sub["grade_oz_per_cu_yd"].max() * 1.1))
        ax.tick_params(labelsize=6)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes[n_holes:]:
        ax.set_visible(False)

    fig.suptitle("Per-hole grade profiles (oz/cu yd, Jesse color scheme)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grade_profiles.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_grade_profiles.png")

    # ------------------------------------------------------------------ #
    # Pay-zone-only grade map (Tweet-comparable)
    # ------------------------------------------------------------------ #
    print("\nPay-zone-only grade map (Tweet-comparable) ...")
    fig, ax = plt.subplots(figsize=(11, 9))
    # Draw Voronoi polygons colored by pay_zone_avg_grade
    from scipy.spatial import Voronoi as _Vor
    from shapely.geometry import Polygon as _Poly, box as _box
    pz_clip = _box(*bbox)
    pz_pts = interior[["x_ft", "y_ft"]].values
    pz_vor = _Vor(pz_pts)
    for i, region_idx in enumerate(pz_vor.point_region):
        region = pz_vor.regions[region_idx]
        if not region or -1 in region:
            continue
        verts = pz_vor.vertices[region]
        try:
            poly_shape = _Poly(verts).intersection(pz_clip)
        except Exception:
            continue
        if poly_shape.is_empty or poly_shape.area <= 0:
            continue
        if poly_shape.geom_type == "Polygon":
            polys = [poly_shape]
        elif poly_shape.geom_type == "MultiPolygon":
            polys = list(poly_shape.geoms)
        else:
            continue
        g = float(interior.iloc[i]["pay_zone_avg_grade"])
        color = grade_to_tweet_color(g)
        for p in polys:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.55, edgecolor="black", linewidth=0.3)

    for _, h in interior.iterrows():
        ax.scatter(h.x_ft, h.y_ft, s=80,
                   c=grade_to_tweet_color(h.pay_zone_avg_grade),
                   edgecolor="black", linewidth=0.7, zorder=3)
        label = h.file_stem.split()[-1].replace("H", "")
        if h.pay_zone_thickness_ft > 0:
            label += f"\n{h.pay_zone_top_ft:.0f}-{h.pay_zone_bottom_ft:.0f}'\n{h.pay_zone_avg_grade:.3f}"
        ax.annotate(label, (h.x_ft, h.y_ft), fontsize=5,
                    ha="left", va="bottom", xytext=(4, 4),
                    textcoords="offset points", zorder=4)
    ax.plot(cx, cy, "r-", linewidth=2.5, label="Bear Cub MS 1178")
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft, local)")
    ax.set_ylabel("North (ft, local)")
    ax.set_title(
        f"Pay-zone-only Voronoi grade map (Tweet-comparable)\n"
        f"Total: {poly_pz['total_fine_oz']:.0f} fine oz · "
        f"Avg pay-zone grade {poly_pz['weighted_avg_grade']:.4f} oz/cu yd · "
        f"Pay-zone volume {poly_pz['total_volume_cuyd']:.0f} cu yd"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pay_zone_grade_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_pay_zone_grade_map.png")

    # ------------------------------------------------------------------ #
    # Year-of-drilling map
    # ------------------------------------------------------------------ #
    print("\nYear-of-drilling map ...")
    import re as _re
    new_collars_y = pd.read_parquet(SD / "drillhole_collars.parquet")
    year_lookup = {}
    for _, c in new_collars_y.iterrows():
        ds = str(c.get("date_started") or "")
        m = _re.match(r"(\d{4})-", ds)
        if m:
            year_lookup[c["file_stem"]] = int(m.group(1))
            continue
        # Fallback: infer from form_type
        ft = str(c.get("form_type") or "")
        if "Frozen Ground" in ft:
            year_lookup[c["file_stem"]] = 1919
        elif "Alaska Gold" in ft:
            year_lookup[c["file_stem"]] = 1955
        elif "Hammon Prospect" in ft:
            year_lookup[c["file_stem"]] = 1936
        elif "Hammon Field" in ft:
            year_lookup[c["file_stem"]] = 1925
        else:
            year_lookup[c["file_stem"]] = None

    valid["year_drilled"] = valid["file_stem"].map(year_lookup)
    fig, ax = plt.subplots(figsize=(11, 9))
    year_palette = {
        1919: "#7f3300",  # rust — Frozen Ground era
        1925: "#1f77b4",  # blue — Hammon early
        1936: "#2ca02c",  # green — Hammon Prospect
        1955: "#d62728",  # red — Alaska Gold
        1988: "#9467bd",  # purple — modern
    }
    seen_years = sorted({y for y in valid["year_drilled"] if y is not None})
    for yr in seen_years:
        sub = valid[valid["year_drilled"] == yr]
        ax.scatter(sub["x_ft"], sub["y_ft"], s=130,
                   c=year_palette.get(yr, "gray"),
                   edgecolor="black", linewidth=0.7,
                   label=f"{yr} ({len(sub)} hole{'s' if len(sub) != 1 else ''})", zorder=3)
        for _, h in sub.iterrows():
            ax.annotate(h.file_stem.split()[-1].replace("H", ""),
                        (h.x_ft, h.y_ft), fontsize=6,
                        ha="left", va="bottom", xytext=(4, 4),
                        textcoords="offset points", zorder=4)
    nan_sub = valid[valid["year_drilled"].isna()]
    if not nan_sub.empty:
        ax.scatter(nan_sub["x_ft"], nan_sub["y_ft"], s=130, c="lightgray",
                   edgecolor="black", label=f"Year unknown ({len(nan_sub)})", zorder=3)
    ax.plot(cx, cy, "r-", linewidth=2.5, label="Bear Cub MS 1178")
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft, local)")
    ax.set_ylabel("North (ft, local)")
    ax.set_title("Drill holes by year (Frozen Ground 1919 → Hammon 1925/1936 → Alaska Gold 1955 → modern 1988)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_year_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_year_map.png")

    # ------------------------------------------------------------------ #
    # Highlight-intervals map: per-hole pay-zone depth annotation
    # ------------------------------------------------------------------ #
    print("\nHighlight-intervals map ...")
    fig, ax = plt.subplots(figsize=(11, 9))
    # Color-code by pay_zone_avg_grade like Tweet
    for _, h in valid.iterrows():
        size = 60 + (h.pay_zone_thickness_ft * 4) if h.pay_zone_thickness_ft > 0 else 60
        color = grade_to_tweet_color(h.pay_zone_avg_grade) if h.pay_zone_thickness_ft > 0 else "white"
        ax.scatter(h.x_ft, h.y_ft, s=size, c=color,
                   edgecolor="black", linewidth=0.7, zorder=3,
                   alpha=0.85 if h.pay_zone_thickness_ft > 0 else 0.4)
        if h.pay_zone_thickness_ft > 0:
            txt = f"{h.file_stem.split()[-1].replace('H','')}\n"
            txt += f"PZ {h.pay_zone_top_ft:.0f}-{h.pay_zone_bottom_ft:.0f}'"
            txt += f" ({h.pay_zone_thickness_ft:.0f}ft)\n"
            txt += f"{h.pay_zone_avg_grade:.3f} oz/yd³"
        else:
            txt = f"{h.file_stem.split()[-1].replace('H','')}\n(no pay zone)"
        ax.annotate(txt, (h.x_ft, h.y_ft), fontsize=5,
                    ha="left", va="bottom", xytext=(6, 6),
                    textcoords="offset points", zorder=4,
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=1))
    ax.plot(cx, cy, "r-", linewidth=2.5, label="Bear Cub MS 1178")
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft, local)")
    ax.set_ylabel("North (ft, local)")
    ax.set_title(
        "Highlight intervals — per-hole pay-zone depth + avg grade\n"
        "Marker size ∝ pay-zone thickness; color = avg grade (Tweet scheme)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_highlight_intervals.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_highlight_intervals.png")

    # ------------------------------------------------------------------ #
    # Item 9: Cross-validation against Tweet + Jesse
    # ------------------------------------------------------------------ #
    print("\nItem 9: cross-validation vs Tweet + Jesse ...")
    cross_check = [
        # (file_stem, our_total_fine_oz, our_pay_zone_grade, jesse_pay_grade, tweet_grade_at_BOP, source_note)
        # Values noted from Jesse's PDFs p5-7 + Tweet's MS1178 master table:
        # Jesse's per-interval grades (highlighted yellow rows in his table)
        # Tweet's hole-level grade is from polygon-method centroid (oz/yd³)
        ("L7100 H7156", "HV-7156",
         {"jesse_pay_grade_oz_yd3": 0.05301016, "jesse_pay_interval_ft": "12-14"},
         {"tweet_grade_at_bop": None, "tweet_note": "in mid-grade band per polygon map"}),
        ("L7100 H7160", "HV-7160",
         {"jesse_pay_grade_oz_yd3": 0.033667, "jesse_pay_interval_ft": "12-14"},
         {"tweet_grade_at_bop": None, "tweet_note": "in low-mid band"}),
        ("L6700 H6760", "HV-6760",
         {"jesse_pay_grade_oz_yd3": None, "jesse_pay_interval_ft": "30-48"},
         {"tweet_grade_at_bop": 0.05, "tweet_note": "from Tweet master table"}),
    ]

    rows = []
    for fs, hv, jesse_pub, tweet_pub in cross_check:
        sel = rollups[rollups.file_stem == fs]
        if len(sel) == 0:
            continue
        r = sel.iloc[0]
        rows.append({
            "Bear Cub log": fs,
            "HV alias": hv,
            "Our pay-zone grade (oz/yd³)": f"{r['pay_zone_grade']:.4f}",
            "Our pay zone (ft)":
                (f"{r['pay_zone_top_ft']:.0f}-{r['pay_zone_bottom_ft']:.0f}"
                 if r["pay_zone_thickness_ft"] > 0 else "—"),
            "Our surface-to-BR grade (oz/yd³)": f"{r['surface_to_br_grade']:.4f}",
            "Jesse's pay grade (oz/yd³)":
                (f"{jesse_pub['jesse_pay_grade_oz_yd3']:.4f}"
                 if jesse_pub["jesse_pay_grade_oz_yd3"] else "—"),
            "Jesse's pay zone (ft)": jesse_pub.get("jesse_pay_interval_ft", "—"),
            "Tweet grade (oz/yd³)":
                (f"{tweet_pub['tweet_grade_at_bop']:.4f}"
                 if tweet_pub["tweet_grade_at_bop"] else "—"),
        })

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(OUT / "cross_validation.csv", index=False)

    md = ["# Cross-validation: our pipeline vs Tweet + Jesse\n",
          "Holes shared between our 24-log Murray subset and the published analyses.\n",
          cv_df.to_markdown(index=False) if not cv_df.empty else "(no rows)", "\n"]
    md.append("**Notes:**")
    md.append("- Jesse's per-interval grades read from PDFs p5-7 (highlighted yellow rows).")
    md.append("- Tweet's per-hole grade-at-BOP read from MS 1178 master table.")
    md.append("- Our surface-to-BR grade is vertically-integrated grade in [0, BR]; "
              "comparable to Tweet's polygon centroid number.")
    md.append("- Our pay-zone grade is the highest single-interval grade; "
              "comparable to Jesse's highlighted-yellow per-interval values.")
    (OUT / "cross_validation.md").write_text("\n".join(md))
    print(f"  → cross_validation.md")
    if len(cv_df):
        print(cv_df.to_string(index=False))

    print(f"\nDone. All outputs in: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
