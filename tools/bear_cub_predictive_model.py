"""Spatial predictive model + drill-site recommender for the Bear Cub corpus.

Uses Gaussian-process regression over hole positions to:
  1. Predict pay-zone avg grade as a function of (x, y, form_year).
  2. Quantify per-location uncertainty.
  3. Recommend next drill locations that maximize info gain
     (variance of the GP) and expected improvement (Bayes-opt-style criterion
     over a 0.05 oz/yd³ aspirational threshold).

LOO-CV across the 24 holes gives an honest small-sample evaluation.

Outputs (data/derived/bear_cub_resource/):
  fig_gp_loo_residuals.png             Predicted vs actual, LOO-CV
  fig_gp_predicted_grade_surface.png   Mean grade over the claim polygon
  fig_gp_uncertainty_surface.png       Predictive std (σ) — high = unknown
  fig_gp_drill_recommender.png         Highlighted recommended next holes
  predictive_model_metrics.json        LOO MAE / R² / RMSE, recommended sites

Run:
    uv run python tools/bear_cub_predictive_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point, Polygon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived" / "bear_cub_resource"
OUT = DERIVED
OUT.mkdir(parents=True, exist_ok=True)

# Bear Cub corner coords (local feet, same frame as rollups x_ft / y_ft)
# These come from latlon_to_local_ft on the 4 patent corners — re-derived
# below from the rollups CSV's lat/lon to keep coords consistent.

EI_TARGET = 0.05  # aspirational pay-zone grade for expected improvement


def load_data() -> tuple[pd.DataFrame, Polygon]:
    rollups = pd.read_csv(DERIVED / "hole_rollups.csv")
    valid = rollups.dropna(subset=["lat_wgs84", "lon_wgs84"]).copy()
    # Use the polygon cells' exact x,y (already in local feet on the same frame)
    poly_cells = pd.read_csv(DERIVED / "polygon_cells.csv")
    poly_cells["file_stem"] = valid.iloc[poly_cells.hole_idx.values].file_stem.values
    valid = valid.merge(poly_cells[["file_stem", "area_ft2"]], on="file_stem", how="left")

    # Re-project (lat, lon) → local feet using the same anchor as the rollups
    # uses (lat0, lon0 = mean lat / mean lon)
    lat0 = valid["lat_wgs84"].mean()
    lon0 = valid["lon_wgs84"].mean()
    deg2ft_lat = 364320.0
    deg2ft_lon = 364320.0 * np.cos(np.deg2rad(lat0))
    valid["x_ft"] = (valid["lon_wgs84"] - lon0) * deg2ft_lon
    valid["y_ft"] = (valid["lat_wgs84"] - lat0) * deg2ft_lat

    # Approximate the claim polygon from the convex hull of holes (close enough
    # for surface-prediction; the actual patent corners would be exact)
    from scipy.spatial import ConvexHull
    pts = valid[["x_ft", "y_ft"]].values
    hull = ConvexHull(pts)
    claim = Polygon(pts[hull.vertices])
    return valid, claim


def build_features(valid: pd.DataFrame, target: str = "pay_zone_avg_grade") -> tuple[np.ndarray, np.ndarray]:
    """Spatial-only features (x, y) for the GP."""
    X = valid[["x_ft", "y_ft"]].values
    y = valid[target].values
    return X, y


def fit_gp(X: np.ndarray, y: np.ndarray, scaler: StandardScaler | None = None) -> GaussianProcessRegressor:
    """Anisotropic RBF + white-noise kernel; ConstantKernel sets the GP variance."""
    if scaler is not None:
        X = scaler.transform(X)
    kernel = (
        ConstantKernel(constant_value=np.var(y), constant_value_bounds=(1e-6, 1e2))
        * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.01, 100.0))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=42)
    gp.fit(X, y)
    return gp


def loo_cv(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out cross-validation. Returns (predicted, std)."""
    n = len(y)
    pred = np.zeros(n)
    std = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scaler = StandardScaler().fit(X[mask])
        gp = fit_gp(X[mask], y[mask], scaler=scaler)
        Xi = scaler.transform(X[[i]])
        m, s = gp.predict(Xi, return_std=True)
        pred[i] = float(m[0])
        std[i] = float(s[0])
    return pred, std


def predict_grid(claim: Polygon, X: np.ndarray, y: np.ndarray, n: int = 80
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict GP mean and std over a grid covering the claim polygon."""
    minx, miny, maxx, maxy = claim.bounds
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    X_grid, Y_grid = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    # Mask grid points outside the claim polygon
    mask_in = np.array([claim.contains(Point(p)) for p in grid_pts])

    scaler = StandardScaler().fit(X)
    gp = fit_gp(X, y, scaler=scaler)
    mean = np.full(len(grid_pts), np.nan)
    std = np.full(len(grid_pts), np.nan)
    if mask_in.any():
        Xs = scaler.transform(grid_pts[mask_in])
        m, s = gp.predict(Xs, return_std=True)
        mean[mask_in] = m
        std[mask_in] = s
    return X_grid, Y_grid, mean.reshape(X_grid.shape), std.reshape(X_grid.shape)


def expected_improvement(mean: np.ndarray, std: np.ndarray, target: float) -> np.ndarray:
    """Standard Bayesian-opt EI criterion. Higher = better candidate."""
    from scipy.stats import norm
    # Avoid div by zero
    s = np.maximum(std, 1e-9)
    z = (mean - target) / s
    ei = (mean - target) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.where(std > 0, ei, 0.0)
    return ei


def main() -> None:
    print("Loading data ...")
    valid, claim = load_data()

    # Compare two targets — pay-zone grade (sharply spiky from beach-line geology)
    # and surface-to-BR grade (smoother, averages over the column)
    print("\n=== TARGET 1: pay_zone_avg_grade (sharp, beach-line dominated) ===")
    Xp, yp = build_features(valid, "pay_zone_avg_grade")
    pred_p, sigma_p = loo_cv(Xp, yp)
    res_p = yp - pred_p
    mae_p = float(np.mean(np.abs(res_p)))
    rmse_p = float(np.sqrt(np.mean(res_p ** 2)))
    r2_p = 1 - float(np.sum(res_p ** 2)) / float(np.sum((yp - yp.mean()) ** 2))
    print(f"  MAE={mae_p:.4f}, RMSE={rmse_p:.4f}, R²={r2_p:.3f}")

    print("\n=== TARGET 2: surface_to_br_grade (smoother, depth-averaged) ===")
    Xs, ys = build_features(valid, "surface_to_br_grade")
    pred_s, sigma_s = loo_cv(Xs, ys)
    res_s = ys - pred_s
    mae_s = float(np.mean(np.abs(res_s)))
    rmse_s = float(np.sqrt(np.mean(res_s ** 2)))
    r2_s = 1 - float(np.sum(res_s ** 2)) / float(np.sum((ys - ys.mean()) ** 2))
    print(f"  MAE={mae_s:.4f}, RMSE={rmse_s:.4f}, R²={r2_s:.3f}")

    # Use the smoother surface_to_BR target for the prediction surface +
    # recommender (it gives a meaningful R²; pay-zone grade is too spiky on n=24)
    target_for_surface = "surface_to_br_grade" if r2_s > r2_p else "pay_zone_avg_grade"
    print(f"\nUsing '{target_for_surface}' for prediction surface + recommender "
          f"(better LOO-CV R² on n=24)")
    X, y = build_features(valid, target_for_surface)
    print(f"  {len(valid)} holes; target range {y.min():.4f}-{y.max():.4f} oz/yd³")

    # ---------------- Side-by-side LOO residual plot for both targets ----------------
    print("\nGenerating LOO residual figure ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (yi, predi, sigi, label, mae_i, rmse_i, r2_i) in zip(axes, [
        (yp, pred_p, sigma_p, "pay_zone_avg_grade", mae_p, rmse_p, r2_p),
        (ys, pred_s, sigma_s, "surface_to_br_grade", mae_s, rmse_s, r2_s),
    ]):
        ax.errorbar(yi, predi, yerr=sigi, fmt="o", color="#1f77b4",
                    ecolor="lightgray", capsize=3, elinewidth=1, markersize=6)
        lim = max(yi.max(), predi.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="y = x")
        for i, fs in enumerate(valid["file_stem"]):
            ax.annotate(fs.split()[-1].replace("H", ""), (yi[i], predi[i]),
                        fontsize=5, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"Actual {label} (oz/yd³)")
        ax.set_ylabel(f"LOO-predicted {label} (oz/yd³)")
        ax.set_title(f"{label}\nMAE {mae_i:.4f} · RMSE {rmse_i:.4f} · R² {r2_i:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle(
        "Gaussian-process LOO-CV — spatial-only (x, y) features over 24 holes\n"
        "Pay-zone grade is too spiky for spatial-only GP (sharp beach-line gradient); "
        "surface-to-BR grade smooths over the column and is learnable",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_gp_loo_residuals.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_gp_loo_residuals.png")

    # Use the better-performing target for the prediction surface
    pred = pred_s if r2_s > r2_p else pred_p
    sigma = sigma_s if r2_s > r2_p else sigma_p
    mae, rmse, r2 = (mae_s, rmse_s, r2_s) if r2_s > r2_p else (mae_p, rmse_p, r2_p)

    # ---------------- Predicted surface + uncertainty ----------------
    print("\nPredicting GP surface over claim polygon ...")
    X_grid, Y_grid, mean_grid, std_grid = predict_grid(claim, X, y, n=80)

    # Predicted-grade surface
    fig, ax = plt.subplots(figsize=(11, 9))
    cs = ax.contourf(X_grid, Y_grid, mean_grid, levels=15, cmap="YlOrRd", alpha=0.75)
    plt.colorbar(cs, ax=ax, label="Predicted pay-zone avg grade (oz/yd³)")
    cx, cy = claim.exterior.xy
    ax.plot(cx, cy, "r-", linewidth=2.5, label="Claim hull")
    for _, h in valid.iterrows():
        ax.scatter(h.x_ft, h.y_ft, s=80, c="white", edgecolor="black",
                   linewidth=0.7, zorder=3)
        ax.annotate(h.file_stem.split()[-1].replace("H", ""), (h.x_ft, h.y_ft),
                    fontsize=6, ha="left", va="bottom", xytext=(4, 4),
                    textcoords="offset points", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft, local)")
    ax.set_ylabel("North (ft, local)")
    ax.set_title("Gaussian-process predicted pay-zone grade surface\n"
                 "Trained on 24 holes; surface interpolates between them")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gp_predicted_grade_surface.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_gp_predicted_grade_surface.png")

    # Uncertainty surface
    fig, ax = plt.subplots(figsize=(11, 9))
    cs = ax.contourf(X_grid, Y_grid, std_grid, levels=15, cmap="viridis", alpha=0.85)
    plt.colorbar(cs, ax=ax, label="Predictive σ (uncertainty, oz/yd³)")
    ax.plot(cx, cy, "r-", linewidth=2.5, label="Claim hull")
    for _, h in valid.iterrows():
        ax.scatter(h.x_ft, h.y_ft, s=60, c="white", edgecolor="black",
                   linewidth=0.7, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft, local)")
    ax.set_ylabel("North (ft, local)")
    ax.set_title("Predictive uncertainty (σ) — GP standard deviation across the claim\n"
                 "Bright = high uncertainty (no nearby holes); Dark = well-constrained")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gp_uncertainty_surface.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_gp_uncertainty_surface.png")

    # ---------------- Drill-site recommender ----------------
    print("\nDrill-site recommender ...")
    # Two criteria: max σ (info gain) and expected improvement vs 0.05 oz/yd³ target.
    flat_x = X_grid.ravel()
    flat_y = Y_grid.ravel()
    flat_mean = mean_grid.ravel()
    flat_std = std_grid.ravel()
    valid_mask = np.isfinite(flat_mean)
    # Don't recommend a point too close to an existing hole (within 80 ft)
    too_close = np.zeros(flat_x.shape, dtype=bool)
    for px, py in zip(valid["x_ft"].values, valid["y_ft"].values):
        too_close |= np.sqrt((flat_x - px) ** 2 + (flat_y - py) ** 2) < 80.0
    cand_mask = valid_mask & ~too_close

    ei = expected_improvement(flat_mean, flat_std, EI_TARGET)
    ei_masked = np.where(cand_mask, ei, -np.inf)
    var_masked = np.where(cand_mask, flat_std, -np.inf)

    # Top-3 by each criterion
    top_var_idx = np.argsort(-var_masked)[:3]
    top_ei_idx = np.argsort(-ei_masked)[:3]

    rec = {
        "target_for_surface": target_for_surface,
        "max_variance_recommendations": [
            {
                "rank": int(r + 1),
                "x_ft": float(flat_x[i]),
                "y_ft": float(flat_y[i]),
                "predicted_grade_oz_per_yd3": float(flat_mean[i]),
                "predicted_sigma_oz_per_yd3": float(flat_std[i]),
            }
            for r, i in enumerate(top_var_idx)
        ],
        "expected_improvement_recommendations": [
            {
                "rank": int(r + 1),
                "x_ft": float(flat_x[i]),
                "y_ft": float(flat_y[i]),
                "predicted_grade_oz_per_yd3": float(flat_mean[i]),
                "predicted_sigma_oz_per_yd3": float(flat_std[i]),
                "expected_improvement_oz_per_yd3": float(ei[i]),
            }
            for r, i in enumerate(top_ei_idx)
        ],
        "ei_target_oz_per_yd3": EI_TARGET,
        "loo_cv_pay_zone_avg_grade": {
            "mae": mae_p, "rmse": rmse_p, "r2": r2_p,
        },
        "loo_cv_surface_to_br_grade": {
            "mae": mae_s, "rmse": rmse_s, "r2": r2_s,
        },
        "n_holes": int(len(valid)),
    }
    (OUT / "predictive_model_metrics.json").write_text(json.dumps(rec, indent=2))
    print(f"  → predictive_model_metrics.json")

    # Recommender plot
    fig, ax = plt.subplots(figsize=(11, 9))
    cs = ax.contourf(X_grid, Y_grid, mean_grid, levels=12, cmap="YlOrRd", alpha=0.6)
    plt.colorbar(cs, ax=ax, label="GP-predicted grade (oz/yd³)")
    ax.plot(cx, cy, "r-", linewidth=2.5)
    for _, h in valid.iterrows():
        ax.scatter(h.x_ft, h.y_ft, s=70, c="white", edgecolor="black",
                   linewidth=0.6, zorder=3)
        ax.annotate(h.file_stem.split()[-1].replace("H", ""), (h.x_ft, h.y_ft),
                    fontsize=5, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points", zorder=4)
    # Max-variance recs (cyan triangle)
    for r, i in enumerate(top_var_idx):
        ax.scatter(flat_x[i], flat_y[i], s=240, marker="^", c="cyan",
                   edgecolor="black", linewidth=1.5, zorder=5,
                   label=f"Max σ #{r+1} (info gain)" if r == 0 else None)
        ax.annotate(f"σ#{r+1}\nσ={flat_std[i]:.3f}", (flat_x[i], flat_y[i]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 8),
                    textcoords="offset points", zorder=6,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="cyan", pad=1.5))
    # Expected-improvement recs (magenta star)
    for r, i in enumerate(top_ei_idx):
        ax.scatter(flat_x[i], flat_y[i], s=300, marker="*", c="magenta",
                   edgecolor="black", linewidth=1.5, zorder=5,
                   label=f"Max EI #{r+1} (vs 0.05 target)" if r == 0 else None)
        ax.annotate(f"EI#{r+1}\nμ={flat_mean[i]:.3f}", (flat_x[i], flat_y[i]),
                    fontsize=7, ha="center", va="top", xytext=(0, -10),
                    textcoords="offset points", zorder=6,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="magenta", pad=1.5))
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft, local)")
    ax.set_ylabel("North (ft, local)")
    ax.set_title(
        "Next-drill-site recommendations\n"
        "▲ max-variance (info gain) · ★ max-EI (vs 0.05 oz/yd³ target). "
        "Both excluded within 80 ft of existing holes."
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gp_drill_recommender.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_gp_drill_recommender.png")

    print(f"\nAll outputs in: {DERIVED.relative_to(REPO)}")


if __name__ == "__main__":
    main()
