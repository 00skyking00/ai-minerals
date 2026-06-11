"""BCGT bedrock-top surface from BCGS drillholes (dh2loop-format input).

Reads ``data/derived/bcgs_dh2loop/{Collar,Lithology}.csv``, infers a
``top-of-bedrock`` elevation per hole inside the BCGT AOI by walking
each hole's lithology intervals top-down until the first non-overburden
unit, then renders a two-pane figure: collar elevations on the left,
interpolated bedrock-top elevation surface on the right.

The bedrock surface uses scipy's RBFInterpolator (thin-plate spline)
rather than LoopStructural's foliation interpolator. Discovery from
D.6.A.3: LoopStructural's create_and_add_foliation requires
authentic structural orientation observations (strike/dip vectors)
that the BCGS GeoFile 2025-11 doesn't ship. Without orientations, the
foliation degenerates to a near-constant surface ignoring the contact
elevations. Thin-plate spline is the appropriate tool for the simpler
"interpolate Z at (X, Y) from scattered observations" problem here.
The LoopStructural pipeline becomes meaningful in the D.6.B+ work
when correlated-prior + multi-hypothesis machinery adds the structural
observations that don't currently exist in the raw data.

Top-of-bedrock inference: walk a hole's lithology intervals from the
collar down; while the interval's ``Detailed_Lithology`` matches the
overburden/casing regex below, advance. The first non-overburden
interval's ``FromDepth`` is the top of bedrock. The regex is the
honest-data version of the dh2loop 757-term thesaurus: BCGS operators
log overburden under 7+ codes ("OVB", "OVBD", "OB", "OVER",
"Overburden", "WCAS", "CAS", "Casing", "Soil", "Till", "Alluvium",
"DHCS"). The thesaurus would normalize these to one ``surficial``
group; here we approximate with the regex pattern.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# BCGT AOI bounds in EPSG:3005 BC Albers (matches features_bcgt_500m.parquet).
BCGT_BBOX = {"x_min": 658_250, "x_max": 792_750,
             "y_min": 1_228_750, "y_max": 1_458_250}

# Operator vocabulary for "above bedrock" intervals. Case-insensitive.
# Order matters only for readability; this is an OR-pattern. The list is
# the empirical top-20 from the BCGT-AOI subset's first-interval
# vocabulary — the dirty-data evidence supporting the dh2loop framing
# in the chapter.
OVERBURDEN_RE = (r"\b(OVB|OVBD|OB|OVER|Overburden|"
                 r"WCAS|CAS|CASN|CASE|Casing|"
                 r"Soil|Till|Alluvium|Glacial|"
                 r"DHCS|drillhole casing)\b")

COLLAR_CSV = Path("data/derived/bcgs_dh2loop/Collar.csv")
LITHO_CSV = Path("data/derived/bcgs_dh2loop/Lithology.csv")
OUT_PNG = Path("data/derived/bcgt/fig_bcgs_bedrock_plate.png")


def load_aoi_contacts() -> pd.DataFrame:
    """Return one row per AOI hole: X, Y, RL, bedrock_depth, z_bedrock."""
    collar = pd.read_csv(COLLAR_CSV)
    litho = pd.read_csv(LITHO_CSV)

    aoi = collar[(collar["X"].between(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"])) &
                 (collar["Y"].between(BCGT_BBOX["y_min"], BCGT_BBOX["y_max"]))]
    aoi_litho = litho[litho["CollarID"].isin(aoi["CollarID"])].copy()
    aoi_litho = aoi_litho.sort_values(["CollarID", "FromDepth"])

    # Mark each interval as overburden vs bedrock.
    aoi_litho["is_overburden"] = (aoi_litho["Detailed_Lithology"]
                                  .astype(str)
                                  .str.contains(OVERBURDEN_RE, case=False, regex=True, na=False))

    # For each hole, find the first non-overburden interval (== top of bedrock).
    # Holes whose first interval is already bedrock get depth=0.
    def _first_bedrock(group: pd.DataFrame) -> float:
        bedrock = group[~group["is_overburden"]]
        if bedrock.empty:
            return np.nan  # entire hole logged as overburden — drop
        return float(bedrock.iloc[0]["FromDepth"])

    bedrock_depth = aoi_litho.groupby("CollarID").apply(_first_bedrock,
                                                       include_groups=False)
    bedrock_depth.name = "bedrock_depth"

    out = aoi.merge(bedrock_depth, left_on="CollarID", right_index=True)
    out = out.dropna(subset=["bedrock_depth", "RL"]).copy()
    out["z_bedrock"] = out["RL"] - out["bedrock_depth"]
    print(f"[contacts] AOI holes with bedrock-top inferred: {len(out):,} of {len(aoi):,}")
    print(f"[contacts]   bedrock-top depth (m): p10={out.bedrock_depth.quantile(0.10):.1f}  "
          f"p50={out.bedrock_depth.quantile(0.50):.1f}  "
          f"p90={out.bedrock_depth.quantile(0.90):.1f}")
    print(f"[contacts]   bedrock-top elevation (m asl): p10={out.z_bedrock.quantile(0.10):.0f}  "
          f"p50={out.z_bedrock.quantile(0.50):.0f}  "
          f"p90={out.z_bedrock.quantile(0.90):.0f}")
    return out[["X", "Y", "RL", "bedrock_depth", "z_bedrock"]].reset_index(drop=True)


def build_bedrock_surface(contacts: pd.DataFrame,
                          nsteps: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin-plate-spline interpolation of bedrock-top elevation across the AOI.

    Returns (xx_grid, yy_grid, z_grid) where z_grid is the interpolated
    bedrock-top elevation at each (xx, yy) cell.
    """
    from scipy.interpolate import RBFInterpolator

    xs = np.linspace(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"], nsteps)
    ys = np.linspace(BCGT_BBOX["y_min"], BCGT_BBOX["y_max"], nsteps)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    pts = contacts[["X", "Y"]].to_numpy()
    vals = contacts["z_bedrock"].to_numpy()
    print(f"[rbf] fitting thin-plate spline through {len(pts):,} contact points...")
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=10.0,
                          neighbors=200)
    print(f"[rbf] evaluating on {len(grid_pts):,}-cell grid...")
    z_grid = rbf(grid_pts).reshape(xx.shape)
    return xx, yy, z_grid


def render_plate(contacts: pd.DataFrame, xx: np.ndarray, yy: np.ndarray,
                 z_surface: np.ndarray) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: collar elevation at each drillhole, showing BCGT topographic relief.
    ax = axes[0]
    sc = ax.scatter(contacts["X"], contacts["Y"], c=contacts["RL"],
                    s=12, cmap="terrain", alpha=0.85,
                    vmin=contacts["RL"].quantile(0.02),
                    vmax=contacts["RL"].quantile(0.98))
    ax.set_xlim(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"])
    ax.set_ylim(BCGT_BBOX["y_min"], BCGT_BBOX["y_max"])
    ax.set_aspect("equal")
    ax.set_title(f"BCGS drill collars in the BCGT AOI ({len(contacts):,} holes)\n"
                 f"colored by collar elevation")
    ax.set_xlabel("Easting (EPSG:3005, m)")
    ax.set_ylabel("Northing (EPSG:3005, m)")
    plt.colorbar(sc, ax=ax, label="Collar elevation (m asl)")

    # Right: interpolated bedrock-top elevation surface from the same holes.
    ax = axes[1]
    im = ax.pcolormesh(xx, yy, z_surface, cmap="terrain", shading="auto",
                       vmin=contacts["z_bedrock"].quantile(0.02),
                       vmax=contacts["z_bedrock"].quantile(0.98))
    ax.scatter(contacts["X"], contacts["Y"], s=2, c="black", alpha=0.4,
               label=f"{len(contacts):,} drill collars")
    ax.set_xlim(BCGT_BBOX["x_min"], BCGT_BBOX["x_max"])
    ax.set_ylim(BCGT_BBOX["y_min"], BCGT_BBOX["y_max"])
    ax.set_aspect("equal")
    ax.set_title("Bedrock-top elevation, thin-plate-spline interpolation\n"
                 "(scipy RBFInterpolator on contact points from dh2loop tables)")
    ax.set_xlabel("Easting (EPSG:3005, m)")
    ax.set_ylabel("Northing (EPSG:3005, m)")
    ax.legend(loc="lower right", fontsize=9)
    plt.colorbar(im, ax=ax, label="Bedrock elevation (m asl)")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"[plate] wrote {OUT_PNG}")


def main() -> None:
    contacts = load_aoi_contacts()
    xx, yy, z_surface = build_bedrock_surface(contacts)
    render_plate(contacts, xx, yy, z_surface)


if __name__ == "__main__":
    main()
