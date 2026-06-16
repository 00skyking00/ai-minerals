"""Refit the Map B1 rough-affine from Sky's QGIS-bootstrap GCPs.

Reads the .points file Sky saved in the rotated-TIFF frame, converts
pixel coords back to the original (un-rotated) Map B1 frame, fits a
fresh affine in original frame, then regenerates the candidate .points
file for the rotated TIFF using all 229 bearcub control points.

Reports per-bootstrap-GCP residual in meters so we can see whether the
4-5 bootstrap GCPs are mutually consistent before trusting the refit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_POINTS = Path(
    "data/derived/tuck1942/v1p5_v2_refined_gcps/tuck1942_map_b1_bootstrap.tif.points"
)
ROTATION_SIDECAR = Path(
    "data/derived/tuck1942/georef_source_rotated/tuck1942_map_b1_rotation.json"
)
CONTROL_POINTS_CSV = Path("data/raw/bearcub_nome/nome_control_points.csv")
OUT_POINTS = Path(
    "data/derived/tuck1942/v1p5_v2_gcp_candidates/tuck1942_map_b1.tif.points"
)
OUT_CSV = Path(
    "data/derived/tuck1942/v1p5_v2_gcp_candidates/tuck1942_map_b1_gcp_candidates.csv"
)


def parse_points_file(path: Path) -> list[tuple[float, float, float, float]]:
    """Return list of (mapX, mapY, sourceX, sourceY) tuples."""
    rows = []
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or "mapX" in ln or not ln.strip():
            continue
        parts = ln.split(",")
        rows.append(tuple(float(p) for p in parts[:4]))
    return rows


def rotated_to_original(col_r: float, row_r: float, orig_h: int) -> tuple[float, float]:
    """Inverse of PIL.Image.rotate(-90, expand=True): 90 deg CW rotation.

    Original (col, row) -> Rotated (orig_h - 1 - row, col), so
    Rotated (col_r, row_r) -> Original (row_r, orig_h - 1 - col_r).
    """
    return row_r, (orig_h - 1) - col_r


def original_to_rotated(col_o: float, row_o: float, orig_h: int) -> tuple[float, float]:
    """Forward of PIL.Image.rotate(-90, expand=True)."""
    return (orig_h - 1) - row_o, col_o


def fit_affine(
    gcps: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float, float, float]:
    """Fit (col, row) -> (lon, lat) affine via least squares.

    Returns (a, b, d, e, xoff, yoff) where:
        lon = a*col + b*row + xoff
        lat = d*col + e*row + yoff
    """
    X = np.array([[c, r, 1.0] for c, r, _, _ in gcps])
    lon = np.array([g[2] for g in gcps])
    lat = np.array([g[3] for g in gcps])
    coefs_lon, *_ = np.linalg.lstsq(X, lon, rcond=None)
    coefs_lat, *_ = np.linalg.lstsq(X, lat, rcond=None)
    a, b, xoff = coefs_lon
    d, e, yoff = coefs_lat
    return a, b, d, e, xoff, yoff


def predict_lonlat(
    col: float, row: float, aff: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    a, b, d, e, xoff, yoff = aff
    return a * col + b * row + xoff, d * col + e * row + yoff


def predict_pixel_original(
    lon: float, lat: float, aff: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    a, b, d, e, xoff, yoff = aff
    A = np.array([[a, b], [d, e]])
    A_inv = np.linalg.inv(A)
    col, row = A_inv @ np.array([lon - xoff, lat - yoff])
    return float(col), float(row)


def main() -> None:
    sidecar = json.loads(ROTATION_SIDECAR.read_text())
    orig_w, orig_h = sidecar["original_size"]
    rotated_w, rotated_h = sidecar["rotated_size"]

    print(f"Original size: {orig_w} x {orig_h}")
    print(f"Rotated size: {rotated_w} x {rotated_h}")

    raw_gcps = parse_points_file(BOOTSTRAP_POINTS)
    print(f"\nParsed {len(raw_gcps)} bootstrap GCPs:")
    bootstrap_orig: list[tuple[float, float, float, float]] = []
    for mapX, mapY, srcX, srcY in raw_gcps:
        col_r = srcX
        # QGIS .points sourceY is world Y in the y-flipped frame: world Y =
        # rotated_h - pixel_row_rotated, so pixel_row_rotated = rotated_h - srcY.
        row_r = rotated_h - srcY
        col_o, row_o = rotated_to_original(col_r, row_r, orig_h)
        bootstrap_orig.append((col_o, row_o, mapX, mapY))
        print(
            f"  rotated px ({col_r:.1f}, {row_r:.1f}) -> "
            f"original px ({col_o:.1f}, {row_o:.1f})  "
            f"target ({mapX:.5f}, {mapY:.5f})"
        )

    aff = fit_affine(bootstrap_orig)
    print(f"\nFitted affine (lon = a*col + b*row + xoff;  lat = d*col + e*row + yoff):")
    print(f"  a={aff[0]:.4e}  b={aff[1]:.4e}  xoff={aff[4]:.5f}")
    print(f"  d={aff[2]:.4e}  e={aff[3]:.4e}  yoff={aff[5]:.5f}")

    # Per-bootstrap residual (predicted - actual lon/lat, converted to meters).
    print("\nPer-bootstrap residual (after refit):")
    cos_lat = np.cos(np.radians(64.5))
    for col, row, lon, lat in bootstrap_orig:
        plon, plat = predict_lonlat(col, row, aff)
        d_m_x = (plon - lon) * 111_320 * cos_lat
        d_m_y = (plat - lat) * 111_320
        err_m = float(np.hypot(d_m_x, d_m_y))
        print(f"  target=({lon:.5f},{lat:.5f}) pred=({plon:.5f},{plat:.5f}) err={err_m:.0f} m")

    # Regenerate candidates for Map B1 in the rotated frame
    df = pd.read_csv(CONTROL_POINTS_CSV).dropna(subset=["lon", "lat"]).reset_index(drop=True)
    print(f"\nGenerating {len(df)} candidate GCPs against the refit affine ...")

    lines = [
        '#CRS: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
        "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual",
    ]
    records = []
    n_on = 0
    n_off = 0
    n_dup = 0
    # Deduplicate source pixels: adjacent patented claims share corner
    # posts, so multiple bearcub GCPs map to the same pixel. TPS can't
    # invert a matrix with duplicate source pixels.
    seen_pixels: set[tuple[int, int]] = set()
    for _, r in df.iterrows():
        col_o, row_o = predict_pixel_original(r["lon"], r["lat"], aff)
        # Forward-rotate to rotated frame
        col_r, row_r = original_to_rotated(col_o, row_o, orig_h)
        on_sheet = (0 <= col_r < rotated_w) and (0 <= row_r < rotated_h)
        if not on_sheet:
            n_off += 1
            continue
        # Round to 2 px so near-duplicates collapse too
        key = (int(round(col_r / 2.0)), int(round(row_r / 2.0)))
        if key in seen_pixels:
            n_dup += 1
            continue
        seen_pixels.add(key)
        n_on += 1
        # Y-flipped sourceY for the QGIS .points file
        sourceX = col_r
        sourceY = rotated_h - row_r
        lines.append(
            f"{r['lon']:.7f},{r['lat']:.7f},{sourceX:.2f},{sourceY:.2f},1,0,0,0"
        )
        records.append({
            "name": r["name"],
            "type": r["type"],
            "ms": (str(int(r["ms"])) if pd.notna(r["ms"]) else ""),
            "lon": float(r["lon"]),
            "lat": float(r["lat"]),
            "pixel_col_rotated": col_r,
            "pixel_row_rotated": row_r,
            "pixel_col_original": col_o,
            "pixel_row_original": row_o,
            "sourceX": sourceX,
            "sourceY": sourceY,
        })

    print(f"  on sheet, unique pixel: {n_on}")
    print(f"  off sheet:               {n_off}")
    print(f"  duplicate source pixel:  {n_dup}")

    OUT_POINTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_POINTS.write_text("\n".join(lines) + "\n")
    print(f"\nWrote candidates: {OUT_POINTS}")
    pd.DataFrame(records).to_csv(OUT_CSV, index=False)
    print(f"Wrote inspection CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
