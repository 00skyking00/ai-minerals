"""Generate Hammon-grid-anchored synthetic GCPs for each Tuck 1942 plate.

For each plate:
  1. Read existing manual GCPs (Sky's feature-anchored .points)
  2. Fit a 6-parameter affine source-pixel -> WGS84 from those GCPs
  3. Compose with WGS84 -> Hammon (via bearcub converter) to get
     source-pixel -> Hammon affine
  4. Invert: Hammon -> source-pixel
  5. For every 5000-ft Hammon-grid intersection inside the plate's
     source extent, predict its source pixel; pair with the exact
     WGS84 lat/lon from the Hammon -> WGS84 converter
  6. Write a synthetic .points file

The resulting GCPs are perfectly distributed on the Hammon grid, all
anchored to the same Hammon -> WGS84 affine. Seams between adjacent
plates align by construction because shared (E, N) intersections
produce identical lat/lon on both plates.

Absolute accuracy = bearcub converter accuracy (~80 ft in Bear Cub
envelope, drifting at atlas edges). Relative seam accuracy ~= the
affine fit residual on each plate (typically <5 m for Map B).

Run:
    uv run python -m scripts.nome_placer.tuck_hammon_synthetic_gcps

then re-warp via the existing tuck_run_tps_warp script.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ai_minerals.data.bearcub_georef import local_to_wgs84, wgs84_to_local


PLATES = ["tuck1942_map_a", "tuck1942_map_b", "tuck1942_map_c", "tuck1942_map_d", "tuck1942_map_e"]
SOURCE_DIR = Path("data/derived/tuck1942/georef_source_rotated")
EXISTING_POINTS_DIRS = (
    Path("data/derived/tuck1942/v1p5_v3_refined_gcps"),
    Path("data/derived/tuck1942/v1p5_v2_refined_gcps"),
    Path("data/derived/tuck1942/v1p5_v2_gcp_candidates"),
)
OUTPUT_POINTS_DIR = Path("data/derived/tuck1942/v1p5_v4_hammon_synthetic_gcps")
HAMMON_STEP_FT = 5000  # Hammon grid spacing as printed on the plates

POINTS_HEADER = (
    '#CRS: GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],'
    'ANGLEUNIT["degree",0.0174532925199433]],'
    'AXIS["geodetic longitude (Lon)",east,ORDER[2],'
    'ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]\n'
    "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual\n"
)


def _find_existing_points(plate: str) -> Path:
    name = f"{plate}.tif.points"
    cands = [d / name for d in EXISTING_POINTS_DIRS if (d / name).exists()]
    if not cands:
        raise FileNotFoundError(f"No existing .points for {plate} in any candidate dir")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _read_points(path: Path) -> list[tuple[float, float, float, float]]:
    """Return list of (px, py, lon, lat) for enabled GCPs."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("mapX"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 5 or parts[4] != "1":
                continue
            lon, lat, px, py = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            rows.append((px, py, lon, lat))
    return rows


def _source_dimensions(source_tif: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["gdalinfo", str(source_tif)], capture_output=True, text=True, check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Size is"):
            w, h = line.replace("Size is", "").strip().split(",")
            return int(w), int(h)
    raise RuntimeError(f"Could not parse size from gdalinfo on {source_tif}")


def _fit_affine_once(gcps: list[tuple[float, float, float, float]]):
    A = np.array([[px, py, 1.0] for px, py, _, _ in gcps])
    b_lon = np.array([lon for _, _, lon, _ in gcps])
    b_lat = np.array([lat for _, _, _, lat in gcps])
    coef_lon, *_ = np.linalg.lstsq(A, b_lon, rcond=None)
    coef_lat, *_ = np.linalg.lstsq(A, b_lat, rcond=None)
    res = []
    for px, py, lon, lat in gcps:
        pred_lon = coef_lon[0]*px + coef_lon[1]*py + coef_lon[2]
        pred_lat = coef_lat[0]*px + coef_lat[1]*py + coef_lat[2]
        dx_m = (pred_lon - lon) * 111000 * np.cos(np.radians(lat))
        dy_m = (pred_lat - lat) * 111000
        res.append(np.hypot(dx_m, dy_m))
    return coef_lon, coef_lat, np.array(res)


def _fit_affine(
    gcps: list[tuple[float, float, float, float]],
    *,
    residual_threshold_m: float = 50.0,
    min_keep: int = 4,
):
    """Iteratively drop high-residual GCPs until all are below threshold.

    Returns (coef_lon, coef_lat, residual_summary, kept_gcps, dropped_indices).
    """
    indexed = list(enumerate(gcps))
    dropped_indices: list[int] = []
    while True:
        active = [(p[0], p[1], p[2], p[3]) for _, p in indexed]
        coef_lon, coef_lat, res = _fit_affine_once(active)
        if len(indexed) <= min_keep or res.max() <= residual_threshold_m:
            break
        worst = int(res.argmax())
        dropped_indices.append(indexed[worst][0])
        indexed.pop(worst)
    summary = {
        "max_m": float(res.max()), "mean_m": float(res.mean()),
        "rms_m": float(np.sqrt((res**2).mean())),
        "kept": len(indexed), "dropped": dropped_indices,
    }
    return coef_lon, coef_lat, summary


def _plate_hammon_extent(coef_lon, coef_lat, w: int, h: int) -> tuple[float, float, float, float]:
    """Map the 4 source-pixel corners to Hammon (E, N) via affine + converter."""
    corners_px = [(0, 0), (w, 0), (0, h), (w, h)]
    E_vals, N_vals = [], []
    for px, py in corners_px:
        lon = coef_lon[0]*px + coef_lon[1]*py + coef_lon[2]
        lat = coef_lat[0]*px + coef_lat[1]*py + coef_lat[2]
        e, n = wgs84_to_local(lat, lon)
        E_vals.append(e)
        N_vals.append(n)
    return min(E_vals), max(E_vals), min(N_vals), max(N_vals)


def _generate_synthetic_gcps(plate: str) -> dict:
    src_tif = SOURCE_DIR / f"{plate}.tif"
    existing_pts = _find_existing_points(plate)
    print(f"\n=== {plate} ===")
    print(f"  source: {src_tif}")
    print(f"  existing .points: {existing_pts}")

    w, h = _source_dimensions(src_tif)
    print(f"  source dims: {w} x {h} px")
    gcps = _read_points(existing_pts)
    print(f"  manual GCPs: {len(gcps)}")

    coef_lon, coef_lat, residuals = _fit_affine(gcps)
    print(f"  kept {residuals['kept']}/{len(gcps)} GCPs after outlier rejection "
          f"(dropped indices: {residuals['dropped']})")
    print(f"  affine fit residuals: max {residuals['max_m']:.1f} m, "
          f"mean {residuals['mean_m']:.1f} m, rms {residuals['rms_m']:.1f} m")

    e_min, e_max, n_min, n_max = _plate_hammon_extent(coef_lon, coef_lat, w, h)
    print(f"  Hammon extent: E [{e_min:.0f}, {e_max:.0f}], N [{n_min:.0f}, {n_max:.0f}]")

    # Inverse affine: WGS84 -> source pixel
    M = np.array([[coef_lon[0], coef_lon[1]], [coef_lat[0], coef_lat[1]]])
    t = np.array([coef_lon[2], coef_lat[2]])
    M_inv = np.linalg.inv(M)

    # Snap Hammon range to 5000-ft grid, with a small inward pad to skip
    # intersections that land near the edge of the source plate
    e0 = int(np.ceil(e_min / HAMMON_STEP_FT) * HAMMON_STEP_FT)
    e1 = int(np.floor(e_max / HAMMON_STEP_FT) * HAMMON_STEP_FT)
    n0 = int(np.ceil(n_min / HAMMON_STEP_FT) * HAMMON_STEP_FT)
    n1 = int(np.floor(n_max / HAMMON_STEP_FT) * HAMMON_STEP_FT)

    PAD = 80  # pixels of margin from source-image edge
    rows = []
    skipped = 0
    for e in range(e0, e1 + 1, HAMMON_STEP_FT):
        for n in range(n0, n1 + 1, HAMMON_STEP_FT):
            lat, lon = local_to_wgs84(e, n)
            px_arr = M_inv @ (np.array([lon, lat]) - t)
            px, py = float(px_arr[0]), float(px_arr[1])
            if not (PAD <= px <= w - PAD and PAD <= py <= h - PAD):
                skipped += 1
                continue
            rows.append((px, py, lon, lat, e, n))

    OUTPUT_POINTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_POINTS_DIR / f"{plate}.tif.points"
    with open(out_path, "w") as f:
        f.write(POINTS_HEADER)
        for px, py, lon, lat, _, _ in rows:
            f.write(f"{lon:.10f},{lat:.10f},{px:.4f},{py:.4f},1,0,0,0\n")
    print(f"  synthetic GCPs: {len(rows)} (skipped {skipped} outside plate)")
    print(f"  wrote: {out_path}")
    return {
        "plate": plate, "n_gcps": len(rows),
        "residuals_m": residuals, "out": str(out_path),
    }


def main() -> None:
    print("Hammon synthetic-GCP generator")
    print(f"Step: {HAMMON_STEP_FT} ft, output dir: {OUTPUT_POINTS_DIR}")
    summary = []
    for plate in PLATES:
        try:
            summary.append(_generate_synthetic_gcps(plate))
        except Exception as exc:
            print(f"  ERROR: {exc}")
    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"  {s['plate']}: {s['n_gcps']} synthetic GCPs, "
              f"max residual on manual fit = {s['residuals_m']['max_m']:.1f} m")


if __name__ == "__main__":
    main()
