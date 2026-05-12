"""Magnetic-field derivatives for prospectivity feature engineering.

Standard derivatives computed from a residual total-intensity (TI)
magnetic grid. References: Blakely (1995) "Potential Theory in Gravity
and Magnetic Applications" Ch. 7-12; Verduzco et al. (2004) "Tilt
derivative makes magnetic interpretation easier".

Implemented derivatives:

- **1VD** (first vertical derivative): F^{-1}(|k| · F(B)). Computed in
  the wavenumber domain via FFT. Sharpens shallow-source edges.
- **HGM** (horizontal gradient magnitude): sqrt(dB/dx² + dB/dy²) via
  Sobel-style finite-difference. Marks density / susceptibility
  contrasts.
- **AS** (analytic signal): sqrt(HGM² + 1VD²). Orientation-independent
  edge detector.
- **Tilt**: atan2(1VD, HGM). Equalizes anomaly amplitudes; weak
  shallow sources become as visible as strong ones.

All four are standard MPM features. Skipped from v3.1: pseudo-gravity
(needs density-susceptibility scaling), reduce-to-pole (needs field
declination + inclination at the AOI). Both are v3.2 items.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


def _vertical_derivative_fft(grid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """First vertical derivative via FFT: F^{-1}(|k| · F(B)).

    Implements the wavenumber-domain operator. Boundaries get a tapering
    Hann window to reduce edge artifacts. Input cells set to NaN are
    replaced with the grid mean before transforming and re-NaN'd at the
    end.
    """
    valid = np.isfinite(grid)
    g = np.where(valid, grid, np.nanmean(grid))

    ny, nx = g.shape
    win_y = np.hanning(ny)[:, None]
    win_x = np.hanning(nx)[None, :]
    g_t = (g - g.mean()) * win_y * win_x

    G = np.fft.fft2(g_t)
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # Apply |k| operator. The result is the 1VD plus a small filtering
    # bias from the Hann window; for prospectivity-feature purposes
    # this is acceptable and standard in the MPM literature.
    out = np.real(np.fft.ifft2(K * G))
    out[~valid] = np.nan
    return out


def _horizontal_gradient_magnitude(grid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """HGM = sqrt((dB/dx)² + (dB/dy)²) via Sobel."""
    valid = np.isfinite(grid)
    g = np.where(valid, grid, 0.0)
    gx = ndimage.sobel(g, axis=1) / dx / 8.0  # Sobel kernel sum
    gy = ndimage.sobel(g, axis=0) / dy / 8.0
    out = np.sqrt(gx**2 + gy**2)
    out[~valid] = np.nan
    return out


def _analytic_signal(hgm: np.ndarray, vd1: np.ndarray) -> np.ndarray:
    return np.sqrt(hgm**2 + vd1**2)


def _tilt(hgm: np.ndarray, vd1: np.ndarray) -> np.ndarray:
    """Tilt = atan2(1VD, HGM); equalizes anomaly amplitudes."""
    out = np.arctan2(vd1, hgm + 1e-12)
    out[np.isnan(hgm) | np.isnan(vd1)] = np.nan
    return out


def compute_all(magnetic_path: Path) -> dict[str, np.ndarray]:
    """Return a dict of derivative-name -> 2D array, all aligned with the
    input grid's shape and CRS.
    """
    with rasterio.open(magnetic_path) as src:
        grid = src.read(1)
        nodata = src.nodata
        transform = src.transform
        dx = abs(transform.a)
        dy = abs(transform.e)

    if nodata is not None:
        grid = np.where(grid == nodata, np.nan, grid).astype(np.float32)
    grid = grid.astype(np.float64)

    vd1 = _vertical_derivative_fft(grid, dx, dy)
    hgm = _horizontal_gradient_magnitude(grid, dx, dy)
    asig = _analytic_signal(hgm, vd1)
    tilt = _tilt(hgm, vd1)

    return {
        "magnetic_1vd": vd1.astype(np.float32),
        "magnetic_hgm": hgm.astype(np.float32),
        "magnetic_analytic_signal": asig.astype(np.float32),
        "magnetic_tilt": tilt.astype(np.float32),
    }


def write_derivatives(magnetic_path: Path, out_dir: Path) -> dict[str, Path]:
    """Compute and write all four derivative GeoTIFFs next to the magnetic grid."""
    derivs = compute_all(magnetic_path)

    with rasterio.open(magnetic_path) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", nodata=np.nan, count=1)

    paths = {}
    for name, arr in derivs.items():
        out = out_dir / f"{name}_{magnetic_path.stem.split('_', 1)[1]}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr, 1)
        paths[name] = out
        print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    return paths


if __name__ == "__main__":
    from ai_minerals.regions.motherlode import MOTHERLODE
    out = write_derivatives(
        MOTHERLODE.raw_paths["magnetic"],
        MOTHERLODE.raw_paths["magnetic"].parent,
    )
    print(out)
