"""Verify the v3 Phase B.5 gaussian_falloff soft-distance helper.

Properties checked:
  1. g(0) = 1.0
  2. g(sigma) ≈ exp(-1/2) ≈ 0.6065
  3. g(2*sigma) ≈ exp(-2) ≈ 0.1353
  4. g(NaN) = 0
  5. Vector inputs return per-element gaussian-falloff scores.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_minerals.features.hydrology import gaussian_falloff


def test_falloff_at_zero_is_one() -> None:
    out = gaussian_falloff(np.array([0.0]), sigma_m=1000.0)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(1.0, abs=1e-6)


def test_falloff_at_sigma_is_e_minus_half() -> None:
    sigma = 1000.0
    out = gaussian_falloff(np.array([sigma]), sigma_m=sigma)
    expected = math.exp(-0.5)  # ≈ 0.6065
    assert out[0] == pytest.approx(expected, rel=1e-4)


def test_falloff_at_two_sigma_is_e_minus_two() -> None:
    sigma = 1000.0
    out = gaussian_falloff(np.array([2.0 * sigma]), sigma_m=sigma)
    expected = math.exp(-2.0)  # ≈ 0.1353
    assert out[0] == pytest.approx(expected, rel=1e-4)


def test_nan_input_returns_zero() -> None:
    out = gaussian_falloff(np.array([np.nan]), sigma_m=1000.0)
    assert out.shape == (1,)
    assert out[0] == 0.0


def test_vector_input() -> None:
    sigma = 500.0
    d = np.array([0.0, sigma, 2.0 * sigma, 3.0 * sigma, np.nan])
    out = gaussian_falloff(d, sigma_m=sigma)
    assert out.shape == d.shape
    assert out[0] == pytest.approx(1.0, abs=1e-6)
    assert out[1] == pytest.approx(math.exp(-0.5), rel=1e-4)
    assert out[2] == pytest.approx(math.exp(-2.0), rel=1e-4)
    # At 3*sigma the falloff is ~exp(-4.5) ≈ 0.0111 ≈ "5%-ish".
    assert out[3] == pytest.approx(math.exp(-4.5), rel=1e-4)
    assert out[4] == 0.0


def test_output_dtype_is_float32() -> None:
    out = gaussian_falloff(np.array([0.0, 1000.0]), sigma_m=1000.0)
    assert out.dtype == np.float32


def test_2d_input_preserves_shape() -> None:
    sigma = 1000.0
    d = np.array([[0.0, sigma], [2.0 * sigma, np.nan]])
    out = gaussian_falloff(d, sigma_m=sigma)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert out[0, 1] == pytest.approx(math.exp(-0.5), rel=1e-4)
    assert out[1, 0] == pytest.approx(math.exp(-2.0), rel=1e-4)
    assert out[1, 1] == 0.0
