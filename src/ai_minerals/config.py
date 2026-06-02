"""Centralized v3 hyperparameters.

Each constant documents its tuning history. Keep this file as the single
source of truth; update downstream code to import from here rather than
hardcoding.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Training / CV (currently hardcoded in
# scripts/northern_sierra_placer_train_predict_250m.py)
# ---------------------------------------------------------------------------

# Mordelet-Vert PU bagging: number of bootstrap bags used to derate the
# unlabeled set into pseudo-negatives. 30 has been the v1/v2/v3 setting;
# untuned beyond "enough that the std across bags is small relative to the
# mean score." Worth a v3.5 sensitivity check.
N_PU_BAGS = 30

# Spatial-block CV block size in meters (working CRS units). 20 km was the
# v2 default carried into v3 without empirical justification; Phase E.6 in
# the v3 plan runs a 10/15/20/25/30 km ablation to validate.
BLOCK_SIZE_M = 20_000.0

# Below this many positives, calibration falls back to Platt (sigmoid)
# instead of isotonic. Isotonic needs enough positives per bin (~10 per bin
# minimum, ~3 bins, plus CV) before its piecewise-constant fit stops being
# noise-dominated. 30 is a heuristic; Phase E.3 runs the empirical
# isotonic-vs-Platt ablation.
ISOTONIC_MIN_POSITIVES = 30

# CalibratedClassifierCV fold count. 5 is the sklearn default; the actual
# fold count gets capped at int(y_train.sum()) when positives are very
# sparse (see the sparsity check in train_predict_250m.py).
CALIBRATION_CV = 5


# ---------------------------------------------------------------------------
# Hawkes dual-decay catchment aggregation
# (currently hardcoded as defaults in
# src/ai_minerals/features/placer_geology.py::hawkes_dual_decay_catchment)
# ---------------------------------------------------------------------------

# Far-decay length scale for the Hawkes catchment kernel. Controls how far
# upstream a geochem anomaly's influence propagates. 15 km matches the
# Hawkes 1976 empirical placer-distance falloff for Sierra-scale watersheds.
HAWKES_DECAY_KM = 15.0

# Mixing weight between the near-channel and far-network kernels:
#     w(d) = exp(-d / near_decay_km) + alpha * exp(-d / far_decay_km)
# 0.3 puts most weight on the near-channel term (modern channel proximity)
# while keeping the far-network term non-trivial for distal placers.
HAWKES_ALPHA = 0.3

# Near-channel decay length scale. 2 km matches typical Sierra reach
# length between gold-trap sites (riffles, knickpoints, contact-zone
# breaks) within a single tributary.
HAWKES_NETWORK_DECAY_KM = 2.0


# ---------------------------------------------------------------------------
# SPI bandpass thresholds
# (currently hardcoded as default in
# src/ai_minerals/features/hydrology.py::stream_power_index_band)
# ---------------------------------------------------------------------------

# Lower edge of the SPI bandpass [ln(flow_acc * tan(slope))] below which
# the cell is "too slow" to concentrate placer gold (Roy/Upton/Craw 2018).
SPI_BANDPASS_LO = 3.0

# Upper edge of the SPI bandpass above which the cell is "too fast" and
# transports gold through without depositing.
SPI_BANDPASS_HI = 6.0
