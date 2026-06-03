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


# ---------------------------------------------------------------------------
# v3 Phase B per-population feature stacks
# (consumed by scripts/northern_sierra_placer_train_predict_250m.py
# train_one_population to filter the assembled feature frame down to the
# subset matched to each population's geomorphic signature)
#
# Features present in the per-population tuple but absent from the assembled
# parquet are skipped at training time (graceful for v3 features not yet
# assembled). Lithology one-hot columns (lithology_class_<N>, major1/2/3_
# class_<N>) are passed through outside this filter because they're expanded
# from the single `lithology_class` column by add_lithology_onehot.
# ---------------------------------------------------------------------------

# Tertiary deep-gravel hydraulic-mine population. Positives = 158 hydraulic
# pit centroids in Orlando 2016; geology = paleo-channels exposed by hydraulic
# mining on hillside benches above modern drainage.
#
# INCLUDED:
#   - morphometrics (elevation, slope, tri, tpi): basic terrain context.
#   - flow_acc: basin-size proxy.
#   - paleochannel_likelihood: v3 uses the same composite raster for both
#     populations; per-population REM/LRM/GMI split is deferred to v3.5
#     (Phase B.3 in the plan, but unlanded as of this commit).
#   - tertiary_terrace_likelihood: bench geometry signal.
#   - distance_to_lode_m: omnidirectional lode proximity (sigma 12 km).
#   - is_quaternary_alluvium: ANTITHESIS — Tertiary deep-gravels are NOT in
#     modern alluvium; the model should learn the negative association.
#   - lithology_class: bedrock context (expanded to one-hot at training).
#   - magnetic, gravity: lode-source-favorability inputs.
#   - catchment_au_hawkes: secondary geochem signal.
#   - hydraulic_pit_proximity_m_buffered: NaN within 1 km of any pit polygon
#     to prevent the v2 label leak (positives ARE pits; unbuffered distance
#     equals the inverse of the label at the positive cell). Buffered variant
#     keeps the generalization signal away from the polygon footprint.
#   - plan_curvature, profile_curvature: v3 B.6 second-derivative DEM features
#     for bench identification. Optional.
#   - distance_to_lithological_contact_m: v3 B.6 contact-zone proximity.
#
# DROPPED (relative to v2 shared stack):
#   - hydraulic_pit_proximity_m (unbuffered): the v2 label-leak feature.
#   - spi_band, ksn, twi: Quaternary-modern-channel features that do not
#     characterize Tertiary terraces.
#   - geomorphon_terrace_mask: duplicate of TPI signal at lower resolution.
#   - distance_downstream_from_lode_m: NaN at >98% of Tertiary cells (deep
#     gravels are paleo-channels offset from the modern Mother Lode trend,
#     not flow-routed downstream of it).
#   - *_5km buffered geochem (au_mean_5km, etc.): superseded by catchment
#     Hawkes.
#
# HELD FOR FUTURE (v3.5 / v4):
#   - paleochannel_likelihood_tertiary (per-population REM composite per B.3)
#   - huc12_id categorical aggregation
#   - depth_to_bedrock_cm (gSSURGO, Phase D.6 if fetched in time)
#   - parent_material_code (gSSURGO)
TERTIARY_FEATURE_COLUMNS: tuple[str, ...] = (
    "elevation", "slope", "tri", "tpi",
    "flow_acc",
    "paleochannel_likelihood",  # v3 same composite; per-pop split is v3.5
    "tertiary_terrace_likelihood",
    "distance_to_lode_m",
    "is_quaternary_alluvium",  # antithesis signal for Tertiary
    "lithology_class",
    "magnetic", "gravity",
    "catchment_au_hawkes",
    "hydraulic_pit_proximity_m_buffered",  # buffered to prevent label leak
    # v3 B.6 new features (optional; if absent in the parquet, skipped gracefully)
    "plan_curvature", "profile_curvature",
    "distance_to_lithological_contact_m",
)


# Quaternary modern-channel placer population. Positives = 573 MRDS placer
# points along modern streams; geology = active stream alluvium.
#
# INCLUDED:
#   - morphometrics (elevation, slope, tri, tpi): basic terrain context.
#   - flow_acc: drainage-area / stream-order proxy.
#   - spi_band: Roy/Upton/Craw 2018 stream-power-competence band.
#   - twi: wetness / saturation index.
#   - ksn: knickpoint steepness, gold-trap signal.
#   - paleochannel_likelihood: same composite raster (v3.5 will split).
#   - distance_downstream_from_lode_m: flow-routed lode proximity
#     (sigma 20 km). Catches Sacramento Valley distal placers.
#   - distance_to_lode_m: omnidirectional companion (sigma 12 km).
#   - catchment_au_hawkes, catchment_as_hawkes, catchment_sb_hawkes:
#     pathfinder-element catchment aggregations.
#   - lithology_class: bedrock context (one-hot at training).
#   - magnetic, gravity: lode-source-favorability inputs.
#   - distance_to_lithological_contact_m: v3 B.6 contact-zone proximity.
#
# DROPPED (relative to v2 shared stack):
#   - is_quaternary_alluvium: LABEL LEAK. MRDS placer records sit in Qa/Qal
#     polygons by definition; the feature trivially encodes the label.
#   - tertiary_terrace_likelihood: opposite-signal feature; high values mean
#     "not in modern channel."
#   - hydraulic_pit_proximity_m AND hydraulic_pit_proximity_m_buffered:
#     Tertiary-relevance only; pit polygons are deep-gravel features.
#   - geomorphon_terrace_mask: duplicate of TPI.
#
# HELD FOR FUTURE (v3.5 / v4):
#   - paleochannel_likelihood_quaternary (per-population REM composite)
#   - huc12_id categorical
#   - nearest_reach_stream_order, nearest_reach_arbolate_sum_km,
#     nearest_reach_slope, distance_to_nearest_reach_m (NHD VAA, Phase D.2)
#   - distance_to_quaternary_fault_m (Jennings 2010, Phase D.3)
#   - depth_to_bedrock_cm (gSSURGO)
QUATERNARY_FEATURE_COLUMNS: tuple[str, ...] = (
    "elevation", "slope", "tri", "tpi",
    "flow_acc",
    "spi_band", "twi", "ksn",
    "paleochannel_likelihood",
    "distance_downstream_from_lode_m",
    "distance_to_lode_m",
    "catchment_au_hawkes", "catchment_as_hawkes", "catchment_sb_hawkes",
    "lithology_class",
    "magnetic", "gravity",
    "distance_to_lithological_contact_m",
)
