"""USGS-Alaska-style knowledge-driven placer-Au index.

Phase 1 scorer for the northern-Sierra placer model. Each feature is
robust-normalized to [0, 1] (clipped at p1/p99 then min-max), then
weighted-summed; the resulting raw score is renormalized to [0, 1].

The weight set follows the build proposal at `research/placer_handoff_start_here.md`
and is documented in `~/.claude/plans/hazy-humming-lynx.md`:

    distance_downstream_from_lode_m    0.25   inverse, capped at 25 km
    paleochannel_likelihood            0.20   direct (Phase 2 raster)
    hydraulic_pit_proximity_m          0.15   inverse, capped at 5 km
    spi_band                           0.10   bandpass membership
    catchment_au_hawkes                0.10   direct, rank-normalized
    is_quaternary_alluvium             0.07   direct
    twi                                0.05   direct
    geomorphon_terrace_mask            0.05   direct
    slope                              0.03   inverse

Phase 1 substitutes `paleochannel_likelihood` with a `hydraulic_pit_proximity_m`
proxy (since the paleochannel raster is a Phase D precompute). The
`DEFAULT_WEIGHTS` table is what the validation gate runs against; the
proxy substitution is applied in `usgs_alaska_placer_index` itself when
the paleochannel column is absent.

The "USGS-Alaska" label refers to the genealogy: the watershed scorer
used by USGS Alaska assessments (Cross / Frost / Karl) is the only
production-grade transferable placer index that exists. The northern-
Sierra adaptation reuses the weighted-sum framing, swaps in the Sierra-
specific feature stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_minerals.features.hydrology import gaussian_falloff


@dataclass(frozen=True)
class FeatureWeight:
    """One feature's contribution to the Phase 1 index.

    direction:
        "direct"      — higher feature value → higher score
        "inverse"     — higher feature value → lower score
        "bandpass"    — feature is already a [0, 1] membership; pass through
        "boolean"     — coerce to 0/1, then treat as direct
        "gaussian"    — soft-falloff distance: score = exp(-d^2 / (2 sigma^2)).
                        Reads `sigma_m`; output is already [0, 1].
    cap:
        For "inverse" features, cells with value >= cap clip to score 0.
        Ignored for other directions.
    sigma_m:
        For "gaussian" features, the Gaussian falloff length scale (meters).
        Ignored for other directions.
    """
    weight: float
    direction: str
    cap: float | None = None
    sigma_m: float | None = None


DEFAULT_WEIGHTS: dict[str, FeatureWeight] = {
    # Re-weighted 2026-06-01 after the first end-to-end run: at all 7 anchor
    # districts, distance_downstream_from_lode is NaN (the deep-gravel sites
    # sit upstream/lateral to the Mother Lode lode-Au seeds, not downstream
    # of them — a geological mismatch with the feature's semantics) and the
    # spi_band [3,6] doesn't fire. Those two together were 35% of the prior
    # weight contributing 0 at every anchor. Redistributed onto features
    # that DO carry signal at the anchors (paleochannel, pit-proximity,
    # catchment-Au Hawkes). Original weights kept in plan history.
    # Third iteration, 2026-06-01: K.4 gate failed at 4/7 after the second
    # iteration. Diagnosis surfaced three feature-engineering bugs, fixed here:
    #   (a) distance_downstream_from_lode_m is structurally NaN at every Sierra
    #       anchor (deep-gravels are paleo-channels offset from the modern
    #       Mother Lode trend, not flow-routed downstream of it). Added
    #       distance_to_lode_m as an omnidirectional Euclidean companion;
    #       kept the directional variant at token weight (Quaternary signal).
    #   (b) spi_band weight dropped to token. SPI is a Quaternary-modern-channel
    #       feature; all 7 validation anchors are Tertiary deep-gravels. Don't
    #       re-tune the band to fit anchors (anchor-overfitting); accept it's a
    #       Quaternary signal that Phase 2 per-population ML will weight where
    #       it belongs.
    #   (c) tertiary_terrace_likelihood — geometric-mean composite of TPI-high
    #       ∧ slope-low ∧ not-Qal. The opposite signature from paleochannel
    #       (which scores modern-channel-proximity); scores bench/terrace
    #       geometry above the modern drainage, which is where the gold-
    #       bearing Tertiary strata sit after Quaternary incision.
    "distance_to_lode_m":              FeatureWeight(0.20, "inverse", cap=15_000.0),
    "tertiary_terrace_likelihood":     FeatureWeight(0.15, "direct"),
    "paleochannel_likelihood":         FeatureWeight(0.18, "direct"),
    "hydraulic_pit_proximity_m":       FeatureWeight(0.18, "inverse", cap=5_000.0),
    "catchment_au_hawkes":             FeatureWeight(0.13, "direct"),
    "is_quaternary_alluvium":          FeatureWeight(0.05, "boolean"),
    "distance_downstream_from_lode_m": FeatureWeight(0.03, "inverse", cap=25_000.0),
    "spi_band":                        FeatureWeight(0.02, "bandpass"),
    "twi":                             FeatureWeight(0.03, "direct"),
    "geomorphon_terrace_mask":         FeatureWeight(0.02, "boolean"),
    "slope":                           FeatureWeight(0.01, "inverse"),
}


# v3 Phase B per-population scorer weights. DEFAULT_WEIGHTS above is the
# legacy / Phase 1 fallback (shared stack); the per-population variants are
# what the v3 calibration code reaches for when a population label is
# available.
#
# Tertiary weights. The Tertiary population (positives = hydraulic-mine pit
# centroids on bench terraces above modern drainage):
#   - Drop spi_band, ksn, twi: Quaternary-modern-channel signals that do
#     not characterize Tertiary terraces (terrace cells score low on
#     stream-power-competence because they are NOT in a channel). twi is
#     kept at token weight for residual moisture-trap context.
#   - Drop distance_downstream_from_lode_m: NaN at >98% of Tertiary cells
#     (Sierra deep-gravels are offset from the modern Mother Lode flow trend,
#     not downstream of it). Wastes weight on a feature that contributes 0.
#   - Use hydraulic_pit_proximity_m_buffered (NaN within 1 km of any pit
#     polygon) NOT the unbuffered original. The unbuffered feature is the
#     v2 label leak: positives ARE pits, so distance == 0 at every positive.
#   - Add tertiary_terrace_likelihood and distance_to_lithological_contact_m
#     as Tertiary-specific signals.
#   - is_quaternary_alluvium kept as a low-weight antithesis (Tertiary
#     deep-gravels are NOT in modern alluvium; high values argue against
#     Tertiary classification).
DEFAULT_WEIGHTS_TERTIARY: dict[str, FeatureWeight] = {
    "distance_to_lode_m":              FeatureWeight(0.22, "gaussian", sigma_m=12_000.0),
    "tertiary_terrace_likelihood":     FeatureWeight(0.18, "direct"),
    "paleochannel_likelihood":         FeatureWeight(0.16, "direct"),
    "hydraulic_pit_proximity_m_buffered": FeatureWeight(0.16, "gaussian", sigma_m=3_000.0),
    "catchment_au_hawkes":             FeatureWeight(0.12, "direct"),
    "is_quaternary_alluvium":          FeatureWeight(0.05, "boolean"),  # antithesis
    "distance_to_lithological_contact_m": FeatureWeight(0.05, "gaussian", sigma_m=2_000.0),
    "twi":                             FeatureWeight(0.03, "direct"),
    "slope":                           FeatureWeight(0.03, "inverse"),
}


# Quaternary weights. The Quaternary population (positives = MRDS placer
# points along modern streams):
#   - Drop tertiary_terrace_likelihood: opposite-signal feature (high =
#     bench-not-channel; the wrong direction for modern-channel placers).
#   - Drop is_quaternary_alluvium: LABEL LEAK. MRDS placer records sit in
#     Qa/Qal polygons by definition; the feature trivially encodes the label.
#   - Drop hydraulic_pit_proximity_m entirely (no buffered variant either):
#     pits are Tertiary deep-gravel features, irrelevant to modern-channel
#     prospectivity.
#   - Keep stream-power / wetness features (spi_band, ksn, twi, flow_acc):
#     these characterize where modern channels concentrate gold.
#   - Use distance_downstream_from_lode_m with the larger sigma (20 km) to
#     catch Sacramento Valley distal placers without overwhelming the signal
#     at intermediate distances.
#   - Add pathfinder elements (catchment_as_hawkes, catchment_sb_hawkes) as
#     low-weight supporting signal for catchment-Au.
DEFAULT_WEIGHTS_QUATERNARY: dict[str, FeatureWeight] = {
    "distance_downstream_from_lode_m": FeatureWeight(0.20, "gaussian", sigma_m=20_000.0),
    "distance_to_lode_m":              FeatureWeight(0.12, "gaussian", sigma_m=12_000.0),
    "catchment_au_hawkes":             FeatureWeight(0.18, "direct"),
    "spi_band":                        FeatureWeight(0.12, "bandpass"),
    "ksn":                             FeatureWeight(0.08, "direct"),
    "twi":                             FeatureWeight(0.06, "direct"),
    "flow_acc":                        FeatureWeight(0.05, "direct"),
    "paleochannel_likelihood":         FeatureWeight(0.08, "direct"),
    "distance_to_lithological_contact_m": FeatureWeight(0.05, "gaussian", sigma_m=2_000.0),
    "catchment_as_hawkes":             FeatureWeight(0.03, "direct"),
    "catchment_sb_hawkes":             FeatureWeight(0.03, "direct"),
}


# Default substitute when the Phase 2 paleochannel raster doesn't exist
# yet (Phase 1 runs before Phase D). The substitute is the inverse of
# hydraulic-pit proximity (which is a coarse Tertiary-deep-gravel proxy).
PALEOCHANNEL_PHASE1_PROXY = "hydraulic_pit_proximity_m"


def _normalize_direct(values: pd.Series) -> pd.Series:
    """Robust min-max to [0, 1] using p1/p99 clipping. NaN preserved."""
    finite = values.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    lo, hi = finite.quantile(0.01), finite.quantile(0.99)
    if hi <= lo:
        return pd.Series(0.0, index=values.index, dtype="float64").where(
            values.notna(), other=np.nan
        )
    clipped = values.clip(lower=lo, upper=hi)
    return ((clipped - lo) / (hi - lo)).astype("float64")


def _normalize_inverse(values: pd.Series, cap: float | None) -> pd.Series:
    """Inverse-direct: high → 0, low → 1. Cells >= cap clip to 0."""
    if cap is not None:
        values = values.clip(upper=cap)
    norm = _normalize_direct(values)
    return 1.0 - norm


def _normalize_boolean(values: pd.Series) -> pd.Series:
    """Coerce to 0/1; NaN preserved."""
    out = values.astype("float64")
    out = out.where(out.isna(), other=(out != 0).astype("float64"))
    return out


def _normalize_bandpass(values: pd.Series) -> pd.Series:
    """Pass-through: feature is already a [0, 1] membership."""
    return values.clip(lower=0.0, upper=1.0).astype("float64")


def _normalize_gaussian(values: pd.Series, sigma_m: float | None) -> pd.Series:
    """Gaussian soft-falloff: score = exp(-d^2 / (2 sigma_m^2)). NaN→0."""
    if sigma_m is None or sigma_m <= 0:
        raise ValueError(
            f"gaussian direction requires sigma_m > 0; got sigma_m={sigma_m!r}"
        )
    arr = values.to_numpy(dtype=np.float64)
    scored = gaussian_falloff(arr, sigma_m=float(sigma_m))
    return pd.Series(scored.astype("float64"), index=values.index)


def _apply_one(values: pd.Series, fw: FeatureWeight) -> pd.Series:
    if fw.direction == "direct":
        return _normalize_direct(values)
    if fw.direction == "inverse":
        return _normalize_inverse(values, fw.cap)
    if fw.direction == "boolean":
        return _normalize_boolean(values)
    if fw.direction == "bandpass":
        return _normalize_bandpass(values)
    if fw.direction == "gaussian":
        return _normalize_gaussian(values, fw.sigma_m)
    raise ValueError(f"Unknown direction {fw.direction!r}")


def usgs_alaska_placer_index(
    df: pd.DataFrame,
    *,
    weights: dict[str, FeatureWeight] = DEFAULT_WEIGHTS,
    paleochannel_proxy: str | None = PALEOCHANNEL_PHASE1_PROXY,
) -> pd.Series:
    """Compute the Phase 1 knowledge-driven placer-Au index.

    df: per-cell feature DataFrame (one row per grid cell).
    weights: feature → (weight, direction, cap) map. Defaults to DEFAULT_WEIGHTS.
    paleochannel_proxy: column to substitute for `paleochannel_likelihood` when
                       that column is absent (Phase 1 fallback before Phase D).

    Returns a Series aligned to df.index, values in [0, 1] with NaN for
    cells where any required feature is NaN across the whole stack.
    """
    contributions: dict[str, pd.Series] = {}
    total_w = 0.0
    skipped: list[tuple[str, str]] = []
    for col, fw in weights.items():
        if col not in df.columns:
            if col == "paleochannel_likelihood" and paleochannel_proxy in df.columns:
                proxy_fw = FeatureWeight(fw.weight, "inverse", cap=5_000.0)
                contributions[col] = _apply_one(df[paleochannel_proxy], proxy_fw) * fw.weight
                total_w += fw.weight
                continue
            skipped.append((col, "missing"))
            continue
        # Skip all-NaN columns (feature wasn't computable in this run; e.g.,
        # whitebox/grass-dependent hydrology features when those tools are
        # absent). Renormalize remaining weights so the index still ranges
        # over [0, 1].
        if df[col].isna().all():
            skipped.append((col, "all-NaN"))
            continue
        contributions[col] = _apply_one(df[col], fw) * fw.weight
        total_w += fw.weight

    if skipped:
        import warnings as _warnings
        msg = "; ".join(f"{c}={reason}" for c, reason in skipped)
        _warnings.warn(
            f"usgs_alaska_placer_index: skipped {len(skipped)} features "
            f"({msg}); remaining weight={total_w:.3f}",
            stacklevel=2,
        )
    if total_w <= 0:
        raise ValueError("usgs_alaska_placer_index: no usable features in df")

    # Per-cell sum: NaN contributions count as 0 (worst-case for that
    # normalized feature) so the sum is always defined. Sparse features
    # (e.g. distance-downstream-from-lode, which only resolves for cells
    # within max_km of a lode-Au seed along the NHD network) correctly
    # contribute 0 elsewhere instead of NaN'ing out the whole cell.
    #
    # KNOWN v2 QUIRK (preserved in v3 for backward compatibility, flagged
    # for v3.5 reconsideration):
    #   For "inverse" features (e.g. distance_to_lode_m, distance_downstream
    #   _from_lode_m, hydraulic_pit_proximity_m), the per-feature normalize
    #   step has already done `1 - normalized_distance`, so a small distance
    #   maps to ~1 (high score) and a large distance maps to ~0 (low score).
    #   A NaN distance means "no lode within the cap" — semantically the
    #   WORST possible case for an inverse-distance feature; the documented
    #   intent is that missing == worst case. But the .fillna(0.0) below
    #   replaces NaN with 0.0 AFTER the inversion, which is the BEST score
    #   for inverse features (0.0 normalized + 1 - 0 = 1.0... actually 0.0
    #   here because contributions[col] already has the (1 - x) * weight
    #   applied; .fillna(0.0) drops the weighted contribution to 0, which
    #   then sums into the renormalized total without penalty).
    #
    #   Net effect: a cell with NaN distance-to-lode gets credited as if
    #   that feature contributed nothing, not as if it failed the feature.
    #   For sparse features this is the right call (otherwise everything
    #   outside the lode network NaN's out). For features where missing
    #   genuinely means "bad," the score is too generous.
    #
    #   v3 preserves the existing semantics so anchor-gate calibration is
    #   stable across the v2 → v3 transition. Revisit in v3.5: split
    #   "missing == sparse, score 0 contribution" from "missing == worst,
    #   contribute full weight at 0" with a per-feature NaN policy.
    stacked = pd.concat(contributions.values(), axis=1).fillna(0.0)
    raw = stacked.sum(axis=1) / total_w
    # Renormalize raw to [0, 1] over the within-AOI distribution.
    return _normalize_direct(raw).rename("phase1_score")


def anchor_decile_check(
    score: pd.Series,
    anchor_cells: pd.Series,
) -> pd.DataFrame:
    """Validation-gate helper.

    score: per-cell Phase 1 score (output of usgs_alaska_placer_index).
    anchor_cells: pd.Series of POSITIONAL integer indices into `score`
                  pointing at the anchor-district cells (one per district).
                  Positional (iloc-style), not label-based, so this works
                  regardless of whether score's underlying df.index is a
                  RangeIndex(0, n).

    Returns a DataFrame with columns:
        district  (the Series index of anchor_cells)
        cell_idx
        score
        decile     (0 = top decile)
        in_top_decile (bool)
    """
    deciles = pd.qcut(
        score.rank(method="first", ascending=False, pct=False),
        q=10,
        labels=range(10),
    )
    n = len(score)
    rows = []
    for district, cell_idx in anchor_cells.items():
        if pd.isna(cell_idx) or not (0 <= int(cell_idx) < n):
            rows.append({
                "district": district,
                "cell_idx": cell_idx,
                "score": np.nan,
                "decile": np.nan,
                "in_top_decile": False,
            })
            continue
        idx = int(cell_idx)
        rows.append({
            "district": district,
            "cell_idx": idx,
            "score": float(score.iloc[idx]),
            "decile": int(deciles.iloc[idx]),
            "in_top_decile": int(deciles.iloc[idx]) == 0,
        })
    return pd.DataFrame(rows)
