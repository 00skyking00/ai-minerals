"""Phase 1 fuzzy-overlay population scorer for the Nome placer model.

Five populations:

- **BL (true beach)**: cells sitting near a true-beach stand elevation
  (Present, Second, Third per Metcalfe and Tuck 1942). Wave-sorted fine
  gold on the beach face.
- **AP (abrasion platform / sloughover)**: cells sitting near an
  abrasion-platform stand elevation (Submarine outer, Submarine inner,
  Submarine deep, Intermediate, Monroeville). Coarse gold lag-behind on
  the sub-aqueous platform. Tuck 1942 distinguishes these from true
  beaches; they account for most district production.
- **BC (beach-stream confluence)**: BL or AP cells intersected by a
  stream channel. Sky's calibration: Bear Cub MS 1178 sits on Third
  Beach + Bear Cub Creek confluence. Not scorable in v0; needs the
  stream-distance feature.
- **QM (off-beach modern creek)**: cells inside an Iron Creek or Nome
  River drift polygon within a stream-distance band. Not scorable in
  v0; needs the drift mask + stream-distance feature.
- **TB (Tertiary buried high-bench)**: cells sitting near the +300 ft or
  +600 ft Tuck stands. The +600 ft stand is the high-bench-gravel level
  at the head of Dexter Creek (Molasses MS 1179 + Honey MS 1181 setting).

Each population's score is a Gaussian fuzzy membership over
elevation-relative-to-stand:

  score_pop(cell) = max over s in stands(pop) of
                    exp(-(elev_rel_to_stand_s(cell))^2 / (2 * sigma_pop^2))

Final per-cell score combines populations with max(), mirroring the
Sierra placer two-population fusion. Population identities are
preserved per-cell so the chapter can show per-population maps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ai_minerals.features.coastal import (
    STAND_ELEVATIONS_FT,
    elevation_relative_to_stand_m,
)
from ai_minerals.grid import Grid


#: True-beach population stands. Tuck 1942 names Present + Second + Third
#: as the only "true" beach concentrations. Fourth Beach is included here
#: as well: Tuck on p.28 describes the four flat terrace ridges at +120 ft
#: between the major rivers as planed-off marine surfaces, so the cells
#: at Fourth Beach elevation are legitimately near a planed-off marine
#: bench. ``present`` is not in STAND_ELEVATIONS_FT because the present
#: sea level is the modern shoreline; the BL scorer treats elevation 0
#: as a stand.
BL_STANDS: tuple[str, ...] = ("second", "third", "fourth")
BL_PRESENT_ELEVATION_M: float = 0.0  # modern shoreline

#: Abrasion-platform / sloughover stands (Tuck 1942: Submarine + Intermediate
#: + Monroeville). Coarse-gold lag-behind on the sub-aqueous platform.
AP_STANDS: tuple[str, ...] = (
    "submarine_outer",
    "submarine_inner",
    "submarine_deep",
    "intermediate",
    "monroeville",
)

#: Tertiary buried high-bench stands. The +600 ft stand is the head-of-
#: Dexter-Creek high-bench-gravel level; +300 ft is the Anvil-left-limit
#: bench cut in limestone. Sky's Molasses / Honey claims map onto +600 ft.
TB_STANDS: tuple[str, ...] = ("tuck_high_300", "tuck_high_600")



@dataclass(frozen=True)
class CoastalScores:
    """Per-cell population scores from the Phase 1 fuzzy-overlay model.

    Each Series is indexed identically to ``grid.centroid_gdf()``.
    Values are in [0, 1]; higher means stronger membership.
    """
    bl: pd.Series      # true beach
    ap: pd.Series      # abrasion-platform / sloughover
    tb: pd.Series      # Tertiary buried high-bench
    composite: pd.Series  # per-cell max over scored populations


def _gaussian_membership(
    elev_rel_m: np.ndarray, sigma_m: float
) -> np.ndarray:
    """Gaussian fuzzy membership: 1 where elev_rel == 0, falls off with sigma."""
    return np.exp(-(elev_rel_m.astype(np.float64) ** 2) / (2.0 * sigma_m ** 2))


def _score_population_over_stands(
    dem: xr.DataArray,
    grid: Grid,
    stand_names: tuple[str, ...],
    sigma_m: float,
    extra_zero_elevations_m: tuple[float, ...] = (),
) -> pd.Series:
    """Max over stands of Gaussian membership in elevation-relative space.

    ``extra_zero_elevations_m`` lets the caller add ad-hoc stand
    elevations (e.g. the modern shoreline at 0 m) that aren't in the
    canonical STAND_ELEVATIONS_FT dict.
    """
    all_rel: list[np.ndarray] = []
    for stand in stand_names:
        rel = elevation_relative_to_stand_m(dem, grid, stand)
        all_rel.append(rel.to_numpy())
    for elev_m in extra_zero_elevations_m:
        # Build an elev_relative_to_<elev_m> series by reusing the
        # nearest-pixel sampling of any one stand's helper (cheap; the
        # sampling work is the same).
        dummy_stand = next(iter(STAND_ELEVATIONS_FT))
        rel_to_dummy = elevation_relative_to_stand_m(
            dem, grid, dummy_stand
        ).to_numpy()
        # rel_to_dummy = dem - dummy_elev_m; we want rel_to_target =
        # dem - target_elev_m = rel_to_dummy + dummy_elev_m - target_elev_m.
        from ai_minerals.features.coastal import stand_elevation_m
        offset = stand_elevation_m(dummy_stand) - elev_m
        all_rel.append(rel_to_dummy + offset)

    stacked = np.stack(all_rel, axis=0)
    membership = _gaussian_membership(stacked, sigma_m=sigma_m)
    best = membership.max(axis=0)
    return pd.Series(best.astype(np.float32), index=grid.centroid_gdf().index)


def score_bl(
    dem: xr.DataArray, grid: Grid, sigma_m: float = 5.0
) -> pd.Series:
    """True-beach (BL) population score, max over (Present, Second, Third).

    Default sigma = 5 m: cells within ~10 m of one BL stand elevation get
    membership >= 0.14. Higher sigma broadens; lower sharpens.
    """
    return _score_population_over_stands(
        dem,
        grid,
        stand_names=BL_STANDS,
        sigma_m=sigma_m,
        extra_zero_elevations_m=(BL_PRESENT_ELEVATION_M,),
    ).rename("score_bl")


def score_ap(
    dem: xr.DataArray, grid: Grid, sigma_m: float = 6.0
) -> pd.Series:
    """Abrasion-platform / sloughover (AP) population score.

    Default sigma = 6 m: AP bands are wider because the sub-aqueous
    sloughover zone is less sharp than a beach.
    """
    return _score_population_over_stands(
        dem, grid, stand_names=AP_STANDS, sigma_m=sigma_m,
    ).rename("score_ap")


def score_tb(
    dem: xr.DataArray, grid: Grid, sigma_m: float = 12.0
) -> pd.Series:
    """Tertiary buried high-bench (TB) population score.

    Default sigma = 12 m: high-bench gravels at the +300 ft and +600 ft
    Tuck stands cover a broader elevation envelope than true beaches
    because the planed-off surfaces are wider terrace surfaces, not
    sharp wave benches.
    """
    return _score_population_over_stands(
        dem, grid, stand_names=TB_STANDS, sigma_m=sigma_m,
    ).rename("score_tb")


def score_all_populations(
    dem: xr.DataArray,
    grid: Grid,
    *,
    bl_sigma_m: float = 5.0,
    ap_sigma_m: float = 6.0,
    tb_sigma_m: float = 12.0,
) -> CoastalScores:
    """Score BL, AP, TB populations and the per-cell max() composite.

    BC and QM require the stream-distance feature and are scored in a
    later phase; the v0 composite covers only BL, AP, TB.
    """
    bl = score_bl(dem, grid, sigma_m=bl_sigma_m)
    ap = score_ap(dem, grid, sigma_m=ap_sigma_m)
    tb = score_tb(dem, grid, sigma_m=tb_sigma_m)
    composite = pd.Series(
        np.stack([bl.to_numpy(), ap.to_numpy(), tb.to_numpy()], axis=0).max(axis=0),
        index=grid.centroid_gdf().index,
        name="score_composite",
    )
    return CoastalScores(bl=bl, ap=ap, tb=tb, composite=composite)
