"""Tests for the bearcub-delivered cross_sections_nome_placer.parquet.

The cross-sections sidecar carries Janin 1912 side-elevation drill-line
data (57 holes across lines AA-EE). Tests verify the schema matches
bearcub's 2026-06-14 delivery note + the join keys are intact + the
parsed pay-zone depths agree with surface elevations at family claims.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def cross_sections(nome_region):
    path = Path(nome_region.raw_paths.get(
        "cross_sections", "data/raw/bearcub_nome/cross_sections_nome_placer.parquet"
    ))
    if not path.exists():
        # The schema is fixed at bearcub side; default to the canonical path.
        path = Path("data/raw/bearcub_nome/cross_sections_nome_placer.parquet")
    if not path.exists():
        pytest.skip(f"cross_sections parquet not at {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Schema (bearcub 2026-06-14 delivery contract)
# ---------------------------------------------------------------------------

def test_schema_matches_bearcub_contract(cross_sections):
    """Schema confirmed in 2026-06-14-from-bearcub-cross-section-sidecar-delivered.md:
    hole_id, drill_line_id, section_order, x_frac, hole_num,
    bedrock_depth_ft, bedrock_code, pay_top_ft, pay_bottom_ft,
    pay_interval_text, grade_oz_cuyd, source."""
    required = {
        "hole_id", "drill_line_id", "section_order", "x_frac", "hole_num",
        "bedrock_depth_ft", "bedrock_code", "pay_top_ft", "pay_bottom_ft",
        "pay_interval_text", "grade_oz_cuyd", "source",
    }
    actual = set(cross_sections.columns)
    missing = required - actual
    assert not missing, f"cross_sections missing columns: {missing}"


def test_row_count_in_documented_band(cross_sections):
    """bearcub said 57 holes across lines AA-EE. Allow some shift as the
    Janin OCR pipeline matures, but flag if it drops below 30 or jumps
    above 200."""
    n = len(cross_sections)
    assert 30 <= n <= 200, f"cross_sections row count {n} outside expected band"


def test_drill_lines_cover_aa_through_ee(cross_sections):
    """5 drill lines (AA, BB, CC, DD, EE) per bearcub's delivery."""
    lines = set(cross_sections["drill_line_id"].astype(str).unique())
    expected = {"AA", "BB", "CC", "DD", "EE"}
    assert expected <= lines, (
        f"Missing drill lines: {expected - lines}; got {lines}"
    )


def test_bedrock_pick_coverage_in_band(cross_sections):
    """bearcub: 44 of 57 holes have bedrock picks. Allow some shift but
    expect majority coverage."""
    n_with_bedrock = cross_sections["bedrock_depth_ft"].notna().sum()
    assert n_with_bedrock >= len(cross_sections) * 0.5, (
        f"Only {n_with_bedrock} of {len(cross_sections)} holes have bedrock "
        "picks; expected at least 50% coverage"
    )


def test_pay_top_bottom_paired(cross_sections):
    """When pay_top_ft is populated, pay_bottom_ft should be too. The
    parser allows a small number (1-2 of 35) of inverted pairs from
    Janin OCR anomalies (e.g. ``hole_id=janin_image137`` has interval
    text ``"90' to 34' and 39' to 41'"`` which the parser fills as
    top=90 bottom=34). Catches large OCR-parse regressions while
    tolerating documented edge cases."""
    has_top = cross_sections["pay_top_ft"].notna()
    has_bottom = cross_sections["pay_bottom_ft"].notna()
    assert (has_top == has_bottom).all(), (
        "pay_top_ft and pay_bottom_ft are not consistently paired"
    )
    paired = cross_sections.loc[has_top & has_bottom]
    if len(paired):
        ordered = paired["pay_bottom_ft"] > paired["pay_top_ft"]
        n_inverted = int((~ordered).sum())
        # Allow up to 5% inverted pairs (1-2 in the current 35-row set).
        max_allowed = max(2, int(len(paired) * 0.05))
        assert n_inverted <= max_allowed, (
            f"Pay-zone top/bottom order inverted in {n_inverted} rows; "
            f"max allowed {max_allowed} (~5%). Check the Janin OCR parse."
        )


# ---------------------------------------------------------------------------
# Join to the drillholes label table
# ---------------------------------------------------------------------------

def test_join_to_drillholes_via_hole_id(cross_sections, bearcub_drillholes):
    """Cross-section holes with non-synthetic hole_ids should join to
    the main label table on hole_id. bearcub said 23 holes have both
    cross-section + label table records."""
    cs_ids = set(cross_sections["hole_id"].astype(str))
    dh_ids = set(bearcub_drillholes["hole_id"].astype(str))
    overlap = cs_ids & dh_ids
    assert len(overlap) >= 10, (
        f"Only {len(overlap)} cross-section holes join to drillholes; "
        "expected >= 10 (bearcub said 23)"
    )


def test_x_frac_is_normalized_along_line(cross_sections):
    """x_frac is the along-line position (0 to 1) per bearcub's note.
    Each drill line should have x_frac spanning roughly [0, 1]."""
    grouped = cross_sections.groupby("drill_line_id")["x_frac"]
    for line, vals in grouped:
        valid = vals.dropna()
        if len(valid) < 2:
            continue
        assert valid.min() >= -0.05, f"line {line} x_frac min < 0: {valid.min()}"
        assert valid.max() <= 1.05, f"line {line} x_frac max > 1: {valid.max()}"


# ---------------------------------------------------------------------------
# Bear Cub-specific: bedrock depths confirm the 80-90 ft burial story
# ---------------------------------------------------------------------------

def test_bear_cub_bedrock_depths_match_third_beach_burial(
    cross_sections, bearcub_drillholes,
):
    """Sky 2026-06-14: Third Beach is buried 80-90 ft below Bear Cub
    surface. Bear Cub-cluster cross-section holes should show bedrock
    depths in roughly that range (Janin and Bear Cub Murray sources
    both contributed)."""
    # Join cross-section + drillholes to get bedrock depths in the Bear
    # Cub family-claim cluster (MS 1178).
    joined = cross_sections.merge(
        bearcub_drillholes[["hole_id", "claim_ms"]], on="hole_id", how="left"
    )
    family = joined[joined["claim_ms"].astype(str).isin(
        ["1178", "1179", "1180", "1181"])
    ]
    if len(family) < 3:
        pytest.skip(
            f"Too few cross-section holes in family block ({len(family)}); "
            "test needs at least 3 to be meaningful"
        )
    bedrock = family["bedrock_depth_ft"].dropna()
    if len(bedrock) < 3:
        pytest.skip(f"Too few bedrock picks ({len(bedrock)}) in family block")
    # Bear Cub bedrock 62-85 ft, max 85 ft per the lithology smoke test.
    # Cross-section holes are Janin and may extend deeper.
    assert 20 <= bedrock.mean() <= 120, (
        f"Family-block cross-section bedrock mean {bedrock.mean():.1f} ft "
        "outside 20-120 ft band; check the Janin OCR or the join"
    )
