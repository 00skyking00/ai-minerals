"""Kenorland Minerals Tanacross drill collars — manually digitized from press releases.

These are held out from training and used only for external blind-test validation
(Day 5). Kenorland's 2023-24 Tanacross drill program results postdate the MRDS
label cutoffs used for training.

Source: Kenorland Minerals press release dated March 28, 2024 — "Announces
Termination of Tanacross Project Earn-in Agreement with Antofagasta PLC and
Highlights Exploration Upside at South Taurus" -- and related 2023 releases.

Hole 23ETD062 is disclosed in the March 28, 2024 release with explicit assay
intervals; collar coords are from the accompanying figure. Other holes are
stubbed; populate from individual press releases / NI 43-101 reports as
available.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "kenorland"

# Placeholder: coordinates are approximate — figure in the press release is
# small-scale. Before using these for blind-test evaluation, verify from
# Kenorland's NI 43-101 Technical Report if published, or from their 2024
# exploration-update figures (plan maps with collar coords).
COLLARS: list[dict] = [
    {
        "hole_id": "23ETD062",
        "target": "East Taurus",
        "lon": -142.25,  # APPROXIMATE — verify
        "lat": 63.45,     # APPROXIMATE — verify
        "cu_pct": 0.14,
        "mo_pct": 0.02,
        "au_gpt": 0.05,
        "interval_m": 174.22,
        "source_url": (
            "https://www.kenorlandminerals.com/news/2024/kenorland-minerals-"
            "announces-termination-of-tanacross-project-earn-in-agreement-"
            "with-antofagasta-plc-and-highlights-exploration-upside-at-"
            "south-taurus/"
        ),
    },
]


def fetch() -> Path:
    """Write the collar stub CSV + SOURCE.md."""
    out_dir = dataset_dir(NAME)
    csv_path = out_dir / "kenorland_tanacross_collars.csv"

    fields = ["hole_id", "target", "lon", "lat", "cu_pct", "mo_pct", "au_gpt", "interval_m", "source_url"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in COLLARS:
            w.writerow({k: row.get(k, "") for k in fields})

    write_source_md(
        NAME,
        title="Kenorland Minerals Tanacross drill collars (manually digitized)",
        url=(
            "https://www.kenorlandminerals.com/projects/project-overview/?"
            "turl=tanacross"
        ),
        license=(
            "Press-release disclosures by a TSXV-listed issuer, public-domain "
            "summaries of regulatory filings. Use is limited to research/"
            "validation purposes; attribute Kenorland Minerals Ltd."
        ),
        notes=(
            "Coordinates in this stub are APPROXIMATE — extracted visually from "
            "small-scale figures in press releases. Before using for blind-test "
            "evaluation, verify against Kenorland NI 43-101 technical reports or "
            "their high-resolution plan maps."
        ),
    )
    return csv_path


if __name__ == "__main__":
    path = fetch()
    print(f"Wrote {path}")
