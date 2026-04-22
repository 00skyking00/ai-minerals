"""Shared helpers for data fetchers."""

from __future__ import annotations

import datetime as dt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DERIVED = PROJECT_ROOT / "data" / "derived"


def dataset_dir(name: str) -> Path:
    d = DATA_RAW / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_source_md(
    dataset: str,
    *,
    title: str,
    url: str,
    license: str,
    notes: str = "",
) -> Path:
    """Record where a dataset came from. Every fetch writes one of these."""
    path = dataset_dir(dataset) / "SOURCE.md"
    today = dt.date.today().isoformat()
    body = f"""# {title}

- **Dataset key:** `{dataset}`
- **Source URL:** <{url}>
- **Retrieved:** {today}
- **License:** {license}
"""
    if notes:
        body += f"\n## Notes\n\n{notes.strip()}\n"
    path.write_text(body)
    return path
