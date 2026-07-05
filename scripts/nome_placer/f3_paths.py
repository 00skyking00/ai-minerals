"""Shared KG-input paths for the F3 marginal-lift experiment.

Both F3 drivers (``f3_kg_marginal_lift`` and ``f3_kg_loo``) read the fossick
knowledge-graph exports from the same two locations. Keep them in ONE place so
the cross-repo path cannot drift between the scripts. It did once: the drivers
pointed at a stale ``fossick-f2`` directory that no longer existed after the
fossick lane consolidated its exports back to ``fossick/``.

``KG_FEATURES_DIR`` is a READ-ONLY cross-repo read from the sibling fossick lane.
F3 does not copy the KG feature exports into ai-minerals.
"""
from __future__ import annotations

from pathlib import Path

# fossick KG feature exports (feature_table.csv, prose_features.jsonl, ...);
# read-only cross-repo read, not copied into this repo.
KG_FEATURES_DIR = Path("/home/sky/src/learning/fossick/exports/features")

# local raw copy of the KG JSON-LD, used only to resolve ground geometries
# (point / polygon); load_feature_entities falls back to longitude/latitude.
KG_JSONLD = Path("data/raw/fossick_kg/kg_nome.jsonld")
