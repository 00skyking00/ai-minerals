"""Thin importer for bearcub's standalone Hammon-local <-> WGS84 converter.

bearcub publishes a zero-dep pure-Python module at
``data/raw/bearcub_nome/bearcub_nome_georef.py`` (the published 2026-06-15
"Option A" delivery). This wrapper loads it via ``importlib`` and
re-exports the public surface, so callers in ai-minerals can write:

    from ai_minerals.data.bearcub_georef import (
        local_to_wgs84, wgs84_to_local, in_validated_envelope,
        VALIDATED_EXTENT_LOCAL_FT,
    )

instead of building a spec each time. The bearcub module stays under
``data/raw/`` because bearcub owns it and ai-minerals only reads.

Accuracy and envelope notes live on the bearcub module's docstring.
The validated extent is the Bear Cub collar bbox + 6000 ft. Calls
outside it (Molasses / Honey ~5 km north) should be masked with
``in_validated_envelope`` before use.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable


_BEARCUB_GEOREF_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "raw" / "bearcub_nome" / "bearcub_nome_georef.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_bearcub_nome_georef", _BEARCUB_GEOREF_PATH,
    )
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"bearcub_nome_georef.py not at expected path: {_BEARCUB_GEOREF_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()

local_to_wgs84: Callable[[float, float], tuple[float, float]] = _mod.local_to_wgs84
wgs84_to_local: Callable[[float, float], tuple[float, float]] = _mod.wgs84_to_local
in_validated_envelope: Callable[[float, float], bool] = _mod.in_validated_envelope
VALIDATED_EXTENT_LOCAL_FT: dict = _mod.VALIDATED_EXTENT_LOCAL_FT
