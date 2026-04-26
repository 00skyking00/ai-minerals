"""Bear Cub placer-Au pilot subproject (Nome Placer Fields).

Stand-alone helpers for the 24-hole Murray drill-log archive: georeference
helpers, IDW bedrock-surface interpolation, PyVista 3D scene assembly.

This module deliberately does NOT depend on the regional `ai_minerals.regions`
machinery — the Bear Cub pilot operates on a single small AOI with bespoke
local-grid coordinates, and shoehorning it into the porphyry-MPM Region
framework would obscure the workflow.
"""
