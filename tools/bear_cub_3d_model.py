"""Driver: render Bear Cub 3D model views.

Outputs to data/derived/bear_cub_resource/:

  - fig_3d_oblique.png       — oblique-from-south
  - fig_3d_plan.png          — plan/top-down
  - fig_3d_profile_east.png  — profile from west, looking east
  - bear_cub_3d.html         — interactive (rotate/pan/zoom in browser)

Usage:
    uv run python tools/bear_cub_3d_model.py
"""

from __future__ import annotations

from pathlib import Path

from ai_minerals.bear_cub.model_3d import load_inputs, render_views

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "derived" / "bear_cub_resource"


def main() -> None:
    print("Loading Bear Cub inputs ...")
    inputs = load_inputs(REPO)
    print(f"  {inputs.n_holes} holes, "
          f"{len(inputs.intervals)} intervals, "
          f"4 patent corners")

    print("\nRendering 3D views (PyVista offscreen) ...")
    paths = render_views(inputs, OUT)
    for label, p in paths.items():
        print(f"  → {label:>12s}: {p.relative_to(REPO)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
