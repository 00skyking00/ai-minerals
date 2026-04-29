"""Vision-based OCR of drill-hole positions on the Bear Cub dh map.

Loads the 6 tiles produced by `tools/bear_cub_drillhole_picker`'s tile prep,
sends each to Claude Opus 4.7 (hi-res vision), and parses structured JSON
back into per-tile (hole_id, dot_x, dot_y) records. Translates tile-pixel
coords into 1.5x-render baseline coords (the system used everywhere else
in the Bear Cub pipeline), matches to bear_cub_collars.csv, and solves an
overdetermined affine fit local-grid-feet → dh-map pixels.

Output:
    data/raw/bear_cub/dhmap_ocr_picks.json — raw per-tile OCR + translated coords
    data/raw/bear_cub/dhmap_ocr_fit.json   — overdetermined affine + residuals

Run:
    uv run python tools/bear_cub_dhmap_ocr.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

# Tile geometry (matches what the picker emits — 3 wide × 2 tall, 1579×1456 each)
TILE_OFFSETS_8X: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (4456, 4626),
    (0, 1): (5885, 4626),
    (0, 2): (7314, 4626),
    (1, 0): (4456, 5932),
    (1, 1): (5885, 5932),
    (1, 2): (7314, 5932),
}
TILE_PATH = REPO / "data" / "raw" / "bear_cub" / "tiles_for_ocr"
COLLARS_CSV = REPO / "data" / "raw" / "bear_cub" / "bear_cub_collars.csv"
OUT_PICKS = REPO / "data" / "raw" / "bear_cub" / "dhmap_ocr_picks.json"
OUT_FIT = REPO / "data" / "raw" / "bear_cub" / "dhmap_ocr_fit.json"

# 8x→1.5x baseline scale
EIGHT_X_TO_15X = 1.5 / 8.0

# Already-rendered tile paths (from prior step in the picker setup)
EXISTING_TILES = {
    (0, 0): "/tmp/bc_q_0_0.png",
    (0, 1): "/tmp/bc_q_0_1.png",
    (0, 2): "/tmp/bc_q_0_2.png",
    (1, 0): "/tmp/bc_q_1_0.png",
    (1, 1): "/tmp/bc_q_1_1.png",
    (1, 2): "/tmp/bc_q_1_2.png",
}


class Hole(BaseModel):
    hole_id: str = Field(description="Numeric hole identifier as a string, no letter prefixes")
    dot_x: int = Field(description="Pixel x of the dot center; (0,0) top-left, x increases right")
    dot_y: int = Field(description="Pixel y of the dot center; (0,0) top-left, y increases down")


class TileResult(BaseModel):
    holes: list[Hole]


SYSTEM_PROMPT = """You are an expert at reading historical mining property maps and extracting drill-hole locations to pixel-level precision. Your task is to identify every drill-hole marker on a section of a 1908-era property map of the Cape Nome mining district, Alaska, and return its pixel coordinates as structured data.

# Map context

The map is a hand-drafted black-line drawing on white paper depicting drill holes from multiple early-1900s drilling campaigns across a placer-gold property. It has been digitized at high resolution; the section you'll see is one tile from a 6-tile grid covering the Bear Cub claim area.

# Drill-hole label format

A drill-hole label is a small open circle "○" (the position marker) accompanied by a numeric identifier and metadata. Common forms:

- `H ○ 6554 85 8.1` — Hammon Field Log: H prefix, dot, 4-digit hole ID, then bedrock depth and total depth.
- `H ○ 7754 61 10.0` — same convention.
- `AK ○ 30-36 9.3` — Alaska Gold Co.: AK prefix. The "30-36" is a depth range, NOT a hole ID. Skip these unless you can clearly identify a separate 4-digit hole number.
- `A ○ 5 76 12.5` — Auxiliary single-digit hole convention. Report as hole_id "5".
- `7754` — bare hole ID near a dot.

# Output rules

Return ONLY drill holes — for each:
- `hole_id`: the numeric identifier as a string. Strip prefix letters (H, AK, A) and trailing metadata. The hole ID is the FIRST number adjacent to the dot, not the second.
- `dot_x`, `dot_y`: integer pixel coordinates of the dot's geometric center. Origin (0,0) at upper-left, x rightward, y downward.

SKIP:
- Claim labels: "M.S. 1178", "M.S. 1349", "BEAR CUB", "URANUS FRACTION", "DRY CR", "NEWTON GULCH", "NO. 1 BELOW", etc.
- Survey-line markers: "LINE 7+700", "L 9+19", "Sta 1+25" — these are chainages, not holes.
- Section corners / property monuments (triangles or crosses, not circles).
- "AK"-prefix labels with NO 4-digit hole ID — they're depth readings.
- Holes where the dot is cropped off the tile edge.
- Holes where you can't confidently read the ID — better to omit than guess.

# Coordinate precision

Aim for ±5 px of the true dot center. Dots are 12-20 px in diameter. Place the coordinate at the visual centroid of the OPEN CIRCLE, NOT the centroid of the label text. The text is offset from the dot (usually to the right or below).

Common errors to avoid:
- Confusing a "0" digit in the label text for the dot symbol.
- Reading depth/grade numbers as the hole ID — the hole ID is the FIRST number after the dot.
- Reporting the label-text centroid instead of the dot centroid.

Be exhaustive: scan top-to-bottom, left-to-right. Report every drill hole you can confidently identify. Do not return duplicates. Return integer coordinates only."""


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def ocr_tile(client: anthropic.Anthropic, tile_path: Path) -> TileResult:
    image_b64 = encode_image(tile_path)
    response = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Tile dimensions: 1579 wide × 1456 tall. Extract every drill-hole "
                            f"position. Return all holes you can identify with confident dot-center "
                            f"localization."
                        ),
                    },
                ],
            }
        ],
        output_format=TileResult,
    )
    return response.parsed_output, response.usage


def tile_to_15x(r: int, c: int, tx: int, ty: int) -> tuple[float, float]:
    ox, oy = TILE_OFFSETS_8X[(r, c)]
    return (ox + tx) * EIGHT_X_TO_15X, (oy + ty) * EIGHT_X_TO_15X


def main() -> None:
    client = anthropic.Anthropic()
    df = pd.read_csv(COLLARS_CSV)
    df["hole_id_str"] = df["hole_id"].astype(str)
    csv_ids = set(df["hole_id_str"])

    all_picks: dict[str, dict] = {}
    raw_per_tile: dict[str, list[dict]] = {}
    total_in = total_out = 0

    for (r, c), tile_path_str in sorted(EXISTING_TILES.items()):
        tile_path = Path(tile_path_str)
        if not tile_path.exists():
            print(f"  [skip] tile ({r},{c}) not found at {tile_path}")
            continue

        print(f"  OCR'ing tile ({r},{c}) — {tile_path.name}")
        result, usage = ocr_tile(client, tile_path)
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        print(
            f"    → {len(result.holes)} holes  "
            f"(in={usage.input_tokens} out={usage.output_tokens} cache_read={cache_read})"
        )

        tile_key = f"{r},{c}"
        raw_per_tile[tile_key] = [h.model_dump() for h in result.holes]

        for h in result.holes:
            x15, y15 = tile_to_15x(r, c, h.dot_x, h.dot_y)
            entry = {
                "hole_id": h.hole_id,
                "tile": tile_key,
                "tile_xy": [h.dot_x, h.dot_y],
                "px15": [round(x15, 1), round(y15, 1)],
                "in_csv": h.hole_id in csv_ids,
            }
            # Prefer the lower-residual pick if same hole appears in two tiles (overlap zones)
            if h.hole_id not in all_picks:
                all_picks[h.hole_id] = entry

    OUT_PICKS.write_text(
        json.dumps({"raw_per_tile": raw_per_tile, "consolidated": all_picks}, indent=2)
    )
    print(f"\n  Saved raw picks → {OUT_PICKS.relative_to(REPO)}")
    print(
        f"  Total tokens: in={total_in} ({total_in*5/1e6:.4f}$), "
        f"out={total_out} ({total_out*25/1e6:.4f}$)"
    )

    # Solve overdetermined affine on holes that match the CSV
    M, T, hids = [], [], []
    for hid, entry in all_picks.items():
        if not entry["in_csv"]:
            continue
        row = df[df.hole_id_str == hid].iloc[0]
        M.append(
            [float(row["easting_local_ft"]), float(row["northing_local_ft"]), 1.0]
        )
        T.append(entry["px15"])
        hids.append(hid)

    if len(hids) < 3:
        print(f"\n  Only {len(hids)} CSV-matching picks — need ≥3 for affine. Skipping fit.")
        return

    M = np.array(M)
    T = np.array(T, dtype=float)
    A, *_ = np.linalg.lstsq(M, T, rcond=None)
    resid = np.linalg.norm(M @ A - T, axis=1)

    # Diagnostics
    ax, ay = A[0]
    bx, by = A[1]
    scale_e = float(np.hypot(ax, ay))
    scale_n = float(np.hypot(bx, by))
    cos_angle = (ax * bx + ay * by) / (scale_e * scale_n + 1e-12)
    shear_deg = 90.0 - float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    print(f"\n  Affine fit ({len(hids)} control points):")
    print(
        f"    E-scale {scale_e:.4f} px/ft   N-scale {scale_n:.4f} px/ft   "
        f"ratio {scale_e/scale_n:.3f}   shear {shear_deg:+.2f}°"
    )
    print(f"    Mean resid {resid.mean():.1f} px   median {np.median(resid):.1f} px   "
          f"max {resid.max():.1f} px")
    print("  Per-pick residuals:")
    for hid, r_px in sorted(zip(hids, resid), key=lambda x: -x[1]):
        flag = " ← OUTLIER" if r_px > 30 else ""
        print(f"    {hid}: {r_px:6.1f} px{flag}")

    OUT_FIT.write_text(
        json.dumps(
            {
                "affine_15x": A.tolist(),
                "control_points": {h: list(p) for h, p in zip(hids, T.tolist())},
                "residuals_px": dict(zip(hids, resid.tolist())),
                "scale_e_px_per_ft": scale_e,
                "scale_n_px_per_ft": scale_n,
                "shear_deg": shear_deg,
                "mean_residual_px": float(resid.mean()),
            },
            indent=2,
        )
    )
    print(f"\n  Saved fit → {OUT_FIT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
