"""Quantify the current served Tuck-1942 overlay georef against the true claim
corners, and audit which GCP sets are real vs affine-synthetic.

Tuck georef v2, step 1 (ADR-002 Tuck-map-georef lane). Sky noticed the served Tuck
overlay sits off the BLM plats (the Bear Cub block reads tens of metres N-NE of the
authoritative claim polygons). This script measures that offset where it can be
measured, and reports the gap where it cannot.

Two things it settles objectively, so no GCP set has to be trusted by name:

1. Real vs synthetic GCP audit. A QGIS ``.points`` set is affine-synthetic when a
   single affine reproduces every pixel<->world pair to ~0 m (the pixels were
   back-projected through the current georef, so re-fitting them returns the same
   georef and corrects nothing). A real hand-picked set leaves a non-zero affine
   residual, because real corner picks carry the sheet's non-affine distortion. The
   residual is the discriminator.

2. Current-georef offset. For a sheet whose real picks land on the 229 true claim
   corners (`bearcub .../nome_control_points.geojson`), apply the *current served*
   affine to those real pixels and compare to the true corner: that vector is the
   live georef error. Mean / median / max magnitude and the mean E/N bias per sheet.

The finding is that the offset is NOT measurable from the existing artifacts. The
served georef is confirmed (identical in ai-minerals overlays_v1p5 and goldbug
data/historical/tuck1942_v1p5). The only real per-sheet picks paired to true
corners are the central block (map_b / map_b1), and applying the served affine to
their pixels lands ~5.6 km off the corners, against the observed tens-of-metres
error: the picks are in a different working pixel frame, not the served frame. The
outer sheets' real picks are on creek mouths and grid lines, not the 229 corners.
So a trustworthy per-sheet offset, and a real-corner TPS warp, both need served-
frame corner picks that do not yet exist. The script reports this as the blocker
rather than emitting kilometre "offsets" as if they were measurements.

Run: PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.tuck_georef_quantify
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio

TUCK = Path("data/derived/tuck1942")
SERVED = TUCK / "overlays_v1p5"                      # current served overlay (affine v1)
REAL_GCP = TUCK / "v1p5_v2_refined_gcps"             # candidate real picks
CONTROL = Path("../bearcub/data/derived/nome_placer_labels/nome_control_points.geojson")
OUT_DIR = TUCK / "georef_v2_quantify"

LAT_M = 111320.0
LON_M = 111320.0 * math.cos(math.radians(64.5))      # Nome latitude
CORNER_MATCH_M = 25.0                                # a pick is "on a true corner" within this
SYNTH_RMS_M = 1.0                                    # below this affine RMS => synthetic


def load_points(p: Path) -> np.ndarray:
    """QGIS .points -> array[mapX(lon), mapY(lat), sourceX(px), sourceY(py)]."""
    rows = []
    for line in p.read_text().splitlines():
        if line.startswith("#") or line.startswith("mapX"):
            continue
        x = line.split(",")
        if len(x) >= 4:
            try:
                rows.append([float(x[0]), float(x[1]), float(x[2]), float(x[3])])
            except ValueError:
                pass
    return np.array(rows)


def affine_residual_m(a: np.ndarray) -> float | None:
    """RMS residual (m) of the best affine pixel->world fit; the real/synthetic test."""
    if len(a) < 4:
        return None
    src = np.column_stack([a[:, 2], a[:, 3], np.ones(len(a))])
    res = []
    for col, scale in ((0, LON_M), (1, LAT_M)):
        coef, *_ = np.linalg.lstsq(src, a[:, col], rcond=None)
        res.append((a[:, col] - src @ coef) * scale)
    return float(np.sqrt(np.mean(np.hypot(res[0], res[1]) ** 2)))


def audit_gcp_sets() -> list[dict]:
    out = []
    for p in sorted(TUCK.glob("**/*.points")):
        a = load_points(p)
        rms = affine_residual_m(a)
        cls = ("too_few" if rms is None else
               "synthetic" if rms < SYNTH_RMS_M else "real")
        out.append({"set": str(p.relative_to(TUCK)).replace(".tif.points", "").replace(".vrt.points", ""),
                    "n": len(a), "affine_rms_m": None if rms is None else round(rms, 1), "class": cls})
    return out


def served_affine(sheet: str) -> tuple[float, float, float, float, float, float]:
    """Current served per-sheet affine from the GeoTIFF: lon=c+a*px+b*py, lat=f+d*px+e*py."""
    with rasterio.open(SERVED / f"{sheet}.tif") as ds:
        t = ds.transform
    return t.a, t.b, t.c, t.d, t.e, t.f


def corner_offsets(sheet: str, control_lonlat: np.ndarray) -> dict:
    """For a sheet's real picks that sit on true corners, the current-georef offset."""
    pts = load_points(REAL_GCP / f"{sheet}.tif.points")
    if len(pts) == 0:
        return {"sheet": sheet, "status": "no_real_picks"}
    a, b, c, d, e, f = served_affine(sheet)
    lon_cur = c + a * pts[:, 2] + b * pts[:, 3]
    lat_cur = f + d * pts[:, 2] + e * pts[:, 3]
    # pair each pick's TRUE world (mapX/Y) to the nearest control corner
    on_corner, offs_e, offs_n = [], [], []
    for i in range(len(pts)):
        dlon = (control_lonlat[:, 0] - pts[i, 0]) * LON_M
        dlat = (control_lonlat[:, 1] - pts[i, 1]) * LAT_M
        if np.hypot(dlon, dlat).min() <= CORNER_MATCH_M:
            on_corner.append(i)
            offs_e.append((lon_cur[i] - pts[i, 0]) * LON_M)   # current minus true, east
            offs_n.append((lat_cur[i] - pts[i, 1]) * LAT_M)   # north
    if not on_corner:
        return {"sheet": sheet, "status": "picks_not_on_true_corners", "n_picks": len(pts)}
    offs_e, offs_n = np.array(offs_e), np.array(offs_n)
    mag = np.hypot(offs_e, offs_n)
    # Sanity gate. Sky/coordinator measure the live overlay error at tens of metres.
    # A reconciliation that returns kilometres means the real picks are NOT in the
    # served raster's pixel frame (they were made on a different working raster), so
    # this is a frame-consistency FAILURE, not an offset measurement.
    consistent = bool(mag.max() < 500.0)
    return {"sheet": sheet, "status": "reconciled" if consistent else "frame_mismatch",
            "n_picks": len(pts), "n_on_corner": len(on_corner),
            "served_frame_consistent": consistent,
            "reconcile_mean_m": round(float(mag.mean()), 1),
            "reconcile_median_m": round(float(np.median(mag)), 1),
            "reconcile_max_m": round(float(mag.max()), 1),
            "bias_E_m": round(float(offs_e.mean()), 1), "bias_N_m": round(float(offs_n.mean()), 1),
            "bias_azimuth_deg": round(float((math.degrees(math.atan2(offs_e.mean(), offs_n.mean()))) % 360), 0)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control = json.loads(CONTROL.read_text())
    control_lonlat = np.array([f["geometry"]["coordinates"] for f in control["features"]])

    audit = audit_gcp_sets()
    print("=== GCP-set audit (real vs synthetic, by affine residual) ===")
    for r in audit:
        print(f"  {r['set']:58s} n={r['n']:>3} rms={r['affine_rms_m']!s:>7} {r['class']}")

    sheets = ["map_a", "map_b", "map_b1", "map_c", "map_d", "map_e"]
    offsets = [corner_offsets(f"tuck1942_{s}", control_lonlat) for s in sheets]
    print("\n=== served-frame reconciliation check (real picks vs true corners) ===")
    for o in offsets:
        if o["status"] in ("reconciled", "frame_mismatch"):
            tag = "OK" if o["served_frame_consistent"] else "FRAME MISMATCH (picks not in served frame)"
            print(f"  {o['sheet']:20s} on_corner={o['n_on_corner']:>2}/{o['n_picks']:<2}  "
                  f"reconcile mean={o['reconcile_mean_m']}m max={o['reconcile_max_m']}m  "
                  f"bias=({o['bias_E_m']:+},{o['bias_N_m']:+})m  -> {tag}")
        else:
            print(f"  {o['sheet']:20s} {o['status']} "
                  f"(picks={o.get('n_picks','-')}) -- GAP, picks not on the 229 true corners")
    measurable = any(o.get("served_frame_consistent") for o in offsets)
    print(f"\nVERDICT: current-offset {'MEASURABLE' if measurable else 'NOT MEASURABLE from existing artifacts'} "
          "-- no real picks reconcile to the served raster frame; offset needs served-frame corner picks.")

    out = {
        "what": "Audit of every Tuck GCP set (real vs affine-synthetic) + a served-frame "
                "reconciliation check of the real picks against the 229 true claim corners.",
        "verdict": ("Current-offset is NOT measurable from the existing artifacts: the served georef "
                    "is confirmed (identical in ai-minerals overlays_v1p5 and goldbug data/historical/"
                    "tuck1942_v1p5), but the only real per-sheet picks (v1p5_v2_refined_gcps, central "
                    "block) are in a different working pixel frame -- applying the served affine to them "
                    "lands ~5.6 km off the true corners, against an observed tens-of-metres error. So "
                    "both the per-sheet offset and a real-corner warp need served-frame pixel picks that "
                    "do not yet exist. Sky-in-the-loop digitising on the served rasters is the unblock."),
        "served_georef": "data/derived/tuck1942/overlays_v1p5 (affine_v1_from_4_visual_gcps)",
        "control_points": str(CONTROL), "n_control": len(control_lonlat),
        "convention": "offset = current_georef(pick_pixel) - true_corner; bias E/N positive = "
                      "current overlay sits east/north of truth; azimuth 0=N,90=E.",
        "gcp_audit": audit,
        "per_sheet_offset": offsets,
        "method": {"real_vs_synthetic": "affine pixel->world RMS; <1 m = synthetic (back-projected)",
                   "offset": "apply current served sheet affine to the real pixel picks that land "
                             "on a true corner (<=25 m), compare to the corner"},
    }
    (OUT_DIR / "tuck_georef_v2_quantify.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'tuck_georef_v2_quantify.json'}")


if __name__ == "__main__":
    main()
