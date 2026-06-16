"""Generate Tuck 1942 GCP candidate files for the v1.5 v2 survey-grade pass.

Programmatic v1.5 v2 v0 ("best effort, verify in QGIS"):

- Take bearcub's 148 control points (27 collars + 121 claim corners
  in the validated Hammon envelope).
- Predict each point's pixel coordinate on each Tuck sheet via the
  current v1.5 v1.1 rough-affine.
- Emit a QGIS-Georeferencer-compatible ``.points`` file per sheet,
  plus a per-sheet CSV for inspection.

Sky's workflow then:
1. Open Tuck Map B1 in QGIS Georeferencer.
2. Load the predicted ``.points`` file from this script's output.
3. For each candidate, click the actual visible feature (collar dot
   or claim corner) on the map; QGIS records the corrected pixel
   coordinate.
4. Save the refined ``.points`` file; run ``gdalwarp -tps`` over it
   to produce the v1.5 v2 survey-grade GeoTIFF.

This script is the productivity multiplier for the QGIS pass: Sky
moves each GCP by hundreds of pixels (the rough-affine's +/- 200-500
m error mapped to ~6-15 pixels at 300 DPI) instead of picking 30+
GCPs from scratch.

Without this, an interactive pass takes about an hour per sheet. With
this it should take 10-15 minutes per sheet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ai_minerals.data.tuck1942_map_b1 import (
    DEFAULT_RENDER_DPI,
    PIXEL_GCPS_AT_200DPI,
    _fit_rough_affine,
)


CONTROL_POINTS_CSV = Path("data/raw/bearcub_nome/nome_control_points.csv")

SHEETS = {
    "tuck1942_map_b1": "Map B1.pdf",
    "tuck1942_nome_district_contours": "Nome District Map (Contours).pdf",
    "tuck1942_nome_district_general": "Nome district Map.pdf",
    "tuck1942_map_a": "Map A.pdf",
    "tuck1942_map_b": "Map B.pdf",
    "tuck1942_map_c": "Map C.pdf",
    "tuck1942_map_d": "Map D.pdf",
    "tuck1942_map_e": "Map E.pdf",
}


def _invert_affine_for_pixel_prediction(
    affine_4326: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Given the forward affine pixel -> (lon, lat) at the target DPI,
    return (A_inv, t) such that (col, row) = A_inv @ (lon - t_lon, lat - t_lat)
    """
    a, b, d, e, xoff, yoff = affine_4326
    A = np.array([[a, b], [d, e]])
    A_inv = np.linalg.inv(A)
    return A_inv, np.array([xoff, yoff])


def _scale_affine_to_dpi(
    affine_at_200: tuple[float, ...], target_dpi: int,
) -> tuple[float, ...]:
    a, b, d, e, xoff, yoff = affine_at_200
    s = 200.0 / target_dpi
    return (a * s, b * s, d * s, e * s, xoff, yoff)


def predict_pixel(
    lon: float, lat: float, affine_inv: np.ndarray, offset: np.ndarray,
) -> tuple[float, float]:
    """Inverse-apply the rough-affine to predict (pixel_col, pixel_row)."""
    delta = np.array([lon, lat]) - offset
    col_row = affine_inv @ delta
    return float(col_row[0]), float(col_row[1])


def write_qgis_points_file(
    out_path: Path,
    gcps: list[dict],
    sheet_pixel_height: int,
) -> None:
    """Write a QGIS Georeferencer ``.points`` file.

    Format (QGIS 4.0): sourceY is the POSITIVE pixel row (origin top-left,
    Y growing downward). QGIS 3.x used a negative-Y convention; 4.0
    dropped it.
    """
    lines = ["#CRS: GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]"]
    lines.append("mapX,mapY,sourceX,sourceY,enable,dX,dY,residual")
    for g in gcps:
        qgis_x = g["pixel_col"]
        qgis_y = g["pixel_row"]
        lines.append(
            f"{g['lon']:.7f},{g['lat']:.7f},"
            f"{qgis_x:.2f},{qgis_y:.2f},1,0,0,0"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _rotate_around_center(
    col: float, row: float, theta_deg: float, cx: float, cy: float,
) -> tuple[float, float]:
    """Rotate (col, row) by theta_deg degrees CCW around (cx, cy)."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    dc, dr = col - cx, row - cy
    return cx + c * dc - s * dr, cy + s * dc + c * dr


def generate_one_sheet(
    sheet_key: str,
    pdf_name: str,
    control_points: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    rotation_deg: float = 0.0,
) -> dict:
    """Produce GCP candidate files for one Tuck sheet.

    ``rotation_deg`` rotates the rough-affine pixel predictions by that
    many degrees CCW around the image centre. Used when the rendered
    sheet is rotated relative to the north-up frame the rough-affine
    GCPs were placed in (Tuck Map B1 is rotated ~+45 deg in the source
    PDF).
    """
    affine_200 = _fit_rough_affine(PIXEL_GCPS_AT_200DPI)
    affine_target = _scale_affine_to_dpi(affine_200, dpi)
    A_inv, offset = _invert_affine_for_pixel_prediction(affine_target)

    # We need the sheet pixel dimensions to filter out GCPs that fall
    # outside the image. Use the v1.5 v1.1 manifest if present; otherwise
    # render a quick PDF probe.
    sheet_meta_path = Path(
        "data/derived/tuck1942/overlays_v1p5/tuck1942_overlay_metadata_v1p5.json"
    )
    if sheet_meta_path.exists():
        meta = json.loads(sheet_meta_path.read_text())
        sheet_meta = meta.get("sheets", {}).get(sheet_key, {})
        width = sheet_meta.get("width_px", 0)
        height = sheet_meta.get("height_px", 0)
    else:
        # Fallback: render the PDF page to peek at dimensions
        import fitz
        d = fitz.open(f"/tmp/tuck/{pdf_name}")
        pix = d[0].get_pixmap(dpi=dpi)
        width, height = pix.width, pix.height
        d.close()

    cx, cy = width / 2.0, height / 2.0
    gcps: list[dict] = []
    n_on_sheet = 0
    n_off_sheet = 0
    for _, row in control_points.iterrows():
        pix_col, pix_row = predict_pixel(row["lon"], row["lat"], A_inv, offset)
        if rotation_deg != 0.0:
            pix_col, pix_row = _rotate_around_center(
                pix_col, pix_row, rotation_deg, cx, cy,
            )
        on_sheet = (0 <= pix_col < width) and (0 <= pix_row < height)
        if on_sheet:
            n_on_sheet += 1
        else:
            n_off_sheet += 1
            continue
        gcps.append({
            "name": row["name"],
            "type": row["type"],
            "ms": (str(int(row["ms"])) if pd.notna(row["ms"]) else ""),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "easting_local_ft": (float(row["easting_local_ft"])
                                  if pd.notna(row["easting_local_ft"]) else None),
            "northing_local_ft": (float(row["northing_local_ft"])
                                   if pd.notna(row["northing_local_ft"]) else None),
            "pixel_col": pix_col,
            "pixel_row": pix_row,
        })

    # Write QGIS .points file
    points_path = out_dir / f"{sheet_key}.tif.points"
    write_qgis_points_file(points_path, gcps, height)

    # Write CSV for inspection
    csv_path = out_dir / f"{sheet_key}_gcp_candidates.csv"
    pd.DataFrame(gcps).to_csv(csv_path, index=False)

    return {
        "sheet_key": sheet_key,
        "image_width": width,
        "image_height": height,
        "n_gcps_on_sheet": n_on_sheet,
        "n_gcps_off_sheet": n_off_sheet,
        "qgis_points_file": str(points_path),
        "csv_file": str(csv_path),
    }


def main() -> None:
    out_dir = Path("data/derived/tuck1942/v1p5_v2_gcp_candidates")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bearcub control points from {CONTROL_POINTS_CSV} ...")
    df = pd.read_csv(CONTROL_POINTS_CSV)
    # Use only points with WGS84 (lat/lon); local-feet may be null for
    # Molasses / Honey outside the validated envelope (those are still
    # candidate GCPs via their WGS84 coords).
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    print(f"  {len(df)} control points total")
    print(f"  with local feet: {df['easting_local_ft'].notna().sum()}")

    sheet_stats = []
    for key, pdf_name in SHEETS.items():
        # Match the v1.5 v1 overlay rendering DPI so predicted pixel
        # positions line up with the unreferenced TIFFs in georef_source/.
        # The Tuck sheets are printed rotated ~+45 deg relative to
        # north-up (the Hammon survey grid is the page's vertical axis).
        stats = generate_one_sheet(
            key, pdf_name, df, out_dir, dpi=300, rotation_deg=45.0,
        )
        sheet_stats.append(stats)
        print(f"  {key}: {stats['n_gcps_on_sheet']} GCPs on sheet, "
              f"{stats['n_gcps_off_sheet']} off")

    # Manifest with QGIS workflow instructions
    manifest = {
        "schema_version": "1.0",
        "phase": "1.5 v2 v0 (GCP candidates for QGIS refinement)",
        "source_control_points": str(CONTROL_POINTS_CSV),
        "n_control_points": len(df),
        "sheets": sheet_stats,
        "next_steps": {
            "owner": "Sky (interactive QGIS pass)",
            "steps": [
                "Open the target Tuck sheet (TIFF) in QGIS Georeferencer.",
                "Settings -> Load GCP Points: pick the .tif.points file.",
                "For each loaded point, click the actual visible feature "
                "(collar dot or claim corner) on the map; QGIS updates "
                "the pixel coordinate of that point.",
                "After refining all points, Settings -> Transformation: "
                "Thin-Plate-Spline (TPS), Output: GeoTIFF EPSG:4326.",
                "Save the refined .points file as ai-minerals/data/derived/"
                "tuck1942/v1p5_v2_refined_gcps/<sheet>.tif.points.",
                "Run gdalwarp -tps to produce the survey-grade GeoTIFF.",
            ],
        },
        "accuracy_note": (
            "These candidates inherit the v1.5 v1.1 rough-affine's +/- "
            "200-500 m positional error. At 300 DPI that maps to ~6-15 "
            "pixels of offset, so each candidate needs a corrective click "
            "to land on the actual visible feature on the sheet. The "
            "candidates are the STARTING POSITION; the survey-grade "
            "accuracy comes from Sky's clicks."
        ),
    }
    manifest_path = out_dir / "v1p5_v2_gcp_candidates_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest -> {manifest_path}")
    print(f"GCP candidates ready at {out_dir}")


if __name__ == "__main__":
    main()
