"""v1.5 v2 pipeline for all 8 Tuck sheets.

Sky pinned 10 GCPs on Map B1 in QGIS this morning, giving us a refit
affine that captures the cartographer's true rotation + scale + offset
on the printed plates. All 8 sheets in the Tuck 1942 atlas are scans
of separate plates showing essentially the same Cape Nome district at
the same orientation; the per-sheet crops differ slightly. So this
script:

1. Rotates each of the 7 remaining sheets -90 deg CCW (PIL) to match
   the B1 north-up orientation.
2. For each rotated sheet, generates candidate GCPs by applying the
   B1-refit affine to the 229 bearcub control points and scaling
   predictions to that sheet's pixel dimensions. Deduplicates by
   pixel position (adjacent claim corners share pixels and would
   break TPS).
3. Runs gdalwarp -tps -dstalpha per sheet to produce a warped EPSG:4326
   GeoTIFF with the rotated-corner fill landed as transparent alpha.
4. Packages everything for goldbug delivery.

B1 was already pinned by Sky directly; we reuse the existing
output rather than re-warp. Other 7 sheets get this scripted-bootstrap
pass.

Per-sheet drift inherits from the cross-sheet dimension assumption: if
Map A actually shows a slightly different sub-crop than B1, Bear Cub
will drift on that sheet until Sky drops a few sheet-specific GCPs
to refine. The B1-affine pass gets us "good enough to view"; the
chapter / viewer surfaces any sheet that needs a QGIS-pinned refit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import Affine


Image.MAX_IMAGE_PIXELS = None
ROTATION_DEG_CCW = -90.0  # 90 deg CW; brings Tuck's N from "left" to "up"

CONTROL_POINTS_CSV = Path("data/raw/bearcub_nome/nome_control_points.csv")
B1_BOOTSTRAP_POINTS = Path(
    "data/derived/tuck1942/v1p5_v2_refined_gcps/tuck1942_map_b1_bootstrap.tif.points"
)
ROTATED_SOURCE_DIR = Path("data/derived/tuck1942/georef_source_rotated")
GCP_CANDIDATES_DIR = Path("data/derived/tuck1942/v1p5_v2_gcp_candidates")
REFINED_DIR = Path("data/derived/tuck1942/v1p5_v2_refined")
UNREF_SOURCE_DIR = Path("data/derived/tuck1942/georef_source")
GOLDBUG_INBOX = Path(
    "/home/sky/src/learning/gldbg/handoff/inbox/2026-06-16-from-ai-minerals-tuck-all-sheets-v1p5-v2"
)

# Skip Map B1: Sky already pinned it directly today.
SHEETS_TO_BOOTSTRAP = [
    "tuck1942_nome_district_contours",
    "tuck1942_nome_district_general",
    "tuck1942_map_a",
    "tuck1942_map_b",
    "tuck1942_map_c",
    "tuck1942_map_d",
    "tuck1942_map_e",
]


def rotate_one(sheet_key: str) -> dict:
    """PIL-rotate the unreferenced sheet -90 CCW, embed y-flipped transform."""
    in_path = UNREF_SOURCE_DIR / f"{sheet_key}.tif"
    out_path = ROTATED_SOURCE_DIR / f"{sheet_key}.tif"
    sidecar_path = ROTATED_SOURCE_DIR / f"{sheet_key}_rotation.json"

    img = Image.open(in_path)
    orig_w, orig_h = img.size
    rotated = img.rotate(
        ROTATION_DEG_CCW,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor="white",
    )
    new_w, new_h = rotated.size
    rotated.save(out_path, compression="tiff_lzw")

    with rasterio.open(out_path) as src:
        profile = src.profile.copy()
        arr = src.read()
    profile.update(
        crs=None,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(new_h)),
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)

    sidecar = {
        "source_tif": str(in_path),
        "rotation_deg_ccw": ROTATION_DEG_CCW,
        "original_size": [orig_w, orig_h],
        "rotated_size": [new_w, new_h],
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"  {sheet_key}: rotated {orig_w}x{orig_h} -> {new_w}x{new_h}")
    return sidecar


def fit_b1_affine() -> tuple[tuple[float, ...], tuple[int, int]]:
    """Re-fit the B1 affine from Sky's bootstrap GCPs.

    Returns the affine + B1's original-frame dims (so per-sheet scaling
    can normalize).
    """
    b1_sidecar = json.loads(
        (ROTATED_SOURCE_DIR / "tuck1942_map_b1_rotation.json").read_text()
    )
    b1_orig_w, b1_orig_h = b1_sidecar["original_size"]
    b1_rotated_h = b1_sidecar["rotated_size"][1]

    # Parse Sky's bootstrap .points (rotated-frame source coords)
    raw_gcps = []
    for ln in B1_BOOTSTRAP_POINTS.read_text().splitlines():
        if ln.startswith("#") or "mapX" in ln or not ln.strip():
            continue
        parts = ln.split(",")
        raw_gcps.append((float(parts[0]), float(parts[1]),
                         float(parts[2]), float(parts[3])))
    # Convert to original-frame pixels:
    # rotated (col_r, row_r) -> original (row_r, orig_h - 1 - col_r)
    # and sourceY in .points is world_y in y-flipped frame = rotated_h - row_r
    src_pts = []  # (col_o, row_o)
    dst_pts = []  # (lon, lat)
    for mapX, mapY, sX, sY in raw_gcps:
        col_r = sX
        row_r = b1_rotated_h - sY
        col_o = row_r
        row_o = (b1_orig_h - 1) - col_r
        src_pts.append([col_o, row_o, 1.0])
        dst_pts.append([mapX, mapY])
    X = np.asarray(src_pts)
    lon = np.asarray([d[0] for d in dst_pts])
    lat = np.asarray([d[1] for d in dst_pts])
    coefs_lon, *_ = np.linalg.lstsq(X, lon, rcond=None)
    coefs_lat, *_ = np.linalg.lstsq(X, lat, rcond=None)
    a, b, xoff = coefs_lon
    d, e, yoff = coefs_lat
    return (a, b, d, e, xoff, yoff), (b1_orig_w, b1_orig_h)


def predict_pixel_original(
    lon: float, lat: float,
    aff: tuple[float, ...],
) -> tuple[float, float]:
    a, b, d, e, xoff, yoff = aff
    A = np.array([[a, b], [d, e]])
    A_inv = np.linalg.inv(A)
    col, row = A_inv @ np.array([lon - xoff, lat - yoff])
    return float(col), float(row)


def generate_candidates_for_sheet(
    sheet_key: str,
    sheet_sidecar: dict,
    b1_dims: tuple[int, int],
    b1_aff: tuple[float, ...],
    bearcub_df: pd.DataFrame,
) -> int:
    """Generate a candidate .points file for one sheet via B1-affine + scaling."""
    sheet_orig_w, sheet_orig_h = sheet_sidecar["original_size"]
    sheet_rotated_w, sheet_rotated_h = sheet_sidecar["rotated_size"]
    b1_orig_w, b1_orig_h = b1_dims

    lines = [
        '#CRS: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
        "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual",
    ]
    seen_pixels: set[tuple[int, int]] = set()
    n_on, n_off, n_dup = 0, 0, 0

    for _, r in bearcub_df.iterrows():
        # Predict in B1's original frame via the refit affine.
        col_o_b1, row_o_b1 = predict_pixel_original(r["lon"], r["lat"], b1_aff)
        # Scale to THIS sheet's original-frame dims (assumes both sheets
        # show the same geography at the same fractional placement).
        col_o = col_o_b1 * (sheet_orig_w / b1_orig_w)
        row_o = row_o_b1 * (sheet_orig_h / b1_orig_h)
        # Forward-rotate to rotated frame: 90 CW (PIL.rotate(-90, expand=True))
        # maps original (col_o, row_o) -> rotated (orig_h - 1 - row_o, col_o)
        col_r = (sheet_orig_h - 1) - row_o
        row_r = col_o
        # On-sheet?
        if not (0 <= col_r < sheet_rotated_w and 0 <= row_r < sheet_rotated_h):
            n_off += 1
            continue
        # Dedup
        key = (int(round(col_r / 2.0)), int(round(row_r / 2.0)))
        if key in seen_pixels:
            n_dup += 1
            continue
        seen_pixels.add(key)
        n_on += 1
        # Y-flipped sourceY
        sourceX = col_r
        sourceY = sheet_rotated_h - row_r
        lines.append(
            f"{r['lon']:.7f},{r['lat']:.7f},{sourceX:.2f},{sourceY:.2f},1,0,0,0"
        )

    out_path = GCP_CANDIDATES_DIR / f"{sheet_key}.tif.points"
    out_path.write_text("\n".join(lines) + "\n")
    print(
        f"  {sheet_key}: {n_on} candidates on sheet, {n_off} off-sheet, {n_dup} dups"
    )
    return n_on


def warp_one_sheet(sheet_key: str) -> Path:
    """Run gdalwarp -tps -dstalpha on one rotated sheet's candidate GCPs."""
    src_tif = ROTATED_SOURCE_DIR / f"{sheet_key}.tif"
    points_file = GCP_CANDIDATES_DIR / f"{sheet_key}.tif.points"
    out_tif = REFINED_DIR / f"{sheet_key}.tif"
    vrt = Path(f"/tmp/{sheet_key}_with_gcps.vrt")

    # Parse the candidate .points to assemble -gcp args (in pixel coords).
    sidecar = json.loads(
        (ROTATED_SOURCE_DIR / f"{sheet_key}_rotation.json").read_text()
    )
    rotated_h = sidecar["rotated_size"][1]

    gcp_args: list[str] = []
    for ln in points_file.read_text().splitlines():
        if ln.startswith("#") or "mapX" in ln or not ln.strip():
            continue
        parts = ln.split(",")
        if int(float(parts[4])) != 1:
            continue
        mapX, mapY, sX, sY = (float(x) for x in parts[:4])
        pixel_row = rotated_h - sY
        gcp_args += ["-gcp", f"{sX}", f"{pixel_row}", f"{mapX}", f"{mapY}"]

    subprocess.run(
        ["gdal_translate", "-of", "VRT", "-a_srs", "", *gcp_args,
         str(src_tif), str(vrt)],
        check=True, capture_output=True,
    )
    result = subprocess.run(
        ["gdalwarp", "-tps", "-r", "cubic",
         "-t_srs", "EPSG:4326", "-dstalpha",
         "-co", "COMPRESS=LZW", "-co", "TILED=YES",
         "-overwrite",
         str(vrt), str(out_tif)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  {sheet_key} STDERR: {result.stderr}")
        raise SystemExit(result.returncode)
    print(f"  {sheet_key}: warped -> {out_tif.name} ({out_tif.stat().st_size:,} B)")
    return out_tif


def main() -> None:
    ROTATED_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    GCP_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    REFINED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: rotate the 7 remaining sheets
    print("Step 1: rotating 7 sheets -90 deg CCW ...")
    sidecars: dict[str, dict] = {}
    for sheet in SHEETS_TO_BOOTSTRAP:
        sidecars[sheet] = rotate_one(sheet)

    # Step 2: fit B1 affine + generate per-sheet candidate GCPs
    print("\nStep 2: refit B1 affine + generate per-sheet candidates ...")
    b1_aff, b1_dims = fit_b1_affine()
    bearcub_df = (
        pd.read_csv(CONTROL_POINTS_CSV)
        .dropna(subset=["lon", "lat"])
        .reset_index(drop=True)
    )
    print(f"  B1 affine: a={b1_aff[0]:.3e} b={b1_aff[1]:.3e} d={b1_aff[2]:.3e} e={b1_aff[3]:.3e}")
    print(f"  bearcub control points: {len(bearcub_df)}")
    for sheet in SHEETS_TO_BOOTSTRAP:
        generate_candidates_for_sheet(
            sheet, sidecars[sheet], b1_dims, b1_aff, bearcub_df,
        )

    # Step 3: warp each sheet with gdalwarp -tps -dstalpha
    print("\nStep 3: warp each sheet ...")
    for sheet in SHEETS_TO_BOOTSTRAP:
        warp_one_sheet(sheet)

    # Step 4: package + deliver
    print(f"\nStep 4: deliver to {GOLDBUG_INBOX}")
    GOLDBUG_INBOX.mkdir(parents=True, exist_ok=True)
    manifest_sheets = {}
    for sheet in ["tuck1942_map_b1"] + SHEETS_TO_BOOTSTRAP:
        tif = REFINED_DIR / f"{sheet}.tif"
        if not tif.exists():
            print(f"  WARN: {tif} missing; skipping")
            continue
        shutil.copy(tif, GOLDBUG_INBOX / tif.name)
        with rasterio.open(tif) as src:
            manifest_sheets[sheet] = {
                "tif_relative_url": tif.name,
                "width_px": src.width,
                "height_px": src.height,
                "bounds_4326": list(src.bounds),
                "bands": src.count,
                "has_alpha": src.count == 4,
            }
        print(f"  delivered {tif.name} ({tif.stat().st_size:,} B)")

    manifest = {
        "schema_version": "1.0",
        "phase": "1.5 v2 (Sky-pinned B1 + B1-affine-scaled 7 other sheets)",
        "rendering_dpi": 300,
        "rotation_deg_ccw": ROTATION_DEG_CCW,
        "georef_method_b1": "gdalwarp -tps via Sky's 10 QGIS-pinned GCPs",
        "georef_method_other_sheets": (
            "B1 refit affine applied to bearcub's 229 control points, "
            "predictions scaled to each sheet's pixel dimensions, "
            "deduplicated, warped via gdalwarp -tps -dstalpha. Inherits "
            "Bear Cub block accuracy from B1; perimeter drifts wherever "
            "the cross-sheet dimension assumption holds less well."
        ),
        "sheets": manifest_sheets,
    }
    (GOLDBUG_INBOX / "tuck1942_overlay_metadata_v1p5_v2.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"  manifest written")
    print(f"\nAll 8 sheets at v1.5 v2 delivered to goldbug.")


if __name__ == "__main__":
    main()
