"""Correct flat<->geo mapping for Tuck-1942 Map C, recovered by image registration.

Why this exists
---------------
fossick's Map C hole drain was blocked: its flat<->geo affine was fit from
`v1p5_v2_gcp_candidates/tuck1942_map_c.tif.points`, whose GCP geo hull sits in
the town-of-Nome area (lon -165.43..-165.33), ~2 km WEST of and disjoint from the
Map C v3 raster extent (lon -165.28..-164.96, Cape Nome). The disjoint fit sent
the coastal hole band OFF the flat scan (negative rows).

What the mapping actually is
----------------------------
The Map C v3 raster is a rotated, anisotropically-scaled warp of the source scan:
the map content is a tilted parallelogram inside v3's north-up box (only ~60% of
v3 is valid data). So v3<->flat is NOT a simple corner map; it carries a real
rotation. The relationship v3 pixel <-> rotated-scan pixel was recovered by SIFT
feature registration between the v3 raster and the rotated source scan
(georef_source_rotated/tuck1942_map_c.tif), RANSAC-fit to an affine:

    v3 pixel --(affine A, from image registration)--> rotated-scan pixel
             --(exact 90-deg rotation)--------------> flat-scan pixel

A is ~ scale (0.68 x, 1.64 y), rotation ~ -17 deg. The fit residual on 38
well-distributed control points is ~3 rot-px (~2.8 m median, ~4.4 m max on a
300-dpi, 1"=800ft scan). A 2nd-order polynomial barely improves on the affine,
so residual nonlinearity is small. Validated visually at the hole field (the red
"(P.109,112)" annotation and "Sec. II" label), the Cape Nome east end (contour
blobs), and the west worked-beach claim strip.

Delivery to fossick
-------------------
fossick's FlatPositioner composes geo->flat (fit from a .points file) with its
exact v3<->geo geotransform (V3_GT['c']). Because v3<->geo, v3<->rot and rot<->flat
are all affine, geo<->flat is affine too, and the intermediate geo (v3's own,
only bulk-accurate) cancels in the v3->flat round-trip. So this writes a .points
grid whose (lon,lat) are v3_to_geo(rot_to_v3(col_rot,row_rot)); fed to fossick's
existing loader it reproduces the registered v3->flat exactly. Point fossick's
GCP_DIR at the output directory, or use the explicit affines in the JSON.

Run:
    .venv/bin/python -m scripts.nome_placer.tuck_map_c_flatgeo            # from CSV (deterministic)
    .venv/bin/python -m scripts.nome_placer.tuck_map_c_flatgeo --rederive # re-run SIFT (needs opencv)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

DERIVED = Path("data/derived/tuck1942")
V3_TIF = DERIVED / "v1p5_v3_refined" / "tuck1942_map_c.tif"
ROT_TIF = DERIVED / "georef_source_rotated" / "tuck1942_map_c.tif"
ROT_JSON = DERIVED / "georef_source_rotated" / "tuck1942_map_c_rotation.json"
OUT_DIR = DERIVED / "v1p5_v6_flatgeo_gcps"
CORR_CSV = OUT_DIR / "map_c_v3_rot_correspondences.csv"

POINTS_HEADER = (
    '#CRS: GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],'
    'ANGLEUNIT["degree",0.0174532925199433]],'
    'AXIS["geodetic longitude (Lon)",east,ORDER[2],'
    'ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]\n'
    "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual\n"
)


def frame() -> dict:
    with rasterio.open(V3_TIF) as ds:
        t = ds.transform
    rot = json.loads(ROT_JSON.read_text())
    wr, hr = rot["rotated_size"]      # 15795 x 9692
    ow, oh = rot["original_size"]     # 9692 x 15795 (flat scan)
    return {"v3_ox": t.c, "v3_oy": t.f, "v3_px": t.a, "v3_py": t.e,
            "Wr": wr, "Hr": hr, "flat_w": ow, "flat_h": oh}


def v3_to_geo(col: np.ndarray, row: np.ndarray, f: dict) -> tuple[np.ndarray, np.ndarray]:
    return f["v3_ox"] + col * f["v3_px"], f["v3_oy"] + row * f["v3_py"]


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 mapping src(N,2) -> dst(N,2) as [[a,b,c],[d,e,f]]."""
    a = np.hstack([src, np.ones((len(src), 1))])
    m, *_ = np.linalg.lstsq(a, dst, rcond=None)   # (3,2)
    return m.T


def read_correspondences() -> tuple[np.ndarray, np.ndarray]:
    v3, rot = [], []
    with open(CORR_CSV) as fh:
        for r in csv.DictReader(fh):
            v3.append([float(r["v3_col"]), float(r["v3_row"])])
            rot.append([float(r["rot_col"]), float(r["rot_row"])])
    return np.array(v3), np.array(rot)


def derive_correspondences() -> None:
    """Re-run SIFT registration v3 <-> rotated scan and overwrite the CSV (needs opencv)."""
    import cv2
    from rasterio.enums import Resampling
    np.random.seed(0)

    def load(path: Path, out_w: int, alpha: bool = False):
        with rasterio.open(path) as ds:
            s = out_w / ds.width
            d = ds.read(out_shape=(ds.count, int(ds.height * s), out_w), resampling=Resampling.bilinear)
        g = cv2.cvtColor(d[:3].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        m = (d[3] > 10).astype(np.uint8) * 255 if (alpha and d.shape[0] >= 4) else None
        return cv2.createCLAHE(3.0, (16, 16)).apply(g), s, m

    gv3, sv3, mv3 = load(V3_TIF, 6000, alpha=True)
    grot, srot, _ = load(ROT_TIF, 4000)
    mv3 = cv2.erode(mv3, np.ones((15, 15), np.uint8))
    sift = cv2.SIFT_create(nfeatures=60000, contrastThreshold=0.012, edgeThreshold=20)
    k1, d1 = sift.detectAndCompute(gv3, mv3)
    k2, d2 = sift.detectAndCompute(grot, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=128))
    p1 = np.float32([kp.pt for kp in k1]); p2 = np.float32([kp.pt for kp in k2])
    # bootstrap seed affine from a loose global match, then guided-filter to kill repeats
    boot = [m for m, n in flann.knnMatch(d1, d2, k=2) if m.distance < 0.72 * n.distance]
    s0 = np.float32([p1[m.queryIdx] for m in boot]); d0 = np.float32([p2[m.trainIdx] for m in boot])
    Mseed, _ = cv2.estimateAffine2D(s0, d0, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    src, dst = [], []
    for m, n in flann.knnMatch(d1, d2, k=2):
        if m.distance < 0.85 * n.distance:
            proj = Mseed[:, :2] @ p1[m.queryIdx] + Mseed[:, 2]
            if np.linalg.norm(proj - p2[m.trainIdx]) < 40:
                src.append(p1[m.queryIdx]); dst.append(p2[m.trainIdx])
    src = np.float32(src); dst = np.float32(dst)
    _, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=80000)
    inl = inl.ravel().astype(bool)
    v3f = (src[inl] / sv3); rotf = (dst[inl] / srot)
    Afit, inl2 = cv2.estimateAffine2D(v3f.astype(np.float32), rotf.astype(np.float32),
                                      method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=80000)
    inl2 = inl2.ravel().astype(bool)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORR_CSV, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["v3_col", "v3_row", "rot_col", "rot_row"])
        for (a, b), (c, d) in zip(v3f[inl2], rotf[inl2]):
            w.writerow([f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{d:.2f}"])
    print(f"re-derived {int(inl2.sum())} correspondences -> {CORR_CSV}")


def build() -> dict:
    f = frame()
    v3c, rotc = read_correspondences()
    a_v3_rot = fit_affine(v3c, rotc)                 # v3 -> rot (2x3)
    A, t = a_v3_rot[:, :2], a_v3_rot[:, 2]
    Ainv = np.linalg.inv(A)
    resid = np.linalg.norm((v3c @ A.T + t) - rotc, axis=1)

    # rot -> flat is the exact 90-deg map fossick uses: col_f=row_rot, row_f=(flat_h-1)-col_rot
    Rflip = np.array([[0.0, 1.0], [-1.0, 0.0]])
    off = np.array([0.0, f["flat_h"] - 1.0])
    A_v3_flat = Rflip @ A                              # v3 -> flat linear
    t_v3_flat = Rflip @ t + off

    # .points grid over the validated rot envelope (fossick convention).
    r0, r1 = rotc[:, 0].min(), rotc[:, 0].max()
    s0, s1 = rotc[:, 1].min(), rotc[:, 1].max()
    cols = np.linspace(r0, r1, 12)
    rows = np.linspace(s0, s1, 9)
    lines = []
    for cr in cols:
        for rr in rows:
            v3col, v3row = Ainv @ (np.array([cr, rr]) - t)   # rot -> v3
            lon, lat = v3_to_geo(np.array(v3col), np.array(v3row), f)
            lines.append((float(lon), float(lat), float(cr), float(f["Hr"] - rr)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "tuck1942_map_c.tif.points", "w") as fh:
        fh.write(POINTS_HEADER)
        for lon, lat, sx, sy in lines:
            fh.write(f"{lon:.10f},{lat:.10f},{sx:.4f},{sy:.4f},1,0,0,0\n")

    sx = float(np.hypot(*A[:, 0])); sy = float(np.hypot(*A[:, 1]))
    ang = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    meta = {
        "map": "c",
        "purpose": "Correct flat<->geo for Tuck-1942 Map C (fixes fossick georef_blocker: v2 GCPs were 2 km off in town-of-Nome).",
        "method": "SIFT+RANSAC image registration v3<->rotated-scan (affine) composed with the exact 90-deg rotated->flat map.",
        "control_points": int(len(v3c)),
        "residual_rot_px": {"median": float(np.median(resid)), "max": float(resid.max())},
        "residual_ground_m_approx": {"median": float(np.median(resid) * 0.8), "max": float(resid.max() * 0.8),
                                     "note": "300-dpi scan of a 1in=800ft sheet -> ~0.8 m per rotated pixel."},
        "affine_v3_to_rot_2x3": a_v3_rot.tolist(),
        "affine_v3_to_flat_2x3": np.hstack([A_v3_flat, t_v3_flat.reshape(2, 1)]).tolist(),
        "v3_to_rot_decomposition": {"scale_x": sx, "scale_y": sy, "rotation_deg": ang},
        "v3_geotransform_origin_px": [f["v3_ox"], f["v3_oy"], f["v3_px"], f["v3_py"]],
        "flat_scan": "data/derived/tuck1942/georef_source/tuck1942_map_c.tif",
        "flat_scan_size_wh": [f["flat_w"], f["flat_h"]],
        "rotated_size_wh": [f["Wr"], f["Hr"]],
        "validated_v3_envelope": {"col": [float(v3c[:, 0].min()), float(v3c[:, 0].max())],
                                  "row": [float(v3c[:, 1].min()), float(v3c[:, 1].max())]},
        "points_convention": "mapX=lon, mapY=lat, sourceX=col_rot, sourceY=Hr-row_rot (fossick FlatPositioner drop-in).",
        "fossick_usage": "set FlatPositioner GCP_DIR to this dir (V3_GT['c'] already exact), or use affine_v3_to_flat_2x3 directly.",
        "provenance_note": "Supersedes the earlier analytic 4-corner map, which wrongly assumed v3 was an axis-aligned corner warp; v3 actually carries a ~-17deg rotation (60% valid, tilted parallelogram).",
    }
    (OUT_DIR / "tuck1942_map_c_flatgeo.json").write_text(json.dumps(meta, indent=2))
    return {"n_pts": len(lines), "resid_med": float(np.median(resid)), "resid_max": float(resid.max()),
            "scale": (sx, sy), "ang": ang}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rederive", action="store_true", help="re-run SIFT registration (needs opencv)")
    a = ap.parse_args()
    if a.rederive:
        derive_correspondences()
    r = build()
    print(f"wrote {r['n_pts']} GCPs + JSON -> {OUT_DIR}")
    print(f"v3->rot: scale {r['scale'][0]:.4f}/{r['scale'][1]:.4f}  rot {r['ang']:.2f} deg")
    print(f"registration residual: median {r['resid_med']:.2f} px  max {r['resid_max']:.2f} px  (~{r['resid_med']*0.8:.1f}/{r['resid_max']*0.8:.1f} m)")
