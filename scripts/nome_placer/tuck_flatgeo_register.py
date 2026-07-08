"""Correct flat<->geo for Tuck-1942 sheets by v3<->rotated-scan image registration.

One systematic repair for the flat-drain basis on sheets A, C, D, E. fossick's
audit (flat_georef_health.json) found the flat<->geo GCP sets are unusable:
A and C reuse one town-of-Nome control block (byte-identical, so at most one
could be right), D has no .points at all, E has 5 points at ~1160 px RMS. The
root cause is a GCP-candidate build that reused a single Nome control set.

The fix does not depend on any of those GCP sets. Each v3 raster
(`v1p5_v3_refined`, Sky-signed-off, exact geotransform) is a rotated,
anisotropically-scaled warp of that sheet's source scan; the map content is a
tilted parallelogram inside the north-up v3 box. So flat<->v3 is not a corner
map, it is a real rotation that must be measured. This registers each v3 raster
to its rotated source scan (georef_source_rotated/) with SIFT+RANSAC:

    v3 pixel --(affine, from registration)--> rotated-scan pixel
             --(exact 90-deg rotation)-------> flat-scan pixel

and writes a fossick-convention .points grid whose (lon,lat) come from
v3_to_geo(rot_to_v3(...)). fossick composes that geo->flat fit with its exact
v3<->geo geotransform; the intermediate geo cancels in v3->flat, so v3's
absolute geo accuracy is irrelevant to positioning. fossick points its GCP_DIR
at the output and adds the per-sheet V3_GT tuple emitted in each JSON.

Run:
    .venv/bin/python -m scripts.nome_placer.tuck_flatgeo_register            # all sheets
    .venv/bin/python -m scripts.nome_placer.tuck_flatgeo_register --sheets a d e
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

DERIVED = Path("data/derived/tuck1942")
V3_DIR = DERIVED / "v1p5_v3_refined"
ROT_DIR = DERIVED / "georef_source_rotated"
FLAT_DIR = DERIVED / "georef_source"
OUT_DIR = DERIVED / "v1p5_v6_flatgeo_gcps"
SHEETS = ["a", "c", "d", "e"]
POSITION_BAR_PX = 50.0   # fossick's positioning bar

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


def sheet_frame(s: str) -> dict:
    with rasterio.open(V3_DIR / f"tuck1942_map_{s}.tif") as ds:
        t = ds.transform
        v3w, v3h = ds.width, ds.height
    rot = json.loads((ROT_DIR / f"tuck1942_map_{s}_rotation.json").read_text())
    wr, hr = rot["rotated_size"]
    ow, oh = rot["original_size"]
    return {"v3_ox": t.c, "v3_oy": t.f, "v3_px": t.a, "v3_py": t.e, "v3w": v3w, "v3h": v3h,
            "Wr": wr, "Hr": hr, "flat_w": ow, "flat_h": oh}


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = np.hstack([src, np.ones((len(src), 1))])
    m, *_ = np.linalg.lstsq(a, dst, rcond=None)
    return m.T


def register(s: str) -> dict:
    """SIFT+RANSAC register v3 <-> rotated scan; return full-res correspondences + stats."""
    import cv2
    from rasterio.enums import Resampling
    np.random.seed(0)

    def load(path: Path, out_w: int, alpha: bool = False):
        with rasterio.open(path) as ds:
            sc = out_w / ds.width
            d = ds.read(out_shape=(ds.count, int(ds.height * sc), out_w), resampling=Resampling.bilinear)
        g = cv2.cvtColor(d[:3].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        m = (d[3] > 10).astype(np.uint8) * 255 if (alpha and d.shape[0] >= 4) else None
        return cv2.createCLAHE(3.0, (16, 16)).apply(g), sc, m

    gv3, sv3, mv3 = load(V3_DIR / f"tuck1942_map_{s}.tif", 6000, alpha=True)
    grot, srot, _ = load(ROT_DIR / f"tuck1942_map_{s}.tif", 4000)
    if mv3 is not None:
        mv3 = cv2.erode(mv3, np.ones((15, 15), np.uint8))
    sift = cv2.SIFT_create(nfeatures=60000, contrastThreshold=0.012, edgeThreshold=20)
    k1, d1 = sift.detectAndCompute(gv3, mv3)
    k2, d2 = sift.detectAndCompute(grot, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=128))
    p1 = np.float32([kp.pt for kp in k1]); p2 = np.float32([kp.pt for kp in k2])
    knn = flann.knnMatch(d1, d2, k=2)
    # bootstrap a seed affine from a strict global match
    boot = [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < 0.72 * n.distance]
    if len(boot) < 8:
        return {"sheet": s, "ok": False, "reason": f"too few bootstrap matches ({len(boot)})"}
    s0 = np.float32([p1[q] for q, _ in boot]); d0 = np.float32([p2[t] for _, t in boot])
    Mseed, seedinl = cv2.estimateAffine2D(s0, d0, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=40000)
    if Mseed is None or seedinl.sum() < 6:
        return {"sheet": s, "ok": False, "reason": "seed affine failed"}
    # guided filter against the seed to kill repetitive false matches
    src, dst = [], []
    for m, n in knn:
        if m.distance < 0.85 * n.distance:
            proj = Mseed[:, :2] @ p1[m.queryIdx] + Mseed[:, 2]
            if np.linalg.norm(proj - p2[m.trainIdx]) < 40:
                src.append(p1[m.queryIdx]); dst.append(p2[m.trainIdx])
    src = np.float32(src); dst = np.float32(dst)
    if len(src) < 8:
        return {"sheet": s, "ok": False, "reason": f"too few guided matches ({len(src)})"}
    _, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=80000)
    inl = inl.ravel().astype(bool)
    v3f = src[inl] / sv3; rotf = dst[inl] / srot
    Afit, inl2 = cv2.estimateAffine2D(v3f.astype(np.float32), rotf.astype(np.float32),
                                      method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=80000)
    inl2 = inl2.ravel().astype(bool)
    v3f, rotf = v3f[inl2], rotf[inl2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"map_{s}_v3_rot_correspondences.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["v3_col", "v3_row", "rot_col", "rot_row"])
        for (a, b), (c, d) in zip(v3f, rotf):
            w.writerow([f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{d:.2f}"])
    return {"sheet": s, "ok": True, "n": int(len(v3f))}


def build(s: str) -> dict:
    f = sheet_frame(s)
    v3c, rotc = [], []
    with open(OUT_DIR / f"map_{s}_v3_rot_correspondences.csv") as fh:
        for r in csv.DictReader(fh):
            v3c.append([float(r["v3_col"]), float(r["v3_row"])])
            rotc.append([float(r["rot_col"]), float(r["rot_row"])])
    v3c, rotc = np.array(v3c), np.array(rotc)
    A2x3 = fit_affine(v3c, rotc)
    A, t = A2x3[:, :2], A2x3[:, 2]
    Ainv = np.linalg.inv(A)
    resid = np.linalg.norm((v3c @ A.T + t) - rotc, axis=1)

    Rflip = np.array([[0.0, 1.0], [-1.0, 0.0]])
    off = np.array([0.0, f["flat_h"] - 1.0])
    A_v3_flat = Rflip @ A
    t_v3_flat = Rflip @ t + off

    # grid over the validated rot envelope, fossick convention
    cols = np.linspace(rotc[:, 0].min(), rotc[:, 0].max(), 12)
    rows = np.linspace(rotc[:, 1].min(), rotc[:, 1].max(), 9)
    lines = []
    for cr in cols:
        for rr in rows:
            v3col, v3row = Ainv @ (np.array([cr, rr]) - t)
            lon = f["v3_ox"] + v3col * f["v3_px"]
            lat = f["v3_oy"] + v3row * f["v3_py"]
            lines.append((float(lon), float(lat), float(cr), float(f["Hr"] - rr)))
    with open(OUT_DIR / f"tuck1942_map_{s}.tif.points", "w") as fh:
        fh.write(POINTS_HEADER)
        for lon, lat, sx, sy in lines:
            fh.write(f"{lon:.10f},{lat:.10f},{sx:.4f},{sy:.4f},1,0,0,0\n")

    # v3 center -> flat, in-bounds sanity
    cc, cr = f["v3w"] / 2, f["v3h"] / 2
    fc, fr = A_v3_flat @ np.array([cc, cr]) + t_v3_flat
    in_bounds = bool((0 <= fc <= f["flat_w"]) and (0 <= fr <= f["flat_h"]))
    sx = float(np.hypot(*A[:, 0])); sy = float(np.hypot(*A[:, 1]))
    ang = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    meta = {
        "map": s,
        "method": "SIFT+RANSAC image registration v3<->rotated-scan (affine) composed with exact 90-deg rotated->flat.",
        "control_points": int(len(v3c)),
        "residual_rot_px": {"median": float(np.median(resid)), "max": float(resid.max())},
        "residual_ground_m_approx": {"median": float(np.median(resid) * 0.8), "max": float(resid.max() * 0.8)},
        "within_position_bar_50px": bool(resid.max() < POSITION_BAR_PX),
        "affine_v3_to_rot_2x3": A2x3.tolist(),
        "affine_v3_to_flat_2x3": np.hstack([A_v3_flat, t_v3_flat.reshape(2, 1)]).tolist(),
        "v3_to_rot_decomposition": {"scale_x": sx, "scale_y": sy, "rotation_deg": ang},
        "fossick_V3_GT_tuple": [f["v3_ox"], f["v3_oy"], f["v3_px"], f["v3_py"], f["v3w"], f["v3h"]],
        "flat_scan_size_wh": [f["flat_w"], f["flat_h"]],
        "rotated_size_wh": [f["Wr"], f["Hr"]],
        "v3_center_to_flat": [float(fc), float(fr)], "v3_center_in_flat_bounds": in_bounds,
        "validated_v3_envelope": {"col": [float(v3c[:, 0].min()), float(v3c[:, 0].max())],
                                  "row": [float(v3c[:, 1].min()), float(v3c[:, 1].max())]},
        "points_convention": "mapX=lon, mapY=lat, sourceX=col_rot, sourceY=Hr-row_rot (fossick FlatPositioner drop-in).",
        "fossick_usage": "GCP_DIR = this dir; add fossick_V3_GT_tuple to V3_GT['" + s + "'].",
    }
    (OUT_DIR / f"tuck1942_map_{s}_flatgeo.json").write_text(json.dumps(meta, indent=2))
    return {"sheet": s, "n": int(len(v3c)), "resid_med": float(np.median(resid)),
            "resid_max": float(resid.max()), "scale": (sx, sy), "ang": ang,
            "in_bounds": in_bounds, "within_bar": bool(resid.max() < POSITION_BAR_PX)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", default=SHEETS)
    ap.add_argument("--build-only", action="store_true", help="skip SIFT, rebuild from existing CSVs")
    a = ap.parse_args()
    print(f"{'sheet':6s} {'n':>4s} {'resid_med':>9s} {'resid_max':>9s} {'scale_x/y':>14s} {'rot':>7s} {'inbnd':>6s} {'<=50px':>7s}")
    for s in a.sheets:
        if not a.build_only:
            reg = register(s)
            if not reg.get("ok"):
                print(f"{s:6s}  REGISTRATION FAILED: {reg.get('reason')}")
                continue
        r = build(s)
        print(f"{s:6s} {r['n']:>4d} {r['resid_med']:>9.2f} {r['resid_max']:>9.2f} "
              f"{r['scale'][0]:>6.3f}/{r['scale'][1]:<7.3f} {r['ang']:>7.2f} "
              f"{str(r['in_bounds']):>6s} {str(r['within_bar']):>7s}")


if __name__ == "__main__":
    main()
