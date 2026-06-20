"""Autonomous georeference of a GLO base MTP via section-grid detection + affine to
the township's known CadNSDI corners. No human GCP picking.

detect_grid: find the 6x6 section-grid bounding box in pixels (longest dark
horizontal/vertical lines, excluding the right title block).
affine: least-squares pixel<->geo from the 4 grid corners <-> 4 township corners.
"""
from __future__ import annotations
import json, urllib.parse, urllib.request
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def township_corners(plssid_like: str):
    p = {"where": f"PLSSID LIKE '{plssid_like}'", "outFields": "PLSSID",
         "returnGeometry": "true", "outSR": "4326", "f": "geojson"}
    u = ("https://gis.blm.gov/akarcgis/rest/services/PLSS/BLM_AK_PLSS_CadNSDI/FeatureServer/3/query?"
         + urllib.parse.urlencode(p))
    d = json.load(urllib.request.urlopen(u, timeout=90))
    g = d["features"][0]["geometry"]
    ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
    pts = np.array([(x, y) for x, y, *_ in ring])
    lon, lat = pts[:, 0], pts[:, 1]
    # corner = ring vertex nearest each bbox corner
    def nearest(tx, ty): return pts[np.argmin((lon - tx) ** 2 + (lat - ty) ** 2)]
    NW = nearest(lon.min(), lat.max()); NE = nearest(lon.max(), lat.max())
    SE = nearest(lon.max(), lat.min()); SW = nearest(lon.min(), lat.min())
    return dict(NW=NW, NE=NE, SE=SE, SW=SW)


def detect_grid_bbox(path):
    im = np.asarray(Image.open(path).convert("L"))
    H, W = im.shape
    dark = im < 110
    # search region: exclude right title block (x>0.62W) and top/bottom margins
    xr = int(0.62 * W)
    yb0, yb1 = int(0.10 * H), int(0.92 * H)
    xb0, xb1 = int(0.04 * W), xr
    # vertical grid lines: columns (x<xr) with dark over most of the grid height band
    colp = dark[yb0:yb1, :xr].sum(axis=0) / (yb1 - yb0)
    rowp = dark[:, xb0:xb1].sum(axis=1) / (xb1 - xb0)
    vcols = np.where(colp > 0.55)[0]
    rrows = np.where(rowp > 0.55)[0]
    if len(vcols) < 2 or len(rrows) < 2:
        # relax threshold
        vcols = np.where(colp > 0.40)[0]; rrows = np.where(rowp > 0.40)[0]
    x0, x1 = int(vcols.min()), int(vcols.max())
    y0, y1 = int(rrows.min()), int(rrows.max())
    return (x0, y0, x1, y1), (W, H), (colp, rowp)


def affine_px2geo(grid_bbox, corners):
    x0, y0, x1, y1 = grid_bbox
    # pixel corners (y down): NW=(x0,y0) NE=(x1,y0) SE=(x1,y1) SW=(x0,y1)
    P = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float)
    G = np.array([corners["NW"], corners["NE"], corners["SE"], corners["SW"]], float)
    A, *_ = np.linalg.lstsq(P, G, rcond=None)   # 3x2 : [px,py,1]@A = [lon,lat]
    return A  # px->geo


def geo2px_fn(A):
    # invert the 2D affine (lon,lat)->(px,py)
    M = np.array([[A[0, 0], A[1, 0]], [A[0, 1], A[1, 1]]])  # [[a,d],[b,e]] for lon,lat
    c = np.array([A[2, 0], A[2, 1]])
    Minv = np.linalg.inv(M)
    return lambda lon, lat: Minv @ (np.array([lon, lat]) - c)


if __name__ == "__main__":
    import sys
    path, plss = sys.argv[1], sys.argv[2]
    gb, (W, H), _ = detect_grid_bbox(path)
    cor = township_corners(plss)
    A = affine_px2geo(gb, cor)
    print("grid bbox px:", gb, "of", (W, H))
    print("township corners:", {k: [round(v[0], 4), round(v[1], 4)] for k, v in cor.items()})
    # report implied scale + residual
    g2p = geo2px_fn(A)
    for k, v in cor.items():
        px = g2p(v[0], v[1]); print(f"  {k} geo->px {px.round(0)}")
