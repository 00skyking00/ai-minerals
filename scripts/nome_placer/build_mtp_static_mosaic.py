"""Download the BLM MTP raster for the Nome claim area as ONE georeferenced PNG
(stitched from <=4096px MapServer exports, Web Mercator), to serve as a static
ImageOverlay in goldbug -- no per-view BLM round-trips, no tiling.
"""
import math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# claim-area bbox (W,S,E,N) — where the 373 cluster (not the empty 1.2-deg config bbox)
W,S,E,N = -165.62, 64.44, -165.05, 64.66
TARGET_W = 5500          # total mosaic width in px
EXPORT = ("https://gis.blm.gov/akarcgis/rest/services/Land_Status/"
          "BLM_AK_Master_Title_Plats/MapServer/export")
OUT = Path("data/derived/nome_placer/mtp/mtp_nome_static.png")

def m(lon,lat):
    return lon*20037508.342789244/180, math.log(math.tan((90+lat)*math.pi/360))*6378137.0
xW,yS=m(W,S); xE,yN=m(E,N)
wm,hm=xE-xW, yN-yS
total_w=TARGET_W; total_h=int(round(total_w*hm/wm))
cols=math.ceil(total_w/4000); rows=math.ceil(total_h/4000)
cw=total_w//cols; ch=total_h//rows
print(f"mosaic {total_w}x{total_h}  grid {cols}x{rows}  cell {cw}x{ch}  ground~{wm*math.cos(math.radians((S+N)/2))/total_w:.1f} m/px")
mosaic=Image.new("RGB",(cw*cols,ch*rows),"white")
for i in range(cols):
    for j in range(rows):
        bx0=xW+i*wm/cols; bx1=xW+(i+1)*wm/cols
        by1=yN-j*hm/rows; by0=yN-(j+1)*hm/rows   # row 0 = top (north)
        url=(f"{EXPORT}?bbox={bx0},{by0},{bx1},{by1}&bboxSR=3857&imageSR=3857"
             f"&size={cw},{ch}&format=png&dpi=200&layers=show:0&f=image")
        tmp=f"/tmp/_mtpcell_{i}_{j}.png"
        subprocess.run(["curl","-s","--max-time","120","-o",tmp,url],check=True)
        mosaic.paste(Image.open(tmp).convert("RGB"),(i*cw,j*ch))
OUT.parent.mkdir(parents=True,exist_ok=True)
mosaic.save(OUT,optimize=True)
kb=OUT.stat().st_size//1024
print(f"wrote {OUT}  {mosaic.size}  {kb} KB")
# bounds for the ImageOverlay (lat/lon)
print(f"BOUNDS W,S,E,N = {W},{S},{E},{N}")
