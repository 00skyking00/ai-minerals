"""Pull the native 5 m IFSAR DTM (+ DSM) over the Nome MPM district, clip, QA figure.

District extent = the MPM district grid template bounds in EPSG:3338
(-577100..-473300, 1629300..1700200), which covers Nome west to Solomon and
east to the Big Hurrah lode anchor. At 5 m that is 20760x14180 px, an exact
20x super-resolution of the 100 m MPM covariate grid.

Source: State of Alaska DGGS IFSAR_DTM / IFSAR_DSM ImageServers (native 5 m,
EPSG:3338). DTM is the bare-earth product for drainage/slope; DSM is captured
alongside for reference. TNM Access was the documented alternative; DGGS
resolves cleanly to the genuine 5 m grid in one CRS, so it is the route used.

Emits a hillshade + AOI-outline QA PNG so coverage voids/seams are eye-checkable.

Run:  uv run python -m scripts.nome_placer.fetch_ifsar_5m_district
      (heavy; prefer scripts/run_capped.sh --mem 8G --swap 0 around it)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from ai_minerals.data import ifsar_dggs

# Canonical MPM district extent (EPSG:3338), == _district_grid_template_3338.tif
DISTRICT_BOUNDS_3338 = (-577100.0, 1629300.0, -473300.0, 1700200.0)
RAW = Path("data/raw/ifsar_dggs")
FIG = Path("data/derived/nome_mpm/ifsar_5m_district_hillshade.png")
NODATA = float(ifsar_dggs.NODATA)


def qa_hillshade(dtm_path: Path, fig_path: Path, max_px: int = 2200) -> None:
    """Decimated hillshade of the DTM with the AOI outline, for a coverage eye-check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    with rasterio.open(dtm_path) as src:
        scale = max(1, int(max(src.width, src.height) / max_px))
        out_w, out_h = src.width // scale, src.height // scale
        dem = src.read(
            1, out_shape=(out_h, out_w), resampling=Resampling.average
        ).astype("float32")
        b = src.bounds
        res_m = abs(src.transform.a)

    valid = dem != NODATA
    z = np.where(valid, dem, np.nan)
    ls = LightSource(azdeg=315, altdeg=45)
    # cellsize scaled up by the decimation factor
    hs = ls.hillshade(np.nan_to_num(z, nan=float(np.nanmin(z))), dx=res_m * scale, dy=res_m * scale)
    hs = np.where(valid, hs, np.nan)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(hs, cmap="gray", extent=(b.left, b.right, b.bottom, b.top), origin="upper")
    elev = ax.imshow(
        z, cmap="terrain", alpha=0.45, extent=(b.left, b.right, b.bottom, b.top), origin="upper"
    )
    ax.plot(
        [b.left, b.right, b.right, b.left, b.left],
        [b.bottom, b.bottom, b.top, b.top, b.bottom],
        color="red", lw=1.5, label="district AOI",
    )
    cov = float(valid.mean())
    ax.set_title(
        f"Nome district IFSAR 5 m DTM (native), {res_m:.0f} m source\n"
        f"on-land coverage {cov:.1%} of AOI bbox  ·  EPSG:3338",
        fontsize=11,
    )
    ax.set_xlabel("Easting (m, EPSG:3338)")
    ax.set_ylabel("Northing (m, EPSG:3338)")
    fig.colorbar(elev, ax=ax, shrink=0.7, label="elevation (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"wrote {fig_path}  (on-land coverage {cov:.1%})")


def main() -> None:
    retrieved = dt.date.today().isoformat()
    metas = []
    dtm_path = RAW / "ifsar_dtm_5m_nome_district_3338.tif"
    metas.append(
        ifsar_dggs.fetch_native(
            DISTRICT_BOUNDS_3338, product="DTM", out_path=dtm_path, aoi_name="nome_district"
        )
    )
    dsm_path = RAW / "ifsar_dsm_5m_nome_district_3338.tif"
    metas.append(
        ifsar_dggs.fetch_native(
            DISTRICT_BOUNDS_3338, product="DSM", out_path=dsm_path, aoi_name="nome_district"
        )
    )
    ifsar_dggs.write_provenance(metas, retrieved)
    qa_hillshade(dtm_path, FIG)

    print("\n=== summary ===")
    for m in metas:
        print(
            f"{m['product']}: {m['size_px'][0]}x{m['size_px'][1]} @ 5 m, "
            f"coverage {m['coverage_fraction']:.1%}, sha256 {m['sha256'][:12]}, {m['output']}"
        )


if __name__ == "__main__":
    main()
