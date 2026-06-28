"""H2 redesign step 0: pull the native 5 m IFSAR DTM over the RI 2024-7 map area.

Study extent = the Big Hurrah-Council-Bluff bedrock map (DGGS RI 2024-7, Werdon
et al. 2024), padded ~2 km, in EPSG:3338. The existing Nome-district 5 m DTM
stops at x=-473300 and y=1700200, so the eastern third (Council/Bluff) and the
northern strip of the RI map are uncovered; this fetch gets the whole map area
on one cell-aligned grid so the confined-reach + down-channel-distance work has
a single fine DEM to run on.

Source: State of Alaska DGGS IFSAR_DTM ImageServer (native 5 m, EPSG:3338), the
same bare-earth product used for the Nome district. Tiled + streamed to disk by
ai_minerals.data.ifsar_dggs.fetch_native.

Run:  uv run python -m scripts.nome_placer.h2_confined_reach.fetch_dem
      (heavy network pull; resumable -- reruns skip completed tiles)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from ai_minerals.data import ifsar_dggs

# RI 2024-7 map extent in EPSG:3338 is x[-500004,-450930] y[1656148,1716142];
# pad to clean 1 km multiples so the grid is tidy and covers upstream contacts.
STUDY_BOUNDS_3338 = (-502000.0, 1654000.0, -449000.0, 1718000.0)
RAW = Path("data/raw/ifsar_dggs")
OUT = RAW / "ifsar_dtm_5m_bighurrah_council_bluff_3338.tif"


def main() -> None:
    retrieved = dt.date.today().isoformat()
    meta = ifsar_dggs.fetch_native(
        STUDY_BOUNDS_3338,
        product="DTM",
        out_path=OUT,
        aoi_name="bighurrah_council_bluff",
    )
    ifsar_dggs.write_provenance([meta], retrieved)
    print("\n=== summary ===")
    print(
        f"DTM: {meta['size_px'][0]}x{meta['size_px'][1]} @ 5 m, "
        f"coverage {meta['coverage_fraction']:.1%}, sha256 {meta['sha256'][:12]}, "
        f"{meta['output']}"
    )


if __name__ == "__main__":
    main()
