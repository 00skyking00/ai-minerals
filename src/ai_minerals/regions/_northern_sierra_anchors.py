"""Anchor districts for the northern-Sierra placer Phase 1 validation gate.

Seven historic hydraulic-mining districts on the Tertiary deep-gravel belt.
Phase 1 (`scorers/usgs_alaska_placer.py`) must place every one of them in
the top decile of the within-AOI index, without any of them appearing in
the training labels. Phase 2 (`scripts/northern_sierra_placer_train_predict_250m.py`)
masks the cells containing these centroids out of every training fold,
every PU bag, and every calibration fold.

Centroids are approximate (decimal degrees, WGS84) — at 250 m grid
resolution, sub-arcsecond precision isn't meaningful. Sources: USGS GNIS
populated-place / mining-district feature records, the Lindgren 1911
PP73 plates, and the Hydraulic Mine Pits of California polygon set
(DOI 10.5066/F7J38QMD). A reviewer wanting tighter centroids should
sample the corresponding pit polygon from `data/raw/hydraulic_pits/`.
"""

from __future__ import annotations


ANCHOR_DISTRICTS: dict[str, tuple[float, float]] = {
    # (longitude, latitude) — (x, y) order to match shapely/rasterio conventions.
    # Malakoff Diggins and North Bloomfield are <2 km apart and form a single
    # hydraulic-mine complex; consolidated under the Malakoff centroid.
    "Malakoff Diggins / North Bloomfield":  (-120.910, 39.371),
    "North San Juan":                       (-121.105, 39.371),
    "Dutch Flat":                           (-120.842, 39.207),
    "You Bet":                              (-120.913, 39.241),
    "Iowa Hill":                            (-120.851, 39.103),
    "Forest Hill":                          (-120.829, 39.020),
    "Michigan Bluff":                       (-120.741, 39.039),
}
