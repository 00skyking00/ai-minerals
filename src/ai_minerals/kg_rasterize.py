"""Extent-aware, capped rasterization of the fossick F2 KG feature export (F2).

Turns fossick's per-entity knowledge-graph feature table into covariate rasters
on a model grid, so the F3 driver can measure whether the KG actually helps the
Nome placer/lode models under F1's leak-guarded cross-validation.

What this module enforces (the over-attribution guards from the run plan and the
F2 manifest, ``feature_manifest.json``):

* **Extent-aware attribution.** Each ground gets its covariate vector written to
  cells according to its extent, never blindly to one cell:
  ``point`` -> the single cell at the centroid;
  ``claim_polygon`` -> every cell the surveyed polygon covers;
  ``area_footprint`` -> spread across the property footprint (a buffered disc
  when the export carries only a centroid, see the note below), not a centroid;
  ``district`` -> dropped from cell attribution entirely (a region footprint is a
  regional covariate, pinning it to cells is the ecological fallacy).

* **Per-cell cap (USGS Alaska OFR 2021-1041).** Where several grounds land on the
  same cell, the content attributes come from the single highest-scoring ground
  only (score = ``source_grade_numeric`` x ``mean_confidence``), and the
  occurrence count is capped at one. Known-deposit clusters and PU positives
  therefore cannot pile onto already-known ground.

Two covariate kinds come out of here, and the distinction matters for reading the
F3 marginal-AUC result straight:

* **Spatial fields** (defined at every cell): distance to nearest KG claim,
  claim density in a radius, distance to nearest KG occurrence. These carry the
  kind of proximity signal a prospectivity map can use, and they are exactly what
  F1's 1 km dead-zone is built to grade without leakage.
* **Ground-attribute fields** (gathered at the ground's cells, a fill value
  elsewhere): source grade, mean confidence, and the prose host/alteration/
  structure/workings flags. These are non-null only where a record exists, so
  "has a KG attribute" is collinear with "is a known site". That coverage
  confound is real; F3 measures it with a coverage-flag control rather than
  hiding it.

CRS contract: geometries come out of the export as bare (CRS-unqualified) WKT,
which GeoSPARQL reads as CRS84/WGS84. The label loaders place occurrences the
same way (``kg`` adapter default ``source_crs=EPSG:4326``), so we follow that
convention here too -- a presence point and its KG ground land in the same cell.
The ~155 m NAD27/WGS84 offset documented in ``adapters/occurrences/kg`` is shared
by labels and covariates alike, so it cancels for the marginal-AUC measurement.

Area-footprint geometry note: the current F2 export resolves the 178 KARDEX
``area_footprint`` grounds to point centroids, not polygons (no surveyed claim
footprint is carried). "Spread across the footprint" is approximated by a
``area_buffer_m`` disc around the centroid (default 200 m ~ a 40-acre claim
half-width). Where a ground resolves to a real polygon it is used directly. This
is flagged in the F3 report as a fossick export gap, not a silent choice.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely import wkt as shapely_wkt
from shapely.geometry import base as shapely_base

WORKING_CRS = "EPSG:3338"
EXPORT_CRS = "EPSG:4326"  # bare WKT read as WGS84, matching the label loaders

_HAS_GEOM = "http://www.opengis.net/ont/geosparql#hasGeometry"
_AS_WKT = "http://www.opengis.net/ont/geosparql#asWKT"
_CRS_PREFIX_RE = re.compile(r"^<[^>]+>\s*")

# --- the admissibility split (the F2 modeling decision, documented above) ----
# Ground-character attributes a model may legitimately use to find NEW ground.
KG_CONTENT_COLUMNS: tuple[str, ...] = (
    "source_grade_numeric",
    "mean_confidence",
)
# Prose ontology flags describing the ROCK (host / alteration / structure /
# workings), not the commodity -- admissible context.
PROSE_CONTEXT_FLAGS: tuple[str, ...] = (
    "geo_host_schist", "geo_host_marble", "geo_host_carbonate", "geo_host_gneiss",
    "geo_host_granite", "geo_host_metavolc", "geo_host_clastic", "geo_host_unconsolidated",
    "geo_struct_vein", "geo_struct_shear", "geo_struct_contact",
    "geo_alt_silicic", "geo_alt_sericite", "geo_alt_chlorite", "geo_alt_carbonate",
    "geo_alt_oxide", "geo_alt_sulfide",
    "geo_work_underground", "geo_work_surface", "geo_work_hydraulic",
    "geo_metamorph",
)
# Columns that RESTATE the label (placer/lode identity, commodity, site grade,
# distance to the labelled occurrences). Excluded from the KG arm; used only in
# the F3 leakage-demo arm to quantify the trap.
KG_LABEL_COLUMNS: tuple[str, ...] = (
    "is_placer", "is_lode", "deposit_model_confidence", "commodity_main_confidence",
)
PROSE_LABEL_FLAGS: tuple[str, ...] = (
    "geo_min_placer", "geo_min_lode", "geo_ore_gold", "geo_ore_sulfide", "geo_ore_oxide",
)

_GRADE_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


# --------------------------------------------------------------------------- #
# Geometry resolution from the KG JSON-LD
# --------------------------------------------------------------------------- #
def resolve_ground_geometries(kg_path: Path) -> dict[str, shapely_base.BaseGeometry]:
    """Map each subject URI to its WGS84 shapely geometry via
    ``geo:hasGeometry`` -> ``geo:asWKT`` in ``kg_nome.jsonld``."""
    doc = json.loads(Path(kg_path).read_text())
    graph = doc if isinstance(doc, list) else doc.get("@graph", [])
    by_id = {n["@id"]: n for n in graph if isinstance(n, dict) and "@id" in n}

    def _wkt_for(uri: str) -> str | None:
        node = by_id.get(uri)
        if not node:
            return None
        gref = node.get(_HAS_GEOM)
        if not gref:
            return None
        gid = gref[0].get("@id") if isinstance(gref, list) else gref
        lit = by_id.get(gid, {}).get(_AS_WKT)
        if not lit:
            return None
        return lit[0].get("@value") if isinstance(lit, list) else lit

    out: dict[str, shapely_base.BaseGeometry] = {}
    for uri in by_id:
        w = _wkt_for(uri)
        if w is None:
            continue
        try:
            out[uri] = shapely_wkt.loads(_CRS_PREFIX_RE.sub("", str(w)))
        except Exception:  # malformed WKT in a node we don't need; skip it
            continue
    return out


def load_feature_entities(features_dir: Path, kg_path: Path) -> gpd.GeoDataFrame:
    """Read the F2 feature table + prose flags, attach each entity's resolved
    ground geometry (by ``ground_key``), and return a GeoDataFrame in
    ``WORKING_CRS``. ``extent_kind`` is the ground extent that drives attribution.
    """
    features_dir = Path(features_dir)
    table = pd.read_csv(features_dir / "feature_table.csv")
    prose = pd.read_json(features_dir / "prose_features.jsonl", lines=True)
    keep_prose = ["corpus", "entity_id", "n_geoterm_families", "embedding",
                  *PROSE_CONTEXT_FLAGS, *PROSE_LABEL_FLAGS]
    prose = prose[[c for c in keep_prose if c in prose.columns]]
    df = table.merge(prose, on=["corpus", "entity_id"], how="left")

    geoms = resolve_ground_geometries(kg_path)
    g: list[shapely_base.BaseGeometry | None] = []
    for _, r in df.iterrows():
        geom = geoms.get(r["ground_key"]) or geoms.get(r["entity_uri"])
        if geom is None and pd.notna(r.get("longitude")) and pd.notna(r.get("latitude")):
            from shapely.geometry import Point
            geom = Point(float(r["longitude"]), float(r["latitude"]))
        g.append(geom)
    gdf = gpd.GeoDataFrame(df, geometry=g, crs=EXPORT_CRS)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    # the ground extent that drives attribution (falls back to the entity extent)
    gdf["attr_extent"] = gdf["ground_extent_kind"].fillna(gdf["extent_kind"])
    # capping score: source completeness x mean extraction confidence
    grade_num = gdf["source_grade"].map(_GRADE_NUM).fillna(
        gdf.get("source_grade_numeric")).fillna(1.0)
    gdf["score"] = grade_num.astype(float) * gdf["mean_confidence"].fillna(0.0).astype(float)
    # bool flags -> 0/1 floats
    for c in (*PROSE_CONTEXT_FLAGS, *PROSE_LABEL_FLAGS, "is_placer", "is_lode"):
        if c in gdf.columns:
            gdf[c] = gdf[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(0.0)
    return gdf.to_crs(WORKING_CRS).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Grid template + extent-aware cell attribution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridTemplate:
    """Raster grid the KG covariates are rendered onto."""

    transform: rasterio.Affine
    shape: tuple[int, int]
    crs: str

    @classmethod
    def from_raster(cls, path: Path) -> "GridTemplate":
        with rasterio.open(path) as ds:
            return cls(ds.transform, (ds.height, ds.width), str(ds.crs))

    @property
    def res_m(self) -> float:
        return float(abs(self.transform.a))


def _cells_for_geometry(
    geom: shapely_base.BaseGeometry, extent: str, tpl: GridTemplate, *, area_buffer_m: float
) -> np.ndarray:
    """Return the (row, col) cells a ground occupies under its extent rule."""
    if extent == "district":
        return np.empty((0, 2), int)  # regional covariate, never pinned to cells
    if geom.geom_type == "Point":
        if extent == "area_footprint":
            shape_geom = geom.buffer(area_buffer_m)  # absent footprint -> claim-sized disc
        else:  # point -> the single containing cell
            row, col = rasterio.transform.rowcol(tpl.transform, geom.x, geom.y)
            h, w = tpl.shape
            return (np.array([[row, col]], int)
                    if 0 <= row < h and 0 <= col < w else np.empty((0, 2), int))
    else:
        shape_geom = geom  # claim_polygon / resolved area polygon -> covered cells
    mask = rasterize([(shape_geom, 1)], out_shape=tpl.shape, transform=tpl.transform,
                     fill=0, all_touched=True, dtype="uint8").astype(bool)
    return np.argwhere(mask)


# --------------------------------------------------------------------------- #
# The capped, extent-aware covariate stack
# --------------------------------------------------------------------------- #
@dataclass
class KGStack:
    """Rendered KG covariate stack + the layers the F2 verification reads."""

    bands: dict[str, np.ndarray]      # covariate name -> (H, W) float array
    template: GridTemplate
    winner_idx: np.ndarray            # (H, W) winning entity iloc per cell, -1 = none
    raw_count: np.ndarray             # (H, W) uncapped grounds-per-cell
    capped_coverage: np.ndarray       # (H, W) 1 where any ground (cap = at most 1)
    content_columns: list[str]
    fill_value: float

    def sample_at(self, xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
        """Sample every band at projected (x, y) point coords (WORKING_CRS)."""
        rows, cols = rasterio.transform.rowcol(self.template.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        h, w = self.template.shape
        inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        rr, cc = np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)
        out = {}
        for name, band in self.bands.items():
            v = band[rr, cc]
            out[name] = np.where(inb, v, self.fill_value)
        return pd.DataFrame(out)


def build_kg_stack(
    entities: gpd.GeoDataFrame,
    template: GridTemplate,
    *,
    content_columns: list[str] | None = None,
    embedding: bool = False,
    area_buffer_m: float = 200.0,
    claim_radius_m: float = 400.0,
    fill_value: float = 0.0,
) -> KGStack:
    """Render the extent-aware, capped KG covariate stack onto ``template``.

    Spatial fields (distance to claim/occurrence, claim density) are computed at
    every cell from de-duplicated ground centroids. Content attributes (and,
    optionally, the prose embedding) are gathered from the single highest-scoring
    ground per cell -- the per-cell cap.
    """
    content_columns = list(content_columns if content_columns is not None
                           else (*KG_CONTENT_COLUMNS, *PROSE_CONTEXT_FLAGS, "n_geoterm_families"))
    h, w = template.shape

    # --- per-cell winner (cap): paint entity iloc in ascending score order ----
    order = np.argsort(entities["score"].to_numpy(), kind="stable")
    winner = np.full((h, w), -1, int)
    raw_count = np.zeros((h, w), int)
    for iloc in order:
        ent = entities.iloc[iloc]
        cells = _cells_for_geometry(ent.geometry, ent["attr_extent"], template,
                                    area_buffer_m=area_buffer_m)
        if len(cells):
            r, c = cells[:, 0], cells[:, 1]
            winner[r, c] = iloc          # higher score painted later -> wins (cap)
            raw_count[r, c] += 1         # uncapped tally (for the verification)
    capped_coverage = (winner >= 0).astype(np.uint8)

    bands: dict[str, np.ndarray] = {}
    # coverage flag (the F3 confound control): 1 where any ground covers the cell
    bands["kg_present"] = capped_coverage.astype(np.float32)

    # --- spatial fields, defined everywhere -----------------------------------
    cents = entities.geometry.representative_point()
    ex, ny = cents.x.to_numpy(), cents.y.to_numpy()
    is_claim = entities["attr_extent"].isin(["claim_polygon", "area_footprint"]).to_numpy()
    is_point = (entities["attr_extent"] == "point").to_numpy()
    cols_x = template.transform.c + (np.arange(w) + 0.5) * template.transform.a
    rows_y = template.transform.f + (np.arange(h) + 0.5) * template.transform.e
    gx, gy = np.meshgrid(cols_x, rows_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    if is_claim.any():
        tree = cKDTree(np.column_stack([ex[is_claim], ny[is_claim]]))
        d, _ = tree.query(grid_pts, k=1)
        bands["kg_dist_claim_m"] = d.reshape(h, w).astype(np.float32)
        bands["kg_claim_density"] = np.array(
            [len(i) for i in tree.query_ball_point(grid_pts, r=claim_radius_m)]
        ).reshape(h, w).astype(np.float32)
    if is_point.any():
        d, _ = cKDTree(np.column_stack([ex[is_point], ny[is_point]])).query(grid_pts, k=1)
        bands["kg_dist_occurrence_m"] = d.reshape(h, w).astype(np.float32)

    # --- content attributes, gathered from the per-cell winner (cap) ----------
    flat = winner.ravel()
    have = flat >= 0
    for col in content_columns:
        if col not in entities.columns:
            continue
        vals = entities[col].to_numpy(np.float32)
        band = np.full(h * w, fill_value, np.float32)
        band[have] = vals[flat[have]]
        bands[col] = band.reshape(h, w)

    if embedding and "embedding" in entities.columns:
        emb = np.vstack(entities["embedding"].to_numpy()).astype(np.float32)  # (n, 96)
        for j in range(emb.shape[1]):
            band = np.full(h * w, fill_value, np.float32)
            band[have] = emb[flat[have], j]
            bands[f"prose_emb_{j:02d}"] = band.reshape(h, w)

    return KGStack(bands=bands, template=template, winner_idx=winner,
                   raw_count=raw_count, capped_coverage=capped_coverage,
                   content_columns=content_columns, fill_value=fill_value)


# --------------------------------------------------------------------------- #
# Fold-aware, leave-one-out spatial fields (the leak-free F3 arm, kg_loo)
# --------------------------------------------------------------------------- #
# The covariate stack above is rendered once from the FULL ground set, so a test
# cell's distance-to-occurrence sees the test cell's own record (distance 0) --
# the coverage confound the F3 report diagnoses. These helpers recompute the
# three spatial fields per CV fold from the fold's TRAINING-region grounds only,
# never from a ground in the query cell, so a held-out positive cannot see its
# own record. That is the only KG arm that can speak to real proximity signal.


def block_xy(coords: np.ndarray, x0: float, y0: float, block_size_m: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """Square-block (bx, by) indices for points, using the same origin/edge as
    ``spatial_cv._grid_blocks`` so ground blocks line up with sample folds."""
    c = np.asarray(coords, float)
    bx = np.floor((c[:, 0] - x0) / block_size_m).astype(int)
    by = np.floor((c[:, 1] - y0) / block_size_m).astype(int)
    return bx, by


def ground_points_by_extent(entities: gpd.GeoDataFrame, template: GridTemplate
                            ) -> dict[str, np.ndarray]:
    """Occurrence (point-extent) and claim (claim/area-extent) ground points,
    deduped to unique grid cells on ``template``.

    Returns ``occ_xy``/``claim_xy`` (cell-centroid coords, (m, 2)) and
    ``occ_rc``/``claim_rc`` ((row, col), (m, 2) int) for each kind. District-extent
    grounds are excluded, matching the cell-attribution guard."""
    reps = entities.geometry.representative_point()
    ex, ny = reps.x.to_numpy(), reps.y.to_numpy()
    ext = entities["attr_extent"].to_numpy()
    masks = {"occ": ext == "point",
             "claim": np.isin(ext, ["claim_polygon", "area_footprint"])}
    out: dict[str, np.ndarray] = {}
    for kind, mask in masks.items():
        xy = np.column_stack([ex[mask], ny[mask]]) if mask.any() else np.empty((0, 2))
        if not len(xy):
            out[f"{kind}_xy"], out[f"{kind}_rc"] = xy, np.empty((0, 2), int)
            continue
        rows, cols = rasterio.transform.rowcol(template.transform, xy[:, 0], xy[:, 1])
        rc = np.column_stack([np.asarray(rows), np.asarray(cols)]).astype(int)
        _, keep = np.unique(rc, axis=0, return_index=True)  # one ground per cell
        keep.sort()
        rc_u = rc[keep]
        cx = template.transform.c + (rc_u[:, 1] + 0.5) * template.transform.a
        cy = template.transform.f + (rc_u[:, 0] + 0.5) * template.transform.e
        out[f"{kind}_xy"], out[f"{kind}_rc"] = np.column_stack([cx, cy]), rc_u
    return out


def sample_cell_centroids(coords: np.ndarray, template: GridTemplate
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(cell-centroid xy, (row, col)) for sample points on ``template`` -- the
    grain at which "the cell's own record" is excluded in the leave-one-out."""
    c = np.asarray(coords, float)
    rows, cols = rasterio.transform.rowcol(template.transform, c[:, 0], c[:, 1])
    rc = np.column_stack([np.asarray(rows), np.asarray(cols)]).astype(int)
    cx = template.transform.c + (rc[:, 1] + 0.5) * template.transform.a
    cy = template.transform.f + (rc[:, 0] + 0.5) * template.transform.e
    return np.column_stack([cx, cy]), rc


def _nearest_excluding_own_cell(samp_xy: np.ndarray, samp_rc: np.ndarray,
                                g_xy: np.ndarray, g_rc: np.ndarray, fill: float
                                ) -> np.ndarray:
    """Distance from each sample to the nearest ground NOT in the sample's own
    cell. Grounds are deduped to one per cell, so the second neighbour suffices."""
    out = np.full(len(samp_xy), fill, float)
    if not len(g_xy):
        return out
    k = min(len(g_xy), 2)
    dd, ii = cKDTree(g_xy).query(samp_xy, k=k)
    if dd.ndim == 1:
        dd, ii = dd[:, None], ii[:, None]
    for i in range(len(samp_xy)):
        for j in range(dd.shape[1]):
            gi = ii[i, j]
            if g_rc[gi, 0] == samp_rc[i, 0] and g_rc[gi, 1] == samp_rc[i, 1]:
                continue  # the cell's own record -> leave it out
            out[i] = dd[i, j]
            break
    return out


def loo_spatial_features(
    sample_xy: np.ndarray,
    sample_rc: np.ndarray,
    occ_xy: np.ndarray,
    occ_rc: np.ndarray,
    claim_xy: np.ndarray,
    claim_rc: np.ndarray,
    *,
    claim_radius_m: float = 400.0,
    fill_dist: float = 1.0e6,
) -> pd.DataFrame:
    """The three leak-free spatial fields at each sample, excluding any ground in
    the sample's own grid cell. Ground arrays must be pre-filtered to the visible
    (training-region) set for the fold. Mirrors the leaky ``kg_spatial`` columns:
    ``kg_dist_occurrence_m``, ``kg_dist_claim_m``, ``kg_claim_density``."""
    rc = np.asarray(sample_rc)
    dist_occ = _nearest_excluding_own_cell(sample_xy, rc, np.asarray(occ_xy),
                                           np.asarray(occ_rc), fill_dist)
    dist_claim = _nearest_excluding_own_cell(sample_xy, rc, np.asarray(claim_xy),
                                             np.asarray(claim_rc), fill_dist)
    density = np.zeros(len(sample_xy), float)
    if len(claim_xy):
        crc = np.asarray(claim_rc)
        nbr = cKDTree(np.asarray(claim_xy)).query_ball_point(sample_xy, r=claim_radius_m)
        for i, idxs in enumerate(nbr):
            if not idxs:
                continue
            same = (crc[idxs, 0] == rc[i, 0]) & (crc[idxs, 1] == rc[i, 1])
            density[i] = len(idxs) - int(same.sum())  # exclude the own-cell claim
    return pd.DataFrame({
        "kg_dist_occurrence_m": dist_occ.astype(np.float32),
        "kg_dist_claim_m": dist_claim.astype(np.float32),
        "kg_claim_density": density.astype(np.float32),
    })


def write_stack_geotiff(stack: KGStack, out_path: Path) -> list[str]:
    """Write the covariate stack to a multiband GeoTIFF; return the band order."""
    names = list(stack.bands)
    h, w = stack.template.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=len(names),
        dtype="float32", crs=stack.template.crs, transform=stack.template.transform,
        compress="deflate",
    ) as ds:
        for i, name in enumerate(names, 1):
            ds.write(stack.bands[name].astype(np.float32), i)
            ds.set_band_description(i, name)
    return names
