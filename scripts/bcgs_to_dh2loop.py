"""BCGS Assessment Report Drillhole Database -> dh2loop standardized tables.

Reads ``data/raw/bcgs_drillholes/bcgs_ardh.gpkg`` (BC Geological Survey
GeoFile 2025-11, ~6,460 collars / ~107k lithology intervals / ~121k
samples / ~5.5M long-format assays across multiple EPSG codes) and writes
dh2loop-compatible tables ready for LoopStructural ingestion:

  data/derived/bcgs_dh2loop/
    Collar.csv       — one row per hole (HoleID, X, Y, RL, MaxDepth, etc.)
    Survey.csv       — one row per hole at depth 0 (azimuth, dip); BCGS
                       doesn't ship downhole deviation surveys, so this is
                       the trivial single-row-per-hole survey table dh2loop
                       requires for pipeline compatibility.
    Lithology.csv    — per-interval lithology with operator vocabulary
                       (no thesaurus mapping; that's downstream work).
    Assay.parquet    — per-sample assays pivoted wide by element symbol,
                       joined to collar for location. Written as parquet
                       (not CSV) because the 121k × ~150-element matrix is
                       large; dh2loop's Assay sidecar pattern doesn't fix
                       a wire format.

All collars get reprojected to EPSG:3005 BC Albers (the BC-wide standard)
plus WGS84 lat/lon, so the Collar.csv carries both projected meters and
geodetic lat/lon as dh2loop expects.

Lithology mapping note: the operator text in ``litho.litho_unit``
("Quartz Biotite Gneiss", "Calc Silicate", etc.) is hard-rock vocabulary,
distinct from the placer terms in the Bear Cub crosswalk. The dh2loop
757-term thesaurus would map these to standardized Detailed_Lithology /
Lithology_Subgroup / Lithology_Group; that mapping is downstream of this
converter (it requires the dh2loop Python package). For now the operator
text passes through to Detailed_Lithology unchanged and the two grouping
columns get the literal "unmapped" sentinel so the schema is intact.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


GPKG = Path("data/raw/bcgs_drillholes/bcgs_ardh.gpkg")
OUT_DIR = Path("data/derived/bcgs_dh2loop")
TARGET_CRS = "EPSG:3005"  # BC Albers — BC-wide standard
WGS84 = "EPSG:4326"


def load_collars() -> gpd.GeoDataFrame:
    """Load + reproject collar layer.

    BCGS stores each hole in its own UTM zone per ``epsg_srid`` (mix of
    EPSG:3155/3156/3157/2955/26708/26709/26710/4326/4269). We groupby
    source CRS, reproject each group to EPSG:3005, then concatenate.
    """
    df = gpd.read_file(GPKG, layer="collar")
    print(f"[collar] raw: {len(df):,} rows across {df['epsg_srid'].nunique()} source CRS codes")

    pieces = []
    for srid, sub in df.groupby("epsg_srid"):
        geom = [Point(x, y) for x, y in zip(sub["easting"], sub["northing"])]
        gdf = gpd.GeoDataFrame(sub.drop(columns=["geometry"], errors="ignore"),
                               geometry=geom, crs=f"EPSG:{srid}")
        pieces.append(gdf.to_crs(TARGET_CRS))
    out = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=TARGET_CRS)
    print(f"[collar] reprojected: {len(out):,} rows in {TARGET_CRS}")
    return out


def build_collar_table(collars: gpd.GeoDataFrame) -> pd.DataFrame:
    """dh2loop Collar table.

    Columns per the dh2loop GMD paper:
      CollarID, HoleID, Longitude, Latitude, X, Y, RL, MaxDepth.
    BCGS adds two carry-throughs the v2.0 POMDP validation step (D.6.B)
    needs — ``DrillStart`` for the date-based train/test split, and
    ``ArNumber`` (BC Assessment Report number) for the provenance trail.
    """
    ll = collars.to_crs(WGS84)
    out = pd.DataFrame({
        "CollarID":   collars["hole_id"].astype("Int64").map(lambda x: f"BCGS-{x}"),
        "HoleID":     collars["hole_name"].astype(str),
        "Longitude":  ll.geometry.x.values,
        "Latitude":   ll.geometry.y.values,
        "X":          collars.geometry.x.values,
        "Y":          collars.geometry.y.values,
        "RL":         collars["elevation"].astype(float),  # already metres for length_unit_code=9
        "MaxDepth":   collars["length"].astype(float),     # metres for length_unit_code=9
        "DrillStart": collars["drill_start_dt"],
        "ArNumber":   collars["ar_number"].astype(str),
    })
    # Sanity: a handful of holes record length in feet (length_unit_code=10). Convert.
    ft_mask = collars["length_unit_code"] == 10
    if ft_mask.any():
        out.loc[ft_mask, "MaxDepth"] *= 0.3048
        print(f"[collar]   {ft_mask.sum()} holes had length in feet -> converted to metres")
    return out


def build_survey_table(collars: gpd.GeoDataFrame) -> pd.DataFrame:
    """dh2loop Survey table.

    BCGS GeoFile 2025-11 carries no downhole-deviation surveys, only the
    initial collar azimuth + dip. The minimal dh2loop-compatible survey
    is one row per hole at depth 0; downstream consumers that need
    full deviation surveys treat this as a straight-hole approximation.
    """
    return pd.DataFrame({
        "CollarID":   collars["hole_id"].astype("Int64").map(lambda x: f"BCGS-{x}"),
        "Depth":      0.0,
        "Azimuth":    collars["azimuth"].astype(float),
        "Inclination": collars["dip"].astype(float),
    })


def build_lithology_table(collar_keys: pd.Series) -> pd.DataFrame:
    """dh2loop Lithology table.

    Operator vocabulary in ``litho_unit`` passes through to
    ``Detailed_Lithology``; the two grouping columns get the literal
    "unmapped" sentinel pending the thesaurus pass downstream.
    """
    df = gpd.read_file(GPKG, layer="litho")
    print(f"[litho] raw: {len(df):,} rows")

    out = pd.DataFrame({
        "CollarID":           df["hole_id"].astype("Int64").map(lambda x: f"BCGS-{x}"),
        "FromDepth":          df["from_depth"].astype(float),
        "ToDepth":            df["to_depth"].astype(float),
        "Detailed_Lithology": df["litho_unit"].astype(str),
        "Lithology_Subgroup": "unmapped",
        "Lithology_Group":    "unmapped",
        "Comments":           df["comments"],
    })
    # Same belt-and-suspenders unit conversion as collar; BCGS uses
    # unit_code=9 (metres) almost universally but a tail of records ship
    # in feet (unit_code=10).
    ft_mask = df["unit_code"] == 10
    if ft_mask.any():
        out.loc[ft_mask, ["FromDepth", "ToDepth"]] *= 0.3048
        print(f"[litho]   {ft_mask.sum()} intervals had depths in feet -> converted to metres")

    # Drop intervals whose CollarID isn't in the collar table (orphans).
    orphans = ~out["CollarID"].isin(collar_keys)
    if orphans.any():
        print(f"[litho]   dropping {orphans.sum()} intervals with no matching collar")
        out = out.loc[~orphans].reset_index(drop=True)
    return out


def build_assay_table(collar_keys: pd.Series) -> pd.DataFrame:
    """dh2loop-style Assay sidecar (BCGS variant).

    Joins ``sample`` (per-interval location) to ``determin`` (long-format
    assay results) to ``code_analytes`` (analyte_code -> element symbol),
    then pivots wide so each row is one sample-interval with one column
    per element.
    """
    sample = gpd.read_file(GPKG, layer="sample")
    determin = gpd.read_file(GPKG, layer="determin")
    analytes = gpd.read_file(GPKG, layer="code_analytes")
    print(f"[assay] sample={len(sample):,}  determin={len(determin):,}  "
          f"analytes={len(analytes)}")

    determin = determin.merge(analytes[["analyte_code", "analyte_abbr"]],
                              on="analyte_code", how="left")
    determin["abundance_f"] = pd.to_numeric(determin["abundance"], errors="coerce")
    # Pivot wide. Take the FIRST measurement per (sample_id, element); a
    # small number of samples have repeat determinations across multiple
    # cert/method combinations — picking the first is the conservative
    # default. Downstream consumers that want multi-method support read
    # the long `determin` table directly.
    wide = (determin
            .dropna(subset=["analyte_abbr", "abundance_f"])
            .groupby(["sample_id", "analyte_abbr"])["abundance_f"]
            .first()
            .unstack(level="analyte_abbr"))
    print(f"[assay]   pivoted to {wide.shape[0]:,} samples × {wide.shape[1]} elements")

    out = sample[["sample_id", "hole_id", "sample_name", "from_depth", "to_depth"]].copy()
    out["CollarID"] = out["hole_id"].astype("Int64").map(lambda x: f"BCGS-{x}")
    out = out.merge(wide, left_on="sample_id", right_index=True, how="left")
    # Unit conversion (handful of samples in feet).
    ft_mask = sample["unit_code"] == 10
    if ft_mask.any():
        out.loc[ft_mask, ["from_depth", "to_depth"]] *= 0.3048
        print(f"[assay]   {ft_mask.sum()} samples had depths in feet -> converted to metres")

    orphans = ~out["CollarID"].isin(collar_keys)
    if orphans.any():
        print(f"[assay]   dropping {orphans.sum()} samples with no matching collar")
        out = out.loc[~orphans].reset_index(drop=True)
    return out.rename(columns={"from_depth": "FromDepth", "to_depth": "ToDepth"})


def summary(collar: pd.DataFrame, survey: pd.DataFrame,
            litho: pd.DataFrame, assay: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Collar:    {len(collar):>9,} holes")
    print(f"Survey:    {len(survey):>9,} survey points (one per hole)")
    print(f"Lithology: {len(litho):>9,} intervals")
    print(f"Assay:     {len(assay):>9,} samples × {assay.shape[1] - 6} elements")
    print()
    print("Geographic extent (WGS84):")
    print(f"  Longitude: {collar['Longitude'].min():.3f} … {collar['Longitude'].max():.3f}")
    print(f"  Latitude:  {collar['Latitude'].min():.3f} … {collar['Latitude'].max():.3f}")
    print()
    print("Depth distribution (collar MaxDepth, metres):")
    q = collar["MaxDepth"].quantile([0.10, 0.50, 0.90]).to_dict()
    print(f"  p10={q[0.10]:.1f}  p50={q[0.50]:.1f}  p90={q[0.90]:.1f}  "
          f"max={collar['MaxDepth'].max():.1f}")
    print()
    print("Top 20 lithology terms (operator vocabulary, raw):")
    for term, count in litho["Detailed_Lithology"].value_counts().head(20).items():
        print(f"  {count:>6,}  {term}")
    print()
    print("Drill-date span (for the D.6.B retrospective POMDP train/test split):")
    dt = pd.to_datetime(collar["DrillStart"], errors="coerce")
    print(f"  earliest: {dt.min().date() if dt.notna().any() else 'n/a'}")
    print(f"  latest:   {dt.max().date() if dt.notna().any() else 'n/a'}")
    pre_2010 = (dt < "2010-01-01").sum()
    post_2010 = (dt >= "2010-01-01").sum()
    print(f"  pre-2010 collars:  {pre_2010:,}")
    print(f"  2010+ collars:     {post_2010:,}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    collars = load_collars()
    collar_table = build_collar_table(collars)
    survey_table = build_survey_table(collars)
    litho_table = build_lithology_table(collar_table["CollarID"])
    assay_table = build_assay_table(collar_table["CollarID"])

    collar_table.to_csv(OUT_DIR / "Collar.csv", index=False)
    survey_table.to_csv(OUT_DIR / "Survey.csv", index=False)
    litho_table.to_csv(OUT_DIR / "Lithology.csv", index=False)
    assay_table.to_parquet(OUT_DIR / "Assay.parquet", index=False)

    print()
    print(f"Wrote {OUT_DIR}/Collar.csv      ({len(collar_table):,} rows)")
    print(f"Wrote {OUT_DIR}/Survey.csv      ({len(survey_table):,} rows)")
    print(f"Wrote {OUT_DIR}/Lithology.csv   ({len(litho_table):,} rows)")
    print(f"Wrote {OUT_DIR}/Assay.parquet   ({len(assay_table):,} rows)")

    summary(collar_table, survey_table, litho_table, assay_table)


if __name__ == "__main__":
    main()
