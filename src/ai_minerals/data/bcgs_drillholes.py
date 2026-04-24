"""BC GeoFile 2025-11 — Assessment Report Drillhole Database.

Relational GeoPackage (bcgs_ardh.gpkg) with ~6,460 drill-hole collars +
~121k samples + ~5.5M element determinations (long format). This is the
blind-test set for the BC Golden Triangle MPM build.

Fetch downloads the zip, extracts the GeoPackage, filters collars to the
AOI, and pre-pivots the `determin` table to per-hole element maxima so
the downstream adapter is fast.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "bcgs_drillholes"
ZIP_URL = (
    "https://cmscontent.nrs.gov.bc.ca/geoscience/PublicationCatalogue/"
    "GeoFile/BCGS_GF2025-11.zip"
)
LANDING_URL = "https://www2.gov.bc.ca/gov/content/industry/mineral-exploration-mining/british-columbia-geological-survey/publications/digital-geoscience-data"


# Element ppm/ppb conversion thresholds for the `intersected` flag.
# Matches the values documented in adapters/drillholes/bcgs_geofile.py.
CU_INTERSECT_PPM = 2000.0
MO_INTERSECT_PPM = 300.0
AU_INTERSECT_PPB = 500.0
AG_INTERSECT_PPM = 10.0


def fetch(*, force: bool = False) -> Path:
    """Download + extract the GeoFile 2025-11 bundle."""
    out_dir = dataset_dir(NAME)
    zip_path = out_dir / "BCGS_GF2025-11.zip"
    gpkg_path = out_dir / "bcgs_ardh.gpkg"

    if zip_path.exists() and gpkg_path.exists() and not force:
        print(f"GeoFile 2025-11 already present ({gpkg_path.stat().st_size:,} B); skipping.")
    else:
        if not zip_path.exists() or force:
            print(f"Downloading BCGS_GF2025-11.zip from {ZIP_URL}")
            with requests.get(ZIP_URL, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            print(f"  {zip_path.stat().st_size:,} bytes zipped")

        print("Extracting bcgs_ardh.gpkg...")
        with zipfile.ZipFile(zip_path) as zf:
            target = next((n for n in zf.namelist() if n.endswith("bcgs_ardh.gpkg")), None)
            if target is None:
                raise RuntimeError(f"bcgs_ardh.gpkg not found in {zip_path}")
            with zf.open(target) as src, gpkg_path.open("wb") as dst:
                dst.write(src.read())
        print(f"  {gpkg_path.stat().st_size:,} bytes extracted")

    write_source_md(
        NAME,
        title="BC GeoFile 2025-11 — Assessment Report Drillhole Database",
        url=LANDING_URL,
        license="Open Government Licence - British Columbia",
        notes=(
            "Relational GeoPackage. Tables: collar (~6,460), sample (~121k), "
            "determin (~5.5M long-format assays) plus lookups. Post-2015 "
            "holes feed the BCGT blind-test set; per-hole element maxima "
            "are pre-pivoted in clip_to_aoi."
        ),
    )
    return gpkg_path


# Element symbols we need for the blind-test intersection flags + per-hole maxima.
_TARGET_ELEMENTS = ("Cu", "Mo", "Au", "Ag", "Pb", "Zn", "As", "Sb")


def clip_to_aoi(aoi: AOI, *, post_year: int | None = 2015) -> Path:
    """Filter collars to AOI (+ optionally drill year), aggregate max-assay
    per hole across Cu/Mo/Au/Ag/Pb/Zn/As/Sb, emit a flat GeoPackage of collars.

    Columns: hole_id, hole_name, drill_date, total_depth_m, ar_number,
    max_cu_ppm, max_mo_ppm, max_au_ppb, max_ag_ppm, max_pb_ppm, max_zn_ppm,
    max_as_ppm, max_sb_ppm, geometry (WGS84).
    """
    gpkg_path = dataset_dir(NAME) / "bcgs_ardh.gpkg"
    if not gpkg_path.exists():
        fetch()

    # Use the WGS84 view for collar geometry so the bbox filter is simple.
    collars = gpd.read_file(gpkg_path, layer="vw_data_collars_location_sp_ll84")
    print(f"GeoFile 2025-11 collars (province-wide): {len(collars):,}")

    # AOI + date filters
    west, south, east, north = aoi.bbox
    xs = collars.geometry.x.to_numpy()
    ys = collars.geometry.y.to_numpy()
    in_aoi = (xs >= west) & (xs <= east) & (ys >= south) & (ys <= north)
    collars_in = collars[in_aoi].copy()
    print(f"  in {aoi.name} AOI: {len(collars_in):,}")

    if post_year is not None and "drill_start_dt" in collars_in.columns:
        drill_start = pd.to_datetime(collars_in["drill_start_dt"], errors="coerce")
        post_mask = drill_start.dt.year >= post_year
        collars_in = collars_in[post_mask].copy()
        print(f"  drilled ≥ {post_year}: {len(collars_in):,}")

    hole_ids = set(collars_in["hole_id"].astype(str))
    if not hole_ids:
        raise RuntimeError(f"No collars in AOI {aoi.name} after filters.")

    # Pull in the samples and determinations for just those holes.
    samples = gpd.read_file(gpkg_path, layer="sample", ignore_geometry=True)
    samples = samples[samples["hole_id"].astype(str).isin(hole_ids)].copy()
    print(f"  samples for those holes: {len(samples):,}")

    determin = gpd.read_file(gpkg_path, layer="determin", ignore_geometry=True)
    determin = determin[determin["sample_id"].isin(samples["sample_id"])].copy()
    print(f"  determinations for those samples: {len(determin):,}")

    # Join element symbol + unit from the code lookups.
    analytes = gpd.read_file(gpkg_path, layer="code_analytes", ignore_geometry=True)
    analytes = analytes[["analyte_code", "analyte_abbr"]].rename(
        columns={"analyte_abbr": "element"}
    )
    determin = determin.merge(analytes, on="analyte_code", how="left")

    # Also merge units so we know ppm vs ppb per row (we assume below).
    units = gpd.read_file(gpkg_path, layer="code_unit", ignore_geometry=True)
    units = units[["unit_code", "unit_name"]].rename(columns={"unit_name": "unit"})
    determin = determin.merge(units, on="unit_code", how="left")

    # Attach hole_id
    determin = determin.merge(samples[["sample_id", "hole_id"]], on="sample_id", how="left")

    # Normalize abundance to ppm (or ppb for Au), since the `determin` table
    # carries mixed units (%, ppm, ppb, g/t) across rows. Drop rows in
    # unsupported units.
    unit_lower = determin["unit"].astype(str).str.lower().str.strip()
    determin["value_ppm"] = pd.NA
    abundance = pd.to_numeric(determin["abundance"], errors="coerce")
    factor_to_ppm = {"ppm": 1.0, "g/t": 1.0, "%": 10_000.0, "ppb": 0.001, "ng/g": 0.001}
    for u, f in factor_to_ppm.items():
        m = unit_lower == u
        determin.loc[m, "value_ppm"] = abundance[m] * f
    dropped_units = (determin["value_ppm"].isna() & determin["abundance"].notna()).sum()
    if dropped_units:
        print(f"  dropped {dropped_units:,} determinations in non-ppm-convertible units "
              f"({unit_lower[determin['value_ppm'].isna()].value_counts().head(3).to_dict()})")

    agg_cols = {}
    for el in _TARGET_ELEMENTS:
        sub = determin[determin["element"].astype(str).str.upper() == el.upper()]
        sub = sub[sub["value_ppm"].notna()]
        if sub.empty:
            continue
        mx = sub.groupby("hole_id")["value_ppm"].max()
        if el.upper() == "AU":
            # Au conventionally reported in ppb = 0.001 ppm, so store as ppb.
            mx = mx * 1000.0
            col = "max_au_ppb"
        else:
            col = f"max_{el.lower()}_ppm"
        agg_cols[col] = mx

    if agg_cols:
        max_df = pd.DataFrame(agg_cols).reset_index()
    else:
        max_df = pd.DataFrame({"hole_id": []})

    # Final table: collar attrs + per-element maxima
    keep = ["hole_id", "hole_name", "drill_start_dt", "length", "ar_number"]
    keep = [c for c in keep if c in collars_in.columns]
    out = collars_in[keep + ["geometry"]].merge(max_df, on="hole_id", how="left")

    # Standardize column names for the adapter
    out = out.rename(columns={"drill_start_dt": "drill_date", "length": "total_depth_m"})

    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=collars_in.crs)
    # Force numeric dtype on assay columns so GPKG writes them as REAL, not TEXT.
    for c in gdf.columns:
        if c.startswith("max_"):
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce").astype("float64")
    if "total_depth_m" in gdf.columns:
        gdf["total_depth_m"] = pd.to_numeric(gdf["total_depth_m"], errors="coerce").astype("float64")
    out_path = dataset_dir(NAME) / f"bcgs_drillholes_{aoi.name.lower()}.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    print(f"Wrote {out_path} ({len(gdf):,} holes)")
    return out_path


if __name__ == "__main__":
    from ai_minerals.regions.bcgt import BCGT
    fetch()
    clip_to_aoi(BCGT.aoi)
