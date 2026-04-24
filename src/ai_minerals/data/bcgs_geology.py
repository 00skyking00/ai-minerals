"""Digital Geology of BC — bedrock polygons + faults + Quaternary cover.

Provincial compilation published by BCGS (v2019-12-19). Single GeoPackage
with 3 layers; we pull bedrock + faults, clipped to the AOI.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import requests

from ai_minerals.aoi import AOI
from ai_minerals.data._common import dataset_dir, write_source_md

NAME = "bcgs_geology"
ZIP_URL = (
    "https://cmscontent.nrs.gov.bc.ca/geoscience/bedrock_geology/"
    "BC_digital_geology_gpkg.zip"
)
LANDING_URL = (
    "https://www2.gov.bc.ca/gov/content/industry/mineral-exploration-mining/"
    "british-columbia-geological-survey/geology/bcdigitalgeology"
)


def fetch(*, force: bool = False) -> Path:
    """Download + extract BC_digital_geology.gpkg (~235 MB)."""
    out_dir = dataset_dir(NAME)
    zip_path = out_dir / "BC_digital_geology_gpkg.zip"
    gpkg_path = out_dir / "BC_digital_geology.gpkg"

    if gpkg_path.exists() and not force:
        print(f"BC digital geology present ({gpkg_path.stat().st_size:,} B); skipping.")
    else:
        if not zip_path.exists() or force:
            print(f"Downloading BC Digital Geology from {ZIP_URL}")
            with requests.get(ZIP_URL, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            print(f"  {zip_path.stat().st_size:,} bytes zipped")

        print("Extracting BC_digital_geology.gpkg...")
        with zipfile.ZipFile(zip_path) as zf:
            target = next((n for n in zf.namelist() if n.endswith(".gpkg")), None)
            if target is None:
                raise RuntimeError(f"No .gpkg in {zip_path}")
            with zf.open(target) as src, gpkg_path.open("wb") as dst:
                dst.write(src.read())
        print(f"  {gpkg_path.stat().st_size:,} bytes extracted")

    write_source_md(
        NAME,
        title="BC Digital Geology (BCGS, v2019-12-19)",
        url=LANDING_URL,
        license="Open Government Licence - British Columbia",
        notes=(
            "Provincial bedrock + faults + Quaternary-cover compilation. "
            "Single GeoPackage; clip_to_aoi reads bedrock polygons and "
            "fault lines separately."
        ),
    )
    return gpkg_path


def clip_to_aoi(aoi: AOI) -> tuple[Path, Path]:
    """Clip the bedrock polygons + fault lines to the AOI. Writes two GPKGs."""
    gpkg_path = dataset_dir(NAME) / "BC_digital_geology.gpkg"
    if not gpkg_path.exists():
        fetch()

    # List available layers and pick the bedrock polygon + fault-line ones
    import fiona
    layers = fiona.listlayers(gpkg_path)
    print(f"Digital Geology layers: {layers}")

    bedrock_layer = next(
        (l for l in layers if "bedrock" in l.lower() or "geology" in l.lower() and "fault" not in l.lower()),
        None,
    )
    fault_layer = next((l for l in layers if "fault" in l.lower()), None)
    if bedrock_layer is None or fault_layer is None:
        raise RuntimeError(f"Could not find bedrock + fault layers among {layers}")
    print(f"Using bedrock layer: {bedrock_layer}")
    print(f"Using fault layer: {fault_layer}")

    west, south, east, north = aoi.bbox
    bbox = (west, south, east, north)

    # Read with bbox filter (far cheaper than loading whole province).
    bedrock = gpd.read_file(gpkg_path, layer=bedrock_layer, bbox=bbox)
    faults = gpd.read_file(gpkg_path, layer=fault_layer, bbox=bbox)
    print(f"bedrock polygons in AOI: {len(bedrock):,}")
    print(f"fault lines in AOI: {len(faults):,}")

    bedrock_path = dataset_dir(NAME) / f"bedrock_{aoi.name.lower()}.gpkg"
    faults_path  = dataset_dir(NAME) / f"faults_{aoi.name.lower()}.gpkg"
    bedrock.to_file(bedrock_path, driver="GPKG")
    faults.to_file(faults_path, driver="GPKG")
    print(f"Wrote {bedrock_path}")
    print(f"Wrote {faults_path}")
    return bedrock_path, faults_path


if __name__ == "__main__":
    from ai_minerals.regions.bcgt import BCGT
    fetch()
    clip_to_aoi(BCGT.aoi)
