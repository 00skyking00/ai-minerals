"""Background download of the gSSURGO California subset for v3 Phase D.6.

USDA NRCS gridded Soil Survey Geographic database. The California state subset
is roughly 3-5 GB compressed. Download from the canonical NRCS endpoint with
resumable HTTP-Range support so the fetch survives a connection drop.

If the download completes before v3 Phase D executes, the subsequent
gSSURGO adapter (to be written if and only if the fetch lands) reads
depth-to-bedrock + parent-material features from the FileGDB.

If the download is still in flight when Phase D starts, defer to v4; the
.zip stays on disk for then.

Usage:
    .venv/bin/python scripts/fetch_gssurgo_ca.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "gssurgo"
TARGET = OUT_DIR / "gSSURGO_CA.zip"
PARTIAL = OUT_DIR / "gSSURGO_CA.zip.partial"

# USDA NRCS Box public download. The folder is browsable at
# https://nrcs.app.box.com/v/soils/folder/188053783475 ; the per-state file
# share links point at box.com /shared/static URLs. The exact share URL for
# gSSURGO_CA.zip varies per release (USDA refreshes annually). At plan time
# the canonical alternate-channel URL is:
DOWNLOAD_URLS = [
    # Primary: USDA Box public link (2024 release)
    "https://nrcs.box.com/shared/static/gssurgo_ca_2024.zip",
    # Fallback: NRCS Geospatial Data Gateway (form-driven; this is the direct
    # link the GDG produces after the per-state state selection step)
    "https://gdg.sc.egov.usda.gov/GDGOrder.aspx?order=gSSURGO_CA",
]


def fetch_with_resume(url: str, out: Path, partial: Path) -> bool:
    """Resumable HTTP-Range download. Returns True on success."""
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    pos = partial.stat().st_size if partial.exists() else 0
    if pos > 0:
        headers["Range"] = f"bytes={pos}-"
        print(f"  resuming from byte {pos:,}", flush=True)
    try:
        with requests.get(url, stream=True, headers=headers, timeout=120) as r:
            if r.status_code not in (200, 206):
                print(f"  HTTP {r.status_code} from {url}; trying next URL", flush=True)
                return False
            total = pos + int(r.headers.get("content-length") or 0)
            mode = "ab" if pos > 0 else "wb"
            with open(partial, mode) as f:
                downloaded = pos
                last_pct = -1
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(100 * downloaded / total)
                        if pct != last_pct and pct % 5 == 0:
                            print(f"  {pct}% ({downloaded/1e9:.2f} GB / "
                                  f"{total/1e9:.2f} GB)", flush=True)
                            last_pct = pct
        partial.rename(out)
        return True
    except (requests.RequestException, OSError) as exc:
        print(f"  download error from {url}: {exc!r}", flush=True)
        return False


def main() -> int:
    if TARGET.exists():
        sz = TARGET.stat().st_size
        print(f"already on disk: {TARGET}  ({sz/1e9:.2f} GB)")
        return 0

    print(f"==> fetching gSSURGO California to {TARGET}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for url in DOWNLOAD_URLS:
        print(f"  trying {url}")
        if fetch_with_resume(url, TARGET, PARTIAL):
            print(f"==> success: {TARGET.stat().st_size/1e9:.2f} GB")
            return 0
    print("==> all download URLs failed. Investigate the NRCS Box public folder "
          "manually at https://nrcs.app.box.com/v/soils/folder/188053783475 "
          "for the current per-state share URL, update DOWNLOAD_URLS, retry.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
