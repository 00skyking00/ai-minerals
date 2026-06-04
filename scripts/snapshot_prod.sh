#!/usr/bin/env bash
# Pull a snapshot of the live https://johnsondevco.com/ai-minerals/ from
# Hostinger and rotate the last 3 archives. Run BEFORE every prod deploy
# so a revert is just "extract tarball, rsync back".
#
# Usage:
#   bash scripts/snapshot_prod.sh
#
# Output:
#   snapshots/ai-minerals_<YYYY-MM-DD_HHMM>_<short-sha>.tgz
#
# Restore (after a bad prod deploy):
#   STAMP=$(ls -t snapshots/ai-minerals_*.tgz | head -1)
#   mkdir -p /tmp/restore
#   tar -xzf "${STAMP}" -C /tmp/restore
#   rsync -azP --delete /tmp/restore/ai-minerals_*/ hostinger:public_html/ai-minerals/
#   rm -rf /tmp/restore
#
# snapshots/ is gitignored; archives are local-only.

set -euo pipefail

SNAP_DIR="snapshots"
KEEP=3
STAMP="$(date +%Y-%m-%d_%H%M)"
SHA="$(git rev-parse --short HEAD)"
NAME="ai-minerals_${STAMP}_${SHA}"
REMOTE_HOST="hostinger"
REMOTE_PATH="public_html/ai-minerals/"

mkdir -p "${SNAP_DIR}/staging"

echo "==> Pulling live prod (${REMOTE_HOST}:${REMOTE_PATH}) to ${SNAP_DIR}/staging/${NAME}/"
rsync -az --delete "${REMOTE_HOST}:${REMOTE_PATH}" "${SNAP_DIR}/staging/${NAME}/"

echo "==> Archiving"
tar -czf "${SNAP_DIR}/${NAME}.tgz" -C "${SNAP_DIR}/staging" "${NAME}"
rm -rf "${SNAP_DIR}/staging/${NAME}"
rmdir "${SNAP_DIR}/staging" 2>/dev/null || true

echo "==> Rotating (keeping ${KEEP} newest)"
ls -1t "${SNAP_DIR}"/ai-minerals_*.tgz 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -v

echo
echo "==> Current snapshots:"
ls -lh "${SNAP_DIR}"/ai-minerals_*.tgz
