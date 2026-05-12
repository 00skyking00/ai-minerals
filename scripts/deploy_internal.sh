#!/usr/bin/env bash
# Deploy _site_internal/ to johnsondevco.com/plans/ai-minerals-internal/.
# Sibling path to the portfolio site at /plans/ai-minerals/.
#
# Usage: bash scripts/deploy_internal.sh
#
# Prerequisites: same SSH alias 'hostinger' as deploy_to_hostinger.sh.

set -euo pipefail

REMOTE_HOST="hostinger"
SUBPATH="plans/ai-minerals-internal"
REMOTE_BASE="public_html"
REMOTE_DIR="${REMOTE_BASE}/${SUBPATH}"
LOCAL_SITE="_site_internal"

if [ ! -d "${LOCAL_SITE}" ]; then
  echo "ERROR: ${LOCAL_SITE}/ not found. Run 'bash scripts/render_internal.sh' first."
  exit 1
fi

echo "==> Ensuring ${REMOTE_DIR} exists"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

echo
echo "==> Syncing ${LOCAL_SITE}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/"
rsync -avz --delete --progress \
  "${LOCAL_SITE}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo
echo "==> Done."
echo "    https://johnsondevco.com/${SUBPATH}/"
