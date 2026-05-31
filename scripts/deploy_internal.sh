#!/usr/bin/env bash
# Deploy _site_internal/ to johnsondevco.com/ai-minerals-internal/.
# (Was /plans/ai-minerals-internal/ until the /plans/ namespace was deprecated;
# the old path now 301-redirects here. The public face is the portfolio umbrella
# at /ai-minerals/.)
#
# Usage: bash scripts/deploy_internal.sh
#   Re-render first so the rendered site carries the new site-url:
#   _quarto-internal.yml site-url = https://johnsondevco.com/ai-minerals-internal/
#
# Prerequisites: same SSH alias 'hostinger' as deploy_to_hostinger.sh.

set -euo pipefail

REMOTE_HOST="hostinger"
SUBPATH="ai-minerals-internal"
LEGACY_SUBPATH="plans/ai-minerals-internal"   # 301-redirects here after the move off /plans/
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
echo "==> Updating legacy-path 301 redirect (/plans/ai-minerals-internal/ -> /ai-minerals-internal/)"
# /plans/ is deprecated: no new content there, only this redirect for any old
# deep-notebook links already handed out. Mirrors the /plans/ai-minerals/ -> /ai-minerals/ rule.
ssh "${REMOTE_HOST}" "
  mkdir -p ${REMOTE_BASE}/${LEGACY_SUBPATH}
  cat > ${REMOTE_BASE}/${LEGACY_SUBPATH}/.htaccess <<'EOF'
# 301 redirect /plans/ai-minerals-internal/* -> /ai-minerals-internal/*  (moved off deprecated /plans/)
RewriteEngine On
RewriteRule ^(.*)\$ /ai-minerals-internal/\$1 [R=301,L]
EOF
"

echo
echo "==> Done."
echo "    https://johnsondevco.com/${SUBPATH}/"
echo "    Legacy URL  https://johnsondevco.com/${LEGACY_SUBPATH}/  -> 301 -> /${SUBPATH}/"
