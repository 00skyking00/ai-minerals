#!/usr/bin/env bash
# Deploy the rendered Quarto site to Hostinger.
#
# By default targets https://johnsondevco.com/ai-minerals/ (production).
# Pass `ai-minerals-beta` as the first arg to deploy to
# https://johnsondevco.com/ai-minerals-beta/ for validation before cutover:
#
#   bash scripts/deploy_to_hostinger.sh                 # -> /ai-minerals/
#   bash scripts/deploy_to_hostinger.sh ai-minerals-beta # -> /ai-minerals-beta/
#
# The legacy /plans/ai-minerals/ -> /ai-minerals/ 301 only updates on the
# production deploy; beta paths get no legacy alias.
#
# Prerequisites:
#   - Site rendered at _site/  (run: quarto render --no-execute)
#   - SSH config alias 'hostinger' in ~/.ssh/config pointing at the
#     Hostinger SFTP account, with the deploy key configured.
#   - Public key (~/.ssh/id_ed25519.pub) registered at
#     Hostinger -> Hosting -> Manage -> SSH Access -> SSH Keys.
#
# Once the key is registered, this script runs without password prompts.

set -euo pipefail

REMOTE_HOST="hostinger"          # ~/.ssh/config alias
SUBPATH="${1:-ai-minerals}"
case "${SUBPATH}" in
  ai-minerals|ai-minerals-beta) ;;
  *)
    echo "ERROR: SUBPATH must be 'ai-minerals' or 'ai-minerals-beta' (got '${SUBPATH}')." >&2
    exit 2
    ;;
esac
LEGACY_SUBPATH="plans/ai-minerals"   # 301 redirect target (production only)

# Hostinger's web root for the primary domain. If johnsondevco.com is
# served from domains/johnsondevco.com/public_html/, change this to
# that path.
REMOTE_BASE="public_html"
REMOTE_DIR="${REMOTE_BASE}/${SUBPATH}"

LOCAL_SITE="_site"

if [ ! -d "${LOCAL_SITE}" ]; then
  echo "ERROR: ${LOCAL_SITE}/ not found. Run 'quarto render --no-execute' first."
  exit 1
fi

echo "==> Target: https://johnsondevco.com/${SUBPATH}/"
echo
echo "==> Pre-flight: web root layout"
ssh "${REMOTE_HOST}" "ls -ld ${REMOTE_BASE} 2>&1 || true; ls -la ${REMOTE_BASE} 2>&1 | head -10"

echo
echo "==> Ensuring ${REMOTE_DIR} exists"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# IMPORTANT — private-data guard. Quarto's website render copies everything
# in the project tree that doesn't start with `.` or `_` into _site/, which
# pulls in data/raw/ (FAMILY-PRIVATE drill-log scans, raw geochem samples,
# unredacted shapefiles), plus src/, tools/, scripts/, research/, design/,
# and the raw .qmd source. None of that may be served publicly. The excludes
# below keep them out, and --delete-excluded removes them from the server if
# a prior deploy ever pushed them. Only the rendered chapter HTML +
# site_libs + data/derived (the public per-cell rasters and figures) go live.
echo
echo "==> Syncing ${LOCAL_SITE}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/ (private dirs + source excluded)"
rsync -avz --delete --delete-excluded --progress \
  --exclude='/data/raw' \
  --exclude='/tools' \
  --exclude='/research' \
  --exclude='/src' \
  --exclude='/scripts' \
  --exclude='/design' \
  --exclude='*.qmd' \
  "${LOCAL_SITE}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

if [ "${SUBPATH}" = "ai-minerals" ]; then
  echo
  echo "==> Updating legacy-path 301 redirect (production only)"
  # Ensure the old /plans/ai-minerals/ path only serves a .htaccess that
  # 301-redirects every request to /ai-minerals/, preserving sub-paths.
  ssh "${REMOTE_HOST}" "
    mkdir -p ${REMOTE_BASE}/${LEGACY_SUBPATH}
    cat > ${REMOTE_BASE}/${LEGACY_SUBPATH}/.htaccess <<'EOF'
# 301 redirect /plans/ai-minerals/* -> /ai-minerals/*  (rename 2026-05-28)
RewriteEngine On
RewriteRule ^(.*)\$ /ai-minerals/\$1 [R=301,L]
EOF
  "
fi

echo
echo "==> Done."
echo "    https://johnsondevco.com/${SUBPATH}/"
if [ "${SUBPATH}" = "ai-minerals" ]; then
  echo "    Legacy URL  https://johnsondevco.com/${LEGACY_SUBPATH}/  -> 301 -> /${SUBPATH}/"
fi
