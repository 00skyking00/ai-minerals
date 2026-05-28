#!/usr/bin/env bash
# Deploy the rendered Quarto site to johnsondevco.com/ai-minerals/
# on Hostinger via SFTP/SSH. (Was /plans/ai-minerals/ until 2026-05-28;
# the old path now serves a 301 redirect via .htaccess.)
#
# Usage:
#   bash scripts/deploy_to_hostinger.sh
#
# Prerequisites:
#   - Site rendered at _site/  (run: quarto render --no-execute)
#   - SSH config alias 'hostinger' in ~/.ssh/config pointing at
#     u739994379@157.173.209.26:65002 with the deploy key.
#   - Public key (~/.ssh/id_ed25519.pub) registered at
#     Hostinger -> Hosting -> Manage -> SSH Access -> SSH Keys.
#
# Once the key is registered, this script runs without password prompts.

set -euo pipefail

REMOTE_HOST="hostinger"          # ~/.ssh/config alias
SUBPATH="ai-minerals"
LEGACY_SUBPATH="plans/ai-minerals"   # 301 redirect target after the rename

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

echo "==> Pre-flight: web root layout"
ssh "${REMOTE_HOST}" "ls -ld ${REMOTE_BASE} 2>&1 || true; ls -la ${REMOTE_BASE} 2>&1 | head -10"

echo
echo "==> Ensuring ${REMOTE_DIR} exists"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

echo
echo "==> Syncing ${LOCAL_SITE}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/"
rsync -avz --delete --progress \
  "${LOCAL_SITE}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo
echo "==> Updating legacy-path 301 redirect"
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

echo
echo "==> Done."
echo "    https://johnsondevco.com/${SUBPATH}/"
echo "    Legacy URL  https://johnsondevco.com/${LEGACY_SUBPATH}/  -> 301 -> /${SUBPATH}/"
