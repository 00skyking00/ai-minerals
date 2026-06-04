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

LOCAL_SITE="portfolio/_site"

if [ ! -d "${LOCAL_SITE}" ]; then
  echo "ERROR: ${LOCAL_SITE}/ not found. Run 'uv run quarto render portfolio' first."
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
# in the project tree that doesn't start with `.` or `_` into _site/. With
# the Quarto project at portfolio/ (since 2026-06-01) plus the data/ symlink
# Quarto follows that symlink and pulls in the WHOLE data/derived/ tree —
# multi-GB feature parquets, SHAP .npz blobs, intermediate .tif rasters,
# experiment-result JSON. None of that goes public. /data is excluded
# wholesale here; the public per-region chart PNGs are pushed via the
# explicit `data/derived/<chart subdirs>/*.png` rsync below.
#
# data/raw/ (FAMILY-PRIVATE drill-log scans) isn't currently followed by
# Quarto (no notebook references reach it through the symlink), but we
# keep belt-and-suspenders: the `/data` exclude here covers both data/raw
# AND any accidental data/derived inclusion in a future render.
echo
echo "==> Syncing ${LOCAL_SITE}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/ (private dirs + source excluded)"
rsync -avz --delete --delete-excluded --progress \
  --exclude='/data' \
  --exclude='/tools' \
  --exclude='/research' \
  --exclude='/src' \
  --exclude='/scripts' \
  --exclude='/design' \
  --exclude='*.qmd' \
  "${LOCAL_SITE}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

# Chapter qmds reference figures via absolute URLs of the form
# https://johnsondevco.com/ai-minerals/data/derived/<region>/<fig>.png.
# Quarto does not copy resources referenced by absolute URL into _site/, so
# those URLs 404 after the _site/ sync above. Push the PNGs that prose
# references separately, restricted to chart subdirectories and PNG files
# only — data/derived/ also holds multi-GB feature parquets, SHAP .npz
# blobs, and intermediate .tif rasters that are not for public release.
echo
echo "==> Syncing data/derived/{chart subdirs}/*.png -> ${REMOTE_HOST}:${REMOTE_DIR}/data/derived/"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}/data/derived"
rsync -avz --delete --delete-excluded --progress \
  --include='*/' \
  --include='*.png' \
  --exclude='*' \
  --prune-empty-dirs \
  data/derived/arizona \
  data/derived/bcgt \
  data/derived/eastak \
  data/derived/motherlode \
  data/derived/portfolio_charts \
  data/derived/northern_sierra_placer \
  data/derived/us_carbonatite_ree \
  "${REMOTE_HOST}:${REMOTE_DIR}/data/derived/"

# Make every served asset revalidate so a deploy is visible immediately
# instead of being masked by Hostinger's default ~7-day cache. The site is
# under active development; chapter prose AND figures (PNGs, charts,
# GeoTIFF thumbnails) change frequently and a 7-day cache means recruiters
# see stale results for a week. Once the site stabilizes for hiring, narrow
# this back to HTML-only and add long-cache for fonts / favicon.
#
# Pattern matches anything with a file extension: HTML, PNG, JPG, JS, CSS,
# JSON, CSV, SVG, TIF, ICO, WOFF, etc. Files without extension (rare here)
# get the server default.
#
# The rsync --delete above would wipe a server-side .htaccess, so (re)write
# it here, after the sync. Requires mod_headers (no-op if absent).
echo
echo "==> Writing no-cache .htaccess (covers HTML + assets) at ${REMOTE_DIR}/"
ssh "${REMOTE_HOST}" "cat > ${REMOTE_DIR}/.htaccess <<'EOF'
<IfModule mod_headers.c>
  <FilesMatch \"\\.[A-Za-z0-9]+\$\">
    Header set Cache-Control \"no-cache, must-revalidate, max-age=0\"
    Header unset Expires
    Header unset Pragma
  </FilesMatch>
</IfModule>
EOF
"

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

  # Legacy deep-notebook URLs handed to recruiters previously lived under
  # /ai-minerals-internal/ (commit 629f010) and /plans/ai-minerals-internal/
  # (commit ad2ffcc). Both are now decommissioned; the deep notebooks render
  # under /ai-minerals/notebooks/<region>/<page>.html. A site-wide .htaccess
  # at the public_html root catches either prefix and 301s to the new path,
  # preserving the rest of the URL. Order matters: the /plans/ variant must
  # match before the bare variant, since /ai-minerals-internal/ is a suffix
  # of /plans/ai-minerals-internal/.
  echo
  echo "==> Writing site-wide 301s for legacy /ai-minerals-internal/ URLs"
  ssh "${REMOTE_HOST}" "
    # Append our two RedirectMatch rules to public_html/.htaccess if missing
    # (idempotent — skip if already present from a prior deploy).
    grep -q 'plans/ai-minerals-internal' ${REMOTE_BASE}/.htaccess 2>/dev/null || {
      mkdir -p ${REMOTE_BASE}
      cat >> ${REMOTE_BASE}/.htaccess <<'EOF'

# 301 redirects for decommissioned internal-site URLs handed to recruiters.
# Source commits: ad2ffcc (/plans/ai-minerals-internal/), 629f010 (/ai-minerals-internal/).
# Deep notebooks now render under /ai-minerals/notebooks/<region>/<page>.html.
RedirectMatch 301 ^/plans/ai-minerals-internal/(.*)\$ /ai-minerals/\$1
RedirectMatch 301 ^/ai-minerals-internal/(.*)\$ /ai-minerals/\$1
EOF
    }
  "
fi

echo
echo "==> Done."
echo "    https://johnsondevco.com/${SUBPATH}/"
if [ "${SUBPATH}" = "ai-minerals" ]; then
  echo "    Legacy URL  https://johnsondevco.com/${LEGACY_SUBPATH}/  -> 301 -> /${SUBPATH}/"
fi
