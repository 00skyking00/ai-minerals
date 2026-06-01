#!/usr/bin/env bash
# DEPRECATED — folded into scripts/deploy_to_hostinger.sh on 2026-05-31.
#
# The separate /ai-minerals-internal/ deploy was decommissioned; ai-minerals
# now deploys a single site at /ai-minerals/ (or /ai-minerals-beta/ for
# validation). Deep-review notebooks render under /ai-minerals/notebooks/.
#
# Use instead:
#   bash scripts/deploy_to_hostinger.sh                 # production
#   bash scripts/deploy_to_hostinger.sh ai-minerals-beta # beta
#
# The leak guards (rsync --exclude on data/raw, tools, research, src,
# scripts, design, *.qmd) moved to deploy_to_hostinger.sh.

echo "ERROR: scripts/deploy_internal.sh is deprecated as of 2026-05-31." >&2
echo "       Use: bash scripts/deploy_to_hostinger.sh [ai-minerals|ai-minerals-beta]" >&2
exit 2
