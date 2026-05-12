#!/usr/bin/env bash
# Render the internal review build with all notebooks to _site_internal/.
# Swaps _quarto.yml for _quarto-internal.yml during render, then restores.
#
# Usage: bash scripts/render_internal.sh

set -euo pipefail

if [ ! -f "_quarto-internal.yml" ]; then
  echo "ERROR: _quarto-internal.yml not found"
  exit 1
fi

cp _quarto.yml _quarto.yml.portfolio_bak
cp _quarto-internal.yml _quarto.yml

cleanup() {
  cp _quarto.yml.portfolio_bak _quarto.yml
  rm -f _quarto.yml.portfolio_bak
}
trap cleanup EXIT

QUARTO_PYTHON="$(pwd)/.venv/bin/python3" quarto render

# Block crawlers on the internal review build.
cat > _site_internal/robots.txt <<'EOF'
User-agent: *
Disallow: /
EOF

echo
echo "==> Internal site rendered to _site_internal/"
echo "    Pages: $(find _site_internal -maxdepth 4 -name '*.html' | wc -l)"
