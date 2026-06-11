#!/usr/bin/env bash
# H2.5/H2.6/H2.7 post-training orchestrator for v3.7.
#
# Runs in this order:
#   1. Sanity-check training outputs exist (Quaternary calibrated parquet + TIFs)
#   2. Swap features parquet from v3.6 canonical to v3.7 build
#   3. Fuse Tertiary (from v3.6 backup) + Quaternary (v3.7) -> single-band raster
#   4. Refresh sidecars (coverage-mask, bands JSON, 2-band) for v3.7
#   5. Run H2.5 southern anchor held-out + MRDS per-county gate
#   6. Run F.2 audit suite (calibration, Lindgren blind, north-south split,
#      rationale, leakage-risk, validation)
#   7. Print a summary
#
# Sky reviews the summary, decides PASS/MARGINAL/FAIL per the H2.5 gate,
# then writes the v3.7 chapter section + tags v1.1.0.
#
# Usage:
#   bash scripts/northern_sierra_placer/v37_post_training_runbook.sh        # default: full pipeline
#   bash scripts/northern_sierra_placer/v37_post_training_runbook.sh --skip-audit-suite  # skip the
#                                                                  # 4-6 hour
#                                                                  # F.2 suite
#
# NOT idempotent on the features-parquet swap — running it twice without
# restoring the symlink in between will result in the v3.7 features being
# treated as the v3.6 features in the backup tag.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=.venv/bin/python
DERIVED=data/derived
PLACER=$DERIVED/northern_sierra_placer
RUN_AUDIT_SUITE=true

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-audit-suite) RUN_AUDIT_SUITE=false ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

section() {
  echo
  echo "================================================================"
  echo "== $1"
  echo "================================================================"
}

# --- 1. Sanity check ---
section "1. Sanity check: training outputs"
need=(
  "$PLACER/pop_calibrated_placer_quaternary_250m.parquet"
  "$PLACER/prospectivity_placer_placer_quaternary_250m_calibrated_4326.tif"
  "$PLACER/pop_calibrated_placer_tertiary_250m.parquet"
  "$PLACER/prospectivity_placer_placer_tertiary_250m_calibrated_4326.tif"
)
missing=0
for f in "${need[@]}"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    missing=1
  else
    echo "ok      $f ($(stat -c %s "$f") bytes, mtime $(stat -c %y "$f" | cut -c1-19))"
  fi
done
if [ $missing -ne 0 ]; then
  echo
  echo "ERROR: required v3.7 outputs are missing. Wait for training to complete."
  exit 3
fi

# --- 2. Features-parquet swap ---
section "2. Swap features parquet to v3.7 build"
CANON=$DERIVED/features_northern_sierra_placer_250m.parquet
V37=$DERIVED/features_northern_sierra_placer_250m_v37.parquet
V36_BACKUP=$DERIVED/features_northern_sierra_placer_250m.v36.parquet

if [ ! -f "$V37" ]; then
  echo "ERROR: v3.7 features parquet not found at $V37"
  exit 4
fi

if [ ! -L "$CANON" ] && [ -f "$CANON" ] && [ ! -f "$V36_BACKUP" ]; then
  echo "Backing up v3.6 canonical features to $V36_BACKUP"
  mv "$CANON" "$V36_BACKUP"
elif [ -L "$CANON" ]; then
  echo "Canonical is a symlink already; removing to repoint"
  rm "$CANON"
fi

echo "Pointing canonical -> v3.7"
ln -sf "$(basename "$V37")" "$CANON"
ls -la "$CANON"

# --- 3. Fuse ---
section "3. Fuse Tertiary + Quaternary -> single-band raster"
$PY scripts/northern_sierra_placer/calibrate_and_fuse.py --no-logistic-fusion

# --- 4. Refresh sidecars (coverage_mask + bands_json + 2-band in one CLI) ---
section "4a-c. Coverage-mask + bands JSON + 2-band sidecars"
$PY scripts/build_placer_sidecars.py all

section "4d. Refresh data/ml/ single-band fused (goldbug stable path)"
cp -p "$PLACER/prospectivity_placer_northern_sierra_250m_calibrated_4326.tif" data/ml/

# --- 5. H2.5 held-out validation ---
section "5. Southern anchor + MRDS per-county gate"
$PY scripts/northern_sierra_placer/v37_southern_anchor_held_out.py

# --- 6. F.2 audit suite ---
if [ "$RUN_AUDIT_SUITE" = "true" ]; then
  section "6. F.2 audit suite (4-6 hours; will run unattended)"
  bash scripts/northern_sierra_placer/audit_suite.sh
else
  echo
  echo "(skipping F.2 audit suite per --skip-audit-suite)"
fi

# --- 7. Summary ---
section "7. v3.7.0 summary"
echo "Held-out report: $PLACER/v37_held_out_report.md"
echo "Anchor parquet:  $PLACER/v37_southern_anchor_held_out.parquet"
echo "County parquet:  $PLACER/v37_mrds_per_county_gate.parquet"
echo
echo "Bands recommended for goldbug (re-read AFTER this run):"
$PY -c "import json; b=json.load(open('data/ml/prospectivity_placer_northern_sierra_250m_calibrated_4326.bands.json')); print('  high=%.4f moderate=%.4f weak=%.4f'%(b['recommended_bands']['high'], b['recommended_bands']['moderate'], b['recommended_bands']['weak']))"
echo
echo "Next steps (manual):"
echo "  - Sky reviews v37_held_out_report.md"
echo "  - If county gates pass: write v3.7.0 chapter section in portfolio/placer.qmd, bump ml_versions.json"
echo "  - If county gates fail: decide v3.7.0.1 augmentation patch vs negative-result chapter prose"
echo "  - Quarto render, beta deploy, walk site, prod deploy, tag ai-minerals-v1.1.0"
echo
