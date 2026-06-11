#!/usr/bin/env bash
# Phase F.2 full audit suite for the northern Sierra placer v3 retrain.
#
# Runs the six audit scripts in the order specified by the v3 plan section
# "F.2 Full audit suite". Each script gets a section header echoed before it
# runs so the combined log is greppable.
#
# The two expensive scripts (rationale_250m, validation) run under
# scripts/run_capped.sh with a 12 GB / no-swap cgroup cap so a runaway job
# gets SIGKILL'd in isolation instead of taking down the whole WSL VM (see
# scripts/run_capped.sh header for the 2026-06-01 incident this guards).
#
# This is a post-F.1 harness. Run only after train_predict_250m and
# calibrate_and_fuse have completed and their outputs are on disk.
#
# Usage:
#   scripts/northern_sierra_placer/audit_suite.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=.venv/bin/python
CAPPED="scripts/run_capped.sh --mem 12G --swap 0"

section() {
  echo
  echo "================================================================"
  echo "== $1"
  echo "================================================================"
}

section "F.2.1 calibration_audit.py"
$PY scripts/northern_sierra_placer/calibration_audit.py

section "F.2.2 lindgren_blind_set.py"
$PY scripts/northern_sierra_placer/lindgren_blind_set.py

section "F.2.3 north_south_split.py"
$PY scripts/northern_sierra_placer/north_south_split.py

section "F.2.4 rationale_250m.py (capped 12G / swap 0)"
$CAPPED -- $PY scripts/northern_sierra_placer/rationale_250m.py

section "F.2.5 leakage_risk_audit.py"
$PY scripts/northern_sierra_placer/leakage_risk_audit.py

section "F.2.6 validation.py (capped 12G / swap 0)"
$CAPPED -- $PY scripts/northern_sierra_placer/validation.py

echo
echo "================================================================"
echo "== F.2 audit suite complete"
echo "================================================================"
