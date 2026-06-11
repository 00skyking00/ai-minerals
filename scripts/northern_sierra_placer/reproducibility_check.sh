#!/usr/bin/env bash
# Phase M: reproducibility audit for the northern-Sierra placer model.
#
# Verifies that every artifact named in the train+test+validate plan
# (~/.claude/plans/hazy-humming-lynx.md, Phase M) exists under
# data/derived/northern_sierra_placer/, that the leakage unit tests still
# pass, and that the chapter + internal notebooks still render.
#
# Exits 0 if everything is in place; exits 1 with a list of missing pieces
# otherwise. Intended to be the last gate before pushing the calibrated
# raster to gldbg.
#
# Usage:
#   bash scripts/northern_sierra_placer/reproducibility_check.sh

set -u  # but NOT -e — we want to collect all failures, not stop at the first

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DERIVED="${ROOT}/data/derived/northern_sierra_placer"
CAL_DERIVED="${ROOT}/data/derived/calaveras_placer"
MISSING=()
PASS=0
FAIL=0

check_file() {
  if [[ -f "$1" ]]; then
    PASS=$((PASS + 1))
    echo "OK    $(basename "$1")"
  else
    FAIL=$((FAIL + 1))
    MISSING+=("$1")
    echo "MISS  $1"
  fi
}

check_glob() {
  # check_glob "<description>" <pattern>
  local desc="$1"; shift
  local matches=("$@")
  if [[ -e "${matches[0]}" ]]; then
    PASS=$((PASS + 1))
    echo "OK    ${desc} (${#matches[@]} match)"
  else
    FAIL=$((FAIL + 1))
    MISSING+=("$desc")
    echo "MISS  ${desc}"
  fi
}

run_cmd() {
  # run_cmd "<description>" <cmd...>
  local desc="$1"; shift
  echo
  echo "==> ${desc}"
  if "$@" > /tmp/repro_step_$$.log 2>&1; then
    PASS=$((PASS + 1))
    echo "OK    ${desc}"
  else
    FAIL=$((FAIL + 1))
    MISSING+=("${desc} (exit non-zero; see /tmp/repro_step_$$.log)")
    echo "FAIL  ${desc}"
    tail -5 /tmp/repro_step_$$.log | sed 's/^/      /'
  fi
}

cd "${ROOT}"

echo "==> Phase 1 + scorer artifacts"
check_file "${DERIVED}/phase1_index_250m.parquet"
check_file "${DERIVED}/phase1_index_250m_3310.tif"
check_file "${DERIVED}/phase1_index_250m_4326.tif"
check_file "${DERIVED}/phase1_anchor_decile_report.csv"

echo
echo "==> Phase 2 per-population artifacts"
for pop in placer_tertiary placer_quaternary; do
  check_file "${DERIVED}/pop_predictions_${pop}_250m.parquet"
  check_file "${DERIVED}/pop_calibrated_${pop}_250m.parquet"
  check_file "${DERIVED}/pop_fold_metrics_${pop}.csv"
  check_file "${DERIVED}/prospectivity_placer_${pop}_250m_calibrated_3310.tif"
  check_file "${DERIVED}/prospectivity_placer_${pop}_250m_calibrated_4326.tif"
done

echo
echo "==> Phase I outputs (SHAP rationale + feature importance + calibration audit)"
for pop in placer_tertiary placer_quaternary; do
  check_file "${DERIVED}/rationale_${pop}_250m.parquet"
  check_file "${DERIVED}/feature_importance_${pop}.csv"
  check_file "${DERIVED}/calibration_audit_${pop}.csv"
  check_file "${DERIVED}/calibration_reliability_${pop}.png"
done

echo
echo "==> Phase G + I.4 outputs (the deliverable + validation + headline metrics)"
check_file "${DERIVED}/prospectivity_placer_northern_sierra_250m_calibrated_3310.tif"
check_file "${DERIVED}/prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
check_file "${DERIVED}/success_rate_curve_phase1_vs_phase2.png"
check_file "${DERIVED}/anchor_districts_decile_table.csv"
check_file "${DERIVED}/phase1_vs_phase2_comparison.csv"
check_file "${DERIVED}/headline_metrics.csv"
check_file "${DERIVED}/fusion_meta.json"

echo
echo "==> Phase J outputs (Lindgren/USMIN secondary + N/S split + Calaveras transfer)"
check_file "${DERIVED}/lindgren_secondary_blind_set_results.csv"
check_file "${DERIVED}/lindgren_secondary_summary.csv"
check_file "${DERIVED}/north_south_split_results.csv"
check_file "${CAL_DERIVED}/transfer_metrics.csv"

echo
echo "==> Final model card"
check_file "${DERIVED}/model_card_northern_sierra_placer.md"

echo
echo "==> Static checks (tests, lint, notebook render)"
run_cmd "pytest leakage tests"   .venv/bin/python -m pytest tests/test_distance_downstream_leakage.py -q
run_cmd "quarto render chapter"  quarto render notebooks/northern_sierra_placer/northern_sierra_placer_prospectivity.qmd --no-execute
run_cmd "quarto render internal" quarto render notebooks/northern_sierra_placer/internal.qmd --no-execute

echo
echo "==> Final model card content check (no TBD remaining)"
if [[ -f "${DERIVED}/model_card_northern_sierra_placer.md" ]]; then
  TBD_COUNT=$(grep -c -i '\bTBD\b' "${DERIVED}/model_card_northern_sierra_placer.md" || true)
  if [[ "${TBD_COUNT}" -eq 0 ]]; then
    PASS=$((PASS + 1))
    echo "OK    no TBD in model card"
  else
    FAIL=$((FAIL + 1))
    MISSING+=("model card still has ${TBD_COUNT} TBD entries")
    echo "FAIL  model card still has ${TBD_COUNT} TBD entries (run Phase L + N to fill them)"
  fi
else
  echo "SKIP  model card content check (file missing)"
fi

echo
echo "==> Sandboxed deliverable check (gdalinfo)"
DELIV="${DERIVED}/prospectivity_placer_northern_sierra_250m_calibrated_4326.tif"
if [[ -f "${DELIV}" ]]; then
  CRS=$(gdalinfo "${DELIV}" 2>/dev/null | grep -E '^Coordinate System|AUTHORITY\["EPSG"' | head -3)
  if echo "${CRS}" | grep -q '4326'; then
    PASS=$((PASS + 1))
    echo "OK    deliverable raster is EPSG:4326"
  else
    FAIL=$((FAIL + 1))
    MISSING+=("deliverable raster is not EPSG:4326")
    echo "FAIL  deliverable CRS is not EPSG:4326"
  fi
fi

echo
echo "===================================================="
echo "  Reproducibility audit summary"
echo "===================================================="
echo "  Pass: ${PASS}"
echo "  Fail: ${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  echo
  echo "  Missing or failing:"
  for m in "${MISSING[@]}"; do
    echo "    - ${m}"
  done
  echo
  echo "  Reproduction commands (from the plan):"
  echo "    python scripts/northern_sierra_placer/fetch_all.py"
  echo "    python scripts/northern_sierra_placer/precompute_paleochannel.py"
  echo "    python scripts/northern_sierra_placer/assemble_250m.py"
  echo "    python scripts/northern_sierra_placer/phase1_index.py --check-anchors-top-decile"
  echo "    python scripts/northern_sierra_placer/train_predict_250m.py"
  echo "    python scripts/northern_sierra_placer/calibrate_and_fuse.py"
  echo "    python scripts/northern_sierra_placer/calibration_audit.py"
  echo "    python scripts/northern_sierra_placer/lindgren_blind_set.py"
  echo "    python scripts/northern_sierra_placer/north_south_split.py"
  echo "    python scripts/northern_sierra_placer/calaveras_transfer.py"
  echo "    python scripts/northern_sierra_placer/validation.py"
  exit 1
fi

echo
echo "  All gates passed. Ready to push deliverable to gldbg."
exit 0
