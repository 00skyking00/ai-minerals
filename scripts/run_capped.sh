#!/usr/bin/env bash
# Run a memory-heavy job inside a cgroup with a hard RAM cap and NO swap, so a
# runaway process (e.g. the paleochannel precompute over the 1.4-Gcell DEM) is
# SIGKILL'd the instant it crosses the cap instead of thrashing swap and
# OOM-killing the whole WSL VM out from under the VS Code server.
#
# Why this exists: on 2026-06-01 an un-capped python job grew to ~16 GB, drove
# the VM into a swap death-spiral, and systemd tore down init.scope (taking the
# vscode-server remote agent and every shell with it). A per-job cgroup cap
# turns that VM-wide crash into a clean single-process kill that names itself.
#
# Usage:
#   scripts/run_capped.sh [--mem 11G] [--swap 0] -- <command> [args...]
#
# Examples:
#   scripts/run_capped.sh -- .venv/bin/python scripts/northern_sierra_placer/precompute_paleochannel.py
#   scripts/run_capped.sh --mem 8G -- .venv/bin/python scripts/northern_sierra_placer/train_predict_250m.py
set -euo pipefail

MEM=11G
SWAP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem)  MEM="$2";  shift 2 ;;
    --swap) SWAP="$2"; shift 2 ;;
    --)     shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done
if [[ $# -eq 0 ]]; then
  echo "error: no command after --" >&2
  exit 64
fi

UNIT="capped-$$"
echo ">> capping '$*'" >&2
echo ">> MemoryMax=$MEM  MemorySwapMax=$SWAP  unit=$UNIT.scope" >&2
echo ">> exceeding the cap = clean SIGKILL of this job only; the VM survives." >&2

set +e
systemd-run --user --scope --quiet \
  --unit="$UNIT" \
  -p MemoryMax="$MEM" \
  -p MemorySwapMax="$SWAP" \
  -p MemoryAccounting=1 \
  -- "$@"
rc=$?
set -e

if [[ $rc -eq 137 ]]; then
  echo ">> EXIT 137 (SIGKILL). Almost certainly the cgroup memory cap fired." >&2
  echo ">> The job hit $MEM of RAM with swap=$SWAP and was killed in isolation." >&2
elif [[ $rc -ne 0 ]]; then
  echo ">> EXIT $rc (non-zero, not a memory kill). See the job output above." >&2
else
  echo ">> EXIT 0. Job completed within the $MEM cap." >&2
fi
exit $rc
