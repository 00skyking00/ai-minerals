#!/usr/bin/env bash
# Live busy/idle flag for the program dashboard. Called from Claude Code hooks:
#   SessionStart / UserPromptSubmit  -> busy   (a session is actively working)
#   PostToolUse                      -> beat   (refresh heartbeat, debounced ~15s)
#   Stop / SessionEnd                -> idle
# Writes .project/activity.json next to the repo's status.json. The dashboard
# treats `busy` with a stale heartbeat (> ~3 min) as idle, so a crashed session
# does not show "working" forever. This file is hook-written; do not hand-edit.
#
#   activity_flag.sh busy|idle|beat [module-name]
set -uo pipefail
state="${1:-beat}"
repo_root=$(cd "$(dirname "$0")/.." && pwd)
af="$repo_root/.project/activity.json"
# Module name: explicit arg wins; else the module key from status.json (handles
# repos whose dir != module key, e.g. gldbg -> goldbug); else the dir basename.
if [ -n "${2:-}" ]; then
  mod="$2"
elif [ -f "$repo_root/.project/status.json" ] &&
     mod=$(sed -n 's/.*"module"[: ]*"\([^"]*\)".*/\1/p' "$repo_root/.project/status.json" | head -1) &&
     [ -n "$mod" ]; then
  :
else
  mod=$(basename "$repo_root")
fi
now=$(date -Is)
mkdir -p "$repo_root/.project"

prev_state=""; prev_since=""; prev_beat=""
if [ -f "$af" ]; then
  prev_state=$(sed -n 's/.*"state"[: ]*"\([^"]*\)".*/\1/p' "$af")
  prev_since=$(sed -n 's/.*"since"[: ]*"\([^"]*\)".*/\1/p' "$af")
  prev_beat=$(sed -n 's/.*"heartbeat"[: ]*"\([^"]*\)".*/\1/p' "$af")
fi

case "$state" in
  beat)
    # A tool just ran, so the session is active -> busy. Debounce to ~15s.
    if [ "$prev_state" = "busy" ] && [ -n "$prev_beat" ]; then
      last=$(date -d "$prev_beat" +%s 2>/dev/null || echo 0)
      [ $(( $(date +%s) - last )) -lt 15 ] && exit 0
    fi
    if [ "$prev_state" = "busy" ] && [ -n "$prev_since" ]; then since="$prev_since"; else since="$now"; fi
    state="busy" ;;
  busy)
    if [ "$prev_state" = "busy" ] && [ -n "$prev_since" ]; then since="$prev_since"; else since="$now"; fi ;;
  idle)
    since="$now" ;;
  *) echo "usage: activity_flag.sh busy|idle|beat [module]" >&2; exit 2 ;;
esac

printf '{"module":"%s","state":"%s","since":"%s","heartbeat":"%s"}\n' \
  "$mod" "$state" "$since" "$now" >"$af"
