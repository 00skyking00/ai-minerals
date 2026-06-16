#!/usr/bin/env bash
# Print handoff inbox + outbox state. Exits silently when both are clean.
# Invoked from .claude/settings.local.json hooks (SessionStart,
# UserPromptSubmit, Stop) so Claude sees pending handoffs prepended to
# every conversation step.
#
# Detects "pending" by listing top-level .md files under handoff/inbox/
# and handoff/outbox/. Files under done/ are excluded (already archived).

set -e

repo_root=$(cd "$(dirname "$0")/.." && pwd)
inbox="$repo_root/handoff/inbox"
outbox="$repo_root/handoff/outbox"

[ -d "$inbox" ] || exit 0

# Surface both top-level .md notes and top-level delivery directories
# (data bundles like a re-tile package). Excludes done/ explicitly.
list_entries() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  {
    find "$dir" -maxdepth 1 -mindepth 1 -name "*.md" -type f 2>/dev/null
    find "$dir" -maxdepth 1 -mindepth 1 -type d ! -name "done" 2>/dev/null
  } | sort
}

inbox_entries=$(list_entries "$inbox")
outbox_entries=$(list_entries "$outbox")

[ -z "$inbox_entries" ] && [ -z "$outbox_entries" ] && exit 0

label_entry() {
  local entry="$1"
  if [ -d "$entry" ]; then
    echo "- $(basename "$entry")/ (directory)"
  else
    echo "- $(basename "$entry")"
  fi
}

echo "## Handoff state"
if [ -n "$inbox_entries" ]; then
  n=$(echo "$inbox_entries" | wc -l)
  echo ""
  echo "**Inbox** ($n unprocessed):"
  while IFS= read -r f; do
    label_entry "$f"
  done <<< "$inbox_entries"
fi
if [ -n "$outbox_entries" ]; then
  n=$(echo "$outbox_entries" | wc -l)
  echo ""
  echo "**Outbox** ($n not archived; verify delivery and move to handoff/outbox/done/):"
  while IFS= read -r f; do
    label_entry "$f"
  done <<< "$outbox_entries"
fi
echo ""
