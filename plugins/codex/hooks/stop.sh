#!/usr/bin/env bash
# Stop — append a turn marker with a progressive-disclosure anchor pointing at
# Codex's transcript file, then reindex `.lethe/memory` so the next retrieval
# sees the new entry.
#
# v1 limitation: this hook does NOT summarize the turn (Codex's transcript
# format is not yet documented). The transcript path is captured in the anchor
# so a future `lethe-codex transcript` parser can hydrate it on demand.

set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

_read_stdin_with_timeout || true
_lethe_init_paths

TRANSCRIPT="$(_json_val transcript_path || true)"
SESSION_ID="$(_json_val session_id || true)"
TURN_ID="$(_json_val turn_id || true)"

TODAY="$(date +%Y-%m-%d)"
NOW="$(date +%H:%M)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"
mkdir -p "${LETHE_MEMORY_DIR}"
[ -f "${TODAY_FILE}" ] || printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"

# Dedupe: if the most recent turn anchor already references this turn_id, skip.
# `grep -F` so a turn_id containing regex metacharacters can't misfire.
LAST_ANCHOR="$(grep -E '^<!-- session:' "${TODAY_FILE}" 2>/dev/null | tail -n 1 || true)"
if [ -n "${TURN_ID}" ] && printf '%s' "${LAST_ANCHOR}" | grep -F -q -- "turn:${TURN_ID}"; then
  _log "stop: duplicate turn ${TURN_ID}, skipping"
else
  {
    printf '\n### %s\n' "${NOW}"
    printf '<!-- session:%s turn:%s transcript:%s -->\n' \
      "${SESSION_ID}" "${TURN_ID}" "${TRANSCRIPT}"
  } >>"${TODAY_FILE}"
fi

# No header-sentinel cleanup: Codex has no SessionEnd hook, and clearing the
# sentinel on every Stop would let the next UserPromptSubmit add a fresh
# `## Session HH:MM` heading mid-session. The sentinel is a tiny empty file
# under `<project>/.lethe/`; stale ones only suppress duplicate headers for
# already-closed sessions, which is the correct behaviour.

if [ -n "${LETHE_CLI}" ]; then
  ( cd "${LETHE_GIT_ROOT}" && ${LETHE_CLI} index >/dev/null 2>&1 ) || true
fi

exit 0
