#!/usr/bin/env bash
# SessionEnd — trigger a final reindex so suppression state and any last
# markdown edits land in SQLite before the session is torn down.

set -eu
set -o pipefail

# shellcheck source=common.sh
source "${CLAUDE_PLUGIN_ROOT}/hooks/common.sh"

_read_stdin_with_timeout || true

# Remove this session's header sentinel (written by user-prompt-submit.sh).
# Best-effort — stale sentinels only suppress duplicate headers for a
# session that's already gone, which is harmless. Uses the same id→key
# transformation as user-prompt-submit.sh (_sanitize_session_id in common.sh)
# so the file we rm here matches the one that was created.
SESSION_ID="$(_json_val session_id || true)"
if [ -n "${SESSION_ID}" ]; then
  SESSION_KEY="$(_sanitize_session_id "${SESSION_ID}")"
  rm -f "${LETHE_DIR}/.session-${SESSION_KEY}.header" 2>/dev/null || true
fi

if [ -n "${LETHE_CLI}" ]; then
  ( cd "${LETHE_GIT_ROOT}" && ${LETHE_CLI} index >/dev/null 2>&1 ) || true
fi

exit 0
