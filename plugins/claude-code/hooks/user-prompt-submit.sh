#!/usr/bin/env bash
# UserPromptSubmit — write today's `## Session HH:MM` heading on the first
# user prompt of a session (one header per session_id, no empty sessions).
# The recall skills decide for themselves when memory is relevant (via their
# descriptions); we no longer inject a per-prompt hint.

set -eu
set -o pipefail

# shellcheck source=common.sh
source "${CLAUDE_PLUGIN_ROOT}/hooks/common.sh"

_read_stdin_with_timeout || true

SESSION_ID="$(_json_val session_id || true)"
TODAY="$(date +%Y-%m-%d)"
NOW="$(date +%H:%M)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"

# Ensure today's file exists (first prompt of the day may land here before
# session-start ran, or session-start may have been skipped).
if [ ! -f "${TODAY_FILE}" ]; then
  printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"
fi

# One header per session_id. Sentinel lives under .lethe and is removed by
# session-end.sh; if session-end never fires the sentinel is harmless and
# simply suppresses duplicate headers for an already-closed session. The
# session_id is hashed before it's interpolated into the path (common.sh)
# so a hostile/malformed id can't escape LETHE_DIR.
if [ -n "${SESSION_ID}" ]; then
  SESSION_KEY="$(_sanitize_session_id "${SESSION_ID}")"
  SENTINEL="${LETHE_DIR}/.session-${SESSION_KEY}.header"
  if [ ! -f "${SENTINEL}" ]; then
    printf '\n## Session %s\n' "${NOW}" >>"${TODAY_FILE}"
    : >"${SENTINEL}"
  fi
else
  # No session_id available — fall back to minute-level dedupe so rapid
  # re-fires within the same minute don't duplicate the heading.
  LAST_HEADING="$(grep -E '^## Session ' "${TODAY_FILE}" 2>/dev/null | tail -n 1 || true)"
  if [ "${LAST_HEADING}" != "## Session ${NOW}" ]; then
    printf '\n## Session %s\n' "${NOW}" >>"${TODAY_FILE}"
  fi
fi

exit 0
