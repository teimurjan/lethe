#!/usr/bin/env bash
# UserPromptSubmit — append today's `## Session HH:MM` heading on the first
# user prompt of a Codex session, then surface a one-line hint pointing the
# agent at the recall skills.

set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

_read_stdin_with_timeout || true
_lethe_init_paths

SESSION_ID="$(_json_val session_id || true)"
TODAY="$(date +%Y-%m-%d)"
NOW="$(date +%H:%M)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"

if [ ! -f "${TODAY_FILE}" ]; then
  printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"
fi

# One header per session_id. Codex has no SessionEnd hook, so sentinels are
# not cleaned up — they're tiny empty files under `<project>/.lethe/` and a
# stale one only suppresses duplicate headers for an already-closed session,
# which is harmless.
if [ -n "${SESSION_ID}" ]; then
  SESSION_KEY="$(_sanitize_session_id "${SESSION_ID}")"
  SENTINEL="${LETHE_DIR}/.session-${SESSION_KEY}.header"
  if [ ! -f "${SENTINEL}" ]; then
    printf '\n## Session %s\n' "${NOW}" >>"${TODAY_FILE}"
    : >"${SENTINEL}"
  fi
else
  LAST_HEADING="$(grep -E '^## Session ' "${TODAY_FILE}" 2>/dev/null | tail -n 1 || true)"
  if [ "${LAST_HEADING}" != "## Session ${NOW}" ]; then
    printf '\n## Session %s\n' "${NOW}" >>"${TODAY_FILE}"
  fi
fi

if [ -z "${LETHE_CLI}" ]; then
  exit 0
fi
if ! ls "${LETHE_MEMORY_DIR}"/*.md >/dev/null 2>&1; then
  exit 0
fi

printf '{"systemMessage":"[lethe] Memory available — use the recall skill for this project, or recall-global for all registered projects."}'
exit 0
