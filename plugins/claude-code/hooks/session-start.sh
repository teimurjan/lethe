#!/usr/bin/env bash
# SessionStart — append a session heading to today's memory file and inject
# the last ~30 lines from the two most recent days as additionalContext.

set -eu
set -o pipefail

# shellcheck source=common.sh
source "${CLAUDE_PLUGIN_ROOT}/hooks/common.sh"

_read_stdin_with_timeout || true
_log "session-start: git_root=${LETHE_GIT_ROOT}"

TODAY="$(date +%Y-%m-%d)"
NOW="$(date +%H:%M)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"

# Ensure today's file exists with a header on first write of the day.
if [ ! -f "${TODAY_FILE}" ]; then
  printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"
fi

# Only add a session heading if today's file doesn't already end with the
# same heading — SessionStart can fire repeatedly (/clear, /reload-plugins,
# compaction) and we don't want back-to-back duplicate headers.
LAST_HEADING="$(grep -E '^## Session ' "${TODAY_FILE}" 2>/dev/null | tail -n 1 || true)"
if [ "${LAST_HEADING}" != "## Session ${NOW}" ]; then
  printf '\n## Session %s\n' "${NOW}" >>"${TODAY_FILE}"
fi

# Collect context from the 2 most recent daily files (including today).
CONTEXT=""
if ls "${LETHE_MEMORY_DIR}"/*.md >/dev/null 2>&1; then
  # shellcheck disable=SC2012 — ls -t is the simplest way to sort by mtime.
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    name="$(basename "$f")"
    tail_txt="$(tail -n 30 "$f" 2>/dev/null)"
    CONTEXT+="\n## ${name}\n${tail_txt}"
  done < <(ls -t "${LETHE_MEMORY_DIR}"/*.md 2>/dev/null | head -n 2)
fi

# Background-warm the uvx cache so the first `lethe search` in the session
# doesn't pay the full install cost. Harmless if already cached or missing.
if [ -n "${LETHE_CLI}" ]; then
  ( ${LETHE_CLI} --version >/dev/null 2>&1 ) &
  disown 2>/dev/null || true
fi

# Emit JSON response with additionalContext. Keep it compact so the UI stays
# readable; Claude Code joins additionalContext into the system prompt.
if [ -n "${CONTEXT}" ]; then
  CONTEXT_JSON="$(_json_encode_str "# Recent Memory${CONTEXT}")"
  printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":%s}}' \
    "${CONTEXT_JSON}"
fi

exit 0
