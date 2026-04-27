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
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"

# Ensure today's file exists with a header on first write of the day.
# Session headings are written lazily by user-prompt-submit.sh so empty
# sessions (user starts Claude Code but never types) don't leave dangling
# `## Session HH:MM` markers in the markdown log.
if [ ! -f "${TODAY_FILE}" ]; then
  printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"
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

# Background-warm the lethe binary so the first `lethe search` in the
# session doesn't pay the full ONNX-runtime/DuckDB load cost. Harmless
# if `lethe` isn't installed (LETHE_CLI is empty in that case).
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
