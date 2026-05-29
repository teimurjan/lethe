#!/usr/bin/env bash
# SessionStart — inject the last ~30 lines from the two most recent daily
# memory files as `systemMessage` so Codex sees recent context on startup.

set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

_read_stdin_with_timeout || true
_lethe_init_paths
_log "session-start: git_root=${LETHE_GIT_ROOT}"

# Codex has no SessionEnd hook, so the per-session `.session-*.header`
# sentinels written by user-prompt-submit.sh would accumulate forever.
# Sweep any older than 24h here — well past any reasonable session lifetime,
# so concurrent in-flight sessions are unaffected.
if [ -d "${LETHE_DIR}" ]; then
  find "${LETHE_DIR}" -maxdepth 1 -type f -name '.session-*.header' -mmin +1440 -delete 2>/dev/null || true
fi

TODAY="$(date +%Y-%m-%d)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"

if [ ! -f "${TODAY_FILE}" ]; then
  printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"
fi

CONTEXT=""
if ls "${LETHE_MEMORY_DIR}"/*.md >/dev/null 2>&1; then
  # shellcheck disable=SC2012
  # Strip per-turn anchors and session/turn headings — they're noise to the
  # model and the anchors are still in the markdown for skills that need to
  # drill via `lethe-codex transcript`.
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    day="$(basename "$f" .md)"
    filtered="$(tail -n 30 "$f" 2>/dev/null \
      | grep -vE '^<!-- session:|^### [0-9]{2}:[0-9]{2}$|^## Session [0-9]{2}:[0-9]{2}$|^# [0-9]{4}-[0-9]{2}-[0-9]{2}$' \
      | awk 'NF || prev; {prev=NF}')"
    [ -z "${filtered}" ] && continue
    # Use real newlines (not "\n" literals — those would be JSON-escaped to
    # `\\n` by _json_encode_str and the agent would see a backslash-n token
    # instead of a line break).
    printf -v CONTEXT '%s\n\n## %s\n%s' "${CONTEXT}" "${day}" "${filtered}"
  done < <(ls -t "${LETHE_MEMORY_DIR}"/*.md 2>/dev/null | head -n 2)
fi

if [ -n "${LETHE_CLI}" ]; then
  ( ${LETHE_CLI} --version >/dev/null 2>&1 ) &
  disown 2>/dev/null || true
fi

if [ -n "${CONTEXT}" ]; then
  CTX_JSON="$(_json_encode_str "# Recent memory${CONTEXT}")"
  # `hookSpecificOutput.additionalContext` injects into the model's context
  # window. `systemMessage` only renders as a UI banner — using it for the
  # actual memory payload would make the recall invisible to Codex.
  printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":%s}}' "${CTX_JSON}"
else
  printf '{}'
fi

exit 0
