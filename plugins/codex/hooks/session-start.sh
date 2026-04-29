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

TODAY="$(date +%Y-%m-%d)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"

if [ ! -f "${TODAY_FILE}" ]; then
  printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"
fi

CONTEXT=""
if ls "${LETHE_MEMORY_DIR}"/*.md >/dev/null 2>&1; then
  # shellcheck disable=SC2012
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    name="$(basename "$f")"
    tail_txt="$(tail -n 30 "$f" 2>/dev/null)"
    # Use real newlines (not "\n" literals — those would be JSON-escaped to
    # `\\n` by _json_encode_str and the agent would see a backslash-n token
    # instead of a line break).
    printf -v CONTEXT '%s\n## %s\n%s' "${CONTEXT}" "${name}" "${tail_txt}"
  done < <(ls -t "${LETHE_MEMORY_DIR}"/*.md 2>/dev/null | head -n 2)
fi

if [ -n "${LETHE_CLI}" ]; then
  ( ${LETHE_CLI} --version >/dev/null 2>&1 ) &
  disown 2>/dev/null || true
fi

if [ -n "${CONTEXT}" ]; then
  MSG_JSON="$(_json_encode_str "# Recent Memory${CONTEXT}")"
  printf '{"systemMessage":%s}' "${MSG_JSON}"
fi

exit 0
