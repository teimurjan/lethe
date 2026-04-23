#!/usr/bin/env bash
# Shared helpers for lethe hooks. Sourced, not executed.
#
# Exports:
#   LETHE_GIT_ROOT    — git root of the invoking project (or cwd if none)
#   LETHE_DIR         — $LETHE_GIT_ROOT/.lethe
#   LETHE_MEMORY_DIR  — $LETHE_DIR/memory
#   LETHE_INDEX_DIR   — $LETHE_DIR/index
#   LETHE_STDIN_JSON  — raw JSON payload read from stdin (may be empty)
#   LETHE_CLI         — full command for invoking the lethe CLI (either `lethe`
#                       on PATH or `uvx --from git+... lethe`)
#
# Helpers:
#   _json_val <key>           — extract a string/scalar from LETHE_STDIN_JSON
#   _json_encode_str <string> — JSON-encode a string (for response bodies)
#   _log <msg>                — append timestamped debug line to LETHE_LOG
#   _read_stdin_with_timeout  — populate LETHE_STDIN_JSON with a 2s budget

set -o pipefail

LETHE_INSTALL_SOURCE="${LETHE_INSTALL_SOURCE:-git+https://github.com/teimurjan/lethe}"

# --- Paths -------------------------------------------------------------------

LETHE_GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
LETHE_DIR="${LETHE_GIT_ROOT}/.lethe"
LETHE_MEMORY_DIR="${LETHE_DIR}/memory"
LETHE_INDEX_DIR="${LETHE_DIR}/index"
LETHE_LOG="${LETHE_DIR}/hooks.log"

mkdir -p "${LETHE_MEMORY_DIR}" "${LETHE_INDEX_DIR}" 2>/dev/null || true

_log() {
  [ -z "${LETHE_DEBUG:-}" ] && return 0
  printf '%s %s\n' "$(date +'%Y-%m-%dT%H:%M:%S')" "$*" >>"${LETHE_LOG}" 2>/dev/null || true
}

# --- CLI detection -----------------------------------------------------------

if command -v lethe >/dev/null 2>&1; then
  LETHE_CLI="lethe"
elif command -v uvx >/dev/null 2>&1; then
  LETHE_CLI="uvx --from ${LETHE_INSTALL_SOURCE} lethe"
elif [ -x "$HOME/.local/bin/uvx" ]; then
  LETHE_CLI="$HOME/.local/bin/uvx --from ${LETHE_INSTALL_SOURCE} lethe"
else
  LETHE_CLI=""
fi

export LETHE_GIT_ROOT LETHE_DIR LETHE_MEMORY_DIR LETHE_INDEX_DIR LETHE_CLI

# --- stdin read with timeout -------------------------------------------------
#
# Claude Code hooks receive a JSON payload on stdin. macOS's `read -t` on
# plain bash (no /dev/stdin indicator) can block, so we use perl to read up
# to 64KB within 2 seconds. If nothing is available, LETHE_STDIN_JSON stays
# empty and callers fall back to environment variables.

LETHE_STDIN_JSON=""

_read_stdin_with_timeout() {
  if [ -t 0 ]; then
    LETHE_STDIN_JSON=""
    return 0
  fi
  LETHE_STDIN_JSON="$(perl -e '
    use IO::Select;
    my $s = IO::Select->new; $s->add(\*STDIN);
    my $buf = "";
    while ($s->can_read(2)) {
      my $n = sysread(STDIN, my $chunk, 65536);
      last unless defined $n && $n > 0;
      $buf .= $chunk;
      last if length($buf) >= 1048576;
    }
    print $buf;
  ' 2>/dev/null)"
  export LETHE_STDIN_JSON
}

# --- JSON helpers ------------------------------------------------------------

_json_val() {
  local key="$1"
  [ -z "${LETHE_STDIN_JSON}" ] && return 0
  if command -v python3 >/dev/null 2>&1; then
    LETHE_STDIN_JSON="${LETHE_STDIN_JSON}" python3 - "$key" <<'PY' 2>/dev/null
import json, os, sys
key = sys.argv[1]
try:
    data = json.loads(os.environ.get("LETHE_STDIN_JSON", ""))
except json.JSONDecodeError:
    sys.exit(0)
cur = data
for part in key.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        sys.exit(0)
if isinstance(cur, (dict, list)):
    sys.stdout.write(json.dumps(cur))
else:
    sys.stdout.write("" if cur is None else str(cur))
PY
  fi
}

_sanitize_session_id() {
  # Hash the session_id before it becomes part of a filesystem path, so a
  # hostile or malformed id (containing /, .., spaces, newlines, etc.) can't
  # escape LETHE_DIR or create weird filenames. Deterministic: the same id
  # always maps to the same key, which lets session-end.sh find the sentinel
  # written by user-prompt-submit.sh.
  local id="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "${id}" | sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s' "${id}" | shasum -a 256 | awk '{print $1}'
  else
    # Fallback: replace anything outside the safe allowlist with _.
    printf '%s' "${id}" | tr -c '[:alnum:]._-' '_'
  fi
}

_json_encode_str() {
  local s="$1"
  if command -v python3 >/dev/null 2>&1; then
    LETHE_ENC_S="$s" python3 -c 'import json, os; print(json.dumps(os.environ["LETHE_ENC_S"]), end="")'
  else
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\t'/\\t}"
    printf '"%s"' "$s"
  fi
}
