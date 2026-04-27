#!/usr/bin/env bash
# Shared helpers for lethe hooks. Sourced, not executed.
#
# Exports:
#   LETHE_GIT_ROOT    — git root of the invoking project (or cwd if none)
#   LETHE_DIR         — $LETHE_GIT_ROOT/.lethe
#   LETHE_MEMORY_DIR  — $LETHE_DIR/memory
#   LETHE_INDEX_DIR   — $LETHE_DIR/index
#   LETHE_STDIN_JSON  — raw JSON payload read from stdin (may be empty)
#   LETHE_CLI         — `lethe` if the binary is on PATH, else empty (the
#                       hooks no-op when empty so a missing binary is not
#                       fatal). Install via `brew install teimurjan/lethe/lethe`,
#                       `cargo install lethe-cli`, or a release tarball from
#                       https://github.com/teimurjan/lethe/releases.
#
# Helpers:
#   _json_val <key>           — extract a string/scalar from LETHE_STDIN_JSON
#   _json_encode_str <string> — JSON-encode a string (for response bodies)
#   _log <msg>                — append timestamped debug line to LETHE_LOG
#   _read_stdin_with_timeout  — populate LETHE_STDIN_JSON with a 2s budget

set -o pipefail

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
  # Flat top-level field extractor for the Claude Code hook stdin
  # payload (always a flat object whose values are strings/bools/
  # numbers). Pure awk so the plugin stays Python-free.
  local key="$1"
  [ -z "${LETHE_STDIN_JSON}" ] && return 0
  printf '%s' "${LETHE_STDIN_JSON}" | awk -v k="$key" '
    BEGIN { found = 0 }
    {
      # Strings: "key": "value" (handles \" inside the value).
      pat = "\"" k "\"[ \t]*:[ \t]*\"(\\\\\"|[^\"])*\""
      if (match($0, pat)) {
        s = substr($0, RSTART, RLENGTH)
        sub(/^[^:]*:[ \t]*"/, "", s)
        sub(/"$/, "", s)
        gsub(/\\"/, "\"", s)
        print s
        found = 1
        exit
      }
      # Booleans / numbers: "key": value (no quotes, ends at , or }).
      pat2 = "\"" k "\"[ \t]*:[ \t]*[^,}]*"
      if (match($0, pat2)) {
        s = substr($0, RSTART, RLENGTH)
        sub(/^[^:]*:[ \t]*/, "", s)
        sub(/[ \t]*$/, "", s)
        print s
        found = 1
        exit
      }
    }
  '
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
  # Pure-bash JSON string encoder. The previous Python branch was
  # removed when the plugin went Python-free; this fallback handles
  # the cases that show up in hook payloads (text content with \, ",
  # \n, \t).
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\t'/\\t}"
  printf '"%s"' "$s"
}
