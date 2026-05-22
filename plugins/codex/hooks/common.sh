#!/usr/bin/env bash
# Shared helpers for lethe Codex hooks. Sourced, not executed.
#
# Codex CLI hooks receive a JSON payload on stdin with at least:
#   session_id, transcript_path, cwd, hook_event_name, model, turn_id
#
# Exports:
#   LETHE_GIT_ROOT    — git root resolved from the JSON `cwd` field (or pwd)
#   LETHE_DIR         — $LETHE_GIT_ROOT/.lethe
#   LETHE_MEMORY_DIR  — $LETHE_DIR/memory
#   LETHE_INDEX_DIR   — $LETHE_DIR/index
#   LETHE_STDIN_JSON  — raw JSON payload read from stdin (may be empty)
#   LETHE_CLI         — `lethe` if the binary is on PATH, else empty
#
# Helpers:
#   _json_val <key>           — extract a string/scalar from LETHE_STDIN_JSON
#   _json_encode_str <string> — JSON-encode a string (for response bodies)
#   _log <msg>                — append timestamped debug line to LETHE_LOG
#   _read_stdin_with_timeout  — populate LETHE_STDIN_JSON with a 2s budget
#   _resolve_git_root <dir>   — git rev-parse --show-toplevel under DIR

set -o pipefail

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

_json_val() {
  local key="$1"
  [ -z "${LETHE_STDIN_JSON}" ] && return 0
  printf '%s' "${LETHE_STDIN_JSON}" | awk -v k="$key" '
    BEGIN { found = 0 }
    {
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
  local id="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "${id}" | sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s' "${id}" | shasum -a 256 | awk '{print $1}'
  else
    printf '%s' "${id}" | tr -c '[:alnum:]._-' '_'
  fi
}

_json_encode_str() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\t'/\\t}"
  printf '"%s"' "$s"
}

# Run "$@" with a wall-clock timeout in seconds. Prefers GNU `timeout` /
# `gtimeout`; falls back to a bash watchdog so macOS without coreutils still
# gets bounded execution. `<&0` is critical — bash auto-redirects async
# (`&`) stdin to /dev/null in non-interactive shells, so the explicit
# redirect is what keeps a piped summarizer prompt reaching the child.
_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=5 "${secs}" "$@"
    return $?
  fi
  if command -v gtimeout >/dev/null 2>&1; then
    gtimeout --kill-after=5 "${secs}" "$@"
    return $?
  fi
  "$@" <&0 &
  local pid=$!
  (
    sleep "${secs}"
    kill -TERM "${pid}" 2>/dev/null && sleep 5 && kill -KILL "${pid}" 2>/dev/null
  ) &
  local watcher=$!
  local rc=0
  wait "${pid}" 2>/dev/null || rc=$?
  kill "${watcher}" 2>/dev/null || true
  wait "${watcher}" 2>/dev/null || true
  return ${rc}
}

_resolve_git_root() {
  local dir="$1"
  [ -z "${dir}" ] && dir="$(pwd)"
  local root
  root="$(
    cd "${dir}" 2>/dev/null && git worktree list --porcelain 2>/dev/null \
    | awk '
        /^worktree / { wt = substr($0, 10); bare = 0; next }
        /^bare$/     { bare = 1; next }
        /^$/         { if (wt && !bare) { print wt; wt = ""; exit } }
        END          { if (wt && !bare) print wt }
      '
  )"
  if [ -n "${root}" ]; then
    printf '%s' "${root}"
    return 0
  fi
  ( cd "${dir}" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null ) || printf '%s' "${dir}"
}

# --- Discover paths from stdin payload (must be called after _read_stdin_with_timeout) ---

_lethe_init_paths() {
  local cwd
  cwd="$(_json_val cwd || true)"
  [ -z "${cwd}" ] && cwd="$(pwd)"
  LETHE_GIT_ROOT="$(_resolve_git_root "${cwd}")"
  LETHE_DIR="${LETHE_GIT_ROOT}/.lethe"
  LETHE_MEMORY_DIR="${LETHE_DIR}/memory"
  LETHE_INDEX_DIR="${LETHE_DIR}/index"
  LETHE_LOG="${LETHE_DIR}/hooks.log"
  mkdir -p "${LETHE_MEMORY_DIR}" "${LETHE_INDEX_DIR}" 2>/dev/null || true
  export LETHE_GIT_ROOT LETHE_DIR LETHE_MEMORY_DIR LETHE_INDEX_DIR LETHE_LOG
}

_log() {
  [ -z "${LETHE_DEBUG:-}" ] && return 0
  printf '%s %s\n' "$(date +'%Y-%m-%dT%H:%M:%S')" "$*" >>"${LETHE_LOG:-/dev/null}" 2>/dev/null || true
}

if command -v lethe >/dev/null 2>&1; then
  LETHE_CLI="lethe"
else
  LETHE_CLI=""
fi
export LETHE_CLI
