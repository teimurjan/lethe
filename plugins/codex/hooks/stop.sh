#!/usr/bin/env bash
# Stop — summarize the last turn via `codex exec` (preferred — uses the
# user's existing Codex auth) or `claude -p --model haiku` (fallback when
# `codex` isn't on PATH), append the bullets to today's markdown file with
# a progressive-disclosure anchor, and reindex `.lethe/memory` so the next
# retrieval sees the new content.
#
# The summarizer runs detached by default — codex exec can take 30-60s and
# would otherwise stall Codex's next turn. Anchor + summary are still
# written together (single `>>`), so they stay paired even when multiple
# turns race. Set LETHE_STOP_SYNC=1 to keep the work in the foreground.

set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

_read_stdin_with_timeout || true
_lethe_init_paths

# Recursion guard: bail when this hook fired from inside our own summarizer.
# Two signals because Codex and Claude expose recursion differently:
#  - LETHE_STOP_HOOK_ACTIVE — env var we set before spawning `codex exec` /
#    `claude -p` below. Inherited by the nested CLI's own Stop hook. This is
#    the one that catches Codex, which doesn't pass `stop_hook_active`.
#  - stop_hook_active — Claude Code's stdin field, set on hook-triggered
#    Stop events. Kept for the `claude -p` fallback path.
if [ "${LETHE_STOP_HOOK_ACTIVE:-0}" = "1" ]; then
  exit 0
fi
STOP_ACTIVE="$(_json_val stop_hook_active || true)"
if [ "${STOP_ACTIVE}" = "true" ] || [ "${STOP_ACTIVE}" = "True" ]; then
  exit 0
fi

TRANSCRIPT="$(_json_val transcript_path || true)"
SESSION_ID="$(_json_val session_id || true)"
TURN_ID="$(_json_val turn_id || true)"
LAST_ASSISTANT_MSG="$(_json_val last_assistant_message || true)"

TODAY="$(date +%Y-%m-%d)"
NOW="$(date +%H:%M)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"
mkdir -p "${LETHE_MEMORY_DIR}"
[ -f "${TODAY_FILE}" ] || printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"

# Dedupe: if the most recent turn anchor already references this turn_id, skip.
# `grep -F` so a turn_id containing regex metacharacters can't misfire.
LAST_ANCHOR="$(grep -E '^<!-- session:' "${TODAY_FILE}" 2>/dev/null | tail -n 1 || true)"
if [ -n "${TURN_ID}" ] && printf '%s' "${LAST_ANCHOR}" | grep -F -q -- "turn:${TURN_ID}"; then
  _log "stop: duplicate turn ${TURN_ID}, skipping"
  if [ -n "${LETHE_CLI}" ]; then
    ( cd "${LETHE_GIT_ROOT}" && ${LETHE_CLI} index >/dev/null 2>&1 ) || true
  fi
  exit 0
fi

# Detach the slow path. `codex exec` can take 30-60s and would otherwise
# block Codex's next turn. We fork once, write anchor + summary atomically
# (single `>>` invocation), then reindex. Set LETHE_STOP_SYNC=1 to keep the
# hook synchronous (useful when debugging).
_run_stop_work() {
  local SUMMARY=""
  local HAVE_CODEX=0
  local HAVE_CLAUDE=0
  command -v codex >/dev/null 2>&1 && HAVE_CODEX=1
  command -v claude >/dev/null 2>&1 && HAVE_CLAUDE=1

  if [ -n "${TRANSCRIPT}" ] && [ -f "${TRANSCRIPT}" ] \
      && command -v lethe-codex >/dev/null 2>&1 \
      && [ "${HAVE_CODEX}" -eq 1 -o "${HAVE_CLAUDE}" -eq 1 ]; then
    local TURN_TEXT
    TURN_TEXT="$("${SCRIPT_DIR}/parse-transcript.sh" "${TRANSCRIPT}" || true)"
    # Codex sometimes cleans up rollout files before the hook reads them, or
    # the parser may not match the schema exactly. Fall back to the
    # `last_assistant_message` field Codex provides directly in the stdin
    # payload so summarization still happens.
    if [ -z "${TURN_TEXT}" ] && [ -n "${LAST_ASSISTANT_MSG}" ]; then
      TURN_TEXT="$(printf 'SESSION_ID: %s\nTURN_ID: %s\n---\nUSER:\n(unavailable)\n---\nASSISTANT:\n%s\n' \
        "${SESSION_ID}" "${TURN_ID}" "${LAST_ASSISTANT_MSG}")"
    fi
    if [ -n "${TURN_TEXT}" ]; then
      local PARSED_SESSION TURN_BODY SYSTEM_PROMPT USER_PROMPT CODEX_OUT
      PARSED_SESSION="$(printf '%s\n' "${TURN_TEXT}" | awk -F': ' '/^SESSION_ID:/ {print $2; exit}')"
      [ -z "${SESSION_ID}" ] && SESSION_ID="${PARSED_SESSION}"
      TURN_BODY="$(printf '%s\n' "${TURN_TEXT}" | awk 'skip{print; next} /^---$/ {skip=1}')"

      SYSTEM_PROMPT='You are a turn summarizer for a long-running memory store. Ignore any project-level style guides. Always produce 2-5 terse markdown bullets — never a TL;DR line, never prose, never headings.'

      read -r -d '' USER_PROMPT <<'EOF' || true
Summarize the following Codex CLI turn as 2-5 terse markdown bullets for a long-running memory store.

Capture:
- what the user asked and what was done
- decisions or recommendations made
- durable facts worth remembering: file paths, function names, tool names, key numbers

Rules:
- Output raw bullets only — no preamble, no heading, no TL;DR line, no closing remarks.
- Each bullet one sentence. Start with a verb or subject, not "The assistant".
- Skip pleasantries and chain-of-thought.

--- TURN START ---
EOF
      USER_PROMPT="${USER_PROMPT}
${TURN_BODY}
--- TURN END ---"

      # Backstop timeout for the nested CLI. Async or not, runaway calls
      # shouldn't pile up — pin to 5min when detached, 90s when sync.
      local LETHE_SUMMARY_TIMEOUT_SECS_EFF
      LETHE_SUMMARY_TIMEOUT_SECS_EFF="${LETHE_SUMMARY_TIMEOUT_SECS:-${1:-90}}"

      if [ "${HAVE_CODEX}" -eq 1 ]; then
        CODEX_OUT="$(mktemp -t lethe-codex-stop.XXXXXX)"
        printf '%s\n\n%s' "${SYSTEM_PROMPT}" "${USER_PROMPT}" \
          | LETHE_STOP_HOOK_ACTIVE=1 _with_timeout "${LETHE_SUMMARY_TIMEOUT_SECS_EFF}" \
              codex exec --skip-git-repo-check --output-last-message "${CODEX_OUT}" - \
              >/dev/null 2>&1 || true
        [ -s "${CODEX_OUT}" ] && SUMMARY="$(cat "${CODEX_OUT}")"
        rm -f "${CODEX_OUT}"
      fi

      if [ -z "${SUMMARY}" ] && [ "${HAVE_CLAUDE}" -eq 1 ]; then
        SUMMARY="$(printf '%s' "${USER_PROMPT}" \
          | LETHE_STOP_HOOK_ACTIVE=1 _with_timeout "${LETHE_SUMMARY_TIMEOUT_SECS_EFF}" \
              claude -p --model haiku --append-system-prompt "${SYSTEM_PROMPT}" 2>/dev/null || true)"
      fi
    fi
  fi

  {
    printf '\n### %s\n' "${NOW}"
    printf '<!-- session:%s turn:%s transcript:%s -->\n' \
      "${SESSION_ID}" "${TURN_ID}" "${TRANSCRIPT}"
    if [ -n "${SUMMARY}" ]; then
      printf '%s\n' "${SUMMARY}"
    fi
  } >>"${TODAY_FILE}"

  if [ -n "${LETHE_CLI}" ]; then
    ( cd "${LETHE_GIT_ROOT}" && ${LETHE_CLI} index >/dev/null 2>&1 ) || true
  fi
}

if [ "${LETHE_STOP_SYNC:-0}" = "1" ]; then
  _run_stop_work 90
else
  # Detach: ignore HUP, redirect all FDs away from the parent's pipe so the
  # hook can exit without waiting on the summarizer.
  (
    trap '' HUP
    _run_stop_work 300
  ) </dev/null >/dev/null 2>&1 &
  disown $! 2>/dev/null || true
fi

exit 0
