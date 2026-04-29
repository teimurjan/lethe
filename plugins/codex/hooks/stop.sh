#!/usr/bin/env bash
# Stop — summarize the last turn via `claude -p --model haiku`, append the
# bullets to today's markdown file with a progressive-disclosure anchor, and
# reindex `.lethe/memory` so the next retrieval sees the new content.

set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

_read_stdin_with_timeout || true
_lethe_init_paths

# Recursion guard: Codex sets `stop_hook_active=true` in the stdin payload
# when the Stop event was triggered by a hook (e.g. our own `claude -p`
# summarizer's nested Stop hook would re-fire this). Bail before we spawn
# another summarizer.
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

# Summarize via `claude -p --model haiku` if both the rollout parser and
# claude CLI are available. Without either, fall back to writing just the
# anchor — the transcript path is still captured so a later pass can hydrate.
SUMMARY=""
if [ -n "${TRANSCRIPT}" ] && [ -f "${TRANSCRIPT}" ] \
    && command -v lethe-codex >/dev/null 2>&1 \
    && command -v claude >/dev/null 2>&1; then
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

    SUMMARY="$(printf '%s' "${USER_PROMPT}" \
      | claude -p --model haiku --append-system-prompt "${SYSTEM_PROMPT}" 2>/dev/null || true)"
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

exit 0
