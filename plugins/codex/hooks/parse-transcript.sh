#!/usr/bin/env bash
# parse-transcript.sh ROLLOUT_PATH
#
# Extract the last user turn and the following assistant response from a
# Codex CLI rollout JSONL. Prints a plain-text blob to stdout suitable for
# piping into a downstream LLM enrichment prompt.
#
# Delegates to `lethe-codex transcript`. Output format matches
# `lethe-claude-code transcript` byte-for-byte (SESSION_ID/TURN_ID/USER/
# ASSISTANT block) so the rest of the hook pipeline works unchanged.

set -eu
set -o pipefail

TRANSCRIPT="${1:-}"
if [ -z "${TRANSCRIPT}" ] || [ ! -f "${TRANSCRIPT}" ]; then
  exit 0
fi

if ! command -v lethe-codex >/dev/null 2>&1; then
  exit 0
fi

lethe-codex transcript "${TRANSCRIPT}"
