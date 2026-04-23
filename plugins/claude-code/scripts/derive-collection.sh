#!/usr/bin/env bash
# Prints a stable per-project collection name derived from the git root.
# Used by the recall / recall-global skills so the LLM sees which project's
# memory it is searching.

set -eu
set -o pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
name="$(basename "${root}")"
hash="$(printf '%s' "${root}" | shasum -a 256 | cut -c1-8)"
printf '%s-%s\n' "${name}" "${hash}"
