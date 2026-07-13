#!/usr/bin/env bash
#
# lethe UserPromptSubmit hook — keep the global index fresh in the
# background so cross-project `recall-global` reflects recent turns.
#
# `recall`'s `lethe search` already reindexes the *current* project on
# demand; the unique job of this hook is the other registered projects,
# whose cross-project search opens them read-only and never reindexes.
#
# Design constraints this script satisfies:
#   * Non-blocking — the work is detached (`nohup … &`), so prompt
#     submission never waits on it. This script returns in milliseconds.
#   * Throttled — `lethe index --all` loads the ONNX bi-encoder per
#     project even on a no-op, so it runs at most once per interval
#     (LETHE_REFRESH_INTERVAL seconds, default 900 = 15 min).
#   * Single-flight — a PID lock skips a new run while one is in flight.
#   * Degrades to a no-op when `lethe` isn't installed.
#
# Tunables (env): LETHE_REFRESH_INTERVAL (seconds), LETHE_HOME (state dir).
#
# Identical to the Claude Code plugin's copy — it resolves `lethe` via
# PATH and never references the plugin root, so the two stay in sync.
set -euo pipefail

# Hook shells often have a minimal PATH; add the usual install locations.
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/.cargo/bin:$HOME/.local/bin"
LETHE="$(command -v lethe || true)"
[ -z "$LETHE" ] && exit 0 # not installed → nothing to do

STATE_DIR="${LETHE_HOME:-$HOME/.lethe}"
SENTINEL="$STATE_DIR/.last_refresh"
LOCK="$STATE_DIR/.refresh.lock"
INTERVAL="${LETHE_REFRESH_INTERVAL:-900}"
mkdir -p "$STATE_DIR"

now="$(date +%s)"

# Throttle: skip if a refresh started within INTERVAL.
if [ -f "$SENTINEL" ]; then
  last="$(cat "$SENTINEL" 2>/dev/null || echo 0)"
  [ "$((now - last))" -lt "$INTERVAL" ] && exit 0
fi

# Single-flight: skip if a prior refresh is still running.
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null || echo)" 2>/dev/null; then
  exit 0
fi

# Stamp the attempt now so concurrent prompts within INTERVAL skip too
# (throttle counts from attempt start, not completion, so a failing run
# can't retry-storm on every keystroke).
echo "$now" >"$SENTINEL"

# Detach the actual work. `index --all` first (needs encoders), then
# `dedupe --all` to compact what was just added (cheap, no encoders).
export LETHE LOCK
nohup bash -c '
  echo $$ >"$LOCK"
  trap "rm -f \"$LOCK\"" EXIT
  "$LETHE" index --all >/dev/null 2>&1 && "$LETHE" dedupe --all >/dev/null 2>&1
' >/dev/null 2>&1 &
disown 2>/dev/null || true

exit 0
