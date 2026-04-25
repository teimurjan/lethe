#!/usr/bin/env bash
# Drive the Python↔Rust comparison harness.
# Builds the release Rust binary if missing, then runs the Python
# script that times both implementations and writes the report.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure the release Rust binary is present.
if [ ! -x "target/release/lethe-rs" ]; then
    echo "[run_compare] building release lethe-rs…"
    cargo build --release -p lethe-cli
fi

# Ensure Python lethe is importable (we run inside the repo via uv).
exec uv run python bench/compare_pipeline.py "$@"
