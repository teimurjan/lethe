#!/usr/bin/env bash
# Quick parity validator: confirms the Rust port returns retrieval
# results "close enough" to the Python reference on the demo corpus.
# Runs ~2-4 minutes after the first model download.
#
# Usage:
#   ./bench/validate_parity.sh           # 10 queries, quick
#   ./bench/validate_parity.sh --full    # all 30 queries

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -x "target/release/lethe-rs" ]; then
    echo "[validate_parity] building release lethe-rs…"
    cargo build --release -p lethe-cli
fi

exec uv run python bench/validate_parity.py "$@"
