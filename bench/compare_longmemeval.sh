#!/usr/bin/env bash
# LongMemEval parity check: run the same 5 retrieval configs through
# both the Python pipeline (run_lme_python.py) and the Rust bench
# (`lethe-bench`), then diff the NDCG@10 / Recall@10 numbers.
#
# Usage:
#   ./bench/compare_longmemeval.sh
#
# First run takes ~10–15 minutes (cross-encoder dominates). Subsequent
# runs reuse the cached HuggingFace model.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Prepare flat data files for the Rust bench.
if [ ! -f "data/lme_rust/meta.json" ]; then
    echo "[compare] preparing flat-binary LongMemEval files for Rust…"
    uv run python bench/prepare_lme_data.py
fi

# 2. Build release lethe-bench.
if [ ! -x "target/release/lethe-bench" ]; then
    echo "[compare] building release lethe-bench…"
    cargo build --release -p lethe-bench
fi

mkdir -p bench/results
PY_OUT="bench/results/_lme_python.json"
RS_OUT="bench/results/_lme_rust.json"

echo "[compare] running Python pipeline → ${PY_OUT}"
uv run python bench/run_lme_python.py > "$PY_OUT"

echo "[compare] running Rust pipeline → ${RS_OUT}"
target/release/lethe-bench > "$RS_OUT"

echo "[compare] writing report…"
uv run python bench/diff_lme.py "$PY_OUT" "$RS_OUT"
