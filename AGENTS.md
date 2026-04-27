# lethe — agent guide

Self-improving memory store for LLM agents. BM25 + dense vector hybrid
retrieval with cross-encoder reranking, clustered retrieval-induced
forgetting (RIF), and optional LLM write-time enrichment. Ships as a
single Rust binary (`lethe`) plus PyPI / npm bindings.

## stack

- **Rust workspace** (`crates/`) — production. Toolchain pinned via
  `rust-toolchain.toml`.
  - `lethe-core` — library: tokenize, bm25, faiss_flat, rrf, dedup, rif,
    kmeans, encoders (ONNX via `ort`), DuckDB persistence, npz reader,
    memory_store, union_store, markdown_store.
  - `lethe-cli` — `lethe` binary (clap; embeds the TUI).
  - `lethe-tui` — ratatui library called from the CLI on no-arg invocation.
  - `lethe-py` — PyO3 binding → PyPI `lethe-memory`.
  - `lethe-node` — napi-rs binding → npm `lethe`.
  - `lethe-claude-code` — Claude Code adapter binary (transcript parsing).
  - `lethe-benchmark` — internal parity bench helper (`publish = false`).
- **DuckDB** for entry metadata + embedding BLOBs (single source of truth;
  Python's `embeddings.npz` is migrated in via `lethe migrate`).
- **ONNX Runtime** (via `ort`) for the bi-encoder + cross-encoder.
  Default models: `Xenova/all-MiniLM-L6-v2`, `Xenova/ms-marco-MiniLM-L-6-v2`.
- **Python `legacy/`** kept for the research-journey trail: produces the
  benchmark numbers cited below. Installed as `lethe-memory-legacy` in
  dev venvs; not published.
- **Datasets** (`tmp_data/`, gitignored): LongMemEval (S), NFCorpus.

## commands

```bash
# Rust workspace
cargo build --workspace --release        # → target/release/lethe + lethe-claude-code
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# Pre-commit / pre-push hooks (replaces the deleted ci.yml)
cargo install --git https://github.com/j178/prek
prek install                             # writes .git/hooks/pre-commit + pre-push
prek run --all-files                     # one-off run

# Legacy Python (research trail; library only)
uv pip install -e legacy/
cd legacy && uv run pytest tests/ -v     # 148 + 8 PyO3 parity = 156
uv run python legacy/benchmarks/run_benchmark.py

# Python ↔ Rust parity bench
uv run python migration_benchmarks/prepare.py
uv run python migration_benchmarks/longmemeval.py --compare
uv run python migration_benchmarks/components.py --compare
uv run python migration_benchmarks/latency.py --compare

# CLI surface (the `lethe` binary)
lethe                                    # no args → TUI (in a terminal)
lethe index                              # reindex .lethe/memory
lethe search "query" --top-k 5
lethe search "query" --all --top-k 5     # all registered projects (DuckDB ATTACH)
lethe projects list|add|remove|prune
lethe expand <chunk-id>
lethe status
lethe migrate [--all]                    # convert legacy embeddings.npz → DuckDB
```

`tree -L 2 -I 'target|.venv|node_modules|tmp_*|results|.git|.lethe'` if
you want the current directory layout.

## key architecture decisions

- BM25 + FAISS-equivalent hybrid retrieval (BM25 is the strongest single
  signal on conversation data).
- Cross-encoder rerank on the merged candidate pool; adaptive depth
  (shallow `k=30`, deep `k=100` only when shallow confidence is low —
  see `legacy/benchmarks/results/BENCHMARKS_DEEP_PASS.md`).
- Cosine 0.95 dedup on add (removes ~5% of LongMemEval, +6.5% NDCG).
- Tier lifecycle: naive → gc → memory (with decay and apoptosis).
- RIF: retrieval-induced forgetting suppresses chronic false positives
  at the candidate-selection stage (workload-specific to long-term
  conversational memory).
- DuckDB + npz interop for persistence; no external services.
- Embeddings are *the* source of truth in DuckDB. `MemoryStore::open()`
  rebuilds the in-memory `FlatIp` from BLOBs every cold start
  (~100 ms on a 200k corpus); no separate FAISS index file is written.

## benchmark headline

LongMemEval S (200k turns, 500 questions, 200-query eval sample):

| System | NDCG@10 |
|---|---|
| Vector only | 0.1376 |
| BM25 only | 0.3171 |
| Hybrid RRF | 0.2408 |
| **Hybrid + cross-encoder rerank** | **0.3817** |

Full methodology: `BENCHMARKS.md`. 18-checkpoint research journey:
`RESEARCH_JOURNEY.md`. arXiv preprint source: `paper.tex`.

## git + release

- Conventional commits. release-please bumps version on `feat:` /
  `fix:` only — use either when you want the next release to fan out
  to crates.io / PyPI / npm / Homebrew. Everything else (`chore:`,
  `docs:`, `refactor:`, `test:`, `perf:`) does not trigger a release.
- Workspace version is pinned at `Cargo.toml :: [workspace.package].version`;
  every member crate inherits via `version.workspace = true`. Same value
  is bumped in `crates/lethe-py/pyproject.toml` and
  `crates/lethe-node/package.json` via release-please's `extra-files` /
  linked-versions plugin.
- Multi-platform release artifacts are built **locally** via
  `scripts/release/build.sh` (host / `--macos` / `--all`) and uploaded
  to the GitHub Release as assets. The `release-rust.yml`,
  `release-pypi.yml`, `release-npm.yml` workflows then publish from
  those assets — they do **not** run their own build matrices.
- Required repo secrets for a successful publish:
  - `CARGO_REGISTRY_TOKEN` — crates.io API token, scoped to `lethe-*`
  - `NPM_TOKEN` — npm Granular Access Token, scoped to `lethe`
  - `HOMEBREW_TAP_GITHUB_TOKEN` — fine-grained PAT, Contents R/W on
    `teimurjan/homebrew-lethe`
  - PyPI uses Trusted Publishing (no token; configure on PyPI side
    plus a GitHub `pypi` environment).
