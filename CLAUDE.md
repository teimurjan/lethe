# lethe

Self-improving memory store for LLM agents. BM25 + dense vector hybrid retrieval with cross-encoder reranking, clustered retrieval-induced forgetting, and optional LLM write-time enrichment.

Ships as a **Claude Code plugin** (`plugins/claude-code/`) that drops a `.lethe/` directory into each project, writes daily markdown memory via hooks, and surfaces prior-session context through `recall` (current project) and `recall-global` (all registered projects) skills.

## stack

- Python 3.12, managed with `uv`
- FAISS (faiss-cpu) for dense vector index
- rank_bm25 for sparse keyword index
- sentence-transformers: all-MiniLM-L6-v2 (bi-encoder), ms-marco-MiniLM-L-6-v2 (cross-encoder)
- SQLite for metadata persistence
- numpy for vectors, matplotlib for plots, pytest for tests
- LongMemEval (S variant) + NFCorpus for benchmarks

## layout

```
crates/               # Rust workspace (production)
├── lethe-core/       # Library: tokenize, bm25, faiss_flat, rrf, dedup, rif,
│                     #   kmeans, encoders, db, memory_store, union_store, …
├── lethe-cli/        # `lethe-rs` binary (clap)
├── lethe-tui/        # `lethe-tui` binary (ratatui)
├── lethe-py/         # PyO3 bindings (importable as `lethe_rust`)
├── lethe-node/       # napi-rs bindings (`@lethe/memory-rust`)
└── lethe-bench/      # parity bench helper binary

benchmarks/           # Python ↔ Rust parity bench (1-1)
├── prepare.py        # exports LongMemEval flat files for the Rust side
├── longmemeval.py    # NDCG@10 / Recall@10 across 5 retrieval configs
├── components.py     # BM25 / FlatIP / xenc per-component numerical diff
├── latency.py        # cold-start + warm-retrieve at 500/5k/20k corpus
└── results/          # COMPARE_*_<host>_<date>.md

legacy/               # Pre-Rust Python implementation (kept for research trail)
├── pyproject.toml    # `uv pip install -e legacy/`
├── lethe/            # Production library (178 tests) — same API, still ships
└── benchmarks/       # Per-checkpoint research benchmarks (run_rif_*, etc.)
└── tests/            # Pytest suite (178 + 8 PyO3 parity = 186)

plugins/claude-code/  # Claude Code plugin
├── hooks/            # SessionStart / UserPromptSubmit / Stop / SessionEnd
├── scripts/          # transcript.py + derive-collection.sh helpers
└── skills/
    ├── recall/           # search memories in the current project
    └── recall-global/    # search memories across all registered projects (--all)

.claude-plugin/       # Marketplace manifest (for `/plugin marketplace add`)

scripts/              # Reproducibility utilities (dataset prep, enrichment runner)

research/             # Experimental / non-production code
├── gc_mutation/      # Germinal-center mutation thread (checkpoints 1-10)
└── sdm/              # Sparse Distributed Memory prototype (checkpoint 15)
```

## commands

```bash
# Python (legacy package — still the production surface today)
uv venv --python 3.12 && uv pip install -e legacy/
cd legacy && uv run pytest tests/ -v
uv run python legacy/benchmarks/run_benchmark.py

# Rust port
cargo build --workspace --release
cargo test --workspace

# Python ↔ Rust parity bench
uv run python benchmarks/prepare.py
uv run python benchmarks/longmemeval.py --compare
uv run python benchmarks/components.py --compare
uv run python benchmarks/latency.py --compare

# CLI (exposed as console script `lethe`)
lethe index                              # reindex .lethe/memory (auto-registers project)
lethe search "query" --top-k 5           # single-project
lethe search "query" --all --top-k 5     # all registered projects (DuckDB ATTACH)
lethe projects list|add|remove|prune     # manage ~/.lethe/projects.json
lethe expand <chunk-id>
lethe status
lethe tui                                # interactive TUI (needs `uv pip install -e '.[tui]'`)

# Run CLI without local install
uvx --from git+https://github.com/teimurjan/lethe lethe --version
```

## key architecture decisions

- BM25 + FAISS hybrid retrieval (BM25 is the strongest single signal on conversation data)
- Cross-encoder reranking on the merged candidate pool
- Adaptive search depth: shallow k=30 for confident queries, deep k=100 when unsure (LongMemEval sweep at `legacy/benchmarks/results/BENCHMARKS_DEEP_PASS.md` shows NDCG@10 is flat past rank 100, so the previous k=200 was pure rerank overhead)
- Cosine 0.95 dedup on add (removes 4.6% of LongMemEval, +6.5% NDCG)
- Tier lifecycle: naive → gc → memory (with decay and apoptosis)
- RIF: retrieval-induced forgetting suppresses chronic false positives at candidate selection stage
- SQLite + .npz + FAISS for persistence (no external services)

## benchmark results

LongMemEval S (200k turns, 500 questions, 200-query eval sample):

| System | NDCG@10 |
|--------|---------|
| Vector only | 0.1376 |
| BM25 only | 0.3171 |
| Hybrid RRF (memsearch style) | 0.2408 |
| **Hybrid + cross-encoder rerank** | **0.3817** |

The 0.3817 is from BM25+vector+cross-encoder reranking (standard IR). The GC mechanism does not improve this number. Verified with integrity checks.

BM25 tokenizer was upgraded from `lower().split()` to a regex word-tokenizer on 2026-04-24 (previous headline: 0.3680 → 0.3817, +1.37pp; BM25-only: 0.2420 → 0.3171, +7.51pp). See [legacy/benchmarks/results/BENCHMARKS_BM25_TOKENIZER.md](legacy/benchmarks/results/BENCHMARKS_BM25_TOKENIZER.md).

## research status

18 checkpoints. Checkpoints 1-10 (GC mechanisms): all failed. Checkpoint 11 (global RIF): +1.1% (not significant: p=0.62 under checkpoint 18's bootstrap). Checkpoint 12 (clustered RIF): +5.8% (significant: p=0.002, CI excludes zero). Checkpoint 13 (clustered + rank-gap): +6.5% NDCG, +9.5% recall@30 — retrieval-only best. Checkpoint 14 (exploration + rescue list): negative at full scale. Checkpoint 15 (SDM research prototype in `sdm/`): negative. Checkpoint 16 (extended metrics for checkpoint 13): RIF's gain is primarily from reducing wrong_family (−1.6pp); sibling_confusion and stale_fact unchanged. Checkpoint 17 (LLM enrichment layer, Haiku write-time gist + anticipated queries + entities + temporal markers): **+8.3pp NDCG on covered queries (+21% rel over RIF), largest single-lever gain in the journey**. Overall diluted to +1.2pp by partial coverage (15% of queries). Full corpus enrichment expected ~$16. Checkpoint 18 (statistical rigor + NFCorpus replication): clustered-RIF main claim survives bootstrap CIs and paired permutation tests on LongMemEval; rank-gap refinement is not individually significant over the uniform rule at n=500 (p=0.55 NDCG); mechanism does not transfer to NFCorpus (3 of 4 variants significantly regress). **Mechanism is workload-specific to long-term conversational memory.**

Full research journey: [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md). Statistical rigor and scope claim written up in the arXiv-formatted paper at [arxiv/paper.tex](arxiv/paper.tex).

## git commits

Use conventional commits. release-please is configured to only bump on `feat:` (minor) and `fix:` (patch). Use `feat:` or `fix:` only when you want a PyPI release. For everything else use other types of conventional commits.
