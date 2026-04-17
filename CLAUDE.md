# lethe

Self-improving memory store for LLM agents. BM25 + dense vector hybrid retrieval with cross-encoder reranking, clustered retrieval-induced forgetting, and optional LLM write-time enrichment.

Ships as a **Claude Code plugin** (`plugins/claude-code/`) that drops a `.lethe/` directory into each project, writes daily markdown memory via hooks, and surfaces prior-session context through a `memory-recall` skill.

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
src/lethe/            # Production library (162 tests, ~95% coverage)
├── memory_store.py   # Main API: MemoryStore
├── markdown_store.py # Markdown chunker (##/### headings, content hashes)
├── cli.py            # `lethe` CLI: index/search/expand/status/config/reset/enrich/projects
├── union_store.py    # Read-only cross-project search via DuckDB ATTACH
├── db.py             # DuckDB persistence (lethe.duckdb, unlimited ATTACH)
├── vectors.py        # FAISS + BM25 index management
├── reranker.py       # Cross-encoder + adaptive depth
├── rif.py            # Retrieval-Induced Forgetting (clustered + gap)
├── enrichment.py     # Optional LLM write-time enrichment (Anthropic SDK)
├── dedup.py          # Hash + cosine deduplication
├── encoders.py       # ONNX bi-encoder + cross-encoder adapters (fastembed)
├── _lock.py          # fcntl-based per-project lock for concurrent CLI calls
├── _registry.py      # ~/.lethe/projects.json for cross-project search
└── entry.py          # MemoryEntry dataclass + Tier enum

plugins/claude-code/  # Claude Code plugin
├── hooks/            # SessionStart / UserPromptSubmit / Stop / SessionEnd
├── scripts/          # transcript.py + derive-collection.sh helpers
└── skills/memory-recall/

.claude-plugin/       # Marketplace manifest (for `/plugin marketplace add`)

benchmarks/           # Per-checkpoint benchmark scripts
├── run_*.py
├── _lib/             # Benchmark-only helpers (metrics, NDCG/recall)
└── results/          # Raw per-run output markdowns

scripts/              # Reproducibility utilities (dataset prep, enrichment runner)

research/             # Experimental / non-production code
├── gc_mutation/      # Germinal-center mutation thread (checkpoints 1-10)
└── sdm/              # Sparse Distributed Memory prototype (checkpoint 15)

tests/                # Production unit tests
```

## commands

```bash
uv venv --python 3.12 && uv pip install -e .
uv run pytest tests/ -v
uv run python experiments/data_prep.py --dataset longmemeval
uv run python benchmarks/run_benchmark.py

# CLI (exposed as console script `lethe`)
lethe index                              # reindex .lethe/memory (auto-registers project)
lethe search "query" --top-k 5           # single-project
lethe search "query" --all --top-k 5     # all registered projects (DuckDB ATTACH)
lethe projects list|add|remove|prune     # manage ~/.lethe/projects.json
lethe expand <chunk-id>
lethe status

# Run CLI without local install
uvx --from git+https://github.com/teimurjan/lethe lethe --version
```

## key architecture decisions

- BM25 + FAISS hybrid retrieval (BM25 is the strongest single signal on conversation data)
- Cross-encoder reranking on the merged candidate pool
- Adaptive search depth: shallow k=30 for confident queries, deep k=200 when unsure
- Cosine 0.95 dedup on add (removes 4.6% of LongMemEval, +6.5% NDCG)
- Tier lifecycle: naive → gc → memory (with decay and apoptosis)
- RIF: retrieval-induced forgetting suppresses chronic false positives at candidate selection stage
- SQLite + .npz + FAISS for persistence (no external services)

## benchmark results

LongMemEval S (200k turns, 500 questions, 200-query eval sample):

| System | NDCG@10 |
|--------|---------|
| Vector only | 0.1376 |
| BM25 only | 0.2420 |
| Hybrid RRF (memsearch style) | 0.2171 |
| **Hybrid + cross-encoder rerank** | **0.3680** |

The 0.3680 is from BM25+vector+cross-encoder reranking (standard IR). The GC mechanism does not improve this number. Verified with integrity checks.

## research status

17 checkpoints. Checkpoints 1-10 (GC mechanisms): all failed. Checkpoint 11 (global RIF): +1.1%. Checkpoint 12 (clustered RIF): +5.8%. Checkpoint 13 (clustered + rank-gap): +6.5% NDCG, +9.5% recall@30 — retrieval-only best. Checkpoint 14 (exploration + rescue list): negative at full scale. Checkpoint 15 (SDM research prototype in `sdm/`): negative. Checkpoint 16 (extended metrics for checkpoint 13): RIF's gain is primarily from reducing wrong_family (−1.6pp); sibling_confusion and stale_fact unchanged. Checkpoint 17 (LLM enrichment layer, Haiku write-time gist + anticipated queries + entities + temporal markers): **+8.3pp NDCG on covered queries (+21% rel over RIF), largest single-lever gain in the journey**. Overall diluted to +1.2pp by partial coverage (15% of queries). Full corpus enrichment expected ~$16.

Full research journey: [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md)
