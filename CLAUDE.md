# gc-memory

Self-improving memory store for LLM agents. BM25 + dense vector hybrid retrieval with cross-encoder reranking.

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
src/gc_memory/
├── memory_store.py   # Production API: MemoryStore
├── db.py             # SQLite persistence
├── vectors.py        # FAISS + BM25 index management
├── reranker.py       # Cross-encoder + adaptive depth
├── dedup.py          # Hash + cosine deduplication
├── entry.py          # MemoryEntry dataclass + Tier enum
├── config.py         # Hyperparameters
├── metrics.py        # NDCG, recall, diversity (for benchmarks)
├── store.py          # Research: original GC mutation store
├── graph.py          # Research: co-relevance graph
├── rescue_index.py   # Research: deep mining cache
├── mutation.py       # Research: adapter mutation
├── mlp_adapter.py    # Research: learned MLP adapter
├── segmentation.py   # Research: text split/merge
└── baselines.py      # Research: experiment baselines

benchmarks/           # Benchmark scripts, produces BENCHMARKS.md
experiments/          # Research experiment harness (data_prep, run_experiment, analyze)
research/             # Archived research modules
tests/                # Unit tests
```

## commands

```bash
uv venv --python 3.12 && uv pip install -e .
uv run pytest tests/ -v
uv run python experiments/data_prep.py --dataset longmemeval
uv run python benchmarks/run_benchmark.py
```

## key architecture decisions

- BM25 + FAISS hybrid retrieval (BM25 is the strongest single signal on conversation data)
- Cross-encoder reranking on the merged candidate pool
- Adaptive search depth: shallow k=30 for confident queries, deep k=200 when unsure
- Cosine 0.95 dedup on add (removes 4.6% of LongMemEval, +6.5% NDCG)
- Tier lifecycle: naive → gc → memory (with decay and apoptosis)
- SQLite + .npz + FAISS for persistence (no external services)

## benchmark results

LongMemEval S (200k turns, 500 questions, 200-query eval sample):

| System | NDCG@10 | vs baseline |
|--------|---------|-------------|
| Vector only | 0.1376 | baseline |
| BM25 only | 0.2420 | +76% |
| Memsearch (BM25+vec RRF) | 0.2171 | +58% |
| gc-memory (BM25+vec+xenc) | 0.3395 | +147% |

## research status

The project started as a germinal center biology experiment. Direct embedding mutation, MLP adapters, text segmentation, and graph evolution were all tested. None beat the static baseline significantly. The winning approach is modern IR (BM25 hybrid + cross-encoder reranking + dedup) with bio-inspired lifecycle management (tiers + decay).

Full research journey: [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md)
