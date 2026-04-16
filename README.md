# gc-memory

A memory store for LLM agents. Hybrid BM25 + dense retrieval, cross-encoder reranking, **clustered retrieval-induced forgetting**, and an **optional LLM enrichment layer** at write time.

On LongMemEval S (199,509 conversation turns, 500 questions, full-corpus NDCG@10):

| Stage | NDCG@10 | vs baseline |
|-------|---------|-------------|
| Hybrid BM25 + vector + cross-encoder | 0.293 | — |
| + clustered+gap RIF (checkpoint 13) | 0.312 | +6.5% |
| + LLM enrichment, on covered queries | **0.473** | **+35%** |

The enrichment gain is measured on the 75 queries for which the answer turns are enriched; overall numbers are diluted by uncovered queries. See [BENCHMARKS.md](BENCHMARKS.md).

## Quick start

```python
from gc_memory import MemoryStore
from sentence_transformers import SentenceTransformer, CrossEncoder

store = MemoryStore(
    "./my_memories",
    bi_encoder=SentenceTransformer("all-MiniLM-L6-v2"),
    cross_encoder=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
)

store.add("I prefer window seats on flights", session_id="trip")
store.add("My wife needs aisle seats", session_id="trip")
store.add("I work at Google as a software engineer", session_id="work")

results = store.retrieve("What are my travel preferences?", k=5)
for entry_id, content, score in results:
    print(f"  [{score:.1f}] {content}")

store.save()
store.close()
```

## Architecture

```
Query
  │
  ├── FAISS top-30 (dense vector similarity)
  ├── BM25 top-30 (sparse keyword match)
  │
  └── Merge (RRF)
        │
        └── RIF suppression penalty (per-cluster, gap-based)
              │
              └── Cross-encoder rerank → top-k
                    │
                    └── Update suppression state, affinities, tier
```

**Optional write-time LLM enrichment layer** (`src/gc_memory/enrichment.py`): before indexing, each memory can be processed by an LLM (default `claude-haiku-4-5`) to produce a gist, 3 anticipated queries, entities, and temporal markers. All fields index alongside the original text; cross-encoder still scores against original. Attacks the vocabulary-mismatch failure mode.

### Retrieval-induced forgetting (RIF)

On each retrieval, entries that reach the candidate pool but lose to the cross-encoder accumulate a per-cluster suppression score. On future retrievals in the same query cluster, their scores get penalized before the cross-encoder sees them, freeing slots for entries that were previously crowded out.

Key design points:
- **Clustered** (k-means 30, cue-dependent): an entry suppressed for "travel" queries stays available for "food" queries. 5× stronger than global suppression.
- **Rank-gap competition formula**: `max(0, xenc_rank − initial_rank) / pool × sigmoid(−xenc)`. Only suppresses entries that actually dropped in rank AND were actively rejected, not entries that just lost a close race.

Based on Anderson's inhibition theory (1994) and the SAM competitive-sampling model (Raaijmakers & Shiffrin, 1981). First implementation in an AI memory system as far as I can tell.

### Three storage layers

| Layer | File | Purpose |
|-------|------|---------|
| SQLite | `gc_memory.db` | Entries, suppression state, rescue cache, stats |
| numpy + FAISS | `embeddings.npz`, `faiss.index` | Vector storage |
| BM25 | In-memory, rebuilt on startup | Sparse keyword index |

### Entry lifecycle (germinal-center inspired)

```
NAIVE → GC → MEMORY
              ↓
         APOPTOTIC
```

- **Naive**: new entries, unproven
- **GC**: retrieved 3+ times, actively evaluated
- **Memory**: high affinity + frequently retrieved, stable, exempt from decay
- **Apoptotic**: low affinity + idle > 1000 steps, excluded from search

Useful for long-running agents; doesn't directly improve retrieval quality (that's what RIF and enrichment do).

### Deduplication (on add)

1. **Exact**: SHA-256 content hash (free)
2. **Near-duplicate**: cosine similarity > 0.95 (keeps the longer entry)

## Install

```bash
git clone https://github.com/teimurjan/gc-memory && cd gc-memory
uv venv --python 3.12 && uv pip install -e .
```

## Benchmark

```bash
# prep LongMemEval
uv run python experiments/data_prep.py --dataset longmemeval

# retrieval-only baseline + RIF variants
uv run python benchmarks/run_benchmark.py
uv run python benchmarks/run_rif_benchmark.py

# LLM enrichment layer (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python experiments/enrich_longmemeval.py     # one-time, ~$16 for 10k entries
uv run python benchmarks/run_rif_enriched.py         # 3-arm benchmark
```

See [BENCHMARKS.md](BENCHMARKS.md) for results and methodology.

## Benchmark methodology

All numbers here are **NDCG@10 over turn-level retrieval on the full 199,509-turn LongMemEval S corpus** — needle-in-haystack among 200k candidates.

Other memory tools commonly report numbers on **per-query fresh DBs of ~50 sessions** at **session granularity** with **recall@5**. That's a ~2000× easier task (random baseline 10% vs 0.005%), and some implementations additionally leak ground truth via annotation fields at indexing time. Published numbers in the 95-99% range on that methodology are not directly comparable to numbers here.

A fair head-to-head comparison (either methodology run on both systems) is a separate experiment.

## Project structure

```
src/gc_memory/
├── memory_store.py    # Main API: MemoryStore
├── db.py              # SQLite persistence
├── vectors.py         # FAISS + BM25 index management
├── reranker.py        # Cross-encoder + adaptive depth
├── rif.py             # Retrieval-induced forgetting (clustered + gap)
├── enrichment.py      # Optional LLM write-time enrichment (Anthropic SDK)
├── dedup.py           # Hash + cosine deduplication
├── entry.py           # MemoryEntry dataclass + Tier enum
└── config.py          # Hyperparameters

benchmarks/
├── run_*.py           # Benchmark scripts (one per checkpoint)
└── results/           # Raw per-run output markdowns (historical)

experiments/           # Data prep, enrichment dataset builder
research/              # Archived GC mutation / adapter / graph research
sdm/                   # Sparse Distributed Memory prototype (checkpoint 15)
tests/                 # Unit tests
```

## Research background

This project started as an experiment porting the immune system's germinal-center mechanism to vector memory. 17 checkpoints so far:

1. **Checkpoints 1-10** (GC-mutation approaches): all failed to improve retrieval quality. Useful for lifecycle management, not retrieval.
2. **Checkpoints 11-13** (RIF): cognitive-science inspired. Clustered + gap-formula variant gives +6.5% NDCG, +9.5% recall@30 on the full 500-query eval. First positive learned-retrieval result.
3. **Checkpoints 14-15** (exploration/rescue, Sparse Distributed Memory prototype): both negative at full scale.
4. **Checkpoint 16** (extended behavior metrics for checkpoint 13): RIF's gain is primarily from reducing cross-topic retrieval (−1.6pp wrong_family); sibling confusion and stale-fact rate unchanged.
5. **Checkpoint 17** (LLM enrichment layer): write-time structured extraction via Haiku. On covered queries: +8.3pp NDCG over checkpoint 13 — largest single-lever gain in the journey. Overall diluted by partial coverage; scaling to full coverage is pending.

Full journey in [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

## License

MIT.
