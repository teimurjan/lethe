# Architecture

## Retrieval pipeline

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

**Optional write-time LLM enrichment layer** (`src/lethe/enrichment.py`): before indexing, each memory can be processed by an LLM (default `claude-haiku-4-5`) to produce a gist, 3 anticipated queries, entities, and temporal markers. All fields index alongside the original text; cross-encoder still scores against original. Attacks the vocabulary-mismatch failure mode.

## RIF: technical details

The formula that made it work (checkpoint 13 of 17):

```
competition_strength = max(0, xenc_rank - initial_rank) / pool_size × sigmoid(-xenc_score)
suppression[eid, cluster] += learning_rate × competition_strength
score_adjusted = base_score - alpha × suppression[eid, cluster]
```

The rank-gap term only penalises entries that **dropped in rank** between the initial hybrid retrieval and the cross-encoder rerank AND were actively rejected (negative xenc score). Entries that just lost a close race aren't suppressed. That prevents the system from forgetting legitimately useful context because it happened to rank second once.

k-means runs once (30 clusters) on the bi-encoder query embedding at retrieval time; the cluster assignment is the "cue" coordinate. Based on Anderson's inhibition theory (1994) and the SAM competitive-sampling model (Raaijmakers & Shiffrin, 1981).

## Storage layers

| Layer | File | Purpose |
|-------|------|---------|
| DuckDB | `lethe.duckdb` | Entries, suppression state, rescue cache, stats. Many project DBs can be `ATTACH`ed simultaneously for cross-project search. |
| numpy + FAISS | `embeddings.npz`, `faiss.index` | Vector storage |
| BM25 | In-memory, rebuilt on startup | Sparse keyword index |

## Entry lifecycle (germinal-center inspired)

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

## Deduplication (on add)

1. **Exact**: SHA-256 content hash (free)
2. **Near-duplicate**: cosine similarity > 0.95 (keeps the longer entry)

## Global search across projects

Every `lethe index` registers the project in `~/.lethe/projects.json`. `lethe search --all` then ATTACHes each registered `lethe.duckdb` as a read-only schema and runs the full hybrid + RIF + cross-encoder pipeline across the union.

```bash
lethe search "mongodb pool" --all --top-k 5
lethe projects list
lethe projects prune
```

DuckDB has no cap on attached databases, so the 10/125 ceiling in SQLite's `ATTACH DATABASE` doesn't apply.
