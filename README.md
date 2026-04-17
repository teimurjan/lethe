# lethe

<div align="center">
    <img src="./assets/logo.png" width="300" height="300" />
</div>

> *Λήθη: the ancient Greek personification of forgetfulness, and one of the five rivers of the underworld.*

A memory store for LLM agents. Hybrid BM25 + dense retrieval, cross-encoder reranking, **clustered retrieval-induced forgetting (RIF)**, and an optional LLM enrichment layer at write time.

<div align="center">
    <img src="./assets/demo.gif" />
</div>

> Lethe full stack vs basic hybrid RRF on the 199,509-turn LongMemEval corpus. NDCG@10 over 5,000 queries.

## Read this before the benchmark numbers

Most memory-tool READMEs advertise NDCG or recall in the 95 to 99% range. Those numbers come from evaluating on a per-query fresh database of ~50 sessions, at session granularity, with recall@5. That's a roughly 2000× easier task than "find this specific turn among 200,000". Some implementations additionally leak ground-truth annotations into the index.

lethe's numbers are reported on the full 199,509-turn LongMemEval S corpus, **turn-level retrieval, NDCG@10, no leakage**. Random baseline is 0.005%. Vector-only search gets 0.14. BM25 alone gets 0.24. When other tools are run on *this* setup, they land in the same 0.1 to 0.4 range. The 99% numbers don't translate.

No 99%. No cherry-picking. These are the honest numbers on a hard benchmark:

| Stage | NDCG@10 | notes |
|---|---|---|
| Hybrid BM25 + vector (RRF) | 0.217 | basic retrieval (most popular) |
| + cross-encoder reranking | 0.293 | +35% from semantic reranking |
| + clustered+gap RIF (checkpoint 13) | **0.312** | +6.5% from retrieval-induced forgetting |
| + LLM enrichment, on covered queries | **0.473** | +21% on the 75 queries where the answer turn was Haiku-enriched |

Enrichment only helps where it's been applied; the full-corpus number is diluted by the uncovered 425 queries. Full methodology in [BENCHMARKS.md](BENCHMARKS.md). The research path, including 10 failed approaches before anything worked, is in [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

## What's new here: retrieval-induced forgetting

RIF is a well-studied phenomenon in cognitive psychology (Anderson, 1994). In plain words:

> *When you retrieve one thing, related things that could have been retrieved but weren't get quietly suppressed. Ask someone to name a fruit; they say "apple". A few minutes later, "pear" and "orange" are measurably harder for them to recall, even though nothing asked about them. Remembering is also an act of forgetting.*

lethe brings this mechanism to vector memory:

- Every retrieval surfaces a candidate pool (BM25 ∪ vector top-k).
- The cross-encoder picks the winners.
- Entries that reached the pool but lost don't disappear. They accumulate a per-cluster **suppression score**.
- On future retrievals in the same query cluster ("travel-like queries", "debugging-like queries"), those entries get their scores penalised *before* the cross-encoder sees them, freeing slots for entries that were previously crowded out.

The key detail: suppression is **cue-dependent, not global**. An entry suppressed for travel queries stays fully available for food queries. Global suppression was ~5× weaker in our benchmarks. The clustering is what made it work.

## Why this matters: a memory that evolves

Most memory stores are static caches. You put strings in; you retrieve them by similarity. The retrieval function never changes.

RIF makes retrieval behaviour **adapt over time without retraining any model**:

- Entries that keep losing to more relevant alternatives get quieter in their own cluster.
- Entries that keep winning get amplified via tier promotion (naive → gc → memory) and exemption from decay.
- The system learns "in the context of *this kind* of query, you usually mean *that* chunk, not *this* one" from its own retrieval history. No fine-tuning. No extra LLM calls. Just O(n_clusters × n_entries) bookkeeping.

Concrete example: you store both *"I prefer window seats on flights"* and *"my wife needs aisle seats"*. Travel queries keep picking your preference first. The system quietly penalises your wife's preference for travel-sounding queries, not globally, just in that cluster. Ask about your spouse's preferences and both entries remain equally findable.

Checkpoint 13 (clustered + rank-gap RIF) gives +6.5% NDCG and +9.5% recall@30 on LongMemEval over the baseline hybrid pipeline. As far as I can find, this is the first implementation of clustered RIF in a production retrieval system.

## Claude Code plugin

Lethe ships as a Claude Code plugin that persists memory across sessions per-project:

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

Hooks write daily markdown to `.lethe/memory/YYYY-MM-DD.md`, summarize each turn with `claude -p --model haiku` (no extra API key), reindex via the `lethe` CLI, and a `memory-recall` skill surfaces prior-session context on demand.

See [plugins/claude-code/README.md](plugins/claude-code/README.md) for hook behavior, CLI reference, and local-install instructions.

## Quick start

```python
from lethe import MemoryStore
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

**Optional write-time LLM enrichment layer** (`src/lethe/enrichment.py`): before indexing, each memory can be processed by an LLM (default `claude-haiku-4-5`) to produce a gist, 3 anticipated queries, entities, and temporal markers. All fields index alongside the original text; cross-encoder still scores against original. Attacks the vocabulary-mismatch failure mode.

### RIF: technical details

The formula that made it work (checkpoint 13 of 17):

```
competition_strength = max(0, xenc_rank - initial_rank) / pool_size × sigmoid(-xenc_score)
suppression[eid, cluster] += learning_rate × competition_strength
score_adjusted = base_score - alpha × suppression[eid, cluster]
```

The rank-gap term only penalises entries that **dropped in rank** between the initial hybrid retrieval and the cross-encoder rerank AND were actively rejected (negative xenc score). Entries that just lost a close race aren't suppressed. That prevents the system from forgetting legitimately useful context because it happened to rank second once.

k-means runs once (30 clusters) on the bi-encoder query embedding at retrieval time; the cluster assignment is the "cue" coordinate. Based on Anderson's inhibition theory (1994) and the SAM competitive-sampling model (Raaijmakers & Shiffrin, 1981).

### Three storage layers

| Layer | File | Purpose |
|-------|------|---------|
| DuckDB | `lethe.duckdb` | Entries, suppression state, rescue cache, stats. Many project DBs can be `ATTACH`ed simultaneously for cross-project search. |
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

From PyPI (the distribution name is `lethe-memory`; the import name is still `lethe`):

```bash
pip install lethe-memory
# or
uv pip install lethe-memory
```

As the `lethe` CLI on PATH without a venv:

```bash
uv tool install lethe-memory
lethe --version
```

From source, for development:

```bash
git clone https://github.com/teimurjan/lethe && cd lethe
uv venv --python 3.12 && uv pip install -e .
```

As a Claude Code plugin (see [plugin README](plugins/claude-code/README.md)):

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

## Global search across projects

Every `lethe index` registers the project in `~/.lethe/projects.json`. `lethe search --all` then ATTACHes each registered `lethe.duckdb` as a read-only schema and runs the full hybrid + RIF + cross-encoder pipeline across the union.

```bash
# In each project you care about:
lethe index

# Search across all registered projects:
lethe search "mongodb pool" --all --top-k 5
# [p_site_fix_pentest_8a99] [+4.63] 06e6...  - MongoDB connection pool issue at 100% rollout…
# [p_lethe_77be]            [+3.41] 4670...  - lethe search returning real hybrid results…

# Restrict to a subset:
lethe search "auth flow" --projects p_site_fix_pentest_8a99,p_api_server_12ab

# Inspect / manage the registry:
lethe projects list
lethe projects add /path/to/other-project
lethe projects remove <slug-or-path>
lethe projects prune   # drop entries whose roots no longer exist
```

Opt out of auto-registration in three ways:
- Per invocation: `lethe index --no-register`
- Per project: `lethe config set auto_register false`
- Globally: `export LETHE_DISABLE_GLOBAL_REGISTRY=1`

The registry only stores absolute project paths, never contents. Cross-project search is read-only; it does not mutate RIF state. Local `lethe search` continues to learn as normal.

DuckDB has no cap on attached databases, so the 10/125 ceiling in SQLite's `ATTACH DATABASE` doesn't apply. Real-world bottleneck before you'd feel it is the per-project BM25 rebuild, which scales linearly.

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

See [BENCHMARKS.md](BENCHMARKS.md) for results tables and methodology details.

## Project structure

```
src/lethe/             # Production library (162 tests, ~95% coverage)
├── memory_store.py
├── markdown_store.py  # Markdown chunker for .lethe/memory/*.md
├── cli.py             # `lethe` console script
├── union_store.py     # Cross-project read-only search via DuckDB ATTACH
├── db.py              # DuckDB persistence
├── vectors.py         # FAISS + BM25 index
├── reranker.py        # Cross-encoder + adaptive depth
├── rif.py             # Retrieval-induced forgetting (clustered + gap)
├── enrichment.py      # Optional LLM write-time enrichment (Anthropic SDK)
├── dedup.py           # Hash + cosine deduplication
└── entry.py           # MemoryEntry dataclass + Tier enum

plugins/claude-code/   # Claude Code plugin (hooks + memory-recall skill)
.claude-plugin/        # Marketplace manifest

benchmarks/            # Per-checkpoint benchmark scripts
├── run_*.py
├── _lib/              # Benchmark-only helpers (metrics, etc.)
└── results/           # Raw per-run output markdowns

scripts/               # Reproducibility utilities (dataset prep, enrichment runner)

research/              # Experimental / research code (not production)
├── gc_mutation/       # Germinal-center mutation thread (checkpoints 1-10)
└── sdm/               # Sparse Distributed Memory prototype (checkpoint 15)

tests/                 # Production unit tests
```

## Research background

This project started as an experiment porting the immune system's germinal-center mechanism to vector memory. 17 checkpoints so far:

1. **Checkpoints 1-10** (GC-mutation approaches): all failed to improve retrieval quality. Useful for lifecycle management, not retrieval.
2. **Checkpoints 11-13** (RIF): cognitive-science inspired. Clustered + gap-formula variant gives +6.5% NDCG, +9.5% recall@30 on the full 500-query eval. First positive learned-retrieval result.
3. **Checkpoints 14-15** (exploration/rescue, Sparse Distributed Memory prototype): both negative at full scale.
4. **Checkpoint 16** (extended behavior metrics for checkpoint 13): RIF's gain is primarily from reducing cross-topic retrieval (−1.6pp wrong_family); sibling confusion and stale-fact rate unchanged.
5. **Checkpoint 17** (LLM enrichment layer): write-time structured extraction via Haiku. On covered queries: +8.3pp NDCG over checkpoint 13, the largest single-lever gain in the journey. Overall diluted by partial coverage; scaling to full coverage is pending.

Full journey in [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

## License

MIT.
