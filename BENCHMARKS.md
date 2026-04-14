# Benchmarks

Dataset: LongMemEval S variant (199,509 conversation turns, 500 questions)
Eval: 200-query random sample, seed=0
Date: 2026-04-12

## Retrieval quality

All systems evaluated on the same 200 queries, same qrels, returning exactly top-10 results.

| System | NDCG@10 | Recall@10 | What it tests |
|--------|---------|-----------|---------------|
| Vector only (MiniLM top-10) | 0.1376 | 0.2173 | Bi-encoder baseline |
| BM25 only (top-10) | 0.2420 | 0.3264 | Sparse keyword baseline |
| Hybrid BM25+vector RRF (memsearch style) | 0.2171 | 0.3334 | Rank fusion without reranker |
| Vector + cross-encoder rerank (k=30) | 0.2425 | 0.2892 | Dense + reranker |
| **Hybrid + cross-encoder rerank (k=30)** | **0.3680** | **0.4694** | **Best: hybrid + reranker** |

## Integrity checks (verified)

- Check 1: All 5 systems evaluated on identical 200 queries
- Check 2: Same qrels dict used by all systems
- Check 3: All systems return exactly 10 results to the user
- Check 4: The best result (0.3680) is BM25 + vector + cross-encoder reranking. No GC mechanism contributes to this number. It is a standard IR technique.
- Check 5: RRF + cross-encoder rerank gives 0.3680 (identical to our hybrid + xenc). The advantage is the cross-encoder, not the candidate merging method.
- Check 6: No parameters were tuned on the eval set.

## What the GC mechanism does NOT improve

The GC routing index (mutable query-to-entry associations with tier lifecycle) was tested on top of the hybrid+xenc baseline. After 2000 steps of query-driven learning:

- Hot NDCG: 0.3273 (vs 0.3680 static baseline = **-11%**)
- The routing index adds noise candidates that dilute the cross-encoder pool

Previous GC approaches also tested (embedding mutation, MLP adapter, segmentation, co-relevance graph): none improved retrieval quality over the static baseline. See [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

## What the GC mechanism DOES provide

- **Memory lifecycle**: entries promote from naive → gc → memory based on usage. Memory-tier entries are stable. Unused entries decay and get archived.
- **Deduplication**: cosine 0.95 dedup removes 4.6% of near-duplicate entries on LongMemEval, freeing retrieval slots (+6.5% NDCG when applied to the corpus before indexing).
- **Cost amortization** (theoretical): rescue cache can reduce cross-encoder calls for repeated query patterns by caching deep search results.

## RIF (Retrieval-Induced Forgetting)

First learned mechanism to improve retrieval quality on top of hybrid+xenc.

Dataset: LongMemEval S, 500-query full eval, 5000-step burn-in.

| System | NDCG@10 | Recall@30 |
|--------|---------|-----------|
| Baseline (no RIF) | 0.2960 | 0.4103 |
| Global RIF (original formula) | 0.2993 (+1.1%) | 0.4142 (+0.9%) |
| Global RIF (gap formula) | 0.3037 (+2.6%) | 0.4250 (+3.6%) |
| Clustered RIF 30 (original) | 0.3132 (+5.8%) | 0.4381 (+6.8%) |
| **Clustered RIF 30 + gap formula** | **0.3152 (+6.5%)** | **0.4494 (+9.5%)** |

Config: suppression_rate=0.1, reinforcement_rate=0.05, alpha=0.3, decay_lambda=0.005.

Mechanism: suppress entries that repeatedly compete but lose to the cross-encoder.
- **Clustered**: per-(entry, query_cluster) suppression prevents cross-topic interference.
- **Gap formula**: competition strength = max(0, xenc_rank - initial_rank) / pool × sigmoid(-xenc_score). Targets entries that dropped in rank AND were actively rejected.

Operates at candidate selection stage (before xenc), not scoring stage (where GC failed).

## How to reproduce

```bash
uv run python experiments/data_prep.py --dataset longmemeval
uv run python benchmarks/run_benchmark.py
uv run python benchmarks/run_rif_benchmark.py
```
