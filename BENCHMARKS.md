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

## LLM enrichment layer (checkpoint 17)

Write-time structured extraction via `claude-haiku-4-5`: gist, 3 anticipated queries, entities, temporal markers. Indexed alongside original text in BM25/vector. Cross-encoder still scores original text.

Dataset: LongMemEval S, 500-query full eval, 5000-step burn-in per RIF arm.
Coverage: 975 entries enriched (15% of queries have at least one enriched qrels entry).

**Overall (all 500 queries, diluted by 85% uncovered):**

| Metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.208 | 0.218 | 0.220 | +0.002 |
| ndcg@10 | 0.293 | 0.312 | 0.324 | +0.012 |
| wrong_family | 0.688 | 0.676 | 0.674 | −0.002 |

**Covered bucket (75 queries — where the mechanism applies):**

| Metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.267 | 0.280 | **0.333** | **+0.053 (+19% rel)** |
| ndcg@10 | 0.350 | 0.390 | **0.473** | **+0.083 (+21% rel)** |
| wrong_family | 0.720 | 0.707 | **0.653** | **−0.053** |

The covered-bucket NDCG gain (+8.3pp over RIF) is the largest single-lever improvement in the research journey — **3.6× larger than RIF's own +2.3pp contribution**. Mechanism: anticipated_queries reduces the vocabulary-mismatch problem, so correct sessions enter the candidate pool more reliably.

Within-session sibling confusion (0.013) and stale-fact rate are unchanged — the enrichment lever doesn't reach those failure modes.

Cost: ~$1.60 for 1000 entries on Haiku. Projected ~$16 for full coverage of answer-relevant sessions.

## Benchmark methodology note

All numbers on this page measure **NDCG@10 over turn-level retrieval on the full 199,509-turn LongMemEval S corpus** — i.e. needle-in-haystack among 200k candidates.

Other memory-tool benchmarks commonly report numbers on **per-query fresh DBs** of ~50 sessions, at **session granularity**, with **recall@5**. That's a ~2000× easier task (10% random baseline vs 0.005%), and some implementations additionally leak ground truth via the `has_answer` annotation at indexing time. Published numbers in the 95-99% range on that methodology are state-of-the-art *for that methodology* — they are not directly comparable to the numbers here.

## How to reproduce

```bash
uv run python experiments/data_prep.py --dataset longmemeval
uv run python benchmarks/run_benchmark.py
uv run python benchmarks/run_rif_benchmark.py

# Enrichment layer
export ANTHROPIC_API_KEY=sk-ant-...
uv run python experiments/enrich_longmemeval.py
uv run python benchmarks/run_rif_enriched.py
```
