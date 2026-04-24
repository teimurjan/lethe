# RIF Benchmark

Retrieval-Induced Forgetting: principled competitor suppression.

**Protocol**: Simulate 2000 queries to build suppression state, then evaluate on 500 queries.

**RIF config**: rate=0.1, reinforce=0.05, alpha=0.3, decay=0.005

Date: 2026-04-24

| Metric | Static | RIF | Δ |
|--------|--------|-----|---|
| NDCG@10 | 0.3311 | 0.3369 | +1.8% |
| Recall@30 | 0.4739 | 0.4741 | +0.0% |

Suppression: 10864 entries suppressed, 0 heavily (>0.5)

## Mechanism

After each retrieval, entries that competed (made it into the candidate pool)
but lost (ranked low by cross-encoder) accumulate suppression. Suppression
is applied as a penalty to hybrid search scores before cross-encoder reranking,
pushing chronic distractors down and allowing new candidates in.
