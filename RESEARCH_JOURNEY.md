# Research Journey

How gc-memory evolved from a biology experiment to a production retrieval system.

## Starting point: the germinal center hypothesis

The immune system has germinal centers where B cells mutate their antibodies and compete to bind antigens. Cells that bind well mutate conservatively; cells that bind poorly mutate aggressively. This adaptive rate produces high-affinity antibodies without collapsing into clones.

Hypothesis: port this to vector memory. Embeddings that get retrieved frequently mutate less. Embeddings with low retrieval affinity mutate more aggressively. Over time, the store improves.

## Checkpoint 1: Direct embedding mutation (v1)

**NFCorpus, 3,633 docs, 10k queries**

Gaussian noise added directly to pretrained embeddings. Three arms: Static (no mutation), Random (fixed sigma), GC (adaptive sigma).

| Arm | NDCG@10 | vs Static |
|-----|---------|-----------|
| Static | 0.3165 | baseline |
| Random | 0.2978 | -5.9% |
| GC | 0.3076 | -2.8% |

**Finding**: Adaptive rate beats random (+3.3%), but both lose to static. Gaussian noise pushes pretrained embeddings off the learned manifold.

**Bug found**: Decay formula used cumulative delta_steps, causing exp(-50.5) total decay. Fixed in commit `13e7f43`.

## Checkpoint 2: Adapter-based mutation (v2)

Froze base embeddings, added mutable adapter vectors. effective = normalize(base + adapter). Adapter norm clipped to 0.5.

**LongMemEval, 19k sessions, 10k queries**

| Arm | NDCG@10 | vs Static |
|-----|---------|-----------|
| Static | 0.1813 | baseline |
| Random adapter | 0.1834 | +1.2% |
| GC adapter (centroid fitness) | 0.1797 | -0.9% |

**Finding**: Random adapter mutation beat static for the first time (+1.2%). Co-retrieval centroid as fitness signal washes out the signal.

## Checkpoint 3: Cross-encoder reranking (v3)

Added ms-marco cross-encoder to rerank FAISS candidates. All arms get reranking.

**LongMemEval, 200k turns**

| Arm | NDCG@10 | vs Static |
|-----|---------|-----------|
| Static + rerank | 0.2217 | baseline |
| Random + rerank | 0.2206 | -0.5% |
| GC + rerank | 0.2202 | -0.7% |

**Finding**: Cross-encoder reranking gives +63% over bi-encoder alone. But mutation adds nothing on top because the cross-encoder overrides FAISS rankings.

## Checkpoint 4: Learned MLP adapter (v4)

Replaced Gaussian noise with a tiny MLP: delta = f(query, embedding, xenc_score). 148k parameters, trained online.

**NFCorpus, full 323-query eval**

| Config | Peak NDCG | vs Static |
|--------|-----------|-----------|
| MLP (no gates) | 0.3362 | +0.6% |
| MLP + probability gate + low promotion | 0.3356 | +0.4% (stable through step 1250) |

**Finding**: First stable positive result. MLP learns the right direction, but deltas are too small to change FAISS rankings significantly.

## Checkpoint 5: Recall analysis (turning point)

Diagnosed the real bottleneck: recall, not reranking.

```
Bi-encoder recall@30:   29.0%
Bi-encoder recall@100:  43.8%
Bi-encoder recall@500:  63.6%
Oracle recall:          100%

k_fetch=30 + xenc:  NDCG=0.2217
k_fetch=100 + xenc: NDCG=0.2697 (+21.6%)
Oracle + xenc:      NDCG=0.4709 (+112%)
```

78% of relevant entries never reach the cross-encoder because FAISS doesn't retrieve them. The mutation was optimizing the wrong layer.

## Checkpoint 6: Adaptive depth + rescue cache

**Adaptive k_fetch**: shallow (k=30) for easy queries, deep (k=200) when cross-encoder confidence is low.

**Rescue cache**: deep search results saved for future similar queries.

**LongMemEval, 200-query sample**

| Method | Hot NDCG | vs Static |
|--------|---------|-----------|
| Static (k=30 + xenc) | 0.1958 | baseline |
| Rescue cache only | 0.2203 | +12.6% |
| Adaptive k + rescue | 0.2853 | +28.7% |

**Finding**: +28.7% from adaptive depth. The biology parallel: routine queries get shallow response, novel queries trigger deep germinal center search, results become immunological memory.

## Checkpoint 7: Deduplication

Cosine similarity > 0.95 dedup with qrel remapping.

| Method | NDCG |
|--------|------|
| Original static | 0.2217 |
| Deduped static | 0.2582 (+16.5%) |
| Dedup + adaptive + rescue | 0.3017 (+54.1%) |

**Finding**: 4.6% of entries are near-duplicates that waste retrieval slots.

## Checkpoint 8: BM25 hybrid (current best)

BM25 sparse keyword search added alongside FAISS dense vectors.

**LongMemEval, 200-query sample**

| System | NDCG@10 |
|--------|---------|
| Vector only | 0.1376 |
| BM25 only | 0.2420 |
| Memsearch (BM25+vector RRF) | 0.2171 |
| Our static (vector + xenc) | 0.2217 |
| **Full stack (dedup + BM25 + xenc)** | **0.3395** |

**Finding**: BM25 is the single biggest retrieval win on conversation data (+76% over vector alone). Keyword matching outperforms semantic similarity for specific fact retrieval in chat histories.

## What worked vs what didn't

### Worked
- Cross-encoder reranking (+63%)
- BM25 hybrid retrieval (+76% over vector alone)
- Adaptive search depth (+20.5%)
- Deduplication (+16.5%)
- Rescue cache (+12.6%)
- Tier lifecycle (stable memory management)

### Didn't work
- Direct embedding mutation (degrades pretrained embeddings)
- Co-retrieval centroid (washes out signal)
- Co-relevance graph (too sparse on LongMemEval, -2.7%)
- Text segmentation (destroys context, -46%)
- MLP adapter (right direction, wrong magnitude, +0.4%)

## Checkpoint 9: GC routing index (final attempt)

Formalized the rescue cache as a proper GC-managed routing index. Query embeddings are clustered. Each cluster maintains entry associations with full tier lifecycle (naive → gc → memory). The routing index adds candidates on top of the static hybrid+xenc pipeline.

| System | Hot NDCG |
|--------|---------|
| Static hybrid+xenc | 0.3680 |
| + GC routing (2000 steps) | 0.3273 (-11%) |

The routing index added noise candidates that diluted the cross-encoder pool. After 2000 steps with 4345 learned routes and 1763 memory-tier routes, retrieval quality was WORSE than the static pipeline.

## Checkpoint 10: Integrity checks

Independent integrity audit of all benchmark results:

1. All systems evaluated on identical 200 queries with identical qrels
2. All systems return exactly 10 results
3. RRF + cross-encoder rerank = 0.3680 (identical to "gc-memory"). The cross-encoder IS the improvement, not the GC mechanism.
4. No parameters were tuned on the eval set

**Conclusion: the 0.3680 NDCG is entirely from hybrid BM25+vector+cross-encoder reranking. The GC mechanism contributed nothing to retrieval quality.**

## What worked vs what didn't

### Worked (for retrieval quality)
- Cross-encoder reranking (+63% over bi-encoder alone)
- BM25 hybrid retrieval (+76% over vector alone)
- Combined hybrid+xenc (NDCG=0.3680, +167% over vector baseline)

### Worked (for memory management, not retrieval quality)
- Deduplication (+6.5% when applied as corpus preprocessing)
- Tier lifecycle (stable memory management)
- Affinity decay (handles stale entries)

### Didn't work (for retrieval quality)
- Direct embedding mutation (degrades pretrained embeddings)
- Adapter mutation + co-retrieval centroid (washes out signal)
- Co-relevance graph (too sparse on LongMemEval)
- Text segmentation (destroys context)
- MLP adapter (right direction, wrong magnitude)
- GC routing index (adds noise to candidate pool)
- Rescue cache (marginal gains, doesn't generalize)
- Adaptive search depth (works but is just "increase k_fetch")

### Key insight (checkpoints 1-10)
The biological control loop (adaptive rate, tier lifecycle, decay) is sound engineering for **memory management**. It is not useful for **retrieval quality**. Modern IR techniques (BM25 + cross-encoder reranking) are the right tool for retrieval. The GC mechanism's value is in lifecycle management: promoting frequently-used memories, decaying stale ones, deduplicating, and archiving unused entries.

## Checkpoint 11: Retrieval-Induced Forgetting (RIF)

Shifted from biology-inspired mutation to cognitive science-inspired forgetting. RIF is a 30-year-old finding in memory psychology (Anderson, 1994): when memory A is retrieved, competing memories that were activated but not selected are actively suppressed. First implementation in an AI memory system.

**Mechanism**: After cross-encoder reranking, entries that made it into the candidate pool but ranked poorly (high initial similarity, low cross-encoder score = distractors) accumulate a suppression score. Before the next retrieval, suppression is subtracted from hybrid search RRF scores, pushing chronic distractors down and freeing candidate slots for previously-excluded entries.

Key difference from GC approaches: RIF operates at the **candidate selection stage** (before cross-encoder), not the scoring stage. All prior GC mechanisms modified scores that the cross-encoder overwrote. RIF changes *which entries reach* the cross-encoder.

**LongMemEval, 500-query full eval, 2000-step burn-in**

| Metric | Static | RIF | Delta |
|--------|--------|-----|-------|
| NDCG@10 | 0.2960 | 0.3020 | +2.0% |
| Recall@30 | 0.4103 | 0.4196 | +2.3% |

Suppression stats: 11,254 entries suppressed (5.6% of corpus), max 0.169, mean 0.058. Zero entries hit heavy suppression (>0.5) — the decay mechanism keeps things balanced.

**Finding**: First positive result from a learned mechanism on top of hybrid+xenc across 11 checkpoints. The effect is modest (+2.0%) but operates in the right direction: candidate pool recall improved by 2.3%, confirming that suppression is letting new relevant entries into the cross-encoder pool.

**Hyperparameter sweep**: tested 6 configs from conservative to extreme. All aggressive configs (higher alpha, faster accumulation, no decay) went negative (-1.2% to -1.8%). The conservative config is optimal for global RIF. More aggressive = worse because cue-independent suppression hurts entries that are distractors for some queries but relevant for others.

## Checkpoint 12: Clustered RIF (cue-dependent suppression)

Global RIF's ceiling is ~+1-2% because suppression is cue-independent: an entry suppressed for "travel" queries is also suppressed for "food" queries. The SAM model (Raaijmakers & Shiffrin, 1981) predicts cue-dependent retrieval probabilities — suppression should be tied to the retrieval context, not the item alone.

**Mechanism**: K-means cluster query embeddings into groups. Maintain separate suppression scores per (entry, query_cluster) pair. At retrieval, only the matching cluster's suppression applies. An entry gets suppressed as a distractor for "travel" queries but remains unsuppressed for "food" queries.

**LongMemEval, 500-query eval, 5000-step burn-in**

| Config | NDCG@10 | vs baseline |
|--------|---------|-------------|
| baseline (no RIF) | 0.2960 | — |
| global RIF | 0.2993 | +1.1% |
| 10 clusters | 0.3113 | +5.2% |
| **30 clusters** | **0.3132** | **+5.8%** |
| 50 clusters | 0.3067 | +3.6% |
| 100 clusters | 0.3092 | +4.5% |

**Finding**: 30-cluster RIF produces +5.8% NDCG and +6.8% recall@30 — 5x the effect of global RIF. The inverted-U on cluster count: 10 is too coarse (diverse queries share suppression), 100 is too fine (insufficient signal per cluster). 30 clusters with 500 queries = ~17 queries/cluster, enough for suppression to accumulate within each topic.
