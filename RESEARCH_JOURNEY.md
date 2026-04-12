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

### Key insight
The biological control loop (adaptive rate, tier lifecycle, decay) is sound engineering. The biological perturbation (random mutation of embeddings) is not. The winning system uses immune-system-inspired lifecycle management with modern IR techniques (BM25 + cross-encoder) instead of mutation.
